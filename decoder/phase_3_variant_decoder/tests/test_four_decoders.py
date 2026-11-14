import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace,ModuleType
import unittest
from unittest.mock import Mock,patch

import h5py
import numpy as np
import pandas as pd
import torch

from phase_3_variant_decoder.compare_four_decoders import main,fit_network,predict_saved,resolve_device
from phase_3_variant_decoder.factorial_models import ARMS,PairDecisionModel,binary_targets,parallel_decisions,load_model
from phase_3_variant_decoder.factorial_data import prepare_inputs,ontology_terms
from phase_3_variant_decoder.factorial_reports import OVERVIEW,publish,create_workspace


def fixture(root,missing=False):
    rng=np.random.default_rng(18);source=root/'multiclass';source.mkdir()
    sequences={};all_rows=[]
    for split in ('train','val','test'):
        rows=[]
        for suffix in ('a','b'):
            pid=f'{split}_{suffix}';sequences[pid]='ACDE'
            for snp,label in (('A1G','LOF'),('C2V','GOF'),('D3A','NEUTRAL')):
                rows.append(dict(protein_id=pid,snp=snp,label=label,matched_go_id='GO:0000001' if suffix=='a' else 'GO:0000002',
                    augmented=False,is_combinatorial=False))
        rows.append(dict(protein_id=f'{split}_a',snp='E4A',label='NEUTRAL',matched_go_id='GO:0000001',
                         augmented=True,is_combinatorial=False))
        pd.DataFrame(rows).to_csv(source/f'hard_{split}.tsv',sep='\t',index=False)
        all_rows.extend(rows)
    (root/'seq.json').write_text(json.dumps(sequences))
    np.savez(root/'go.npz',go_ids=np.array(['GO:0000002','GO:0000001']),embeddings=rng.normal(size=(2,5)).astype('float32'))
    (root/'go.obo').write_text('[Term]\nid: GO:0000001\nnamespace: molecular_function\nalt_id: GO:0000011\n\n'
                              '[Term]\nid: GO:0000002\nnamespace: biological_process\n')
    with h5py.File(root/'global.h5','w') as g,h5py.File(root/'local.h5','w') as l:
        for pid in sequences:
            g[pid]=rng.normal(size=1280).astype('float32')
            for pos in range(1,5):l[f'{pid}_{pos}']=rng.normal(size=1280).astype('float32')
        for row in all_rows:
            key=row['protein_id']+'_'+row['snp']
            if missing and key=='train_a_E4A':continue
            g[key]=rng.normal(size=1280).astype('float32');l[key]=rng.normal(size=1280).astype('float32')
    return ['--dataset_dir',str(root),'--sequences_json',str(root/'seq.json'),'--go_embeddings',str(root/'go.npz'),
        '--global_h5',str(root/'global.h5'),'--local_h5',str(root/'local.h5'),'--go_obo',str(root/'go.obo'),
        '--family','hard','--device','cpu','--epochs','2','--early_stop_patience','2','--batch_size','4',
        '--projection_dim','4','--hidden_dims','8','--dropout','0.2','--cpu_threads','1','--no_wandb','--quiet',
        '--out_dir',str(root/'runs')]


class FourDecoderTests(unittest.TestCase):
    def test_device_selection_on_mac_server_and_cpu(self):
        with patch('torch.backends.mps.is_available',return_value=True),patch('torch.cuda.is_available',return_value=False):
            self.assertEqual(str(resolve_device()),'mps')
            self.assertEqual(str(resolve_device('cpu')),'cpu')
            with self.assertRaisesRegex(RuntimeError,'CUDA'):resolve_device('cuda:6')
        with patch('torch.backends.mps.is_available',return_value=False),patch('torch.cuda.is_available',return_value=True),patch('torch.cuda.device_count',return_value=8):
            self.assertEqual(resolve_device().type,'cuda')
            self.assertEqual(str(resolve_device('cuda')),'cuda:0')
            self.assertEqual(str(resolve_device('6')),'cuda:6')
            with self.assertRaises(RuntimeError):resolve_device('cuda:8')
        with patch('torch.backends.mps.is_available',return_value=False),patch('torch.cuda.is_available',return_value=False):
            self.assertEqual(str(resolve_device()),'cpu')
            with self.assertRaises(RuntimeError):resolve_device('mps')
        with self.assertRaises(ValueError):resolve_device('cuda:-1')

    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def test_parallel_targets_and_conflict_decision_are_explicit(self):
        labels=torch.tensor([0,1,2])
        self.assertEqual(binary_targets(labels,0).tolist(),[1,0,0])
        self.assertEqual(binary_targets(labels,1).tolist(),[0,1,0])
        logits=torch.tensor([[-1.,-2.],[2.,-1.],[-1.,2.],[3.,4.],[0.,0.],[2.,2.]])
        pred,scores=parallel_decisions(logits)
        self.assertEqual(pred.tolist(),[2,0,1,1,2,0])
        torch.testing.assert_close(scores.sum(1),torch.ones(6))

    def test_shared_scorer_has_zero_delta_for_identical_pairs_even_with_dropout(self):
        model=PairDecisionModel('score_difference',7,5,3,4,[8],.6).train()
        wt,go=torch.randn(9,7),torch.randn(9,5)
        self.assertTrue(torch.equal(model.comparison(wt,wt,go)['delta'],torch.zeros(9)))
        model.eval();var=torch.randn(9,7)
        torch.testing.assert_close(model.comparison(wt,var,go)['delta'],-model.comparison(var,wt,go)['delta'])
        self.assertEqual(model(wt,var,go).shape,(9,3))

    def test_score_mechanism_preserves_existing_cascade_scalar_comparison(self):
        from phase_3_variant_decoder.cascade_model import PairScorer
        from phase_3_variant_decoder.factorial_models import SharedPairScorer
        old=PairScorer(4,5,[8],.2).eval()
        new=SharedPairScorer(4,5,[8],.2).eval();new.load_state_dict(old.state_dict())
        inputs=(torch.randn(3,4),torch.randn(3,4),torch.randn(3,5))
        for key,value in old(*inputs).items():torch.testing.assert_close(value,new(*inputs)[key])

    def test_parallel_networks_share_no_weights_and_only_selected_network_updates(self):
        a=PairDecisionModel('joint_concat',7,5,1,4,[8],0)
        b=PairDecisionModel('joint_concat',7,5,1,4,[8],0)
        self.assertFalse({p.data_ptr() for p in a.parameters()}&{p.data_ptr() for p in b.parameters()})
        before={k:v.clone() for k,v in b.state_dict().items()}
        optimizer=torch.optim.Adam(a.parameters(),lr=.01)
        a(torch.randn(3,7),torch.randn(3,7),torch.randn(3,5)).sum().backward();optimizer.step()
        for k,v in b.state_dict().items():torch.testing.assert_close(v,before[k])

    def test_three_class_concat_preserves_previous_direct_mlp(self):
        from phase_3_variant_decoder.direct_pair_decoder import DirectPairDecoder
        old=DirectPairDecoder('concat',7,5,4,[8],.2).eval()
        new=PairDecisionModel('joint_concat',7,5,3,4,[8],.2).eval()
        new.load_state_dict(old.state_dict())
        inputs=(torch.randn(4,7),torch.randn(4,7),torch.randn(4,5))
        torch.testing.assert_close(old(*inputs),new(*inputs))

    def test_mf_alias_uses_namespace_without_changing_go_lookup_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root)
            terms=ontology_terms(root/'go.obo')
            self.assertEqual(terms['GO:0000011'],('molecular_function',False))

    def test_missing_inputs_fail_before_training_or_wandb_and_subset_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=fixture(root,missing=True)
            with patch('phase_3_variant_decoder.factorial_reports.start_wandb') as wb,contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError,'missing/invalid inputs'):main(args)
                wb.assert_not_called()
            self.assertFalse(list((root/'runs').rglob('*.pt')))
            with contextlib.redirect_stdout(io.StringIO()):report=main(args+['--allow_missing_embeddings','--preflight_only'])
            self.assertEqual(report['data']['hard']['split_rows'],{'train':3,'val':3,'test':3})
            self.assertEqual(report['data']['hard']['rows_with_missing_or_invalid_input'],1)
            self.assertTrue(report['preflight_only'])

    def test_mf_requires_explicit_ontology(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=fixture(root)
            i=args.index('--go_obo');del args[i:i+2]
            with contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(ValueError,'require --go_obo'):main(args)

    def test_mf_unknown_terms_are_not_silently_classified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=fixture(root)
            source=root/'multiclass/hard_train.tsv'
            frame=pd.read_csv(source,sep='\t');frame.loc[0,'matched_go_id']='GO:9999999'
            frame.to_csv(source,sep='\t',index=False)
            with contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(ValueError,'missing from ontology'):main(args)
            self.assertFalse(list((root/'runs').rglob('*.pt')))

    def test_training_checkpoint_selection_does_not_require_test_view(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root)
            args=SimpleNamespace(dataset_dir=str(root),embedding='seq_phase1',input_mode='both',
                sequences_json=str(root/'seq.json'),global_h5=str(root/'global.h5'),local_h5=str(root/'local.h5'),
                go_embeddings=str(root/'go.npz'),max_seq_len=1022,drop_contradictory=False,exclude_combinatorial=False,
                aspect='MF',go_obo=str(root/'go.obo'),unknown_go='error',allow_missing_embeddings=False,
                projection_dim=4,hidden_dims=[8],dropout=.1,lr=.001,weight_decay=.0001,class_weighting='none',
                epochs=1,early_stop_patience=1,batch_size=4,quiet=True)
            with contextlib.redirect_stdout(io.StringIO()):tensors,norm,audit=prepare_inputs(args,'hard',root/'audit')
            from phase_3_variant_decoder.factorial_data import split_views
            views=split_views(tensors);del views['test']
            model,checkpoint,_,_=fit_network(args,tensors,views,norm,'score_difference',None,42,root/'fit',{})
            self.assertFalse(checkpoint['test_used_for_selection'])
            self.assertTrue(audit['feature_tensors_sha256']['WT'])

    def test_complete_four_arm_training_saved_inference_and_wandb_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=fixture(root)
            with contextlib.redirect_stdout(io.StringIO()):report=main(args)
            self.assertEqual(len(report['results']),4)
            self.assertEqual(report['data']['hard']['split_rows'],{'train':4,'val':3,'test':3})
            self.assertEqual({r['filtered_rows_sha256'] for r in report['results']},{report['data']['hard']['filtered_rows_sha256']})
            run_root=next((root/'runs').glob('four_decoders_*'))
            self.assertEqual(len(list(run_root.rglob('decoder.pt'))),6)
            for result in report['results']:
                paths=[run_root/p for p in result['checkpoints']]
                for path in paths:
                    _,checkpoint=load_model(path)
                    self.assertFalse(checkpoint['test_used_for_selection'])
                    self.assertEqual(int(checkpoint['training_class_counts'].sum()),4)
                # Reconstruct one held-out row and verify checkpoint-only inference.
                with h5py.File(root/'global.h5') as g,h5py.File(root/'local.h5') as l:
                    wt=torch.tensor(np.concatenate((g['test_a'][:],l['test_a_1'][:]))).unsqueeze(0)
                    var=torch.tensor(np.concatenate((g['test_a_A1G'][:],l['test_a_A1G'][:]))).unsqueeze(0)
                from phase_3_variant_decoder.data.go_embeddings import GOEmbeddingTable
                go=GOEmbeddingTable(root/'go.npz').lookup(['GO:0000001'])
                output=predict_saved(paths,wt,var,go)
                frame=pd.read_csv(paths[0].parent.parent/'test_predictions.tsv',sep='\t')
                self.assertEqual(['LOF','GOF','NEUTRAL'][output['prediction'].item()],frame.prediction.iloc[0])
            fake=SimpleNamespace(id='fixture',url='offline:test',summary={},log=Mock(),log_artifact=Mock(),define_metric=Mock(),finish=Mock())
            wb=SimpleNamespace(Table=lambda **kw:kw,Image=lambda path:path,Artifact=lambda *a,**k:SimpleNamespace(add_file=Mock()))
            publish(fake,wb,report,run_root)
            keys=set(fake.log.call_args.args[0])
            for section in (OVERVIEW,*[s['section'] for s in ARMS.values()]):
                self.assertTrue(any(k.startswith(section+'/') for k in keys))
            self.assertEqual(len([k for k in fake.summary if k.endswith('/test/macro_f1')]),4)
            self.assertTrue(fake.log_artifact.called)
            # Uploading saved results restores sections/histories without training.
            with patch('phase_3_variant_decoder.factorial_reports.start_wandb',return_value=(fake,wb)), \
                 patch('phase_3_variant_decoder.compare_four_decoders.fit_network',side_effect=AssertionError('must not retrain')), \
                 contextlib.redirect_stdout(io.StringIO()):
                url=main(['--upload_existing',str(run_root/'four_decoder_comparison.json'),'--wandb_mode','offline'])
            self.assertEqual(url,'offline:test');fake.finish.assert_called()

    def test_ordered_workspace_has_overview_and_four_separate_sections(self):
        import sys
        ws=ModuleType('wandb_workspaces.workspaces');wr=ModuleType('wandb_workspaces.reports.v2')
        ws.Section=lambda **kw:kw;ws.RunsetSettings=lambda **kw:kw
        captured={}
        def workspace(**kw):
            captured.update(kw);return SimpleNamespace(save=lambda:SimpleNamespace(url='saved-view'))
        ws.Workspace=workspace;wr.MediaBrowser=lambda **kw:kw
        modules={'wandb_workspaces':ModuleType('wandb_workspaces'),'wandb_workspaces.workspaces':ws,
                 'wandb_workspaces.reports':ModuleType('wandb_workspaces.reports'),'wandb_workspaces.reports.v2':wr}
        run=SimpleNamespace(id='fixture',entity='e',project='p')
        with patch.dict(sys.modules,modules):self.assertEqual(create_workspace(run,{}),'saved-view')
        self.assertEqual([s['name'] for s in captured['sections']],[OVERVIEW,*[s['section'] for s in ARMS.values()]])


if __name__=='__main__':unittest.main()

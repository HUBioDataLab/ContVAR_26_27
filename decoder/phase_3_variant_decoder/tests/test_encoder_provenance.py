import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import pandas as pd

from phase_3_variant_decoder.trace_encoder_provenance import discover, inspect_evidence, run, source_ids


class EncoderProvenanceTests(unittest.TestCase):
    def test_code_defaults_are_not_runtime_proof(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source=root/'mapper.py'
            source.write_text('output = "uniprot_dict.json"\nmerged = merge(a,b,how="inner")\n')
            rows,_=inspect_evidence(source,10000)
            self.assertTrue(rows)
            self.assertTrue(all(x['evidence_kind']=='source_code_not_execution_proof' for x in rows))
            log=root/'run.log'
            log.write_text('[*] Loading UniProt data from: /data/raw/with_sequences.tsv\n')
            rows,_=inspect_evidence(log,10000)
            self.assertEqual(rows[0]['evidence_kind'],'runtime_loading_message')
            self.assertFalse(rows[0]['proves_target_h5_provenance'])

    def test_wandb_arguments_keep_paths_with_spaces(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'wandb-metadata.json'
            path.write_text(json.dumps({'program':'build_pair_embeddings.py','args':['--sequences_json','/data/my sequences.json','--backbone','/data/phase1.pt']}))
            rows,_=inspect_evidence(path,10000)
            result={x['parameter']:x['value'] for x in rows}
            self.assertEqual(result['sequences_json'],'/data/my sequences.json')
            self.assertEqual(result['program'],'build_pair_embeddings.py')
            self.assertTrue(all(x['evidence_kind']=='recorded_run_metadata' for x in rows))

    def test_metadata_table_does_not_become_a_sequence_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'uniprot.tsv'
            path.write_text('Entry\tLength\nP2\t42\n')
            ids,candidates,info=source_ids(path,{'P2'},False)
            self.assertEqual(ids,{'P2'})
            self.assertFalse(candidates)
            self.assertEqual(info['kind'],'uniprot_metadata_only_no_sequence_column')

    def test_dms_mutants_are_not_recovered_as_wt_sequences(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'proteins 2.csv'
            path.write_text('File_Name,Sequence\nP12345_ASSAY_A1V_1,VAAA\nP12345_ASSAY_WT,AAAA\n')
            ids,candidates,info=source_ids(path,{'P12345'},False)
            self.assertEqual(ids,{'P12345'})
            self.assertEqual([x['sequence'] for x in candidates],['AAAA'])
            self.assertEqual(info['kind'],'dms_filename_sequence_table_wt_rows_only')

    def test_discovery_budget_is_not_consumed_by_mutant_fastas_or_first_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=root/'a';b=root/'b';a.mkdir();b.mkdir()
            for i in range(5): (a/f'ASSAY_P12345_A{i}V.fasta').write_text('>P12345\nAAAA\n')
            (a/'first.py').write_text('x=1\n');(b/'second.py').write_text('x=2\n')
            metadata,seq,_,warnings=discover([a,b],root/'out',2,1)
            self.assertEqual(len(metadata),2)
            self.assertEqual(seq,[])

    def test_inner_join_is_only_a_hypothesis_and_inputs_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            primary=root/'primary.json';primary.write_text(json.dumps({'P1':'AA'}))
            missing=root/'missing.tsv';missing.write_text('protein_id\nP2\nP3\n')
            raw=root/'uniprot_sequences.tsv';raw.write_text('Entry\tSequence\nP1\tAA\nP2\tGG\n')
            meta=root/'uniprot_metadata.tsv';meta.write_text('Entry\tLength\nP3\t40\n')
            goa=root/'goa.gaf';goa.write_text('!gaf-version: 2.2\nUniProtKB\tP1\tSYMBOL\t\tGO:0000001\n')
            sources=[primary,missing,raw,meta,goa];before={p:p.read_bytes() for p in sources}
            args=SimpleNamespace(out_dir=str(root/'out'),primary_sequences=str(primary),missing_ids=str(missing),
                search_root=[str(root)],config_file=[],log_file=[],history_file=[],uniprot_source=[str(raw),str(meta)],
                goa_source=[str(goa)],h5=[],max_depth=2,max_files_per_root=100,max_source_files=20,max_text_bytes=100000)
            with contextlib.redirect_stdout(io.StringIO()): run(args)
            traces=pd.read_csv(root/'out/protein_trace.tsv',sep='\t').set_index('protein_id')
            self.assertEqual(traces.loc['P2','status'],'sequence_found_in_other_source')
            self.assertTrue(traces.loc['P2','inner_join_exclusion_possible'])
            self.assertFalse(traces.loc['P2','historical_input_pair_confirmed'])
            self.assertEqual(traces.loc['P3','status'],'metadata_ID_found_but_sequence_not_recovered')
            self.assertTrue(all(p.read_bytes()==content for p,content in before.items()))


if __name__=='__main__': unittest.main()

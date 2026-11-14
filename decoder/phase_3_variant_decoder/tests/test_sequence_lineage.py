"""Tests for causal distinctions in the read-only sequence lineage audit."""
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import h5py
import numpy as np
import pandas as pd

from phase_3_variant_decoder.audit_sequence_lineage import (
    id_candidates, inspect_h5, read_primary, run, sequence_records, sequence_status,
)


class SequenceLineageTests(unittest.TestCase):
    def test_sequence_filters_do_not_confuse_long_proteins_with_outside_positions(self):
        seq = {'LONG': 'A' * 1500, 'SHORT': 'AG'}
        self.assertEqual(sequence_status('LONG','A2V',seq,1022),'eligible')
        self.assertEqual(sequence_status('LONG','A1100V',seq,1022),'position_above_max_seq_len')
        self.assertEqual(sequence_status('SHORT','A3V',seq,1022),'position_outside_sequence')
        self.assertEqual(sequence_status('SHORT','G1V',seq,1022),'wt_amino_acid_mismatch')
        self.assertEqual(sequence_status('SHORT','G0V',seq,1022),'position_below_1')
        self.assertEqual(sequence_status('MISSING','A1V',seq,1022),'missing_wt_sequence')

    def test_primary_conflicts_are_visible_and_ids_are_not_silently_normalized(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'seq.json'
            p.write_text(json.dumps([{'Entry':'P1','Sequence':'AA'}, {'Entry':'P1','Sequence':'AG'},
                                     {'Entry':' P2 ','Sequence':'GG'}, {'Entry':'P3','Sequence':''}]))
            seq, meta = read_primary(p)
            self.assertEqual(seq['P1'],'AG')
            self.assertNotIn('P2',seq)
            self.assertEqual(meta['conflicting_ids'],['P1'])
            self.assertIn('P3',meta['empty_ids'])
            self.assertIn(('P2','whitespace_trim'),id_candidates(' P2 '))
            self.assertIn(('P1','uniprot_fasta_header_isoform_base'),id_candidates('sp|P1-2|NAME'))

    def test_supported_sequence_formats(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fasta = root/'seq.fasta'
            fasta.write_text('>sp|P1|NAME full header\nAG\nAA\n')
            self.assertEqual(list(sequence_records(fasta)),[('sp|P1|NAME','AGAA',[])])
            table = root/'seq.tsv'
            table.write_text('Entry\tSequence\nP1\tAGAA\n')
            self.assertEqual(list(sequence_records(table)),[('P1','AGAA',[])])
            table.write_text('File_Name,Sequence\nP62593_EXPERIMENT_A1V_1,AGAA\n')
            named = root/'named.csv'
            named.write_text(table.read_text())
            self.assertEqual(list(sequence_records(named)),[('P62593_EXPERIMENT_A1V_1','AGAA',[])])
            self.assertIn(('P62593','accession_in_source_identifier'),id_candidates('P62593_EXPERIMENT_A1V_1'))
            rest = root/'rest.json'
            rest.write_text(json.dumps({'results':[{'primaryAccession':'P1','sequence':{'value':'AG'},'secondaryAccessions':['OLD']}]}))
            self.assertEqual(list(sequence_records(rest)),[('P1','AG',['OLD'])])
            dat = root/'seq.dat'
            dat.write_text('ID   NAME\nAC   P1; OLD;\nSQ   SEQUENCE\n     AG AA\n//\n')
            self.assertEqual(list(sequence_records(dat)),[('P1','AGAA',['OLD'])])

    def test_numeric_h5_defects_are_not_treated_as_missing_sequences(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'global.h5'
            with h5py.File(path,'w') as h:
                h['valid'] = np.ones(4,dtype=np.float32)
                h['nan'] = np.array([1,2,np.nan,4],dtype=np.float32)
                h['zero'] = np.zeros(4,dtype=np.float32)
                h['wrong'] = np.ones(3,dtype=np.float32)
                h.create_group('group')
            states,_ = inspect_h5(path,{'valid','nan','zero','wrong','group','absent'},4,False)
            self.assertEqual(states,{'valid':'valid','nan':'nonfinite','zero':'zero_vector',
                'wrong':'wrong_shape','group':'not_a_dataset','absent':'missing_key'})
            before = path.read_bytes()
            inspect_h5(path,{'valid'},4,True)
            self.assertEqual(before,path.read_bytes())

    def test_missing_or_broken_h5_is_not_a_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'broken.h5'
            states,_ = inspect_h5(p,{'P1'},4,False)
            self.assertEqual(states['P1'],'missing_file')
            p.write_text('not HDF5')
            states,meta = inspect_h5(p,{'P1'},4,False)
            self.assertEqual(states['P1'],'unreadable_file')
            self.assertIn('error',meta)

    def test_end_to_end_candidates_stay_candidates_and_reports_preserve_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data,ref = root/'data',root/'reference'
            data.mkdir();ref.mkdir()
            seq = root/'uniprot.json'
            seq.write_text(json.dumps({'P1':'AGAA'}))
            source = root/'backup.fasta'
            source.write_text('>sp|P2|NAME\nGGAA\n')
            main_rows = [dict(protein_id='P1',snp='A1V',matched_go_id='GO:0000001',label='LOF',augmented=False),
                         dict(protein_id='P2',snp='G1A',matched_go_id='GO:0000001',label='NEUTRAL',augmented=True),
                         dict(protein_id='P1',snp='A1T',matched_go_id='GO:0000001',label='GOF',augmented=False)]
            for split in ('train','val','test'):
                pd.DataFrame(main_rows).to_csv(data/f'easy_{split}.tsv',sep='\t',index=False)
                pd.DataFrame(main_rows[:1]).to_csv(ref/f'easy_{split}.tsv',sep='\t',index=False)
            g,l = root/'global.h5',root/'local.h5'
            with h5py.File(g,'w') as h:
                for key in ('P1','P1_A1V'):
                    h[key] = np.ones(4,dtype=np.float32)
            with h5py.File(l,'w') as h:
                for key in ('P1_1','P1_A1V'):
                    h[key] = np.ones(4,dtype=np.float32)
            alternative = root/'another_global.h5'
            with h5py.File(alternative,'w') as h:
                h['P1_A1T'] = np.ones(4,dtype=np.float32)
            args = SimpleNamespace(out_dir=str(root/'reports'),family='easy',sequences_json=str(seq),
                dataset_dir=str(data),global_h5=str(g),local_h5=str(l),reference_dataset_dir=str(ref),
                sequence_source=[str(source)],search_root=[str(root)],max_depth=2,max_candidate_files=30,
                max_sequence_files=10,max_h5_files=10,max_seq_len=1022,expected_dim=4,keys_only=False,log_file=[],compare_inventory=None)
            sources = [seq,g,l,source,*data.glob('*.tsv'),*ref.glob('*.tsv')]
            before = {p:p.read_bytes() for p in sources}
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run(args),0)
            report = json.loads((root/'reports/summary.json').read_text())
            self.assertEqual(report['unique_counts']['missing_wt_sequences'],1)
            self.assertEqual(report['unique_counts']['missing_wt_sequences_with_candidates'],1)
            self.assertEqual(report['unique_counts']['eligible_missing_variants'],1)
            self.assertFalse(report['ready_for_current_eligible_protocol_rows'])
            diagnosis = pd.read_csv(root/'reports/row_diagnosis.tsv',sep='\t')
            self.assertTrue(diagnosis.loc[diagnosis.protein_id.eq('P2'),'sequence_status'].eq('missing_wt_sequence').all())
            self.assertTrue(diagnosis.loc[diagnosis.snp.eq('A1T'),'variant_new_vs_reference'].all())
            matches = pd.read_csv(root/'reports/alternative_h5_matches.tsv',sep='\t')
            self.assertFalse(matches.usable_replacement_verified.any())
            self.assertTrue(all(p.read_bytes()==content for p,content in before.items()))
            # Add a previously missing vector only to a disposable fixture, then
            # verify the audit distinguishes server availability from a local gap.
            inventory = root/'reports/artifact_inventory.json.gz'
            with h5py.File(g,'a') as h:
                h['P1_A1T'] = np.ones(4,dtype=np.float32)
            args.compare_inventory = str(inventory)
            args.out_dir = str(root/'server_reports')
            with contextlib.redirect_stdout(io.StringIO()):
                run(args)
            diff = pd.read_csv(root/'server_reports/inventory_comparison.tsv',sep='\t')
            record = diff[(diff.role=='global') & (diff.key=='P1_A1T')].iloc[0]
            self.assertEqual(record.comparison,'present_here_missing_or_invalid_in_reference')


if __name__ == '__main__':
    unittest.main()

"""Regressões de proveniência para o replay histórico e as novas aquisições."""
from pathlib import Path
import sys
from unittest import TestCase

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from tcc_campaign import comparison_row, read_plan, units
from ofdm_benchmark.config import load
from generate_gram import measured_4000


class ReplayTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = read_plan()
        cls.reference = cls.plan[0]
        cls.measured = dict(cls.reference, raw_sha256=cls.reference['physical_source_sha256'])

    def test_units(self):
        self.assertEqual(len(units(self.plan)), 5920)

    def test_complete_measurement(self):
        row = comparison_row(self.reference, self.measured, False)
        self.assertEqual(row['historical_comparison'], 'match')

    def test_ratio_difference(self):
        row = comparison_row(self.reference, dict(self.measured, per=0.1234567), False)
        self.assertIn('per', row['mismatch_fields'].split(','))

    def test_missing_measurement(self):
        measured = self.measured.copy()
        del measured['evm_error_energy']
        with self.assertRaises(KeyError):
            comparison_row(self.reference, measured, True)

    def test_new_measurement(self):
        row = comparison_row(self.reference, dict(self.measured, per=0.1234567), True)
        self.assertEqual(row['per'], 0.1234567)
        self.assertEqual(row['historical_comparison'], 'not_applicable_new_capture')

    def test_offline_config(self):
        gram = load(ROOT/'configs/tcc.example.json').methods[-1].parameters['quantum_gram_path']
        self.assertEqual(Path(gram), ROOT/'data/config/cx_4000.npz')

    def test_protected_gram(self):
        with self.assertRaisesRegex(ValueError, 'históricos'):
            measured_4000(ROOT/'data/config/cx_4000.npz', 'cx')

    def test_source_manifest(self):
        import csv
        with (ROOT/'data/campanha_final/fontes_iq.csv').open(newline='') as stream:
            refs = {(r['session'], r['burst']): r for r in csv.DictReader(stream)}
        self.assertEqual(len(refs), 100)
        for row in self.plan:
            source = refs[row['session_id'], row['burst']]
            self.assertEqual(source['raw_sha256'], row['physical_source_sha256'])
            self.assertEqual(len(bytes.fromhex(source['truth_sha256'])), 32)

    def test_radio_guard(self):
        from ofdm_benchmark.benchmark import run_campaign
        with self.assertRaisesRegex(ValueError, 'explícita'):
            run_campaign(None, None)

"""A native-only numerical disagreement must remain distinct from a compiler failure."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from aten_fixture_batch import correctness_summary

class FixtureResults(unittest.TestCase):
    def log(self,bad=0,native=0,guard=0,complete=True,native_reference=0,legacy=False):
        lines=['POLYGEIST_V2_CALL backend=cuda']
        for resident in range(2 if complete else 1):
            ref='' if legacy else f' nativerefbad={native_reference}'
            lines.append(f'CHECK kind=1 parameters=0 resident={resident} count=16 bad={bad} nativebad={native}{ref} guardbad={guard} inputbad=0 maxabs=0')
        if not (bad or native or guard or native_reference):lines.append('PASS correctness')
        return '\n'.join(lines)
    def test_native_only_error_retains_raised_result(self):
        r=correctness_summary(3,self.log(native=1,native_reference=1),16,2)
        self.assertTrue(r['raised_passed']);self.assertFalse(r['native_passed']);self.assertFalse(r['passed'])
    def test_raised_error_cannot_authorize_timing(self):
        r=correctness_summary(3,self.log(bad=1),16,2)
        self.assertFalse(r['raised_passed']);self.assertTrue(r['native_passed']);self.assertFalse(r['passed'])
    def test_guard_error_neither_passes(self):
        r=correctness_summary(3,self.log(guard=1),16,2)
        self.assertFalse(r['raised_passed']);self.assertFalse(r['native_passed'])
    def test_short_output_and_unexpected_exit_fail(self):
        for rc,log in [(0,self.log(complete=False)),(2,self.log()),(3,self.log())]:
            r=correctness_summary(rc,log,16,2)
            self.assertFalse(r['raised_passed']);self.assertFalse(r['native_passed'])
    def test_all_modes_complete(self):
        self.assertTrue(correctness_summary(0,self.log(),16,2)['passed'])
    def test_legacy_pair_disagreement_does_not_implicate_native(self):
        r=correctness_summary(3,self.log(bad=1,native=1,legacy=True),16,2)
        self.assertFalse(r['raised_passed']);self.assertIsNone(r['native_passed'])
    def test_native_reference_comparison_is_independent(self):
        r=correctness_summary(3,self.log(bad=1,native=1),16,2)
        self.assertFalse(r['raised_passed']);self.assertTrue(r['native_passed'])
        r=correctness_summary(3,self.log(native_reference=1),16,2)
        self.assertFalse(r['passed']);self.assertTrue(r['raised_matches_native'])

if __name__=='__main__':unittest.main()

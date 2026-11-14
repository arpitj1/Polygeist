"""Per-operation ABI reuse retains evidence and does not cache across calls."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import spec_match as S


class ABITableReuse(unittest.TestCase):
    def test_reusing_the_same_table_preserves_all_results(self):
        table = ({}, {}, {'missing': 'unsupported declaration'})
        with patch.object(S.ABI, 'build_table', return_value=table) as build:
            expected = (S.capabilities({}, 'cuda'), S.abi_digest(),
                        S.backend_evidence([], {}))
            self.assertEqual(build.call_count, 3)
            build.reset_mock()
            actual = (S.capabilities({}, 'cuda', table=table), S.abi_digest(table),
                      S.backend_evidence([], {}, table=table))
            self.assertEqual(actual, expected)
            build.assert_not_called()

    def test_independent_operations_still_read_current_declarations(self):
        with patch.object(S.ABI, 'build_table', side_effect=[
                ({}, {}, {}), ({}, {}, {'changed': 'different declaration'})]) as build:
            self.assertNotEqual(S.abi_digest(), S.abi_digest())
            self.assertEqual(build.call_count, 2)


if __name__ == '__main__':
    unittest.main()

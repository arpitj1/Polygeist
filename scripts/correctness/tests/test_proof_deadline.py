"""Deadlines interrupt native/blocking work and never leave child processes."""
from pathlib import Path
import sys
import time
import unittest
import json
import tempfile
import subprocess
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from proof_metrics import graph_call, stop_worker, ProofDeadline
import kernel_match as K


class ProofDeadlineTests(unittest.TestCase):
    def tearDown(self):
        stop_worker()

    def test_result_and_exception_are_distinct(self):
        result=graph_call(dict(kind='scalar',left=['In',0],right=['In',1],
                              collect=False,distributivity=False),2)
        self.assertFalse(result['matched'])
        with self.assertRaisesRegex(RuntimeError, 'unknown proof kind'):
            graph_call(dict(kind='invalid',collect=False),2)

    def test_blocking_work_is_killed_and_reaped(self):
        original=subprocess.Popen;children=[]
        def blocking_worker(*args,**kwargs):
            child=original([sys.executable,'-c','import time; time.sleep(60)'],**kwargs)
            children.append(child)
            return child
        started=time.perf_counter()
        with patch('proof_metrics.subprocess.Popen',side_effect=blocking_worker):
            with self.assertRaises(ProofDeadline):
                graph_call({},.1)
        self.assertLess(time.perf_counter()-started,2)
        self.assertIsNotNone(children[0].poll())

    def test_scalar_result_and_counters_return_from_worker(self):
        K.configure_matcher(collect_egraph_sizes=True)
        a,b=K.Term.In(0),K.Term.In(1)
        with patch.dict('os.environ',POLYGEIST_SATURATION_TIMEOUT='1',
                        POLYGEIST_NUMERIC_MODEL='relaxed'):
            self.assertTrue(K.equivalent(a+b,b+a,False))
        counters=K.matcher_telemetry()
        self.assertEqual(counters['proofs_matched'],1)
        self.assertGreater(counters['proof_iterations'],0)
        self.assertGreater(counters['egraph_nodes_max'],0)
        self.assertGreater(counters['saturation_elapsed_ms'],0)

    def test_timeout_cannot_authorize_replacement(self):
        K.configure_matcher()
        with patch.dict('os.environ',POLYGEIST_SATURATION_TIMEOUT='1'):
            with patch('proof_metrics.graph_call',side_effect=ProofDeadline()):
                self.assertFalse(K.equivalent(K.Term.In(0),K.Term.In(0)))
        self.assertEqual(K.matcher_telemetry()['proof_timeouts'],1)
        self.assertEqual(K.matcher_telemetry()['proofs_matched'],0)

    def test_region_proof_emits_size_and_iteration_measurements(self):
        from region_match.egg import check_equal
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'proofs.jsonl'
            with patch.dict('os.environ',POLYGEIST_SATURATION_TIMEOUT='1',
                            POLYGEIST_MATCHER_TELEMETRY_JSONL=str(path)):
                result=check_equal(K.Term.In(0),K.Term.In(0),[],[])
            self.assertTrue(result.ok)
            row=json.loads(path.read_text())
            self.assertEqual(row['phase'],'region_proof')
            self.assertGreater(row['iterations'],0)
            self.assertGreater(row['egraph_nodes'],0)
            self.assertFalse(row['timeout'])


if __name__=='__main__': unittest.main()

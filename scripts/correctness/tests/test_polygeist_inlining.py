"""The upstream inliner may expand views, but not execution-context operations."""
from pathlib import Path
import subprocess
import unittest

OPT = Path(__file__).resolve().parents[3]/'build/bin/polygeist-opt'


class PolygeistInlining(unittest.TestCase):
    def inline(self, text):
        p=subprocess.run([str(OPT),'--inline'],input=text,text=True,capture_output=True)
        self.assertEqual(p.returncode,0,p.stderr)
        return p.stdout

    def test_memory_view_helper_is_inlined(self):
        result=self.inline('''module {
func.func private @view(%x: memref<4xf32>) -> memref<?xf32> {
 %zero = arith.constant 0 : index
 %r = "polygeist.subindex"(%x, %zero) : (memref<4xf32>, index) -> memref<?xf32>
 return %r : memref<?xf32>
}
func.func @use(%x: memref<4xf32>) -> f32 {
 %zero = arith.constant 0 : index
 %view = func.call @view(%x) : (memref<4xf32>) -> memref<?xf32>
 %value = memref.load %view[%zero] : memref<?xf32>
 return %value : f32
}
}''')
        self.assertNotIn('call @view',result)

    def test_barrier_helper_remains_a_call(self):
        result=self.inline('''module {
func.func private @effect(index)
func.func private @barrier(%x: index) -> index {
 %zero = arith.constant 0 : index
 %one = arith.constant 1 : index
 %four = arith.constant 4 : index
 scf.parallel (%i) = (%zero) to (%four) step (%one) {
   func.call @effect(%i) : (index) -> ()
   "polygeist.barrier"(%i) : (index) -> ()
   func.call @effect(%i) : (index) -> ()
   scf.yield
 }
 return %x : index
}
func.func @use(%x: index) -> index {
 %r = func.call @barrier(%x) : (index) -> index
 return %r : index
}
}''')
        self.assertIn('call @barrier',result)


if __name__=='__main__':unittest.main()

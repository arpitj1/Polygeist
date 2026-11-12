"""Execute a source-derived four-dimensional reference through every stage."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'scripts/correctness'), str(ROOT/'scripts/correctness/lib_v2')]
import build_lib_v2 as B
import recipe
import validate_specs as V


class NdReferenceValidation(unittest.TestCase):
    def test_four_dimensional_reference_executes_all_stages(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'issues') as tmp:
            root = Path(tmp)
            ref = root/'ref.c'
            ref.write_text('''void NETLIB_SPEC(int a,int b,int h,int w,
float x[a][b][h][w],float y[a][b][h][w]) {
for(int i=0;i<a;++i)for(int j=0;j<b;++j)
for(int k=0;k<h;++k)for(int l=0;l<w;++l)
y[i][j][k][l]=x[i][j][k][l]*2.0f+0.5f;
}''')
            raw = dict(args=['a','b','h','w','x','y'], validation={
                'scalars': dict(a=2,b=3,h=5,w=7),
                'arrays': dict(x=[2,3,5,7],y=[2,3,5,7])})
            spec = B.Spec('dimension_probe','probe',root/'manifest.json',ref,{},
                          raw['args'],{},[],[],raw=raw)
            (root/'manifest.json').write_text(json.dumps({'specs':[raw]}))
            with patch.object(B,'RAISED',root/'raised'), patch.object(V,'RAISED',root/'raised'), \
                    patch.object(V,'WORK',root/'validate'):
                result = B.build_one(spec,recipe.PRODUCTION,60,True)
                self.assertEqual(result['status'],'RAISED',result.get('reason'))
                dims = result['meta']['array_dimensions']
                self.assertNotIn('a',dims)  # Outer C array annotations are not layout.
                self.assertEqual(dims['w'],[{'array':'x','axis':3},{'array':'y','axis':3}])
                self.assertFalse(result['meta']['leading_dims'])
                for stage in ['affine','linalg','defn']:
                    result = V.validate(spec.name,raw,ref,set(),stage)
                    self.assertEqual(result['verdict'],'EXACT',result)

    def test_missing_or_inconsistent_nd_validation_shapes_fail(self):
        args=[('%x','memref<?x?x?xf32>')]
        for validation in [{},{'arrays':{'x':[2,3]}},{'arrays':{'x':[2,0,3]}}]:
            with self.subTest(validation=validation),self.assertRaises(ValueError):
                V.c_param_kinds('probe',args,['x'],set(),validation=validation)
        with self.assertRaises(ValueError):
            V.c_param_kinds('probe',[(f'%p{i}','i32') for i in range(8)],
                            [f'p{i}' for i in range(8)],set())

    def test_explicit_integer_metadata_values_execute(self):
        args=[('%length','memref<4xi32>')]
        params=V.c_param_kinds('probe',args,['length'],set(),{'length':'int'},
            {'array_values':{'length':[0,1,3,7]}})
        source=V.harness('probe',params,None)
        source+='''\n#include <stdlib.h>
void probe_ref(int *x){int want[]={0,1,3,7};for(int i=0;i<4;++i)if(x[i]!=want[i])exit(17);}
void probe_impl(int *base,int *aligned,int64_t offset,int64_t size,int64_t stride){
 (void)base;if(size!=4||stride!=1)exit(18);probe_ref(aligned+offset);}
'''
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);(d/'check.c').write_text(source)
            subprocess.run(['gcc','-O0',str(d/'check.c'),'-lm','-o',str(d/'check')],check=True,capture_output=True)
            subprocess.run([str(d/'check')],check=True,capture_output=True)
        for values in [[],[2**31],[1.5],[True]]:
            with self.subTest(values=values),self.assertRaises(ValueError):
                V.c_param_kinds('probe',args,['length'],set(),{'length':'int'},
                    {'array_values':{'length':values}})


if __name__=='__main__':
    unittest.main()

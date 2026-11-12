from pathlib import Path
import copy
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'lib_v2'))
import abi_v2 as A
import typed_wrapper as T

class TypedContracts(unittest.TestCase):
    def contract(self,element='float',wrapper_element='float',rank=1,backend='cuda',change=None,readonly=False):
        dims=['n'] if rank==1 else ['h','w'] if rank==2 else ['d'+str(i) for i in range(rank)]
        args=[A.SpecArg(d,'int',ctype='int') for d in dims]
        args += [A.SpecArg('input','array',rank=rank,vla_ld='w' if rank==2 else None,intent='in',ctype=element),A.SpecArg('output','array',rank=rank,vla_ld='w' if rank==2 else None,intent='out',ctype=element)]
        config=dict(schema=1,kind='typed_c_wrapper',entry='wrapper',arguments=dims+['input','output'],arrays={'input':dims,'output':dims},bounds={d:{'min':1,'max':4096} for d in dims})
        if change:change(config)
        target=dict(library='TestLibrary',call='vendorSubmit',backends=[backend],runtime_adapter=config)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'runtime').mkdir()
            declaration=','.join(['int32_t']*len(dims)+['const '+wrapper_element+' *',('const ' if readonly else '')+wrapper_element+' *'])
            (root/'runtime/polygeist_cublas_rt.h').write_text('#include <stdint.h>\n/* POLYGEIST_TYPED_VENDOR_ENTRY TestLibrary vendorSubmit wrapper */\nvoid wrapper('+declaration+');\n')
            with patch.object(A,'REPO',root),patch.object(A,'CLANG','gcc'):
                return T.contract(A,'reference',target,'void',args)

    def test_optional_linkage_is_explicit(self):
        _,_,cfg=self.contract(change=lambda c:c.update(linkage='optional'))
        self.assertEqual(cfg['linkage'],'optional')
        with self.assertRaises(A.AbiError):self.contract(change=lambda c:c.update(linkage='fallback'))

    def test_dimension_divisibility_contract(self):
        _,_,cfg=self.contract(change=lambda c:c['bounds']['n'].update(multiple_of=4))
        self.assertEqual(cfg['bounds']['n']['multiple_of'],4)
        for value in (0,-1,1.5,True,2147483648):
            with self.subTest(value=value),self.assertRaises(A.AbiError):
                self.contract(change=lambda c:c['bounds']['n'].update(multiple_of=value))

    def test_cuda_rows_preserve_exact_types(self):
        params,binding,cfg=self.contract()
        self.assertEqual(cfg['memory_domain'],'cuda_device')
        self.assertEqual([(p.name,p.elem,p.rank) for p in params if p.tkind=='array'],[('input','float',1),('output','float',1)])
        self.assertEqual(binding['output'],('arg',2))

    def test_multidimensional_wrapper_requires_every_extent(self):
        params,_,cfg=self.contract(rank=4)
        self.assertEqual([p.rank for p in params if p.tkind=='array'],[4,4])
        self.assertEqual(cfg['arrays']['input'],['d0','d1','d2','d3'])
        with self.assertRaises(A.AbiError):
            self.contract(rank=4,change=lambda c:c['arrays'].update(input=['d0','d1','d2']))

    def test_pva_unsigned_image_layout(self):
        params,_,cfg=self.contract(element='unsigned char',wrapper_element='uint8_t',rank=2,backend='pva')
        self.assertEqual(cfg['memory_domain'],'host')
        self.assertEqual([p.elem for p in params if p.tkind=='array'],['unsigned char']*2)

    def test_partial_extent_requires_provably_positive_dimension(self):
        def change(c):
            c['bounds']['h']['min']=3
            c['arrays']['output']=[{'dimension':'h','offset':-1},'w']
        _,_,cfg=self.contract(rank=2,backend='pva',change=change)
        self.assertEqual(cfg['arrays']['output'][0]['offset'],-1)
        def invalid(c):
            change(c);c['arrays']['output'][0]['offset']=-3
        with self.assertRaises(A.AbiError):self.contract(rank=2,backend='pva',change=invalid)

    def test_equal_width_signedness_mismatch_rejected(self):
        with self.assertRaises(A.AbiError):self.contract(element='signed char',wrapper_element='uint8_t')

    def test_precision_mismatch_rejected(self):
        with self.assertRaises(A.AbiError):self.contract(element='float',wrapper_element='double')

    def test_written_buffer_cannot_bind_const_pointer(self):
        with self.assertRaises(A.AbiError):self.contract(readonly=True)

    def test_missing_argument_or_extent_rejected(self):
        for change in [lambda c:c['arguments'].pop(),lambda c:c['arrays'].pop('output'),lambda c:c['bounds'].clear()]:
            with self.subTest(change=change),self.assertRaises(A.AbiError):self.contract(change=change)

    def test_layout_dimensions_must_agree_with_vla(self):
        with self.assertRaises(A.AbiError):self.contract(rank=2,change=lambda c:c['arrays'].update(output=['w','h']))

    def test_invalid_bounds_and_alias_capability_rejected(self):
        for change in [lambda c:c.update(alias_policy='exact_or_disjoint'),lambda c:c['bounds']['n'].update(max=1<<40),lambda c:c['arrays'].update(output=['n*2'])]:
            with self.subTest(change=change),self.assertRaises(A.AbiError):self.contract(change=change)

    def test_negative_extent_still_rejected(self):
        with self.assertRaises(A.AbiError):
            self.contract(change=lambda c:c['bounds']['n'].update(min=-1))

    def test_affine_extent_from_two_dimensions(self):
        expression = {'terms': {'d0': 1, 'd1': 1}, 'offset': -1}
        _, _, cfg = self.contract(rank=3, change=lambda c:
            c['arrays']['input'].__setitem__(0, expression))
        self.assertEqual(cfg['arrays']['input'][0], expression)
        self.assertEqual(T.extent_expression(expression),
                         '(((int64_t)d0 * (1)) + ((int64_t)d1 * (1)) + -1)')

    def test_affine_extent_unknown_nonpositive_or_overflow_rejected(self):
        for expression in [
                {'terms': {'unknown': 1}, 'offset': 0},
                {'terms': {'d0': 1, 'd1': -1}, 'offset': 0},
                {'terms': {'d0': 2147483647}, 'offset': 0},
                {'terms': {'d0': 1.5}, 'offset': 0},
                {'terms': {}, 'offset': 3}]:
            with self.subTest(expression=expression), self.assertRaises(A.AbiError):
                self.contract(rank=3, change=lambda c:
                    c['arrays']['input'].__setitem__(0, expression))

    def test_signed_control_is_not_an_extent(self):
        # The existing positive extent is n; a separate scalar seed may use
        # the whole exact int32 range without allowing negative array sizes.
        args=[A.SpecArg('n','int',ctype='int'),A.SpecArg('seed','int',ctype='int'),
              A.SpecArg('input','array',rank=1,intent='in',ctype='int')]
        cfg=dict(schema=1,kind='typed_c_wrapper',entry='control',arguments=['n','seed','input'],
                 arrays={'input':['n']},bounds={'n':{'min':1,'max':4096},
                 'seed':{'min':-2147483648,'max':2147483647}})
        target=dict(library='TestLibrary',call='vendorSubmit',backends=['cuda'],runtime_adapter=cfg)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'runtime').mkdir()
            (root/'runtime/polygeist_cublas_rt.h').write_text(
                '/* POLYGEIST_TYPED_VENDOR_ENTRY TestLibrary vendorSubmit control */\n'
                'int control(int n,int seed,const int *input);\n')
            with patch.object(A,'REPO',root),patch.object(A,'CLANG','gcc'):
                T.contract(A,'reference',target,'int',args)
                cfg['bounds']['seed']['min']=-2147483649
                with self.assertRaises(A.AbiError):T.contract(A,'reference',target,'int',args)

if __name__=='__main__':unittest.main()

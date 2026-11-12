"""Compare imported scalar adapters with their pinned public reference functions.

The original upstream function bodies are extracted and compiled at test time;
this checks the manual source adaptation independently of compiler self-checks.
"""
import ctypes
import itertools
from pathlib import Path
import subprocess
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[3]


def function(source, start):
    begin = source.index(start)
    brace = source.index('{', begin)
    depth = 1
    end = brace + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[begin:end]


def values(count, offset=0):
    return (ctypes.c_float * count)(*(((i * 13 + offset) % 47 - 23) / 16
                                     for i in range(count)))


class PublicSpatialReferences(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        sample = (REPO / 'third_party/cudnn_reference/conv_sample.cpp').read_text()
        bits = [function(sample, needle) for needle in (
            'static void\ngenerateStrides(', 'void\nlin2dim(',
            'static int\ndim2lin(', 'static float\ndoFma(float',
            'static void\ndoEpilog(float',
            'template <typename T_ELEM, typename T_MATH>\nstatic void\nconv_cpu_ref(')]
        # Select the public sample's existing cross-correlation branch, as
        # documented by the imported reference; all arithmetic stays upstream.
        bits[-1] = bits[-1].replace('bool isConv = true;', 'bool isConv = false;')
        cpp = '''#include <algorithm>
#include <cstdint>
#include <optional>
typedef int cudnnTensorFormat_t;
constexpr int CUDNN_TENSOR_NCHW=0, CUDNN_TENSOR_NCHW_VECT_C=2;
''' + '\n'.join(bits) + '''
extern "C" void upstream_depthwise(int B,int C,int H,int W,int RH,int RW,
 const float *x,const float *w,const float *bias,float *y) {
 int in[4]={1,1,H,W},f[4]={1,1,2*RH+1,2*RW+1},out[4]={1,1,H,W};
 int strides[4]={H*W,H*W,W,1},step[2]={1,1},pad[2]={RH,RW};
 for(int b=0;b<B;++b) for(int c=0;c<C;++c) {
  float *result=y+(b*C+c)*H*W;
  std::fill(result,result+H*W,bias[c]);
  conv_cpu_ref<float,float>(x+(b*C+c)*H*W,w+c*(2*RH+1)*(2*RW+1),result,
      1.f,1.f,1,CUDNN_TENSOR_NCHW,in,f,out,strides,strides,step,pad,step,4);
 }
}
namespace at { template<class F> void parallel_for(int64_t a,int64_t b,int64_t,F f) { f(a,b); } }
namespace c10 {
struct range { int64_t a,b; struct it { int64_t v; int64_t operator*() const { return v; }
 it &operator++(){ ++v; return *this; } bool operator!=(it r) const { return v!=r.v; } };
 it begin() const { return {a}; } it end() const { return {b}; } };
range irange(int64_t a,int64_t b) { return {a,b}; }
}
'''
        pool = (REPO / 'third_party/pytorch/aten/src/ATen/native/AveragePool3d.cpp').read_text()
        cpp += function(pool, 'template <typename scalar_t>\nvoid avg_pool3d_backward_out_frame(')
        cpp += '''
extern "C" void upstream_pool(int B,int C,int O0,int O1,int O2,int R0,int R1,int R2,
 const float *dy,float *dx) {
 avg_pool3d_backward_out_frame<float>(dx,dy,B*C,2*O0+R0,2*O2+R2,2*O1+R1,
 O0,O2,O1,2,2,2,2,2,2,0,0,0,true,std::nullopt);
}
'''
        (root/'upstream.cpp').write_text(cpp)
        subprocess.run(['g++','-std=c++17','-O2','-ffp-contract=off','-shared','-fPIC',
                        str(root/'upstream.cpp'),'-o',str(root/'upstream.so')],check=True,
                       capture_output=True,text=True)
        cls.upstream = ctypes.CDLL(str(root/'upstream.so'))
        cls.upstream.upstream_depthwise.restype = None
        cls.upstream.upstream_pool.restype = None
        paths = {
            'depthwise': REPO/'generic_solver/lib_v3/ref_c/cudnn_depthwise_conv2d.c',
            'pool': REPO/'generic_solver/lib_v3/ref_c/aten_avg_pool3d_backward.c',
        }
        cls.adapted = {}
        for name, path in paths.items():
            dest = root/(name+'.so')
            subprocess.run(['gcc','-std=c99','-O2','-ffp-contract=off','-shared','-fPIC',
                            '-DNETLIB_SPEC=adapted',str(path),'-o',str(dest)],check=True,
                           capture_output=True,text=True)
            cls.adapted[name] = ctypes.CDLL(str(dest)).adapted
            cls.adapted[name].restype = None

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_depthwise_group_and_padding_specialization(self):
        for dims in [(1,1,2,3,0,0),(2,3,5,7,1,2),(1,2,3,2,3,1)]:
            with self.subTest(dims=dims):
                B,C,H,W,RH,RW = dims
                x,w,bias = values(B*C*H*W),values(C*(2*RH+1)*(2*RW+1),7),values(C,17)
                n = B*C*H*W
                expected,actual = values(n,6),values(n,8)
                self.upstream.upstream_depthwise(*dims,x,w,bias,expected)
                self.adapted['depthwise'](*dims,x,w,bias,actual)
                self.assertEqual(bytes(expected),bytes(actual))

    def test_pool_remainders_and_unwritten_tail(self):
        for B,C,O0,O1,O2 in [(1,2,2,3,4),(2,3,3,2,1)]:
            for remainder in itertools.product((0,1), repeat=3):
                dims=(B,C,O0,O1,O2,*remainder)
                with self.subTest(dims=dims):
                    R0,R1,R2=remainder
                    n=B*C*(2*O0+R0)*(2*O1+R1)*(2*O2+R2)
                    dy=values(B*C*O0*O1*O2,3)
                    expected,actual=values(n,6),values(n,8)
                    self.upstream.upstream_pool(*dims,dy,expected)
                    self.adapted['pool'](B*C,O0,O1,O2,*remainder,dy,actual)
                    self.assertEqual(bytes(expected),bytes(actual))


if __name__ == '__main__':
    unittest.main()

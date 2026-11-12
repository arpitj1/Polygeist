import os
from pathlib import Path
import subprocess,argparse
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser(description="Compare imported convolution references with the public NVIDIA sample")
p.add_argument("--output",type=Path,required=True)
args=p.parse_args()
D=args.output.resolve();D.mkdir(parents=True,exist_ok=True)
source=(Path(os.environ.get('POLYGEIST_DEPENDENCY_ROOT', ROOT/'third_party'))/'cudnn_reference/conv_sample.cpp').read_text()
def extract(start):
 a=source.index(start);b=source.index('{',a);depth=1;c=b+1
 while depth:
  depth+=(source[c]=='{')-(source[c]=='}');c+=1
 return source[a:c]
parts=[extract(x) for x in ['static void\ngenerateStrides(', 'void\nlin2dim(', 'static int\ndim2lin(', 'static float\ndoFma(float', 'static void\ndoEpilog(float', 'template <typename T_ELEM, typename T_MATH>\nstatic void\nconv_cpu_ref(']]
parts[-1]=parts[-1].replace('bool isConv = true;', 'bool isConv = false;')
header=source.split('#include')[0]+'#include <vector>\n#include <cstring>\n#include <cmath>\n#include <cfloat>\n#include <cstdio>\nusing cudnnTensorFormat_t=int;constexpr int CUDNN_TENSOR_NCHW=0,CUDNN_TENSOR_NCHW_VECT_C=2;\n'
extra_proto='extern "C" void ref_unbatched(int,int,int,int,int,int,int,int,const float*,const float*,float*);\n'
proto='extern "C" void ref_d%d(int,int,int,int,int,int,int,int,int,const float*,const float*,float*);\n'
main=r'''int main(){long checked=0;
 const float edges[]={0.f,-0.f,INFINITY,-INFINITY,NAN,FLT_MAX,-FLT_MAX,FLT_MIN,-FLT_MIN,0x1p-149f,-0x1p-149f,1.f,-1.f,.25f,-.25f};
 for(int d:{1,2})for(int B:{1,2})for(int C:{1,3})for(int O:{2,5})for(int KH:{2,3})for(int KW:{2,4})for(int mode=0;mode<3;++mode){
 int H=d*(KH-1)+5,W=d*(KW-1)+7,OH=5,OW=7;
 std::vector<float>x(B*C*H*W),f(O*C*KH*KW),a(B*O*OH*OW+8,913.25f),b=a;
 for(size_t i=0;i<x.size();++i)x[i]=mode==2?-0.f:mode==1?edges[i%15]:float(int(i*37%101)-50)/32;
 for(size_t i=0;i<f.size();++i)f[i]=mode==2?1.f:mode==1?edges[(i+4)%15]:float(int(i*19%79)-39)/32;
 auto oldx=x,oldf=f;int in[]={B,C,H,W},filter[]={O,C,KH,KW},out[]={B,O,OH,OW},is[]={C*H*W,H*W,W,1},os[]={O*OH*OW,OH*OW,OW,1},stride[]={1,1},pad[]={0,0},dilation[]={d,d};
 conv_cpu_ref<float,float>(x.data(),f.data(),a.data()+4,1,0,1,0,in,filter,out,is,os,stride,pad,dilation,4);
 if(B==1&&d==2)ref_unbatched(C,H,W,O,KH,KW,OH,OW,x.data(),f.data(),b.data()+4);else (d==1?ref_d1:ref_d2)(B,C,H,W,O,KH,KW,OH,OW,x.data(),f.data(),b.data()+4);
 for(size_t i=0;i<a.size();++i){if(!(std::isnan(a[i])&&std::isnan(b[i]))&&std::memcmp(&a[i],&b[i],4)){std::printf("FAIL d=%d i=%zu a=%a b=%a\n",d,i,a[i],b[i]);return 1;}++checked;}
 if(std::memcmp(x.data(),oldx.data(),x.size()*4)||std::memcmp(f.data(),oldf.data(),f.size()*4))return 2;
 }std::printf("PASS %ld source outputs and guards\n",checked);}
'''
(D/'source_check.cpp').write_text(header+'\n'.join(parts)+'\n'+''.join(proto%d for d in [1,2])+extra_proto+main)
for d in [1,2]:subprocess.run(['gcc','-O0','-ffp-contract=off','-DNETLIB_SPEC=ref_d'+str(d),'-DDILATION='+str(d),'-c',str(ROOT/'generic_solver/lib_v3/ref_c/cudnn_conv.c'),'-o',str(D/('d'+str(d)+'.o'))],check=True)
subprocess.run(['gcc','-O0','-ffp-contract=off','-DNETLIB_SPEC=ref_unbatched','-DDILATION=2','-DBATCHED=0','-c',str(ROOT/'generic_solver/lib_v3/ref_c/cudnn_conv.c'),'-o',str(D/'unbatched.o')],check=True)
subprocess.run(['g++','-std=c++17','-O0','-ffp-contract=off',str(D/'source_check.cpp'),str(D/'d1.o'),str(D/'d2.o'),str(D/'unbatched.o'),'-o',str(D/'source_check')],check=True)
subprocess.run([str(D/'source_check')],check=True)

# Validate the same public reference in three spatial dimensions.
main3d='int main(){long checked=0;\n const float edges[]={0.f,-0.f,INFINITY,-INFINITY,NAN,FLT_MAX,-FLT_MAX,FLT_MIN,-FLT_MIN,0x1p-149f,-0x1p-149f,1.f,-1.f,.25f,-.25f};\n for(int C:{1,3})for(int O:{2,5})for(int KD:{2,3})for(int KH:{2,4})for(int KW:{1,3})for(int mode=0;mode<3;++mode){\n int ID=KD+3,H=KH+4,W=KW+5,OD=4,OH=5,OW=6;\n std::vector<float>x(C*ID*H*W),f(O*C*KD*KH*KW),a(O*OD*OH*OW+8,913.25f),b=a;\n for(size_t i=0;i<x.size();++i)x[i]=mode==2?-0.f:mode==1?edges[i%15]:float(int(i*37%101)-50)/32;\n for(size_t i=0;i<f.size();++i)f[i]=mode==2?1.f:mode==1?edges[(i+4)%15]:float(int(i*19%79)-39)/32;\n auto oldx=x,oldf=f;int in[]={1,C,ID,H,W},filter[]={O,C,KD,KH,KW},out[]={1,O,OD,OH,OW},is[]={C*ID*H*W,ID*H*W,H*W,W,1},os[]={O*OD*OH*OW,OD*OH*OW,OH*OW,OW,1},stride[]={1,1,1},pad[]={0,0,0},dilation[]={1,1,1};\n conv_cpu_ref<float,float>(x.data(),f.data(),a.data()+4,1,0,1,0,in,filter,out,is,os,stride,pad,dilation,5);\n ref3d(C,ID,H,W,O,KD,KH,KW,OD,OH,OW,x.data(),f.data(),b.data()+4);\n for(size_t i=0;i<a.size();++i){if(!(std::isnan(a[i])&&std::isnan(b[i]))&&std::memcmp(&a[i],&b[i],4)){std::printf("FAIL i=%zu a=%a b=%a\\n",i,a[i],b[i]);return 1;}++checked;}\n if(std::memcmp(x.data(),oldx.data(),x.size()*4)||std::memcmp(f.data(),oldf.data(),f.size()*4))return 2;\n }std::printf("PASS %ld 3D source outputs and guards\\n",checked);}\n'
proto3d='extern "C" void ref3d(int,int,int,int,int,int,int,int,int,int,int,const float*,const float*,float*);\n'
(D/'source_check_3d.cpp').write_text(header+'\n'.join(parts)+'\n'+proto3d+main3d)
subprocess.run(['gcc','-O0','-ffp-contract=off','-DNETLIB_SPEC=ref3d','-c',str(ROOT/'generic_solver/lib_v3/ref_c/cudnn_conv3d.c'),'-o',str(D/'ref3d.o')],check=True)
subprocess.run(['g++','-std=c++17','-O0','-ffp-contract=off',str(D/'source_check_3d.cpp'),str(D/'ref3d.o'),'-o',str(D/'source_check_3d')],check=True)
subprocess.run([str(D/'source_check_3d')],check=True)

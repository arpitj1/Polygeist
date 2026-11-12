#!/usr/bin/env python3
"""Validate proved FP32, FP64, int8 and int32 fixtures against C and real ATen CUDA.

Native operations below are benchmark baselines, never compiler recognition
rules. A saved specification proof must already authorize every replacement.
The default uses the original fixture dimensions, not paper-sized timings.
"""
import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

import aten_pointwise_graph_silicon as BENCH

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CUDA = Path(os.environ.get('ATEN_NATIVE_CUDA_ROOT', '/usr/local/cuda-12.6/targets/sbsa-linux'))
TORCH = Path(os.environ.get('ATEN_NATIVE_TORCH_ROOT', '/tmp/aten_native_sysroot_24_12/usr/local/lib/python3.12/dist-packages/torch'))
# Each expression calls real ATen with preallocated outputs and cached base
# tensors. Any composition/view construction below is included in its timing.
NATIVE = {
    # Same real ATen composition as native_exact_region_bench.cpp. Its
    # preallocated intermediates and bool-to-int output copy are timed.
    'nested_all_cpu': '''static auto input=t[0].view({B,N});
      static auto lengths=t[1].view({B,1});
      static auto columns=at::arange(N,t[1].options()).view({1,N});
      static auto mask=at::empty({B,N},t[0].options().dtype(at::kBool));
      static auto nonzero=at::empty_like(mask);
      static auto selected=at::empty_like(mask);
      static auto identity=at::ones_like(mask);
      static auto reduced=at::empty({B},mask.options());
      at::lt_out(mask,columns,lengths);at::ne_out(nonzero,input,0);
      at::where_out(selected,mask,nonzero,identity);
      at::all_out(reduced,selected,1,false);t[2].copy_(reduced,true);''',
    'embedding_bag_counts_cpu': 't[1].copy_(at::bincount(t[0],{},E));',
    # COO construction/coalescing and index conversion remain in the native
    # timing, as does its allocation/copy into the extracted stage's output.
    'sparse_addmm_cpu': '''auto matrix=at::sparse_coo_tensor(
      at::stack({t[0],t[1]}).to(at::kLong),t[2],{R,64},
      t[2].options().layout(at::kSparse));
      t[4].view({R,C}).copy_(at::mm(matrix,t[3].view({64,C})));''',
    'hspmm_cpu': '''auto matrix=at::sparse_coo_tensor(
      at::stack({t[0],t[1]}).to(at::kLong),t[2],{R,64},
      t[2].options().layout(at::kSparse));
      t[4].view({R,C}).copy_(at::mm(matrix,t[3].view({64,C})));''',
    'sspaddmm_cpu': '''auto matrix=at::sparse_coo_tensor(
      at::stack({t[0],t[1]}).to(at::kLong),t[2],{R,K},
      t[2].options().layout(at::kSparse));
      t[4].view({R,C}).copy_(at::mm(matrix,t[3].view({K,C})));''',
    'sparse_addmv_csr_cpu': '''static auto matrix=at::sparse_csr_tensor(
      t[0],t[1],t[2],{R,C},t[2].options().layout(at::kSparseCsr));
      at::mv_out(t[4],matrix,t[3]);''',
    # Match sparse_addmv_bsr_exact's public API, long indices, sparse
    # construction outside timing, and preallocated single-column output.
    'sparse_addmv_bsr_cpu': '''static auto matrix=at::sparse_bsr_tensor(
      t[0].to(at::kLong),t[1].to(at::kLong),t[2].view({N,BR,BC}),
      {R*BR,C*BC},t[2].options().layout(at::kSparseBsr));
      auto output=t[4].view({R*BR,1});
      at::mm_out(output,matrix,t[3].view({C*BC,1}));''',
    'sparse_csr_addmm_cpu': '''static auto matrix=at::sparse_csr_tensor(
      t[0],t[1],t[2],{R,K},t[2].options().layout(at::kSparseCsr));
      auto output=t[4].view({R,C});at::mm_out(output,matrix,t[3].view({K,C}));''',
    # Same public API and preallocated sparse output as the publication's
    # whole_sampled_addmm_sparse_csr native harness.
    'sampled_addmm_sparse_csr_cpu': '''static auto self=at::sparse_csr_tensor(
      t[0],t[1],t[2],{R,C},t[2].options().layout(at::kSparseCsr));
      static auto output=at::sparse_csr_tensor(
      t[0],t[1],t[5],{R,C},t[5].options().layout(at::kSparseCsr));
      at::sparse_sampled_addmm_out(output,self,t[3].view({R,K}),t[4].view({K,C}),s[1],s[0]);''',
    'sparse_matmul_cpu': '''static auto matrix=at::sparse_csr_tensor(
      t[0],t[1],t[2],{R,R},t[2].options().layout(at::kSparseCsr));
      auto output=t[4].view({R,C});at::mm_out(output,matrix,t[3].view({R,C}));''',
    # Real ATen expansion of CSR row counts. Allocation and copying the
    # returned indices into the fixture's output are included in timing.
    'convert_csr_to_coo_cpu': '''static auto rows=at::arange(R,t[0].options());
      auto result=at::repeat_interleave(rows,t[0].slice(0,1,R+1)-t[0].slice(0,0,R),0,N);
      t[2].copy_(result);''',
    'sparse_matmul_csr_to_coo_cpu': '''static auto rows=at::arange(R,t[0].options());
      auto result=at::repeat_interleave(rows,t[0].slice(0,1,R+1)-t[0].slice(0,0,R),0,N);
      t[1].copy_(result);''',
    'conv_transpose3d_cpu': '''auto out=t[2].view({1,O,D+2,H+2,W+2});
      at::convolution_out(out,t[0].view({1,C,D,H,W}),t[1].view({C,O,K,K,K}),
        std::nullopt,{1,1,1},{0,0,0},{1,1,1},true,{0,0,0},1);''',
    # Real ATen input gradient; allocation and copy into the fixture output
    # are timed, as for the weight-gradient API below.
    'slow_conv3d_backward_input_cpu': '''static auto input=at::zeros({1,C,D+2,H+2,W+2},t[0].options());
      auto grads=at::convolution_backward(t[0].view({1,O,D,H,W}),input,
        t[1].view({O,C,K,K,K}),{}, {1,1,1},{0,0,0},{1,1,1},
        false,{0,0,0},1,{true,false,false});
      t[2].view({1,C,D+2,H+2,W+2}).copy_(std::get<0>(grads));''',
    # Real ATen weight gradients; returned gradient allocation and the copy
    # into the fixture output remain inside the timed native invocation.
    'slow_conv3d_backward_weight_cpu': '''static auto weight=at::empty({O,C,K,K,K},t[0].options());
      auto grads=at::convolution_backward(t[1].view({1,O,D,H,W}),
        t[0].view({1,C,D+2,H+2,W+2}),weight,{}, {1,1,1},{0,0,0},{1,1,1},
        false,{0,0,0},1,{false,true,false});
      t[2].view({O,C,K,K,K}).copy_(std::get<1>(grads));''',
    'conv_transpose3d_grad_weight_cpu': '''static auto weight=at::empty({C,O,K,K,K},t[0].options());
      auto grads=at::convolution_backward(t[1].view({1,O,D+2,H+2,W+2}),
        t[0].view({1,C,D,H,W}),weight,{}, {1,1,1},{0,0,0},{1,1,1},
        true,{0,0,0},1,{false,true,false});
      t[2].view({C,O,K,K,K}).copy_(std::get<1>(grads));''',
    'clamp_min_scalar_cpu': 'at::clamp_min_out(t[1],t[0],s[0]);',
    'clamp_max_scalar_cpu': 'at::clamp_max_out(t[1],t[0],s[0]);',
    'cumprod_cpu': 'auto out=t[1].view({R,K}); at::cumprod_out(out,t[0].view({R,K}),1);',
    'transpose_copy': 't[1].view({N,M}).copy_(t[0].view({M,N}).transpose(0,1));',
    'sum': 'at::sum_out(t[1],t[0].view({M,N}),{1},false);',
    'mm': 'auto out=t[2].view({M,N}); at::mm_out(out,t[0].view({M,K}),t[1].view({K,N}));',
    'cpu_blas_gemm_cpu': 'auto out=t[2].view({M,N}); at::mm_out(out,t[0].view({M,K}),t[1].view({K,N}));',
    'bmm': 'auto out=t[2].view({BATCH,M,N}); at::bmm_out(out,t[0].view({BATCH,M,K}),t[1].view({BATCH,K,N}));',
    'nested_bmm_cpu': 'auto out=t[2].view({B,M,N}); at::bmm_out(out,t[0].view({B,M,K}),t[1].view({B,K,N}));',
    'sumproduct_pair_cpu': 'auto out=t[2].view({B,M,N}); at::bmm_out(out,t[0].view({B,M,K}),t[1].view({B,K,N}));',
    'cpu_blas_gemm_batched_cpu': 'auto out=t[2].view({B,M,N}); at::bmm_out(out,t[0].view({B,M,K}),t[1].view({B,K,N}));',
    'cpu_blas_gemm_strided_batched_cpu': 'auto out=t[2].view({B,M,N}); at::bmm_out(out,t[0].view({B,M,K}),t[1].view({B,K,N}));',
    'nested_matmul_broadcast_cpu': 'auto out=t[2].view({B*M,N}); at::mm_out(out,t[0].view({B*M,K}),t[1].view({K,N}));',
    'nested_sum_dim_cpu': '''static auto columns=at::arange(N,t[1].options());
      static auto mask=at::empty({B,N},t[0].options().dtype(at::kBool));
      static auto selected=at::empty({B,N},t[0].options());
      static auto zero=at::zeros({},t[0].options());
      at::lt_out(mask,columns.view({1,N}),t[1].view({B,1}));
      at::where_out(selected,mask,t[0].view({B,N}),zero); at::sum_out(t[2],selected,{1},false);''',
    'int_mm_out_cpu': 'auto out=t[2].view({M,N}); at::_int_mm_out(out,t[0].view({M,K}),t[1].view({K,N}));',
    'quant_col_offsets_cpu': 'at::sum_out(t[1],t[0].view({K,N}),{0},false,at::kInt); t[1].sub_((int32_t)s[0]*K);',
    'allany_dims_cpu': '''static auto reduced=at::empty({R},t[0].options().dtype(at::kBool));
      if(s[0]!=0)at::all_out(reduced,t[0].view({R,C}),1,false);
      else at::any_out(reduced,t[0].view({R,C}),1,false);
      t[1].copy_(reduced);''',
    'nansum_cpu': 'at::nansum_out(t[1],t[0].view({R,K}),{1},false,std::nullopt);',
    'outer': 'auto out=t[2].view({M,N}); at::outer_out(out,t[0],t[1]);',
    'addmm': 'auto out=t[2].view({M,N}); at::addmm_out(out,out,t[0].view({M,K}),t[1].view({K,N}),s[0],s[1]);',
    'exponential_cpu': '''static auto scratch=at::empty_like(t[0]);
      at::neg_out(scratch,t[0]); at::log1p_out(scratch,scratch);
      at::neg_out(scratch,scratch); at::div_out(t[1],scratch,s[0]);''',
    # The installed libtorch predates hash_tensor. Keep this explicit ATen
    # composition distinct from a timing of the newer xor_sum CUDA kernel.
    'xor_sum_cpu': '''static std::vector<at::Tensor> scratch;
      if(scratch.empty())for(int64_t width=K;width>1;width=(width+1)/2)
        scratch.push_back(at::empty({R,(width+1)/2},t[0].options()));
      auto current=t[0].view({R,K});size_t stage=0;
      for(int64_t width=K;width>1;width=(width+1)/2){
        auto next=scratch[stage++];auto pairs=next.slice(1,0,width/2);
        at::bitwise_xor_out(pairs,current.slice(1,0,width-1,2),current.slice(1,1,width,2));
        if(width%2)next.slice(1,width/2,width/2+1).copy_(current.slice(1,width-1,width));
        current=next;
      }
      t[1].view({R,1}).copy_(current);''',
    'avg_pool2d': 'auto out=t[1].view({B,C,H/2,W/2}); at::avg_pool2d_out(out,t[0].view({B,C,H,W}),{2,2},{2,2},{0,0},false,true,std::nullopt);',
    'avg_pool3d': 'auto out=t[1].view({B,C,D/2,H/2,W/2}); at::avg_pool3d_out(out,t[0].view({B,C,D,H,W}),{2,2,2},{2,2,2},{0,0,0},false,true,std::nullopt);',
    'adaptive_avg_pool2d': 'auto out=t[1].view({B,C,OH,OW}); at::adaptive_avg_pool2d_out(out,t[0].view({B,C,H,W}),{OH,OW});',
    'adaptive_avg_pool3d': 'auto out=t[1].view({B,C,4,4,4}); at::adaptive_avg_pool3d_out(out,t[0].view({B,C,D,H,W}),{4,4,4});',
    'avg_pool2d_backward_cpu': '''static auto input=at::empty({B,C,I0,I1},t[0].options());
      auto out=t[1].view({B,C,I0,I1}); at::avg_pool2d_backward_out(out,t[0].view({B,C,I0/2,I1/2}),input,{2,2},{2,2},{0,0},false,true,std::nullopt);''',
    'avg_pool3d_backward_cpu': '''static auto input=at::empty({B,C,I0,I1,I2},t[0].options());
      auto out=t[1].view({B,C,I0,I1,I2}); at::avg_pool3d_backward_out(out,t[0].view({B,C,I0/2,I1/2,I2/2}),input,{2,2,2},{2,2,2},{0,0,0},false,true,std::nullopt);''',
    'conv1d': 'auto out=t[3].view({B,OC,W-K+1}); at::convolution_out(out,t[0].view({B,IC,W}),t[1].view({OC,IC,K}),t[2],{1},{0},{1},false,{0},1);',
    'conv2d': 'auto out=t[2].view({B,OC,H-KH+1,W-KW+1}); at::convolution_out(out,t[0].view({B,IC,H,W}),t[1].view({OC,IC,KH,KW}),std::nullopt,{1,1},{0,0},{1,1},false,{0,0},1);',
    'dilated_convolution_cpu': 'auto out=t[2].view({1,O,H-2*D,W-2*D}); at::convolution_out(out,t[0].view({1,C,H,W}),t[1].view({O,C,K,K}),std::nullopt,{1,1},{0,0},{D,D},false,{0,0},1);',
    'conv_transpose3d_backward_cpu': 'auto out=t[2].view({1,C,D,H,W}); at::convolution_out(out,t[0].view({1,O,D+2,H+2,W+2}),t[1].view({C,O,K,K,K}),std::nullopt,{1,1,1},{0,0,0},{1,1,1},false,{0,0,0},1);',
    'conv_transpose2d': 'auto out=t[2].view({B,OC,H+K-1,W+K-1}); at::convolution_out(out,t[0].view({B,IC,H,W}),t[1].view({IC,OC,K,K}),std::nullopt,{1,1},{0,0},{1,1},true,{0,0},1);',
    'conv_tbc_cpu': 'static auto bias=at::zeros({O},t[0].options()); auto out=t[2].view({T-K+1,B,O}); at::conv_tbc_out(out,t[0].view({T,B,I}),t[1].view({K,I,O}),bias,0);',
    # The extracted fixture computes only dInput. Use the real ATen backward
    # API with that output mask; include layout conversion, allocation and
    # the copy back into TBC storage in the measured native invocation.
    'conv_tbc_backward_cpu': '''static auto input=at::zeros({B,I,T+K-1},t[0].options());
      auto grads=at::convolution_backward(t[0].view({T,B,O}).permute({1,2,0}),input,
        t[1].view({K,I,O}).permute({2,1,0}),{}, {1},{0},{1},
        false,{0},1,{true,false,false});
      t[2].view({T+K-1,B,I}).copy_(std::get<0>(grads).permute({2,0,1}));''',
    'count_nonzero_impl_cpu': 'static auto counts=at::empty({R},t[0].options().dtype(at::kLong)); at::count_nonzero_out(counts,t[0].view({R,C}),std::optional<int64_t>{1}); t[1].copy_(counts);',
    'renorm_scale_factor': 'auto iter=at::TensorIteratorConfig().add_output(t[1]).add_input(t[0]).build(); at::native::renorm_scale_factor_stub(at::kCUDA,iter,(double)s[0]);',
    'blas_copy_cpu': 't[1].copy_(t[0]);',
    'copy_cpu': 't[1].copy_(t[0]);',
    'copy_tensor_array_cpu': 't[1].copy_(t[0]);',
    'nested_squeeze_cpu': 't[1].copy_(t[0]);',
    'nested_clone_cpu': 't[1].copy_(t[0]);',
    'unbind_copy_cpu': 't[1].copy_(t[0]);',
    'as_complex_cpu': 't[1].copy_(t[0].slice(0,0,t[0].numel(),2)); t[2].copy_(t[0].slice(0,1,t[0].numel(),2));',
    'complex_scalarized': 't[2].copy_(t[0]); t[3].copy_(t[1]);',
    'conj_complex_scalarized': 't[2].copy_(t[0]); at::neg_out(t[3],t[1]);',
    'angle_complex_scalarized': 'at::atan2_out(t[2],t[1],t[0]);',
    'sparse_mul_cpu': 'at::mul_out(t[2],t[0],t[1]);',
    'sparse_intersection_apply_cpu': 'at::mul_out(t[2],t[0],t[1]);',
    'sparse_intersection_launch_cpu': 'at::mul_out(t[2],t[0],t[1]);',
    'sparse_add_values_cpu': 'at::add_out(t[2],t[0],t[1],s[0]);',
    'blas_axpy_cpu': 't[1].add_(t[0],s[0]);',
    'blas_scale_cpu': 't[0].mul_(s[0]);',
    'cat_serial_cpu': 't[2].slice(0,0,t[0].numel()).copy_(t[0]); t[2].slice(0,t[0].numel(),t[2].numel()).copy_(t[1]);',
    'narrow_copy_dense_cpu': 'auto out=t[1].view({R,L}); out.copy_(t[0].view({R,C}).narrow(1,S,L));',
    'im2col': 'auto out=t[1].view({B,C*KH*KW,(H-KH+1)*(W-KW+1)}); at::im2col_out(out,t[0].view({B,C,H,W}),{KH,KW},{1,1},{0,0},{1,1});',
    'conv2d_columns_cpu': 'auto out=t[1].view({1,C*K*K,(H-2)*(W-2)}); at::im2col_out(out,t[0].view({1,C,H,W}),{K,K},{1,1},{0,0},{1,1});',
    'sparse_bmm_cpu': 'auto out=t[2].view({B,M,N}); at::bmm_out(out,t[0].view({B,M,K}),t[1].view({B,K,N}));',
    'flatten_nd_linear_cpu': 'auto out=t[2].view({B*M,N}); at::mm_out(out,t[0].view({B*M,K}),t[1].view({K,N}));',
    'normal_cpu': 'at::add_out(t[1],t[0]*s[1],s[0]);',
    'uniform_cpu': 'at::add_out(t[1],t[0]*(s[1]-s[0]),s[0]);',
    'log_normal_cpu': 'at::exp_out(t[1],t[0]*s[1]+s[0]);',
    'cross': 'auto out=t[2].view({N,3}); at::linalg_cross_out(out,t[0].view({N,3}),t[1].view({N,3}),1);',
    'cross_cpu_backend': 'auto out=t[2].view({V,3}); at::linalg_cross_out(out,t[0].view({V,3}),t[1].view({V,3}),1);',
    'polar_scalarized': '''static auto z=at::empty_like(t[0],t[0].options().dtype(at::kComplexFloat));
      at::polar_out(z,t[0],t[1]); t[2].copy_(at::real(z)); t[3].copy_(at::imag(z));''',
    'gamma_transform_cpu': 'auto d=t[0]-.3333333f; auto v=1.0f+at::reciprocal(at::sqrt(9.0f*d))*t[1]; at::mul_out(t[3],d*v*v,v);',
    'standard_gamma_grad_cpu': 'at::add_out(t[2],(t[1]-t[0])/(t[0]+.001f),at::log(t[1]+.001f));',
    'dirichlet_grad_cpu': 'at::mul_out(t[3],t[0],at::log(t[0]+.001f)-t[1]/t[2]);',
    'geometric_cpu': 'at::ceil_out(t[1],at::log(1.0f-t[0])/::logf(1.0f-s[0]));',
    'cauchy_cpu': 'at::add_out(t[1],at::tan(3.14159265358979323846f*(t[0]-.5f))*s[1],s[0]);',
    'sparse_norm_cpu': 'auto total=at::sum(t[0]*t[0]).view({1}); at::sqrt_out(t[1],total);',
    'pixel_shuffle': 'auto out=t[1].view({B,C,H*R,W*R}); at::pixel_shuffle_out(out,t[0].view({B,C*R*R,H,W}),R);',
    'pixel_shuffle_cpu_backend': 'auto out=t[1].view({B,C,H*RATIO,W*RATIO}); at::pixel_shuffle_out(out,t[0].view({B,C*RATIO*RATIO,H,W}),RATIO);',
    'pixel_unshuffle_cpu_backend': 'auto out=t[1].view({B,C*RATIO*RATIO,H,W}); at::pixel_unshuffle_out(out,t[0].view({B,C,H*RATIO,W*RATIO}),RATIO);',
    # The public math_channel_shuffle implementation is reshape/permute/copy.
    # Keep its materialization in the preallocated output, as for other views.
    'channel_shuffle': 't[1].view({B,CPG,G,H,W}).copy_(t[0].view({B,G,CPG,H,W}).permute({0,2,1,3,4}));',
    'channel_shuffle_cpu': 't[1].view({B,CPG,G,S}).copy_(t[0].view({B,G,CPG,S}).permute({0,2,1,3}));',
    'stack_serial_cpu': 't[1].view({R,T,K}).copy_(t[0].view({T,R,K}).permute({1,0,2}));',
    'compressed_block_convert_cpu': 't[1].view({R/BR,C/BC,BR,BC}).copy_(t[0].view({R/BR,BR,C/BC,BC}).permute({0,2,1,3}));',
    'repeat_compute_cpu': 'at::repeat_out(t[1],t[0],{R});',
    'repeat_tensor_shape_cpu': 'at::repeat_out(t[1],t[0],{R});',
    'nested_sum_backward_cpu': 't[1].view({B,N}).copy_(t[0].view({B,1}).expand({B,N}));',
    'diff_cpu': 'at::diff_out(t[1],t[0]);',
    'kron_impl_cpu': 'auto out=t[2].view({A*C,B*D}); at::kron_out(out,t[0].view({A,B}),t[1].view({C,D}));',
    'kron_out_cpu': 'auto out=t[2].view({A*C,B*D}); at::kron_out(out,t[0].view({A,B}),t[1].view({C,D}));',
    'zeros_cpu': 't[0].zero_();',
    'and_reduce_cpu': 'static auto logical=at::empty({R},t[0].options().dtype(at::kBool)); at::all_out(logical,t[0].view({R,K}),1,false); t[1].copy_(logical);',
    'or_reduce_cpu': 'static auto logical=at::empty({R},t[0].options().dtype(at::kBool)); at::any_out(logical,t[0].view({R,K}),1,false); t[1].copy_(logical);',
    'batch_norm': 'auto out=t[5].view({B,C,H,W}); at::batch_norm_elemt_out(out,t[0].view({B,C,H,W}),t[1],t[4],t[2],t[3],0.0);',
    'batch_norm_transform_cpu': 'auto out=t[5].view({N,C,H,W}); at::batch_norm_elemt_out(out,t[0].view({N,C,H,W}),t[3],t[4],t[1],t[2],0.0);',
    'depthwise_conv3x3_cpu': '''auto out=t[3].view({B,C,H,W});
      at::convolution_out(out,t[0].view({B,C,H,W}),t[1].view({C,1,3,3}),
        t[2],{1,1},{1,1},{1,1},false,{0,0},C);''',
    'dropout_feature_noise_cpu': 'auto out=t[2].view({B,C,H,W}); at::mul_out(out,t[0].view({B,C,H,W}),t[1].view({B,C,1,1})); out.mul_(s[0]);',
    'transform_bias_rescale_qkv_cpu': 'auto input=t[0].view({B,S,3,H,D}); auto bias=t[1].view({3,H,D}); for(int i=0;i<3;i++){auto out=t[i+2].view({B,H,S,D}); at::add_out(out,input.select(2,i).permute({0,2,1,3}),bias.select(0,i).view({1,H,1,D})); if(i==0)out.mul_(s[0]);}',
}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n'); temp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configuration(name, shape_record=None):
    cfg = BENCH.auto_spec(name) or BENCH.CASES.get(name)
    if cfg and any(a[1] in ('dptr','dscalar') for a in cfg['args']):
        if any(a[1] not in ('dptr','dscalar') for a in cfg['args']):
            raise ValueError('Mixed FP64 storage is not supported by this harness')
        cfg = dict(cfg, dtype='double', args=[
            (n, 'ptr' if k=='dptr' else 'scalar',v,o,t) for n,k,v,o,t in cfg['args']])
    if not cfg or any(a[1] not in ('ptr', 'iptr', 'bptr', 'scalar', 'iscalar') for a in cfg['args']):
        raise ValueError('This harness supports FP32/int8/int32 or uniform FP64 storage')
    text = (BENCH.ATEN/(name+'.c')).read_text()
    cfg['dims'] = {k: int(v) for k,v in re.findall(r'^\s*#\s*define\s+(\w+)\s+(\d+)\s*$', text, re.M)}
    cfg['dimension_scope'] = 'unchanged C fixture dimensions'
    if shape_record is not None:
        from aten_paper_shapes import apply_shape, input_recipe
        cfg = apply_shape(name, cfg, shape_record)
        cfg = input_recipe(name, cfg, shape_record, BENCH)
    if name == 'aten_nested_sum_dim_cpu':
        width=cfg['dims']['N']
        cfg['integer_array_values']={'len':[-1,0,1,width,width//2,max(0,width-1),2,width//3]}
    if name == 'aten_nested_all_cpu':
        width=cfg['dims']['N']
        cfg['integer_array_values']={'len':[0,1,width,width//2,max(0,width-1),min(2,width),width//3]}
    return cfg


def harness(name, cfg):
    sizes, outputs, integers, elements, decl, actual = [], [], [], [], [], []
    integer_parameters = []
    for n,k,value,output,_ in cfg['args']:
        if k in ('ptr', 'iptr', 'bptr'):
            element = {'ptr':'float', 'iptr':'int32_t', 'bptr':'int8_t'}[k]
            actual.append('('+element+'*)p['+str(len(sizes))+']')
            decl.append(element+'*'); integers.append({'ptr':0,'iptr':1,'bptr':2}[k])
            elements.append('sizeof('+element+')')
            sizes.append(value); outputs.append(int(output))
        else:
            actual.append(('(int32_t)' if k=='iscalar' else '')+
                          'parameters['+str(len(integer_parameters))+']')
            decl.append('int32_t' if k=='iscalar' else 'float')
            integer_parameters.append(int(k=='iscalar'))
    ns = len(integer_parameters)
    prefix = '\n'.join('#define '+k+' '+str(v) for k,v in cfg['dims'].items())
    prefix += '''
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime_api.h>
'''
    prefix += f'#define NA {len(sizes)}\n#define NS {ns}\n#define ABS_TOL 1e-5\n#define REL_TOL 1e-4\n'
    samples = cfg.get('timing_samples', 5)
    if type(samples) is not int or samples < 5:
        raise ValueError('timing_samples must be an integer of at least five')
    prefix += f'#define TIMING_SAMPLES {samples}\n'
    prefix += '#define PAPER_RECIPE '+str(int('paper_initializers' in cfg))+'\n'
    prefix += 'static double paper_value(int array,size_t i){switch(array){\n'
    for i,(n,k,_,_,_) in enumerate(a for a in cfg['args'] if a[1] in ('ptr','iptr','bptr')):
        expression = cfg.get('paper_initializers',{}).get(n)
        if expression is not None:
            prefix += f'case {i}:return '+expression+';\n'
    prefix += '} (void)i;return 0;}\n'
    paper_parameters = [str(cfg.get('paper_parameters',{}).get(a[0],0))
                        for a in cfg['args'] if a[1] in ('scalar','iscalar')]
    prefix += 'static const float paper_parameters[]={'+','.join(paper_parameters or ['0'])+'};\n'
    prefix += 'static float parameters['+str(max(1,ns))+'];\n'
    prefix += 'static const int integer_parameters[]={'+','.join(map(str,integer_parameters or [0]))+'};\n'
    prefix += 'static const size_t sizes[]={'+','.join(sizes)+'};\n'
    prefix += 'static const int outputs[]={'+','.join(map(str,outputs))+'};\n'
    prefix += 'static const int integer_arrays[]={'+','.join(map(str,integers))+'};\n'
    prefix += 'static const size_t element_bytes[]={'+','.join(elements)+'};\n'
    prefix += 'static int initialize_integer_metadata(int array,size_t index,int32_t *value){\n'
    for i,(n,k,_,_,_) in enumerate(a for a in cfg['args'] if a[1] in ('ptr','iptr','bptr')):
        values=cfg.get('integer_array_values',{}).get(n)
        if values is not None:
            if k!='iptr' or not values or any(type(v) is not int or not -(1<<31)<=v<(1<<31) for v in values):
                raise ValueError('invalid fixture integer metadata values: '+n)
            prefix+=f' if(array=={i}){{static const int32_t data[]={{'+','.join(map(str,values))+f'}};*value=data[index%{len(values)}];return 1;}}\n'
    prefix += ' (void)array;(void)index;(void)value;return 0;}\n'
    prefix += 'void '+name+'('+','.join(decl)+');\nvoid native_reference('+','.join(decl)+');\n'
    prefix += 'void native_fixture(float**,const float*,const size_t*,int);\n'
    prefix += 'static void invoke(float **p){'+name+'('+','.join(actual)+');}\n'
    prefix += 'static void invoke_reference(float **p){native_reference('+','.join(actual)+');}\n'
    prefix += 'static void invoke_native(float **p){native_fixture(p,parameters,sizes,NA);}\n'
    text=prefix+(HERE/'aten_fixture_batch_harness.c').read_text()
    if cfg.get('dtype') == 'double':
        text=re.sub(r'\bfloat\b','double',text).replace('FLT_MAX','DBL_MAX').replace('FLT_MIN','DBL_MIN')
        text=text.replace('#include <double.h>','#include <float.h>')
        text=text.replace('0x1p-149f','0x1p-1074').replace('#define ABS_TOL 1e-5','#define ABS_TOL 1e-12')
        text=text.replace('#define REL_TOL 1e-4','#define REL_TOL 1e-10')
    return text


def bridge(name, cfg):
    integers = ','.join(str({'ptr':0,'iptr':1,'bptr':2}[a[1]]) for a in cfg['args'] if a[1] in ('ptr','iptr','bptr'))
    text='#define INTEGER_ARRAYS {'+integers+'}\n'+'''#include <ATen/ATen.h>
#include <ATen/native/Normalization.h>
#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <cmath>
#include <cstdio>
#include <vector>
extern "C" void native_fixture(float **p,const float *s,const size_t *sizes,int count){
 static std::vector<at::Tensor> t;
 if(t.empty()){
  at::set_num_threads(1);
  at::globalContext().setAllowTF32CuBLAS(false);
  at::globalContext().setAllowTF32CuDNN(false);
  const int integer_arrays[]=INTEGER_ARRAYS;
  for(int i=0;i<count;i++)t.push_back(at::from_blob(p[i],{(int64_t)sizes[i]},at::TensorOptions().dtype(integer_arrays[i]==2?at::kChar:integer_arrays[i]?at::kInt:at::kFloat).device(at::kCUDA)));
  std::puts("NATIVE_BACKEND real libtorch CUDA; preallocated extracted-stage outputs");
 }
 '''+'\n'.join('constexpr int64_t '+k+'='+str(v)+';' for k,v in cfg['dims'].items())+'\n'+NATIVE[name.removeprefix('aten_')]+ '\n}\n'
    if cfg.get('dtype') == 'double':
        text=re.sub(r'\bfloat\b','double',text).replace('at::kFloat','at::kDouble')
    return text


def build(name, args):
    dest = args.output/name; dest.mkdir(parents=True, exist_ok=True)
    src = BENCH.ATEN/(name+'.c')
    rank = args.rank
    if rank == -1:
        result = json.loads((args.campaign/name/'result.json').read_text())
        variants = [v for v in result['variants'] if v.get('abi_ok') and
                    v.get('emitted') and not v['counts']['linalg']]
        if not variants:
            raise ValueError(name+': no fully matched variant with executable ABI')
        rank = min(variants, key=lambda v: (v['counts']['loops'], len(v['emitted'])))['rank']
    proofdir = args.campaign/name/('rank'+str(rank))
    proof = json.loads((proofdir/'proof.json').read_text())
    if not proof['emitted']:
        raise ValueError(name+': no specification proof')
    if re.search(r'\blinalg\.(generic|reduce|map)\b', (proofdir/'matched.mlir').read_text()):
        raise ValueError(name+': residual computation needs investigation before resident execution')
    shapes = getattr(args, 'shape_specs', None)
    record = None
    if shapes:
        from aten_paper_shapes import load_shapes, materialize
        record = load_shapes(shapes)[name]
    cfg = configuration(name, record)
    cfg['timing_samples'] = getattr(args, 'timing_samples', 5)
    original_src = src
    if record:
        src = dest / (name+'.c')
        materialize(original_src, src, record)
        request = json.loads((proofdir.parent/'request.json').read_text())
        if sha(src) != sha(ROOT / request['source']):
            raise ValueError('Benchmark source differs from the source used for the saved match')
    (dest/'harness.c').write_text(harness(name,cfg)); (dest/'native.cpp').write_text(bridge(name,cfg))
    env=dict(os.environ, PYTHON='/usr/bin/python3', POLYGEIST_SPEC_SUITE='lib_v3',
        POLYGEIST_NUMERIC_MODEL=proof['configuration']['numeric_model'],
        POLYGEIST_BUILD_WORK_DIR=str(dest/'work'), POLYGEIST_MATCH_PROOF=str(proofdir/'proof.json'),
        POLYGEIST_CUSTOM_CUDA_OBJ=str(dest/'reference.o'), POLYGEIST_MINIMAL_CUDNN_RUNTIME='1')
    libraries = {target.get('library') for match in proof['matches']
                 for target in match.get('targets', [])}
    if 'cuSPARSE' in libraries:
        env['POLYGEIST_MINIMAL_CUDNN_RUNTIME'] = '0'
        env['POLYGEIST_MINIMAL_CUSPARSE_RUNTIME'] = '1'
    # Repeated fixtures share the same runtime translation units. Let ccache
    # validate compiler flags and header dependencies instead of recompiling
    # those units for every case. The resulting binary is still hashed below.
    compiler_cache = (Path('/usr/lib/ccache') if shutil.which('ccache') and
                      Path('/usr/lib/ccache/aarch64-linux-gnu-gcc').is_file() else None)
    if compiler_cache:
        env['PATH'] = str(compiler_cache) + os.pathsep + env.get('PATH', '')
        env.setdefault('POLYGEIST_NVCC', str(HERE/'cached_nvcc.sh'))
    if args.cutensor_root:
        env['POLYGEIST_CUTENSOR_ROOT']=str(args.cutensor_root)
    if args.persistent_workspace:
        env['POLYGEIST_PERSISTENT_WORKSPACE_FUNCTION']=name
    commands=[['aarch64-linux-gnu-gcc','-O2','-fno-fast-math','-ffp-contract=off',
        '-D'+name+'=native_reference','-c',str(src),'-o',str(dest/'reference.o')],
        ['aarch64-linux-gnu-g++','-O2','-std=c++17','-DHAVE_SVE256_CPU_DEFINITION',
         '-I'+str(TORCH/'include'),'-I'+str(TORCH/'include/torch/csrc/api/include'),
         '-I'+str(CUDA/'include'),'-c',str(dest/'native.cpp'),'-o',str(dest/'native.o')],
        [str(HERE/'polygeist_build.sh'),'--target=jetson','--function='+name,
         '--semantic-mlir='+str(proofdir/'matched.mlir'),'--harness='+str(dest/'harness.c'),
         '-o',str(dest/'raised-program'),str(src),'-I'+str(CUDA/'include')]]
    # Build the normal compiler/runtime objects, then link the independent
    # native ATen baseline into the same measurement process.
    env['POLYGEIST_CUSTOM_CUDA_OBJ'] = str(dest/'reference.o')+' '+str(dest/'native.o')
    env['POLYGEIST_SKIP_LINK'] = '1'
    rc=0
    with (dest/'build.log').open('w') as log:
        for cmd in commands:
            log.write(shlex.join(cmd)+'\n');log.flush()
            rc=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600).returncode
            if rc:break
        if not rc:
            work = dest/'work'
            objects = [work/(n+'.o') for n in ['kernel','wrapper','harness','rt','mlir_runner_utils','spec_runtime']]
            objects += [p for p in [work/'thrust_runtime.o',work/'cub_runtime.o'] if p.exists()]
            objects += [dest/'reference.o',dest/'native.o']
            cmd=['aarch64-linux-gnu-g++','-O2',*map(str,objects),
                '-L'+str(TORCH/'lib'),'-L'+str(CUDA/'lib'),'-L'+str(CUDA/'lib/stubs'),
                '-L/usr/lib/aarch64-linux-gnu','-Wl,--allow-shlib-undefined','-Wl,--no-as-needed',
                '-ltorch','-ltorch_cpu','-ltorch_cuda','-lc10_cuda','-lc10',
                '-lcudnn','-lcublasLt','-lcublas','-lcusolver','-lcudart','-lm','-lpthread','-ldl',
                '-Wl,--gc-sections','-o',str(dest/'raised-program')]
            if args.cutensor_root:
                cmd += ['-L'+str(args.cutensor_root/'lib'),'-l:libcutensor.so.2']
            if 'cuSPARSE' in libraries:
                cmd += ['-lcusparse']
            commands.append(cmd);log.write(shlex.join(cmd)+'\n');log.flush()
            rc=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120).returncode
    row=dict(kernel=name,rc=rc,configuration=cfg,commands=commands,rank=rank,
             compiler_cache=str(compiler_cache) if compiler_cache else None,
             persistent_workspace=args.persistent_workspace,
             source_sha256=sha(src),proof_sha256=sha(proofdir/'proof.json'),
             baseline='real ATen CUDA calls for the extracted fixture; preallocated outputs',
             numerical_policy=('NaN classes, infinity signs, zero signs; ' +
                 ('FP64 atol=1e-12 rtol=1e-10' if cfg.get('dtype')=='double' else 'FP32 atol=1e-5 rtol=1e-4')))
    if name == 'aten_xor_sum_cpu':
        row['baseline']='ATen CUDA bitwise_xor tree with preallocated intermediates; installed libtorch lacks hash_tensor; not the newer xor_sum kernel'
    if name == 'aten_quant_col_offsets_cpu':
        row['baseline']='ATen CUDA sum and subtract composition; preallocated outputs'
    if name == 'aten_nested_sum_dim_cpu':
        row['baseline']='ATen CUDA length mask, where and sum composition; preallocated intermediates'
    if name == 'aten_nested_all_cpu':
        row['baseline']='ATen CUDA length mask, nonzero, where, all_out and int32 copy; preallocated intermediates, matching historical native_exact_region API scope'
        row['paper_baseline_input_caveat']='Recovered publication harness has zero x and lengths; historical native_exact_region uses patterned boolean x and lengths N-(b%31). Current raised and native use identical recorded inputs; paper speedup has different native inputs.'
    if name == 'aten_polar_scalarized':
        row['baseline']='ATen CUDA polar_out plus real/imaginary copies to the separate fixture outputs'
    if name == 'aten_standard_gamma_grad_cpu':
        row['baseline']='ATen CUDA arithmetic and log composition for the extracted simplified formula; not the full ATen standard-gamma-gradient operator'
    if record:
        row.update(original_source_sha256=sha(original_src), shape_record=record,
                   shape_specs_sha256=sha(shapes))
    if not rc:row['binary_sha256']=sha(dest/'raised-program')
    save(dest/'build.json',row);print(name,'build',rc,flush=True)
    return row


def correctness_summary(rc, log, expected, modes):
    checks=[]
    for line in log.splitlines():
        if line.startswith('CHECK '):
            checks.append({k:float(v) for k,v in re.findall(r'(\w+)=([+\-\d.eE]+)',line)})
    vendor_calls=len(re.findall(r'POLYGEIST_V2_CALL',log))
    # The harness returns 3 only after completing comparisons with mismatches.
    # CUDA errors, crashes, missing output or an unexpected exit still fail.
    complete=(len(checks)==modes and vendor_calls>0 and
              all(c.get('count')==expected and all(k in c for k in
                  ('bad','nativebad','guardbad','inputbad')) for c in checks))
    complete=complete and ((rc==0 and 'PASS correctness' in log) or
        (rc==3 and any(any(c.get(k,0)>0 for k in ('bad','nativebad','nativerefbad','guardbad','inputbad')) for c in checks)))
    raised=complete and all(all(c[k]==0 for k in ('bad','guardbad','inputbad')) for c in checks)
    agreement=complete and all(all(c[k]==0 for k in ('nativebad','guardbad','inputbad')) for c in checks)
    # Historical logs compare raised output to native, not native to C.
    # Numerical tolerance is not transitive, so never infer the third comparison.
    native=(complete and all(all(c[k]==0 for k in ('nativerefbad','guardbad','inputbad')) for c in checks)
            if all('nativerefbad' in c for c in checks) else None)
    return dict(rc=rc,passed=raised and agreement and native is not False,
                raised_passed=raised,native_passed=native,
                raised_matches_native=agreement,checks=checks,vendor_calls=vendor_calls)


def run_gpu(names, args):
    transport = args.transport or os.environ.get('POLYGEIST_REMOTE_TRANSPORT')
    if not transport:
        raise ValueError('--transport or POLYGEIST_REMOTE_TRANSPORT must point to the site SSH adapter')
    loader=importlib.util.spec_from_file_location('aten_board',transport)
    board=importlib.util.module_from_spec(loader); loader.loader.exec_module(board)
    environment = 'LD_LIBRARY_PATH='+shlex.quote(args.remote_library_path)+' '
    if args.cudnn_plan_cache:
        environment += 'POLYGEIST_CUDNN_PLAN_CACHE=1 '
    run_id=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())
    prior=args.output/'gpu-results.json'
    results=[r for r in json.loads(prior.read_text()) if r['kernel'] not in names] if prior.exists() else []
    (args.output/(run_id+'-host-uptime.txt')).write_text(subprocess.check_output(['uptime'],text=True))
    for label in ['before','after']:
        if label=='after':break
        rc,log=board.run("python3 - <<'PY'\n"+(HERE/'fig3_board_preflight.py').read_text()+'\nPY',timeout=40)
        (args.output/(run_id+'-board-'+label+'.json')).write_text(log)
        if rc:raise RuntimeError('Board preflight failed')
    for name in names:
        dest=args.output/name; built=json.loads((dest/'build.json').read_text())
        if built['rc']:
            row=dict(kernel=name, run_id=run_id, fully_validated=False,
                     build_rc=built['rc'], error=built.get('error'), correctness={})
            results.append(row);save(dest/'gpu-result.json',row)
            save(args.output/'gpu-results.json',results)
            continue
        program=dest/'raised-program'
        if sha(program)!=built['binary_sha256']:raise RuntimeError('Binary changed after build')
        remote='/tmp/ll-aten-fixture-'+name+'-'+built['binary_sha256'][:12]
        if board.upload(program,remote):raise RuntimeError('Upload failed: '+name)
        rc,log=board.run('chmod +x '+shlex.quote(remote)+' && sha256sum '+shlex.quote(remote),timeout=20)
        (dest/'remote-hash.log').write_text(log)
        if rc or built['binary_sha256'] not in log:raise RuntimeError('Remote hash check failed')
        row=dict(kernel=name,run_id=run_id,binary_sha256=built['binary_sha256'],
                 scope=built['baseline'],dimension_scope=built['configuration']['dimension_scope'],
                 cudnn_plan_cache=args.cudnn_plan_cache,correctness={})
        results.append(row)
        cfg=built['configuration']; expected=sum(eval(a[2],{'__builtins__':{}},cfg['dims']) for a in cfg['args'] if a[1] in ('ptr','iptr','bptr') and a[3])
        modes=2*(2 if any(a[1] in ('scalar','iscalar') for a in cfg['args']) else 1)
        run_timeout = getattr(args, 'run_timeout', 60)
        kinds = args.correctness_kinds or (['normal','edge','zeros']+
                    (['paper'] if 'paper_initializers' in cfg else []))
        if 'paper' in kinds and 'paper_initializers' not in cfg:
            raise ValueError('Paper-input validation requires a recorded input recipe')
        row['correctness_scope'] = kinds
        for kind in kinds:
            rc,log=board.run(environment+'POLYGEIST_V2_TRACE=1 timeout '+str(run_timeout)+' '+shlex.quote(remote)+' '+kind,timeout=run_timeout+5)
            (dest/(kind+'-gpu.log')).write_text(log)
            row['correctness'][kind]=correctness_summary(rc,log,expected,2 if kind=='paper' else modes)
            print(name,kind,row['correctness'][kind]['passed'],flush=True);save(args.output/'gpu-results.json',results)
        row['fully_validated']=all(v['passed'] for v in row['correctness'].values())
        row['raised_validated']=all(v['raised_passed'] for v in row['correctness'].values())
        row['raised_matches_native']=all(v['raised_matches_native'] for v in row['correctness'].values())
        row['native_matches_reference']=(all(v['native_passed'] for v in row['correctness'].values())
            if all(v['native_passed'] is not None for v in row['correctness'].values()) else None)
        # Timings use normal inputs. Require the compiler to pass every input
        # mode and both implementations to agree on the actual timing inputs.
        # A native-only edge disagreement remains explicit and never turns green.
        timing_input='paper' if 'paper_initializers' in cfg else 'normal'
        if (not args.skip_timing and (row['raised_validated'] or args.time_validated_normal_inputs)
                and row['correctness'].get(timing_input,{}).get('passed')):
            rc,log=board.run(environment+'timeout '+str(run_timeout)+' '+shlex.quote(remote)+' timing',timeout=run_timeout+5)
            (dest/'timing-gpu.log').write_text(log)
            samples=[dict(path=a,sample=int(b),ms=float(c)) for a,b,c in re.findall(r'SAMPLE path=(\w+) sample=(\d+) ms=([\deE.+-]+)',log)]
            count=cfg.get('timing_samples',5)
            passed=rc==0 and 'PASS timing' in log and len(samples)==2*count and all(sorted(s['sample'] for s in samples if s['path']==p)==list(range(count)) for p in ('raised','native_aten_cuda'))
            row['timing']=dict(passed=passed,samples=samples,input_kind=timing_input,
                all_raised_input_modes_passed=row['raised_validated'],
                normal_input_only_override=args.time_validated_normal_inputs,
                native_edge_disagreement=(not row['native_matches_reference']
                    if row['native_matches_reference'] is not None else None),
                statistic=f'minimum of {count} synchronized resident samples after five warmups; alternating order; reset outside timing')
            if passed:
                row['timing']['minimum_ms']={p:min(s['ms'] for s in samples if s['path']==p) for p in ('raised','native_aten_cuda')}
                row['timing']['minimum_first_five_ms']={p:min(s['ms'] for s in samples if s['path']==p and s['sample']<5) for p in ('raised','native_aten_cuda')}
                row['timing']['native_over_raised']=row['timing']['minimum_ms']['native_aten_cuda']/row['timing']['minimum_ms']['raised']
        save(args.output/'gpu-results.json',results)
        save(dest/'gpu-result.json',row)
    rc,log=board.run("python3 - <<'PY'\n"+(HERE/'fig3_board_preflight.py').read_text()+'\nPY',timeout=40)
    (args.output/(run_id+'-board-after.json')).write_text(log)
    if rc:raise RuntimeError('Board postflight failed')
    print('GPU validated',sum(r['fully_validated'] for r in results),'/',len(results),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--cudnn-plan-cache',action='store_true',
                   help='reuse cuDNN descriptors, algorithm and workspace; recorded in run results')
    p.add_argument('--time-validated-normal-inputs',action='store_true',
                   help='allow normal-input timings when all three normal comparisons pass; preserve edge failures and failed full validation')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rank',type=int,default=0)
    p.add_argument('--shape-specs',type=Path,help='Pinned benchmark dimensions; separate source copies preserve the original fixtures')
    p.add_argument('--timing-samples',type=int,default=5)
    p.add_argument('--run-timeout',type=int,default=60)
    p.add_argument('--correctness-kinds', nargs='+', choices=['normal','edge','zeros','paper'],
                   help='Selected input recipes, explicitly recorded in the results')
    p.add_argument('--skip-timing', action='store_true')
    p.add_argument('--cases',nargs='+',required=True)
    p.add_argument('--jobs',type=int,default=2)
    p.add_argument('--persistent-workspace',action='store_true',
                   help='apply the shared private GPU workspace lifetime/residency planners')
    p.add_argument('--cutensor-root',type=Path,
                   default=Path(os.environ['POLYGEIST_CUTENSOR_ROOT']) if os.environ.get('POLYGEIST_CUTENSOR_ROOT') else None)
    p.add_argument('--stage',choices=['build','run'],default='build')
    p.add_argument('--transport')
    p.add_argument('--remote-library-path',default='/home/nvidia/polygeist_aten_native_24_12:/home/nvidia/polygeist_cuda_libs:/home/nvidia/jetson-cuda-libs:/home/nvidia/cuda-12.6/lib64:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu')
    args=p.parse_args();args.output=args.output.resolve();args.campaign=args.campaign.resolve()
    names=[n if n.startswith('aten_') else 'aten_'+n for n in args.cases]
    if any(n.removeprefix('aten_') not in NATIVE for n in names):
        p.error('No independent native baseline defined for a requested case')
    args.output.mkdir(parents=True,exist_ok=True)
    if args.stage=='run':
        run_gpu(names,args);return
    def guarded_build(name):
        try:
            return build(name,args)
        except Exception as error:
            row=dict(kernel=name,rc=1,error=type(error).__name__+': '+str(error))
            save(args.output/name/'build.json',row)
            print(name,'build rejected:',row['error'],flush=True)
            return row
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        rows=list(pool.map(guarded_build,names))
    save(args.output/'build-results.json',rows)
    raise SystemExit(any(r['rc'] for r in rows))


if __name__=='__main__':main()

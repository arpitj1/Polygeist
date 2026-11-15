#map = affine_map<(d0)[s0] -> (-d0 + s0 - 1)>
#map1 = affine_map<(d0)[s0] -> (d0 + s0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @thrust_reverse_negate_window_f32(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = bufferization.to_tensor %arg3 : memref<?xf32>
    %1 = arith.index_cast %arg0 : i32 to index
    %2 = arith.extsi %arg1 : i32 to i64
    %3 = arith.extsi %arg0 : i32 to i64
    %4 = arith.addi %2, %3 : i64
    %5 = arith.index_cast %4 : i64 to index
    %6 = arith.extsi %arg2 : i32 to i64
    %7 = arith.addi %4, %6 : i64
    %8 = arith.index_cast %7 : i64 to index
    %9 = affine.for %arg4 = 0 to %1 iter_args(%arg5 = %0) -> (tensor<?xf32>) {
      %11 = affine.apply #map(%arg4)[%5]
      %extracted = tensor.extract %arg5[%11] : tensor<?xf32>
      %12 = arith.negf %extracted : f32
      %13 = affine.apply #map1(%arg4)[%8]
      %inserted = tensor.insert %12 into %arg5[%13] : tensor<?xf32>
      affine.yield %inserted : tensor<?xf32>
    }
    %10 = bufferization.to_memref %9 : memref<?xf32>
    memref.copy %10, %arg3 : memref<?xf32> to memref<?xf32>
    return
  }
}


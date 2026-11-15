#map = affine_map<()[s0] -> (s0 - 1)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2)[s0, s1] -> (s0, s1, d0, d1, d2)>
#map3 = affine_map<(d0, d1, d2)[s0, s1, s2, s3] -> (s0, s1, s2, s3)>
#map4 = affine_map<(d0, d1, d2)[s0, s1, s2, s3] -> (s0, d0 + s1, d1 + s2, d2 + s3)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @cudnn_conv3d_backward_data_valid_f32(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: memref<?x?x?x?xf32>, %arg9: memref<?x?x?x?x?xf32>, %arg10: memref<?x?x?x?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = bufferization.to_tensor %arg10 : memref<?x?x?x?xf32>
    %1 = bufferization.to_tensor %arg9 : memref<?x?x?x?x?xf32>
    %2 = bufferization.to_tensor %arg8 : memref<?x?x?x?xf32>
    %3 = arith.index_cast %arg1 : i32 to index
    %4 = arith.addi %arg2, %arg5 : i32
    %5 = arith.index_cast %4 : i32 to index
    %6 = arith.addi %arg3, %arg6 : i32
    %7 = arith.index_cast %6 : i32 to index
    %8 = arith.addi %arg4, %arg7 : i32
    %9 = arith.index_cast %8 : i32 to index
    %10 = affine.apply #map()[%9]
    %11 = affine.apply #map()[%7]
    %12 = affine.apply #map()[%5]
    %13 = polygeist.submap(%0, %3, %12, %11, %10) {map = #map1} : (tensor<?x?x?x?xf32>, index, index, index, index) -> tensor<?x?x?x?xf32>
    %14 = linalg.generic {doc = "", indexing_maps = [#map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%13 : tensor<?x?x?x?xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    } -> tensor<?x?x?x?xf32>
    %15 = polygeist.submapInverse(%0, %14, %3, %12, %11, %10) {map = #map1} : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>, index, index, index, index) -> tensor<?x?x?x?xf32>
    %16 = arith.index_cast %arg0 : i32 to index
    %17 = arith.index_cast %arg2 : i32 to index
    %18 = arith.index_cast %arg3 : i32 to index
    %19 = arith.index_cast %arg4 : i32 to index
    %20 = arith.index_cast %arg5 : i32 to index
    %21 = arith.index_cast %arg6 : i32 to index
    %22 = arith.index_cast %arg7 : i32 to index
    %23 = affine.for %arg11 = 0 to %16 iter_args(%arg12 = %15) -> (tensor<?x?x?x?xf32>) {
      %25 = affine.for %arg13 = 0 to %3 iter_args(%arg14 = %arg12) -> (tensor<?x?x?x?xf32>) {
        %26 = affine.for %arg15 = 0 to %17 iter_args(%arg16 = %arg14) -> (tensor<?x?x?x?xf32>) {
          %27 = affine.for %arg17 = 0 to %18 iter_args(%arg18 = %arg16) -> (tensor<?x?x?x?xf32>) {
            %28 = affine.for %arg19 = 0 to %19 iter_args(%arg20 = %arg18) -> (tensor<?x?x?x?xf32>) {
              %29 = polygeist.submap(%1, %arg11, %arg13, %20, %21, %22) {map = #map2} : (tensor<?x?x?x?x?xf32>, index, index, index, index, index) -> tensor<?x?x?xf32>
              %30 = polygeist.submap(%2, %arg11, %arg15, %arg17, %arg19, %20, %21, %22) {map = #map3} : (tensor<?x?x?x?xf32>, index, index, index, index, index, index, index) -> tensor<?x?x?xf32>
              %31 = polygeist.submap(%arg20, %arg13, %arg15, %arg17, %arg19, %20, %21, %22) {map = #map4} : (tensor<?x?x?x?xf32>, index, index, index, index, index, index, index) -> tensor<?x?x?xf32>
              %32 = linalg.generic {doc = "", indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} ins(%29, %30 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%31 : tensor<?x?x?xf32>) {
              ^bb0(%in: f32, %in_0: f32, %out: f32):
                %34 = arith.mulf %in, %in_0 : f32
                %35 = arith.addf %out, %34 : f32
                linalg.yield %35 : f32
              } -> tensor<?x?x?xf32>
              %33 = polygeist.submapInverse(%arg20, %32, %arg13, %arg15, %arg17, %arg19, %20, %21, %22) {map = #map4} : (tensor<?x?x?x?xf32>, tensor<?x?x?xf32>, index, index, index, index, index, index, index) -> tensor<?x?x?x?xf32>
              affine.yield %33 : tensor<?x?x?x?xf32>
            }
            affine.yield %28 : tensor<?x?x?x?xf32>
          }
          affine.yield %27 : tensor<?x?x?x?xf32>
        }
        affine.yield %26 : tensor<?x?x?x?xf32>
      }
      affine.yield %25 : tensor<?x?x?x?xf32>
    }
    %24 = bufferization.to_memref %23 : memref<?x?x?x?xf32>
    memref.copy %24, %arg10 : memref<?x?x?x?xf32> to memref<?x?x?x?xf32>
    return
  }
}


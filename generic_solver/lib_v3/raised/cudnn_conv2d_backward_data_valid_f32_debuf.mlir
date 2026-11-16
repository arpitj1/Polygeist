#map = affine_map<()[s0] -> (s0 - 1)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2)[s0] -> (s0, d0, d1, d2)>
#map3 = affine_map<(d0, d1, d2)[s0, s1, s2, s3] -> (s0, s1, s2, s3)>
#map4 = affine_map<(d0, d1, d2)[s0, s1, s2] -> (s0, d0, d1 + s1, d2 + s2)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @cudnn_conv2d_backward_data_valid_f32(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: memref<?x?x?x?xf32>, %arg8: memref<?x?x?x?xf32>, %arg9: memref<?x?x?x?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = bufferization.to_tensor %arg9 : memref<?x?x?x?xf32>
    %1 = bufferization.to_tensor %arg8 : memref<?x?x?x?xf32>
    %2 = bufferization.to_tensor %arg7 : memref<?x?x?x?xf32>
    %3 = arith.index_cast %arg0 : i32 to index
    %4 = arith.index_cast %arg2 : i32 to index
    %5 = arith.addi %arg3, %arg5 : i32
    %6 = arith.index_cast %5 : i32 to index
    %7 = arith.addi %arg4, %arg6 : i32
    %8 = arith.index_cast %7 : i32 to index
    %9 = affine.apply #map()[%8]
    %10 = affine.apply #map()[%6]
    %11 = polygeist.submap(%0, %3, %4, %10, %9) {map = #map1} : (tensor<?x?x?x?xf32>, index, index, index, index) -> tensor<?x?x?x?xf32>
    %12 = linalg.generic {doc = "", indexing_maps = [#map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%11 : tensor<?x?x?x?xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    } -> tensor<?x?x?x?xf32>
    %13 = polygeist.submapInverse(%0, %12, %3, %4, %10, %9) {map = #map1} : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>, index, index, index, index) -> tensor<?x?x?x?xf32>
    %14 = arith.index_cast %arg1 : i32 to index
    %15 = arith.index_cast %arg3 : i32 to index
    %16 = arith.index_cast %arg4 : i32 to index
    %17 = arith.index_cast %arg5 : i32 to index
    %18 = arith.index_cast %arg6 : i32 to index
    %19 = affine.for %arg10 = 0 to %3 iter_args(%arg11 = %13) -> (tensor<?x?x?x?xf32>) {
      %21 = affine.for %arg12 = 0 to %14 iter_args(%arg13 = %arg11) -> (tensor<?x?x?x?xf32>) {
        %22 = affine.for %arg14 = 0 to %15 iter_args(%arg15 = %arg13) -> (tensor<?x?x?x?xf32>) {
          %23 = affine.for %arg16 = 0 to %16 iter_args(%arg17 = %arg15) -> (tensor<?x?x?x?xf32>) {
            %24 = polygeist.submap(%1, %arg12, %4, %17, %18) {map = #map2} : (tensor<?x?x?x?xf32>, index, index, index, index) -> tensor<?x?x?xf32>
            %25 = polygeist.submap(%2, %arg10, %arg12, %arg14, %arg16, %4, %17, %18) {map = #map3} : (tensor<?x?x?x?xf32>, index, index, index, index, index, index, index) -> tensor<?x?x?xf32>
            %26 = polygeist.submap(%arg17, %arg10, %arg14, %arg16, %4, %17, %18) {map = #map4} : (tensor<?x?x?x?xf32>, index, index, index, index, index, index) -> tensor<?x?x?xf32>
            %27 = linalg.generic {doc = "", indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} ins(%24, %25 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%26 : tensor<?x?x?xf32>) {
            ^bb0(%in: f32, %in_0: f32, %out: f32):
              %29 = arith.mulf %in, %in_0 : f32
              %30 = arith.addf %out, %29 : f32
              linalg.yield %30 : f32
            } -> tensor<?x?x?xf32>
            %28 = polygeist.submapInverse(%arg17, %27, %arg10, %arg14, %arg16, %4, %17, %18) {map = #map4} : (tensor<?x?x?x?xf32>, tensor<?x?x?xf32>, index, index, index, index, index, index) -> tensor<?x?x?x?xf32>
            affine.yield %28 : tensor<?x?x?x?xf32>
          }
          affine.yield %23 : tensor<?x?x?x?xf32>
        }
        affine.yield %22 : tensor<?x?x?x?xf32>
      }
      affine.yield %21 : tensor<?x?x?x?xf32>
    }
    %20 = bufferization.to_memref %19 : memref<?x?x?x?xf32>
    memref.copy %20, %arg9 : memref<?x?x?x?xf32> to memref<?x?x?x?xf32>
    return
  }
}


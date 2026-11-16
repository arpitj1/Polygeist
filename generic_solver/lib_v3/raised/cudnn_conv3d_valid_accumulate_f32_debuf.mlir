#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1, d2, d3, d4)[s0] -> (s0, d1, d2, d3, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4)[s0, s1, s2] -> (s0, d1, d2 + s1, d3 + s2, d4 + d0)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map5 = affine_map<(d0)[s0, s1, s2, s3] -> (s0, s1, s2, s3, d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @cudnn_conv3d_valid_accumulate_f32(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: memref<?x?x?x?x?xf32>, %arg13: memref<?x?x?x?x?xf32>, %arg14: memref<?x?x?x?x?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = bufferization.to_tensor %arg14 : memref<?x?x?x?x?xf32>
    %1 = bufferization.to_tensor %arg13 : memref<?x?x?x?x?xf32>
    %2 = bufferization.to_tensor %arg12 : memref<?x?x?x?x?xf32>
    %3 = arith.index_cast %arg0 : i32 to index
    %4 = arith.index_cast %arg5 : i32 to index
    %5 = arith.index_cast %arg9 : i32 to index
    %6 = arith.index_cast %arg10 : i32 to index
    %7 = arith.index_cast %arg11 : i32 to index
    %8 = arith.index_cast %arg1 : i32 to index
    %9 = arith.index_cast %arg6 : i32 to index
    %10 = arith.index_cast %arg7 : i32 to index
    %11 = arith.index_cast %arg8 : i32 to index
    %12 = affine.for %arg15 = 0 to %3 iter_args(%arg16 = %0) -> (tensor<?x?x?x?x?xf32>) {
      %14 = affine.for %arg17 = 0 to %4 iter_args(%arg18 = %arg16) -> (tensor<?x?x?x?x?xf32>) {
        %15 = affine.for %arg19 = 0 to %5 iter_args(%arg20 = %arg18) -> (tensor<?x?x?x?x?xf32>) {
          %16 = affine.for %arg21 = 0 to %6 iter_args(%arg22 = %arg20) -> (tensor<?x?x?x?x?xf32>) {
            %alloca = memref.alloca(%7) : memref<?xf32>
            %17 = bufferization.to_tensor %alloca : memref<?xf32>
            %18 = polygeist.submap(%17, %7) {map = #map} : (tensor<?xf32>, index) -> tensor<?xf32>
            %19 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel"], library_call = ""} outs(%18 : tensor<?xf32>) {
            ^bb0(%out: f32):
              linalg.yield %cst : f32
            } -> tensor<?xf32>
            %20 = polygeist.submapInverse(%17, %19, %7) {map = #map} : (tensor<?xf32>, tensor<?xf32>, index) -> tensor<?xf32>
            %21 = polygeist.submap(%1, %arg17, %7, %8, %9, %10, %11) {map = #map1} : (tensor<?x?x?x?x?xf32>, index, index, index, index, index, index) -> tensor<?x?x?x?x?xf32>
            %22 = polygeist.submap(%2, %arg15, %arg19, %arg21, %7, %8, %9, %10, %11) {map = #map2} : (tensor<?x?x?x?x?xf32>, index, index, index, index, index, index, index, index) -> tensor<?x?x?x?x?xf32>
            %23 = polygeist.submap(%20, %7, %8, %9, %10, %11) {map = #map3} : (tensor<?xf32>, index, index, index, index, index) -> tensor<?x?x?x?x?xf32>
            %24 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "reduction", "reduction", "reduction", "reduction"], library_call = ""} ins(%21, %22 : tensor<?x?x?x?x?xf32>, tensor<?x?x?x?x?xf32>) outs(%23 : tensor<?x?x?x?x?xf32>) {
            ^bb0(%in: f32, %in_0: f32, %out: f32):
              %30 = arith.mulf %in, %in_0 : f32
              %31 = arith.addf %30, %out : f32
              linalg.yield %31 : f32
            } -> tensor<?x?x?x?x?xf32>
            %25 = polygeist.submapInverse(%20, %24, %7, %8, %9, %10, %11) {map = #map3} : (tensor<?xf32>, tensor<?x?x?x?x?xf32>, index, index, index, index, index) -> tensor<?xf32>
            %26 = polygeist.submap(%25, %7) {map = #map} : (tensor<?xf32>, index) -> tensor<?xf32>
            %27 = polygeist.submap(%arg22, %arg15, %arg17, %arg19, %arg21, %7) {map = #map5} : (tensor<?x?x?x?x?xf32>, index, index, index, index, index) -> tensor<?xf32>
            %28 = linalg.generic {doc = "", indexing_maps = [#map, #map], iterator_types = ["parallel"], library_call = ""} ins(%26 : tensor<?xf32>) outs(%27 : tensor<?xf32>) {
            ^bb0(%in: f32, %out: f32):
              %30 = arith.addf %in, %out : f32
              linalg.yield %30 : f32
            } -> tensor<?xf32>
            %29 = polygeist.submapInverse(%arg22, %28, %arg15, %arg17, %arg19, %arg21, %7) {map = #map5} : (tensor<?x?x?x?x?xf32>, tensor<?xf32>, index, index, index, index, index) -> tensor<?x?x?x?x?xf32>
            affine.yield %29 : tensor<?x?x?x?x?xf32>
          }
          affine.yield %16 : tensor<?x?x?x?x?xf32>
        }
        affine.yield %15 : tensor<?x?x?x?x?xf32>
      }
      affine.yield %14 : tensor<?x?x?x?x?xf32>
    }
    %13 = bufferization.to_memref %12 : memref<?x?x?x?x?xf32>
    memref.copy %13, %arg14 : memref<?x?x?x?x?xf32> to memref<?x?x?x?x?xf32>
    return
  }
}

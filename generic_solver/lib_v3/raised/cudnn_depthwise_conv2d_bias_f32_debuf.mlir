#map = affine_map<()[s0] -> (s0 * 2 + 1)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0, d1, d2)[s0] -> (s0, d1, d2)>
#map3 = affine_map<(d0, d1, d2)[s0, s1, s2, s3, s4] -> (s0, s1, d1 + s2 - s3, d2 + d0 - s4)>
#map4 = affine_map<(d0, d1, d2) -> (d0)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map6 = affine_map<(d0, d1)[s0] -> (d0 + d1 - s0)>
#map7 = affine_map<(d0, d1)[s0, s1] -> (-d0 - d1 + s0 + s1 - 1)>
#map8 = affine_map<(d0)[s0] -> (s0)>
#map9 = affine_map<(d0)[s0, s1, s2] -> (s0, s1, s2, d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @cudnn_depthwise_conv2d_bias_f32(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: memref<?x?x?x?xf32>, %arg7: memref<?x?x?xf32>, %arg8: memref<?xf32>, %arg9: memref<?x?x?x?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = bufferization.to_tensor %arg9 : memref<?x?x?x?xf32>
    %1 = bufferization.to_tensor %arg8 : memref<?xf32>
    %2 = bufferization.to_tensor %arg7 : memref<?x?x?xf32>
    %3 = bufferization.to_tensor %arg6 : memref<?x?x?x?xf32>
    %4 = arith.index_cast %arg5 : i32 to index
    %5 = arith.index_cast %arg4 : i32 to index
    %6 = arith.index_cast %arg0 : i32 to index
    %7 = arith.index_cast %arg1 : i32 to index
    %8 = arith.index_cast %arg2 : i32 to index
    %9 = arith.index_cast %arg3 : i32 to index
    %10 = affine.apply #map()[%4]
    %11 = affine.apply #map()[%5]
    %12 = affine.for %arg10 = 0 to %6 iter_args(%arg11 = %0) -> (tensor<?x?x?x?xf32>) {
      %14 = affine.for %arg12 = 0 to %7 iter_args(%arg13 = %arg11) -> (tensor<?x?x?x?xf32>) {
        %15 = affine.for %arg14 = 0 to %8 iter_args(%arg15 = %arg13) -> (tensor<?x?x?x?xf32>) {
          %alloca = memref.alloca(%9) : memref<?xf32>
          %16 = bufferization.to_tensor %alloca : memref<?xf32>
          %17 = polygeist.submap(%16, %9) {map = #map1} : (tensor<?xf32>, index) -> tensor<?xf32>
          %18 = linalg.generic {doc = "", indexing_maps = [#map1], iterator_types = ["parallel"], library_call = ""} outs(%17 : tensor<?xf32>) {
          ^bb0(%out: f32):
            linalg.yield %cst : f32
          } -> tensor<?xf32>
          %19 = polygeist.submapInverse(%16, %18, %9) {map = #map1} : (tensor<?xf32>, tensor<?xf32>, index) -> tensor<?xf32>
          %20 = polygeist.submap(%2, %arg12, %9, %11, %10) {map = #map2} : (tensor<?x?x?xf32>, index, index, index, index) -> tensor<?x?x?xf32>
          %21 = polygeist.submap(%3, %arg10, %arg12, %arg14, %5, %4, %9, %11, %10) {map = #map3} : (tensor<?x?x?x?xf32>, index, index, index, index, index, index, index, index) -> tensor<?x?x?xf32>
          %22 = polygeist.submap(%19, %9, %11, %10) {map = #map4} : (tensor<?xf32>, index, index, index) -> tensor<?x?x?xf32>
          %23 = linalg.generic {doc = "", indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "reduction", "reduction"], library_call = ""} ins(%20, %21 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%22 : tensor<?x?x?xf32>) {
          ^bb0(%in: f32, %in_0: f32, %out: f32):
            %30 = linalg.index 0 : index
            %31 = linalg.index 1 : index
            %32 = affine.apply #map6(%arg14, %31)[%5]
            %33 = affine.apply #map7(%arg14, %31)[%5, %8]
            %34 = linalg.index 2 : index
            %35 = affine.apply #map7(%30, %34)[%4, %9]
            %36 = arith.cmpi sge, %35, %c0 : index
            %37 = affine.apply #map6(%30, %34)[%4]
            %38 = arith.cmpi sge, %37, %c0 : index
            %39 = arith.andi %36, %38 : i1
            %40 = arith.cmpi sge, %32, %c0 : index
            %41 = arith.andi %39, %40 : i1
            %42 = arith.cmpi sge, %33, %c0 : index
            %43 = arith.andi %41, %42 : i1
            %44 = arith.mulf %in, %in_0 : f32
            %45 = arith.addf %44, %out : f32
            %46 = arith.select %43, %45, %out : f32
            linalg.yield %46 : f32
          } -> tensor<?x?x?xf32>
          %24 = polygeist.submapInverse(%19, %23, %9, %11, %10) {map = #map4} : (tensor<?xf32>, tensor<?x?x?xf32>, index, index, index) -> tensor<?xf32>
          %25 = polygeist.submap(%24, %9) {map = #map1} : (tensor<?xf32>, index) -> tensor<?xf32>
          %26 = polygeist.submap(%1, %arg12, %9) {map = #map8} : (tensor<?xf32>, index, index) -> tensor<?xf32>
          %27 = polygeist.submap(%arg15, %arg10, %arg12, %arg14, %9) {map = #map9} : (tensor<?x?x?x?xf32>, index, index, index, index) -> tensor<?xf32>
          %28 = linalg.generic {doc = "", indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"], library_call = ""} ins(%25, %26 : tensor<?xf32>, tensor<?xf32>) outs(%27 : tensor<?xf32>) {
          ^bb0(%in: f32, %in_0: f32, %out: f32):
            %30 = arith.addf %in, %in_0 : f32
            linalg.yield %30 : f32
          } -> tensor<?xf32>
          %29 = polygeist.submapInverse(%arg15, %28, %arg10, %arg12, %arg14, %9) {map = #map9} : (tensor<?x?x?x?xf32>, tensor<?xf32>, index, index, index, index) -> tensor<?x?x?x?xf32>
          affine.yield %29 : tensor<?x?x?x?xf32>
        }
        affine.yield %15 : tensor<?x?x?x?xf32>
      }
      affine.yield %14 : tensor<?x?x?x?xf32>
    }
    %13 = bufferization.to_memref %12 : memref<?x?x?x?xf32>
    memref.copy %13, %arg9 : memref<?x?x?x?xf32> to memref<?x?x?x?xf32>
    return
  }
}


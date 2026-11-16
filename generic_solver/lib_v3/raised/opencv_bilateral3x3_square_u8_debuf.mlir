#map = affine_map<()[s0] -> (0, s0 - 1, s0)>
#map1 = affine_map<()[s0] -> (0, s0 - 1)>
#map2 = affine_map<()[s0] -> (s0 - 2)>
#map3 = affine_map<(d0) -> (d0)>
#map4 = affine_map<(d0, d1) -> (d0 + 1)>
#map5 = affine_map<(d0, d1) -> (d1 + 1)>
#map6 = affine_map<(d0) -> (d0 - 1)>
#map7 = affine_map<(d0)[s0, s1, s2] -> (s1 + s2, d0 + s0)>
#map8 = affine_map<(d0)[s0] -> (s0)>
#map9 = affine_map<(d0)[s0] -> (s0 + 1, d0 + 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @opencv_bilateral3x3_square_u8(%arg0: i32, %arg1: i32, %arg2: memref<?xi8>, %arg3: memref<?xi8>, %arg4: f32, %arg5: f32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c3 = arith.constant 3 : index
    %c255_i32 = arith.constant 255 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 5.000000e-01 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant -5.000000e-01 : f32
    %0 = arith.index_cast %arg1 : i32 to index
    %1 = arith.index_cast %arg0 : i32 to index
    %2 = arith.mulf %arg4, %arg4 : f32
    %3 = arith.divf %cst_1, %2 : f32
    %4 = arith.mulf %arg5, %arg5 : f32
    %5 = arith.divf %cst_1, %4 : f32
    %6 = affine.max #map()[%1]
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [%6, %0], strides: [%0, 1] : memref<?xi8> to memref<?x?xi8>
    %7 = bufferization.to_tensor %reinterpret_cast : memref<?x?xi8>
    %8 = affine.max #map1()[%1]
    %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [%8, %0], strides: [%0, 1] : memref<?xi8> to memref<?x?xi8>
    %9 = bufferization.to_tensor %reinterpret_cast_2 : memref<?x?xi8>
    %10 = affine.apply #map2()[%0]
    %11 = affine.apply #map2()[%0]
    %12 = affine.apply #map2()[%0]
    %13 = affine.apply #map2()[%0]
    %14 = affine.apply #map2()[%0]
    %15 = affine.for %arg6 = 0 to #map2()[%1] iter_args(%arg7 = %9) -> (tensor<?x?xi8>) {
      %alloca = memref.alloca(%10) : memref<?xf32>
      %17 = bufferization.to_tensor %alloca : memref<?xf32>
      %alloca_3 = memref.alloca(%11) : memref<?xf32>
      %18 = bufferization.to_tensor %alloca_3 : memref<?xf32>
      %19 = polygeist.submap(%17, %12) {map = #map3} : (tensor<?xf32>, index) -> tensor<?xf32>
      %20 = linalg.generic {doc = "", indexing_maps = [#map3], iterator_types = ["parallel"], library_call = ""} outs(%19 : tensor<?xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_0 : f32
      } -> tensor<?xf32>
      %21 = polygeist.submapInverse(%17, %20, %12) {map = #map3} : (tensor<?xf32>, tensor<?xf32>, index) -> tensor<?xf32>
      %22 = polygeist.submap(%18, %13) {map = #map3} : (tensor<?xf32>, index) -> tensor<?xf32>
      %23 = linalg.generic {doc = "", indexing_maps = [#map3], iterator_types = ["parallel"], library_call = ""} outs(%22 : tensor<?xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_0 : f32
      } -> tensor<?xf32>
      %24 = polygeist.submapInverse(%18, %23, %13) {map = #map3} : (tensor<?xf32>, tensor<?xf32>, index) -> tensor<?xf32>
      %25:2 = affine.for %arg8 = 0 to #map2()[%0] iter_args(%arg9 = %21, %arg10 = %24) -> (tensor<?xf32>, tensor<?xf32>) {
        %31 = affine.apply #map4(%arg6, %arg8)
        %32 = affine.apply #map5(%arg6, %arg8)
        %extracted = tensor.extract %7[%31, %32] : tensor<?x?xi8>
        %33 = arith.uitofp %extracted : i8 to f32
        %alloca_4 = memref.alloca(%c3) : memref<?xf32>
        %34 = bufferization.to_tensor %alloca_4 : memref<?xf32>
        %alloca_5 = memref.alloca(%c3) : memref<?xf32>
        %35 = bufferization.to_tensor %alloca_5 : memref<?xf32>
        %36:4 = affine.for %arg11 = 0 to 3 iter_args(%arg12 = %34, %arg13 = %35, %arg14 = %arg9, %arg15 = %arg10) -> (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) {
          %37 = affine.apply #map6(%arg11)
          %extracted_6 = tensor.extract %arg14[%arg8] : tensor<?xf32>
          %extracted_7 = tensor.extract %arg15[%arg8] : tensor<?xf32>
          %38 = arith.index_cast %37 : index to i32
          %39 = arith.muli %38, %38 : i32
          %inserted = tensor.insert %extracted_6 into %arg12[%arg11] : tensor<?xf32>
          %inserted_8 = tensor.insert %extracted_7 into %arg13[%arg11] : tensor<?xf32>
          %40 = polygeist.submap(%7, %arg8, %arg6, %arg11, %c3) {map = #map7} : (tensor<?x?xi8>, index, index, index, index) -> tensor<?xi8>
          %41 = polygeist.submap(%inserted, %arg11, %c3) {map = #map8} : (tensor<?xf32>, index, index) -> tensor<?xf32>
          %42 = polygeist.submap(%inserted_8, %arg11, %c3) {map = #map8} : (tensor<?xf32>, index, index) -> tensor<?xf32>
          %43:2 = linalg.generic {doc = "", indexing_maps = [#map3, #map3, #map3], iterator_types = ["reduction"], library_call = ""} ins(%40 : tensor<?xi8>) outs(%41, %42 : tensor<?xf32>, tensor<?xf32>) {
          ^bb0(%in: i8, %out: f32, %out_13: f32):
            %46 = linalg.index 0 : index
            %47 = affine.apply #map6(%46)
            %48 = arith.index_cast %47 : index to i32
            %49 = arith.uitofp %in : i8 to f32
            %50 = arith.subf %49, %33 : f32
            %51 = arith.muli %48, %48 : i32
            %52 = arith.addi %51, %39 : i32
            %53 = arith.sitofp %52 : i32 to f32
            %54 = arith.mulf %53, %5 : f32
            %55 = arith.mulf %50, %50 : f32
            %56 = arith.mulf %55, %3 : f32
            %57 = arith.addf %54, %56 : f32
            %58 = math.exp %57 : f32
            %59 = arith.addf %out_13, %58 : f32
            %60 = arith.mulf %58, %49 : f32
            %61 = arith.addf %out, %60 : f32
            linalg.yield %61, %59 : f32, f32
          } -> (tensor<?xf32>, tensor<?xf32>)
          %44 = polygeist.submapInverse(%inserted, %43#0, %arg11, %c3) {map = #map8} : (tensor<?xf32>, tensor<?xf32>, index, index) -> tensor<?xf32>
          %45 = polygeist.submapInverse(%inserted_8, %43#1, %arg11, %c3) {map = #map8} : (tensor<?xf32>, tensor<?xf32>, index, index) -> tensor<?xf32>
          %extracted_9 = tensor.extract %44[%arg11] : tensor<?xf32>
          %extracted_10 = tensor.extract %45[%arg11] : tensor<?xf32>
          %inserted_11 = tensor.insert %extracted_9 into %arg14[%arg8] : tensor<?xf32>
          %inserted_12 = tensor.insert %extracted_10 into %arg15[%arg8] : tensor<?xf32>
          affine.yield %44, %45, %inserted_11, %inserted_12 : tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>
        }
        affine.yield %36#2, %36#3 : tensor<?xf32>, tensor<?xf32>
      }
      %26 = polygeist.submap(%25#0, %14) {map = #map3} : (tensor<?xf32>, index) -> tensor<?xf32>
      %27 = polygeist.submap(%25#1, %14) {map = #map3} : (tensor<?xf32>, index) -> tensor<?xf32>
      %28 = polygeist.submap(%arg7, %arg6, %14) {map = #map9} : (tensor<?x?xi8>, index, index) -> tensor<?xi8>
      %29 = linalg.generic {doc = "", indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel"], library_call = ""} ins(%26, %27 : tensor<?xf32>, tensor<?xf32>) outs(%28 : tensor<?xi8>) {
      ^bb0(%in: f32, %in_4: f32, %out: i8):
        %31 = arith.divf %in, %in_4 : f32
        %32 = arith.addf %31, %cst : f32
        %33 = arith.fptosi %32 : f32 to i32
        %34 = arith.cmpi slt, %33, %c0_i32 : i32
        %35 = arith.cmpi sgt, %33, %c255_i32 : i32
        %36 = arith.select %35, %c255_i32, %33 : i32
        %37 = arith.select %34, %c0_i32, %36 : i32
        %38 = arith.trunci %37 : i32 to i8
        linalg.yield %38 : i8
      } -> tensor<?xi8>
      %30 = polygeist.submapInverse(%arg7, %29, %arg6, %14) {map = #map9} : (tensor<?x?xi8>, tensor<?xi8>, index, index) -> tensor<?x?xi8>
      affine.yield %30 : tensor<?x?xi8>
    }
    %16 = bufferization.to_memref %15 : memref<?x?xi8>
    memref.copy %16, %reinterpret_cast_2 : memref<?x?xi8> to memref<?x?xi8>
    return
  }
}


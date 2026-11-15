#map = affine_map<()[s0] -> (0, s0 - 1, s0)>
#map1 = affine_map<()[s0] -> (0, s0 - 1)>
#map2 = affine_map<()[s0] -> (s0 - 2)>
#map3 = affine_map<(d0) -> (d0)>
#map4 = affine_map<(d0) -> (d0 - 1)>
#map5 = affine_map<(d0)[s0, s1, s2] -> (s1 + s2, d0 + s0)>
#map6 = affine_map<(d0)[s0] -> (s0)>
#map7 = affine_map<(d0)[s0] -> (s0 + 1, d0 + 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
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
    %7 = affine.max #map1()[%1]
    %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [%7, %0], strides: [%0, 1] : memref<?xi8> to memref<?x?xi8>
    %8 = affine.apply #map2()[%0]
    %9 = affine.apply #map2()[%0]
    %10 = affine.apply #map2()[%0]
    %11 = affine.apply #map2()[%0]
    %12 = affine.apply #map2()[%0]
    affine.for %arg6 = 0 to #map2()[%1] {
      %alloca = memref.alloca(%8) : memref<?xf32>
      %alloca_3 = memref.alloca(%9) : memref<?xf32>
      %13 = polygeist.submap(%alloca, %10) {map = #map3} : (memref<?xf32>, index) -> memref<?xf32>
      linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel"]} outs(%13 : memref<?xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_0 : f32
      }
      %14 = polygeist.submap(%alloca_3, %11) {map = #map3} : (memref<?xf32>, index) -> memref<?xf32>
      linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel"]} outs(%14 : memref<?xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_0 : f32
      }
      affine.for %arg7 = 0 to #map2()[%0] {
        %18 = affine.load %reinterpret_cast[%arg6 + 1, %arg7 + 1] : memref<?x?xi8>
        %19 = arith.uitofp %18 : i8 to f32
        %alloca_4 = memref.alloca(%c3) : memref<?xf32>
        %alloca_5 = memref.alloca(%c3) : memref<?xf32>
        affine.for %arg8 = 0 to 3 {
          %20 = affine.apply #map4(%arg8)
          %21 = affine.load %alloca[%arg7] : memref<?xf32>
          %22 = affine.load %alloca_3[%arg7] : memref<?xf32>
          %23 = arith.index_cast %20 : index to i32
          %24 = arith.muli %23, %23 : i32
          affine.store %21, %alloca_4[%arg8] : memref<?xf32>
          affine.store %22, %alloca_5[%arg8] : memref<?xf32>
          %25 = polygeist.submap(%reinterpret_cast, %arg7, %arg6, %arg8, %c3) {map = #map5} : (memref<?x?xi8>, index, index, index, index) -> memref<?xi8>
          %26 = polygeist.submap(%alloca_4, %arg8, %c3) {map = #map6} : (memref<?xf32>, index, index) -> memref<?xf32>
          %27 = polygeist.submap(%alloca_5, %arg8, %c3) {map = #map6} : (memref<?xf32>, index, index) -> memref<?xf32>
          linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["reduction"]} ins(%25 : memref<?xi8>) outs(%26, %27 : memref<?xf32>, memref<?xf32>) {
          ^bb0(%in: i8, %out: f32, %out_6: f32):
            %30 = linalg.index 0 : index
            %31 = affine.apply #map4(%30)
            %32 = arith.index_cast %31 : index to i32
            %33 = arith.uitofp %in : i8 to f32
            %34 = arith.subf %33, %19 : f32
            %35 = arith.muli %32, %32 : i32
            %36 = arith.addi %35, %24 : i32
            %37 = arith.sitofp %36 : i32 to f32
            %38 = arith.mulf %37, %5 : f32
            %39 = arith.mulf %34, %34 : f32
            %40 = arith.mulf %39, %3 : f32
            %41 = arith.addf %38, %40 : f32
            %42 = math.exp %41 : f32
            %43 = arith.addf %out_6, %42 : f32
            %44 = arith.mulf %42, %33 : f32
            %45 = arith.addf %out, %44 : f32
            linalg.yield %45, %43 : f32, f32
          }
          %28 = affine.load %alloca_4[%arg8] : memref<?xf32>
          %29 = affine.load %alloca_5[%arg8] : memref<?xf32>
          affine.store %28, %alloca[%arg7] : memref<?xf32>
          affine.store %29, %alloca_3[%arg7] : memref<?xf32>
        }
      } {polygeist.was_parallel}
      %15 = polygeist.submap(%alloca, %12) {map = #map3} : (memref<?xf32>, index) -> memref<?xf32>
      %16 = polygeist.submap(%alloca_3, %12) {map = #map3} : (memref<?xf32>, index) -> memref<?xf32>
      %17 = polygeist.submap(%reinterpret_cast_2, %arg6, %12) {map = #map7} : (memref<?x?xi8>, index, index) -> memref<?xi8>
      linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel"]} ins(%15, %16 : memref<?xf32>, memref<?xf32>) outs(%17 : memref<?xi8>) {
      ^bb0(%in: f32, %in_4: f32, %out: i8):
        %18 = arith.divf %in, %in_4 : f32
        %19 = arith.addf %18, %cst : f32
        %20 = arith.fptosi %19 : f32 to i32
        %21 = arith.cmpi slt, %20, %c0_i32 : i32
        %22 = arith.cmpi sgt, %20, %c255_i32 : i32
        %23 = arith.select %22, %c255_i32, %20 : i32
        %24 = arith.select %21, %c0_i32, %23 : i32
        %25 = arith.trunci %24 : i32 to i8
        linalg.yield %25 : i8
      }
    } {polygeist.was_parallel}
    return
  }
}


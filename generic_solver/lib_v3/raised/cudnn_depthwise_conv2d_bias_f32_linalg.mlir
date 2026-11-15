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
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @cudnn_depthwise_conv2d_bias_f32(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: memref<?x?x?x?xf32>, %arg7: memref<?x?x?xf32>, %arg8: memref<?xf32>, %arg9: memref<?x?x?x?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = arith.index_cast %arg5 : i32 to index
    %1 = arith.index_cast %arg4 : i32 to index
    %2 = arith.index_cast %arg0 : i32 to index
    %3 = arith.index_cast %arg1 : i32 to index
    %4 = arith.index_cast %arg2 : i32 to index
    %5 = arith.index_cast %arg3 : i32 to index
    %6 = affine.apply #map()[%0]
    %7 = affine.apply #map()[%1]
    affine.for %arg10 = 0 to %2 {
      affine.for %arg11 = 0 to %3 {
        affine.for %arg12 = 0 to %4 {
          %alloca = memref.alloca(%5) : memref<?xf32>
          %8 = polygeist.submap(%alloca, %5) {map = #map1} : (memref<?xf32>, index) -> memref<?xf32>
          linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%8 : memref<?xf32>) {
          ^bb0(%out: f32):
            linalg.yield %cst : f32
          }
          %9 = polygeist.submap(%arg7, %arg11, %5, %7, %6) {map = #map2} : (memref<?x?x?xf32>, index, index, index, index) -> memref<?x?x?xf32>
          %10 = polygeist.submap(%arg6, %arg10, %arg11, %arg12, %1, %0, %5, %7, %6) {map = #map3} : (memref<?x?x?x?xf32>, index, index, index, index, index, index, index, index) -> memref<?x?x?xf32>
          %11 = polygeist.submap(%alloca, %5, %7, %6) {map = #map4} : (memref<?xf32>, index, index, index) -> memref<?x?x?xf32>
          linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "reduction", "reduction"]} ins(%9, %10 : memref<?x?x?xf32>, memref<?x?x?xf32>) outs(%11 : memref<?x?x?xf32>) {
          ^bb0(%in: f32, %in_0: f32, %out: f32):
            %15 = linalg.index 0 : index
            %16 = linalg.index 1 : index
            %17 = affine.apply #map6(%arg12, %16)[%1]
            %18 = affine.apply #map7(%arg12, %16)[%1, %4]
            %19 = linalg.index 2 : index
            %20 = affine.apply #map7(%15, %19)[%0, %5]
            %21 = arith.cmpi sge, %20, %c0 : index
            %22 = affine.apply #map6(%15, %19)[%0]
            %23 = arith.cmpi sge, %22, %c0 : index
            %24 = arith.andi %21, %23 : i1
            %25 = arith.cmpi sge, %17, %c0 : index
            %26 = arith.andi %24, %25 : i1
            %27 = arith.cmpi sge, %18, %c0 : index
            %28 = arith.andi %26, %27 : i1
            %29 = arith.mulf %in, %in_0 : f32
            %30 = arith.addf %29, %out : f32
            %31 = arith.select %28, %30, %out : f32
            linalg.yield %31 : f32
          }
          %12 = polygeist.submap(%alloca, %5) {map = #map1} : (memref<?xf32>, index) -> memref<?xf32>
          %13 = polygeist.submap(%arg8, %arg11, %5) {map = #map8} : (memref<?xf32>, index, index) -> memref<?xf32>
          %14 = polygeist.submap(%arg9, %arg10, %arg11, %arg12, %5) {map = #map9} : (memref<?x?x?x?xf32>, index, index, index, index) -> memref<?xf32>
          linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%12, %13 : memref<?xf32>, memref<?xf32>) outs(%14 : memref<?xf32>) {
          ^bb0(%in: f32, %in_0: f32, %out: f32):
            %15 = arith.addf %in, %in_0 : f32
            linalg.yield %15 : f32
          }
        } {polygeist.was_parallel}
      } {polygeist.was_parallel}
    } {polygeist.was_parallel}
    return
  }
}


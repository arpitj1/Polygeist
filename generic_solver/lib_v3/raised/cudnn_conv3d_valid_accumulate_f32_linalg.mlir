#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1, d2, d3, d4)[s0] -> (s0, d1, d2, d3, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4)[s0, s1, s2] -> (s0, d1, d2 + s1, d3 + s2, d4 + d0)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map5 = affine_map<(d0)[s0, s1, s2, s3] -> (s0, s1, s2, s3, d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @cudnn_conv3d_valid_accumulate_f32(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: memref<?x?x?x?x?xf32>, %arg13: memref<?x?x?x?x?xf32>, %arg14: memref<?x?x?x?x?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = arith.index_cast %arg5 : i32 to index
    %2 = arith.index_cast %arg9 : i32 to index
    %3 = arith.index_cast %arg10 : i32 to index
    %4 = arith.index_cast %arg11 : i32 to index
    %5 = arith.index_cast %arg1 : i32 to index
    %6 = arith.index_cast %arg6 : i32 to index
    %7 = arith.index_cast %arg7 : i32 to index
    %8 = arith.index_cast %arg8 : i32 to index
    affine.for %arg15 = 0 to %0 {
      affine.for %arg16 = 0 to %1 {
        affine.for %arg17 = 0 to %2 {
          affine.for %arg18 = 0 to %3 {
            %alloca = memref.alloca(%4) : memref<?xf32>
            %9 = polygeist.submap(%alloca, %4) {map = #map} : (memref<?xf32>, index) -> memref<?xf32>
            linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%9 : memref<?xf32>) {
            ^bb0(%out: f32):
              linalg.yield %cst : f32
            }
            %10 = polygeist.submap(%arg13, %arg16, %4, %5, %6, %7, %8) {map = #map1} : (memref<?x?x?x?x?xf32>, index, index, index, index, index, index) -> memref<?x?x?x?x?xf32>
            %11 = polygeist.submap(%arg12, %arg15, %arg17, %arg18, %4, %5, %6, %7, %8) {map = #map2} : (memref<?x?x?x?x?xf32>, index, index, index, index, index, index, index, index) -> memref<?x?x?x?x?xf32>
            %12 = polygeist.submap(%alloca, %4, %5, %6, %7, %8) {map = #map3} : (memref<?xf32>, index, index, index, index, index) -> memref<?x?x?x?x?xf32>
            linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "reduction", "reduction", "reduction", "reduction"]} ins(%10, %11 : memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>) outs(%12 : memref<?x?x?x?x?xf32>) {
            ^bb0(%in: f32, %in_0: f32, %out: f32):
              %15 = arith.mulf %in, %in_0 : f32
              %16 = arith.addf %15, %out : f32
              linalg.yield %16 : f32
            }
            %13 = polygeist.submap(%alloca, %4) {map = #map} : (memref<?xf32>, index) -> memref<?xf32>
            %14 = polygeist.submap(%arg14, %arg15, %arg16, %arg17, %arg18, %4) {map = #map5} : (memref<?x?x?x?x?xf32>, index, index, index, index, index) -> memref<?xf32>
            linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%13 : memref<?xf32>) outs(%14 : memref<?xf32>) {
            ^bb0(%in: f32, %out: f32):
              %15 = arith.addf %in, %out : f32
              linalg.yield %15 : f32
            }
          } {polygeist.was_parallel}
        } {polygeist.was_parallel}
      } {polygeist.was_parallel}
    } {polygeist.was_parallel}
    return
  }
}

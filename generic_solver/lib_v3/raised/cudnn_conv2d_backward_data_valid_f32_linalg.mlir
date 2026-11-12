#map = affine_map<()[s0] -> (s0 - 1)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2)[s0] -> (s0, d0, d1, d2)>
#map3 = affine_map<(d0, d1, d2)[s0, s1, s2, s3] -> (s0, s1, s2, s3)>
#map4 = affine_map<(d0, d1, d2)[s0, s1, s2] -> (s0, d0, d1 + s1, d2 + s2)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @cudnn_conv2d_backward_data_valid_f32(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: memref<?x?x?x?xf32>, %arg8: memref<?x?x?x?xf32>, %arg9: memref<?x?x?x?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = arith.index_cast %arg2 : i32 to index
    %2 = arith.addi %arg3, %arg5 : i32
    %3 = arith.index_cast %2 : i32 to index
    %4 = arith.addi %arg4, %arg6 : i32
    %5 = arith.index_cast %4 : i32 to index
    %6 = affine.apply #map()[%5]
    %7 = affine.apply #map()[%3]
    %8 = polygeist.submap(%arg9, %0, %1, %7, %6) {map = #map1} : (memref<?x?x?x?xf32>, index, index, index, index) -> memref<?x?x?x?xf32>
    linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%8 : memref<?x?x?x?xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    }
    %9 = arith.index_cast %arg1 : i32 to index
    %10 = arith.index_cast %arg3 : i32 to index
    %11 = arith.index_cast %arg4 : i32 to index
    %12 = arith.index_cast %arg5 : i32 to index
    %13 = arith.index_cast %arg6 : i32 to index
    affine.for %arg10 = 0 to %0 {
      affine.for %arg11 = 0 to %9 {
        affine.for %arg12 = 0 to %10 {
          affine.for %arg13 = 0 to %11 {
            %14 = polygeist.submap(%arg8, %arg11, %1, %12, %13) {map = #map2} : (memref<?x?x?x?xf32>, index, index, index, index) -> memref<?x?x?xf32>
            %15 = polygeist.submap(%arg7, %arg10, %arg11, %arg12, %arg13, %1, %12, %13) {map = #map3} : (memref<?x?x?x?xf32>, index, index, index, index, index, index, index) -> memref<?x?x?xf32>
            %16 = polygeist.submap(%arg9, %arg10, %arg12, %arg13, %1, %12, %13) {map = #map4} : (memref<?x?x?x?xf32>, index, index, index, index, index, index) -> memref<?x?x?xf32>
            linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%14, %15 : memref<?x?x?xf32>, memref<?x?x?xf32>) outs(%16 : memref<?x?x?xf32>) {
            ^bb0(%in: f32, %in_0: f32, %out: f32):
              %17 = arith.mulf %in, %in_0 : f32
              %18 = arith.addf %out, %17 : f32
              linalg.yield %18 : f32
            }
          }
        }
      }
    } {polygeist.was_parallel}
    return
  }
}


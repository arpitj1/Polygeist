#map = affine_map<()[s0] -> (s0 - 1)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2)[s0, s1] -> (s0, s1, d0, d1, d2)>
#map3 = affine_map<(d0, d1, d2)[s0, s1, s2, s3] -> (s0, s1, s2, s3)>
#map4 = affine_map<(d0, d1, d2)[s0, s1, s2, s3] -> (s0, d0 + s1, d1 + s2, d2 + s3)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @cudnn_conv3d_backward_data_valid_f32(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: memref<?x?x?x?xf32>, %arg9: memref<?x?x?x?x?xf32>, %arg10: memref<?x?x?x?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = arith.index_cast %arg1 : i32 to index
    %1 = arith.addi %arg2, %arg5 : i32
    %2 = arith.index_cast %1 : i32 to index
    %3 = arith.addi %arg3, %arg6 : i32
    %4 = arith.index_cast %3 : i32 to index
    %5 = arith.addi %arg4, %arg7 : i32
    %6 = arith.index_cast %5 : i32 to index
    %7 = affine.apply #map()[%6]
    %8 = affine.apply #map()[%4]
    %9 = affine.apply #map()[%2]
    %10 = polygeist.submap(%arg10, %0, %9, %8, %7) {map = #map1} : (memref<?x?x?x?xf32>, index, index, index, index) -> memref<?x?x?x?xf32>
    linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%10 : memref<?x?x?x?xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    }
    %11 = arith.index_cast %arg0 : i32 to index
    %12 = arith.index_cast %arg2 : i32 to index
    %13 = arith.index_cast %arg3 : i32 to index
    %14 = arith.index_cast %arg4 : i32 to index
    %15 = arith.index_cast %arg5 : i32 to index
    %16 = arith.index_cast %arg6 : i32 to index
    %17 = arith.index_cast %arg7 : i32 to index
    affine.for %arg11 = 0 to %11 {
      affine.for %arg12 = 0 to %0 {
        affine.for %arg13 = 0 to %12 {
          affine.for %arg14 = 0 to %13 {
            affine.for %arg15 = 0 to %14 {
              %18 = polygeist.submap(%arg9, %arg11, %arg12, %15, %16, %17) {map = #map2} : (memref<?x?x?x?x?xf32>, index, index, index, index, index) -> memref<?x?x?xf32>
              %19 = polygeist.submap(%arg8, %arg11, %arg13, %arg14, %arg15, %15, %16, %17) {map = #map3} : (memref<?x?x?x?xf32>, index, index, index, index, index, index, index) -> memref<?x?x?xf32>
              %20 = polygeist.submap(%arg10, %arg12, %arg13, %arg14, %arg15, %15, %16, %17) {map = #map4} : (memref<?x?x?x?xf32>, index, index, index, index, index, index, index) -> memref<?x?x?xf32>
              linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%18, %19 : memref<?x?x?xf32>, memref<?x?x?xf32>) outs(%20 : memref<?x?x?xf32>) {
              ^bb0(%in: f32, %in_0: f32, %out: f32):
                %21 = arith.mulf %in, %in_0 : f32
                %22 = arith.addf %out, %21 : f32
                linalg.yield %22 : f32
              }
            }
          }
        }
      } {polygeist.was_parallel}
    }
    return
  }
}


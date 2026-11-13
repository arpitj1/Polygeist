#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0)[s0] -> (s0, d0)>
#map2 = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @cusp_csr_matvecs_beta0_f32(%arg0: i32, %arg1: i32, %arg2: memref<?xi32>, %arg3: i32, %arg4: memref<?xi32>, %arg5: i32, %arg6: memref<?xf32>, %arg7: i32, %arg8: i32, %arg9: memref<?x?xf32>, %arg10: i32, %arg11: i32, %arg12: memref<?x?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = arith.index_cast %arg8 : i32 to index
    %2 = polygeist.submap(%arg12, %0, %1) {map = #map} : (memref<?x?xf32>, index, index) -> memref<?x?xf32>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%2 : memref<?x?xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    }
    affine.for %arg13 = 0 to %0 {
      %3 = affine.load %arg2[%arg13] : memref<?xi32>
      %4 = scf.while (%arg14 = %3) : (i32) -> i32 {
        %5 = affine.load %arg2[%arg13 + 1] : memref<?xi32>
        %6 = arith.cmpi slt, %arg14, %5 : i32
        scf.condition(%6) %arg14 : i32
      } do {
      ^bb0(%arg14: i32):
        %5 = arith.index_cast %arg14 : i32 to index
        %6 = memref.load %arg4[%5] : memref<?xi32>
        %7 = memref.load %arg6[%5] : memref<?xf32>
        %8 = arith.index_cast %6 : i32 to index
        %9 = polygeist.submap(%arg12, %arg13, %1) {map = #map1} : (memref<?x?xf32>, index, index) -> memref<?xf32>
        linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel"]} outs(%9 : memref<?xf32>) {
        ^bb0(%out: f32):
          %11 = linalg.index 0 : index
          %12 = memref.load %arg9[%8, %11] : memref<?x?xf32>
          %13 = arith.mulf %7, %12 : f32
          %14 = arith.addf %out, %13 : f32
          linalg.yield %14 : f32
        }
        %10 = arith.addi %arg14, %c1_i32 : i32
        scf.yield %10 : i32
      }
    }
    return
  }
}


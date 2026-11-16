#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1)[s0] -> (s0, d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @cusp_bsr_matvec_beta0_f32(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: memref<?xi32>, %arg4: i32, %arg5: memref<?xi32>, %arg6: i32, %arg7: memref<?x?x?xf32>, %arg8: i32, %arg9: memref<?x?xf32>, %arg10: i32, %arg11: memref<?x?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = arith.index_cast %arg1 : i32 to index
    %2 = polygeist.submap(%arg11, %0, %1) {map = #map} : (memref<?x?xf32>, index, index) -> memref<?x?xf32>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%2 : memref<?x?xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    }
    affine.for %arg12 = 0 to %0 {
      %3 = affine.load %arg3[%arg12] : memref<?xi32>
      %4 = scf.while (%arg13 = %3) : (i32) -> i32 {
        %5 = affine.load %arg3[%arg12 + 1] : memref<?xi32>
        %6 = arith.cmpi slt, %arg13, %5 : i32
        scf.condition(%6) %arg13 : i32
      } do {
      ^bb0(%arg13: i32):
        %5 = arith.index_cast %arg13 : i32 to index
        %6 = memref.load %arg5[%5] : memref<?xi32>
        %7 = arith.index_cast %6 : i32 to index
        %8 = polygeist.submap(%arg11, %arg12, %1, %1) {map = #map1} : (memref<?x?xf32>, index, index, index) -> memref<?x?xf32>
        linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "reduction"]} outs(%8 : memref<?x?xf32>) {
        ^bb0(%out: f32):
          %10 = linalg.index 0 : index
          %11 = linalg.index 1 : index
          %12 = memref.load %arg7[%5, %10, %11] : memref<?x?x?xf32>
          %13 = memref.load %arg9[%7, %11] : memref<?x?xf32>
          %14 = arith.mulf %12, %13 : f32
          %15 = arith.addf %out, %14 : f32
          linalg.yield %15 : f32
        }
        %9 = arith.addi %arg13, %c1_i32 : i32
        scf.yield %9 : i32
      }
    }
    return
  }
}


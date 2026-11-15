#map = affine_map<(d0, d1) -> (d0, d1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @cusp_coo_matvecs_beta0_f32(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: memref<?xi32>, %arg4: i32, %arg5: memref<?xi32>, %arg6: i32, %arg7: memref<?xf32>, %arg8: i32, %arg9: i32, %arg10: memref<?x?xf32>, %arg11: i32, %arg12: i32, %arg13: memref<?x?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = arith.index_cast %arg9 : i32 to index
    %2 = polygeist.submap(%arg13, %0, %1) {map = #map} : (memref<?x?xf32>, index, index) -> memref<?x?xf32>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%2 : memref<?x?xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    }
    %3 = arith.index_cast %arg1 : i32 to index
    affine.for %arg14 = 0 to %3 {
      affine.for %arg15 = 0 to %1 {
        %4 = affine.load %arg3[%arg14] : memref<?xi32>
        %5 = arith.index_cast %4 : i32 to index
        %6 = affine.load %arg7[%arg14] : memref<?xf32>
        %7 = affine.load %arg5[%arg14] : memref<?xi32>
        %8 = arith.index_cast %7 : i32 to index
        %9 = memref.load %arg10[%8, %arg15] : memref<?x?xf32>
        %10 = arith.mulf %6, %9 : f32
        %11 = memref.load %arg13[%5, %arg15] : memref<?x?xf32>
        %12 = arith.addf %11, %10 : f32
        memref.store %12, %arg13[%5, %arg15] : memref<?x?xf32>
      }
    }
    return
  }
}


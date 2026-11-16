#map = affine_map<(d0)[s0, s1] -> (-s0 + s1 - 1, d0 - s0 + s1)>
#map1 = affine_map<(d0)[s0, s1, s2] -> (s0, d0 - s1 + s2)>
#map2 = affine_map<(d0)[s0] -> (s0)>
#map3 = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @netlib_dsymm_ll_beta0(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: i32, %arg4: memref<?x?xf64>, %arg5: i32, %arg6: memref<?x?xf64>, %arg7: i32, %arg8: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f64
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = arith.index_cast %arg1 : i32 to index
    %2 = arith.subi %0, %c1 : index
    affine.for %arg9 = 0 to %1 {
      %alloca = memref.alloca(%0) : memref<?xf64>
      affine.for %arg10 = 0 to %0 {
        %3 = affine.load %arg6[%arg9, -%arg10 + symbol(%0) - 1] : memref<?x?xf64>
        %4 = arith.mulf %arg2, %3 : f64
        affine.store %cst, %alloca[%arg10] : memref<?xf64>
        %5 = polygeist.submap(%arg4, %arg10, %0, %2) {map = #map} : (memref<?x?xf64>, index, index, index) -> memref<?xf64>
        %6 = polygeist.submap(%arg6, %arg9, %arg10, %0, %2) {map = #map1} : (memref<?x?xf64>, index, index, index, index) -> memref<?xf64>
        %7 = polygeist.submap(%arg4, %arg10, %0, %2) {map = #map} : (memref<?x?xf64>, index, index, index) -> memref<?xf64>
        %8 = polygeist.submap(%arg8, %arg9, %arg10, %0, %2) {map = #map1} : (memref<?x?xf64>, index, index, index, index) -> memref<?xf64>
        %9 = polygeist.submap(%alloca, %arg10, %2) {map = #map2} : (memref<?xf64>, index, index) -> memref<?xf64>
        linalg.generic {indexing_maps = [#map3, #map3, #map3, #map3, #map3], iterator_types = ["reduction"]} ins(%5, %6, %7 : memref<?xf64>, memref<?xf64>, memref<?xf64>) outs(%8, %9 : memref<?xf64>, memref<?xf64>) {
        ^bb0(%in: f64, %in_0: f64, %in_1: f64, %out: f64, %out_2: f64):
          %15 = arith.mulf %4, %in : f64
          %16 = arith.addf %out, %15 : f64
          %17 = arith.mulf %in_0, %in_1 : f64
          %18 = arith.addf %out_2, %17 : f64
          %19 = linalg.index 0 : index
          %20 = arith.cmpi slt, %19, %arg10 : index
          %21 = arith.select %20, %16, %out : f64
          %22 = arith.select %20, %18, %out_2 : f64
          linalg.yield %21, %22 : f64, f64
        }
        %10 = affine.load %alloca[%arg10] : memref<?xf64>
        %11 = affine.load %arg4[-%arg10 + symbol(%0) - 1, -%arg10 + symbol(%0) - 1] : memref<?x?xf64>
        %12 = arith.mulf %4, %11 : f64
        %13 = arith.mulf %arg2, %10 : f64
        %14 = arith.addf %12, %13 : f64
        affine.store %14, %arg8[%arg9, -%arg10 + symbol(%0) - 1] : memref<?x?xf64>
      }
    } {polygeist.was_parallel}
    return
  }
}


#map = affine_map<(d0)[s0, s1] -> (-s0 + s1 - 1, d0 - s0 + s1)>
#map1 = affine_map<(d0)[s0, s1, s2] -> (s0, d0 - s1 + s2)>
#map2 = affine_map<(d0)[s0] -> (s0)>
#map3 = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @netlib_dsymm_ll(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: i32, %arg4: memref<?x?xf64>, %arg5: i32, %arg6: memref<?x?xf64>, %arg7: f64, %arg8: i32, %arg9: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f64
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = arith.index_cast %arg1 : i32 to index
    %2 = arith.subi %0, %c1 : index
    affine.for %arg10 = 0 to %1 {
      %alloca = memref.alloca(%0) : memref<?xf64>
      affine.for %arg11 = 0 to %0 {
        %3 = affine.load %arg6[%arg10, -%arg11 + symbol(%0) - 1] : memref<?x?xf64>
        %4 = arith.mulf %arg2, %3 : f64
        affine.store %cst, %alloca[%arg11] : memref<?xf64>
        %5 = polygeist.submap(%arg4, %arg11, %0, %2) {map = #map} : (memref<?x?xf64>, index, index, index) -> memref<?xf64>
        %6 = polygeist.submap(%arg6, %arg10, %arg11, %0, %2) {map = #map1} : (memref<?x?xf64>, index, index, index, index) -> memref<?xf64>
        %7 = polygeist.submap(%arg4, %arg11, %0, %2) {map = #map} : (memref<?x?xf64>, index, index, index) -> memref<?xf64>
        %8 = polygeist.submap(%arg9, %arg10, %arg11, %0, %2) {map = #map1} : (memref<?x?xf64>, index, index, index, index) -> memref<?xf64>
        %9 = polygeist.submap(%alloca, %arg11, %2) {map = #map2} : (memref<?xf64>, index, index) -> memref<?xf64>
        linalg.generic {indexing_maps = [#map3, #map3, #map3, #map3, #map3], iterator_types = ["reduction"]} ins(%5, %6, %7 : memref<?xf64>, memref<?xf64>, memref<?xf64>) outs(%8, %9 : memref<?xf64>, memref<?xf64>) {
        ^bb0(%in: f64, %in_0: f64, %in_1: f64, %out: f64, %out_2: f64):
          %18 = arith.mulf %4, %in : f64
          %19 = arith.addf %out, %18 : f64
          %20 = arith.mulf %in_0, %in_1 : f64
          %21 = arith.addf %out_2, %20 : f64
          %22 = linalg.index 0 : index
          %23 = arith.cmpi slt, %22, %arg11 : index
          %24 = arith.select %23, %19, %out : f64
          %25 = arith.select %23, %21, %out_2 : f64
          linalg.yield %24, %25 : f64, f64
        }
        %10 = affine.load %alloca[%arg11] : memref<?xf64>
        %11 = affine.load %arg9[%arg10, -%arg11 + symbol(%0) - 1] : memref<?x?xf64>
        %12 = arith.mulf %arg7, %11 : f64
        %13 = affine.load %arg4[-%arg11 + symbol(%0) - 1, -%arg11 + symbol(%0) - 1] : memref<?x?xf64>
        %14 = arith.mulf %4, %13 : f64
        %15 = arith.addf %12, %14 : f64
        %16 = arith.mulf %arg2, %10 : f64
        %17 = arith.addf %15, %16 : f64
        affine.store %17, %arg9[%arg10, -%arg11 + symbol(%0) - 1] : memref<?x?xf64>
      }
    } {polygeist.was_parallel}
    return
  }
}


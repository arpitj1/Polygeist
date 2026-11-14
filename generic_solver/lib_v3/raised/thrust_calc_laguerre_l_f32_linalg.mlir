#map = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @thrust_calc_laguerre_l_f32(%arg0: i32, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %false = arith.constant false
    %true = arith.constant true
    %c0_i8 = arith.constant 0 : i8
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c0_i64 = arith.constant 0 : i64
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = llvm.mlir.undef : f32
    %2 = polygeist.submap(%arg3, %0) {map = #map} : (memref<?xf32>, index) -> memref<?xf32>
    %3 = polygeist.submap(%arg1, %0) {map = #map} : (memref<?xf32>, index) -> memref<?xf32>
    %4 = polygeist.submap(%arg2, %0) {map = #map} : (memref<?xf32>, index) -> memref<?xf32>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%3, %4 : memref<?xf32>, memref<?xf32>) outs(%2 : memref<?xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %5 = arith.fptosi %in_1 : f32 to i64
      %6 = arith.cmpi slt, %5, %c0_i64 : i64
      %7 = arith.cmpi sge, %5, %c0_i64 : i64
      %8 = arith.select %6, %cst_0, %1 : f32
      %9 = math.absf %in : f32
      %10 = arith.cmpf oeq, %9, %cst_0 : f32
      %11 = arith.select %10, %cst, %8 : f32
      %12 = arith.cmpi eq, %5, %c0_i64 : i64
      %13 = arith.cmpi ne, %5, %c0_i64 : i64
      %14 = arith.select %12, %cst, %11 : f32
      %15 = arith.select %10, %false, %13 : i1
      %16 = arith.select %10, %11, %14 : f32
      %17 = arith.select %7, %15, %false : i1
      %18 = arith.select %7, %16, %8 : f32
      %19 = arith.cmpi eq, %5, %c1_i64 : i64
      %20 = arith.andi %17, %19 : i1
      %21 = arith.xori %20, %true : i1
      %22 = arith.andi %21, %17 : i1
      %23 = scf.if %22 -> (f32) {
        %24 = arith.subf %cst, %in : f32
        %25:4 = scf.while (%arg4 = %c1_i64, %arg5 = %24, %arg6 = %24, %arg7 = %cst) : (i64, f32, f32, f32) -> (f32, i64, f32, f32) {
          %26 = arith.cmpi slt, %arg4, %5 : i64
          %27 = arith.cmpf uno, %arg6, %arg6 : f32
          %28 = arith.extui %27 : i1 to i8
          %29 = arith.cmpi eq, %28, %c0_i8 : i8
          %30 = arith.andi %26, %29 : i1
          scf.condition(%30) %arg5, %arg4, %arg6, %arg7 : f32, i64, f32, f32
        } do {
        ^bb0(%arg4: f32, %arg5: i64, %arg6: f32, %arg7: f32):
          %26 = arith.cmpf uno, %arg6, %arg6 : f32
          %27 = arith.extui %26 : i1 to i8
          %28 = arith.cmpi eq, %27, %c0_i8 : i8
          %29 = arith.select %28, %arg6, %arg7 : f32
          %30 = arith.cmpi slt, %arg5, %5 : i64
          %31 = arith.select %30, %29, %arg7 : f32
          %32 = arith.addi %arg5, %arg5 : i64
          %33 = arith.sitofp %32 : i64 to f32
          %34 = arith.addf %33, %24 : f32
          %35 = arith.mulf %34, %arg6 : f32
          %36 = arith.sitofp %arg5 : i64 to f32
          %37 = arith.mulf %36, %arg7 : f32
          %38 = arith.subf %35, %37 : f32
          %39 = arith.addi %arg5, %c1_i64 : i64
          %40 = arith.sitofp %39 : i64 to f32
          %41 = arith.divf %38, %40 : f32
          scf.yield %39, %41, %41, %31 : i64, f32, f32, f32
        }
        scf.yield %25#0 : f32
      } else {
        %24 = arith.subf %cst, %in : f32
        %25 = arith.select %20, %24, %18 : f32
        scf.yield %25 : f32
      }
      linalg.yield %23 : f32
    }
    return
  }
}


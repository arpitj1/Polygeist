#map = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @thrust_calc_laguerre_l_f32(%arg0: i32, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %c1_i64 = arith.constant 1 : i64
    %c0_i8 = arith.constant 0 : i8
    %true = arith.constant true
    %0 = bufferization.to_tensor %arg3 : memref<?xf32>
    %1 = bufferization.to_tensor %arg2 : memref<?xf32>
    %2 = bufferization.to_tensor %arg1 : memref<?xf32>
    %3 = arith.index_cast %arg0 : i32 to index
    %4 = llvm.mlir.undef : f32
    %5 = polygeist.submap(%2, %3) {map = #map} : (tensor<?xf32>, index) -> tensor<?xf32>
    %6 = polygeist.submap(%1, %3) {map = #map} : (tensor<?xf32>, index) -> tensor<?xf32>
    %7 = polygeist.submap(%0, %3) {map = #map} : (tensor<?xf32>, index) -> tensor<?xf32>
    %8 = linalg.generic {doc = "", indexing_maps = [#map, #map, #map], iterator_types = ["parallel"], library_call = ""} ins(%5, %6 : tensor<?xf32>, tensor<?xf32>) outs(%7 : tensor<?xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %11 = arith.fptosi %in_1 : f32 to i64
      %12 = arith.cmpi slt, %11, %c0_i64 : i64
      %13 = arith.cmpi sge, %11, %c0_i64 : i64
      %14 = arith.select %12, %cst, %4 : f32
      %15 = math.absf %in : f32
      %16 = arith.cmpf oeq, %15, %cst : f32
      %17 = arith.cmpi eq, %11, %c0_i64 : i64
      %18 = arith.cmpi ne, %11, %c0_i64 : i64
      %19 = arith.xori %16, %true : i1
      %20 = arith.andi %19, %18 : i1
      %21 = arith.select %17, %cst_0, %14 : f32
      %22 = arith.select %16, %cst_0, %21 : f32
      %23 = arith.andi %13, %20 : i1
      %24 = arith.select %13, %22, %14 : f32
      %25 = arith.cmpi eq, %11, %c1_i64 : i64
      %26 = arith.andi %23, %25 : i1
      %27 = arith.xori %26, %true : i1
      %28 = arith.andi %27, %23 : i1
      %29 = scf.if %28 -> (f32) {
        %30 = arith.subf %cst_0, %in : f32
        %31:4 = scf.while (%arg4 = %c1_i64, %arg5 = %30, %arg6 = %30, %arg7 = %cst_0) : (i64, f32, f32, f32) -> (f32, i64, f32, f32) {
          %32 = arith.cmpi slt, %arg4, %11 : i64
          %33 = arith.cmpf uno, %arg6, %arg6 : f32
          %34 = arith.extui %33 : i1 to i8
          %35 = arith.cmpi eq, %34, %c0_i8 : i8
          %36 = arith.andi %32, %35 : i1
          scf.condition(%36) %arg5, %arg4, %arg6, %arg7 : f32, i64, f32, f32
        } do {
        ^bb0(%arg4: f32, %arg5: i64, %arg6: f32, %arg7: f32):
          %32 = arith.cmpf uno, %arg6, %arg6 : f32
          %33 = arith.extui %32 : i1 to i8
          %34 = arith.cmpi eq, %33, %c0_i8 : i8
          %35 = arith.select %34, %arg6, %arg7 : f32
          %36 = arith.cmpi slt, %arg5, %11 : i64
          %37 = arith.select %36, %35, %arg7 : f32
          %38 = arith.addi %arg5, %arg5 : i64
          %39 = arith.sitofp %38 : i64 to f32
          %40 = arith.addf %39, %30 : f32
          %41 = arith.mulf %40, %arg6 : f32
          %42 = arith.sitofp %arg5 : i64 to f32
          %43 = arith.mulf %42, %arg7 : f32
          %44 = arith.subf %41, %43 : f32
          %45 = arith.addi %arg5, %c1_i64 : i64
          %46 = arith.sitofp %45 : i64 to f32
          %47 = arith.divf %44, %46 : f32
          scf.yield %45, %47, %47, %37 : i64, f32, f32, f32
        }
        scf.yield %31#0 : f32
      } else {
        %30 = arith.subf %cst_0, %in : f32
        %31 = arith.select %26, %30, %24 : f32
        scf.yield %31 : f32
      }
      linalg.yield %29 : f32
    } -> tensor<?xf32>
    %9 = polygeist.submapInverse(%0, %8, %3) {map = #map} : (tensor<?xf32>, tensor<?xf32>, index) -> tensor<?xf32>
    %10 = bufferization.to_memref %9 : memref<?xf32>
    memref.copy %10, %arg3 : memref<?xf32> to memref<?xf32>
    return
  }
}


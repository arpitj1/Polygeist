#map = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @thrust_calc_legendre_p_f32(%arg0: i32, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %c2_i64 = arith.constant 2 : i64
    %cst_1 = arith.constant -1.000000e+00 : f32
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
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %11 = arith.fptosi %in_2 : f32 to i64
      %12 = arith.cmpi slt, %11, %c0_i64 : i64
      %13 = arith.cmpi sge, %11, %c0_i64 : i64
      %14 = arith.select %12, %cst, %4 : f32
      %15 = math.absf %in : f32
      %16 = arith.cmpf oeq, %15, %cst_0 : f32
      %17 = arith.cmpf ogt, %in, %cst : f32
      %18 = arith.remsi %11, %c2_i64 : i64
      %19 = arith.cmpi eq, %18, %c0_i64 : i64
      %20 = arith.xori %17, %true : i1
      %21 = arith.andi %20, %19 : i1
      %22 = arith.ori %17, %21 : i1
      %23 = arith.xori %22, %true : i1
      %24 = arith.select %23, %cst_1, %cst_0 : f32
      %25 = arith.cmpi eq, %11, %c0_i64 : i64
      %26 = arith.cmpi ne, %11, %c0_i64 : i64
      %27 = arith.select %25, %cst_0, %14 : f32
      %28 = arith.xori %16, %true : i1
      %29 = arith.andi %28, %26 : i1
      %30 = arith.select %16, %24, %27 : f32
      %31 = arith.andi %13, %29 : i1
      %32 = arith.select %13, %30, %14 : f32
      %33 = arith.cmpi eq, %11, %c1_i64 : i64
      %34 = arith.cmpi ne, %11, %c1_i64 : i64
      %35 = arith.select %33, %in, %32 : f32
      %36 = scf.if %31 -> (f32) {
        %37 = scf.if %34 -> (f32) {
          %38:4 = scf.while (%arg4 = %c1_i64, %arg5 = %in, %arg6 = %in, %arg7 = %cst_0) : (i64, f32, f32, f32) -> (f32, i64, f32, f32) {
            %39 = arith.cmpi slt, %arg4, %11 : i64
            %40 = arith.cmpf uno, %arg6, %arg6 : f32
            %41 = arith.extui %40 : i1 to i8
            %42 = arith.cmpi eq, %41, %c0_i8 : i8
            %43 = arith.andi %39, %42 : i1
            scf.condition(%43) %arg5, %arg4, %arg6, %arg7 : f32, i64, f32, f32
          } do {
          ^bb0(%arg4: f32, %arg5: i64, %arg6: f32, %arg7: f32):
            %39 = arith.cmpf uno, %arg6, %arg6 : f32
            %40 = arith.extui %39 : i1 to i8
            %41 = arith.cmpi eq, %40, %c0_i8 : i8
            %42 = arith.select %41, %arg6, %arg7 : f32
            %43 = arith.cmpi slt, %arg5, %11 : i64
            %44 = arith.select %43, %42, %arg7 : f32
            %45 = arith.addi %arg5, %arg5 : i64
            %46 = arith.addi %45, %c1_i64 : i64
            %47 = arith.sitofp %46 : i64 to f32
            %48 = arith.mulf %47, %in : f32
            %49 = arith.mulf %48, %arg6 : f32
            %50 = arith.sitofp %arg5 : i64 to f32
            %51 = arith.mulf %50, %arg7 : f32
            %52 = arith.subf %49, %51 : f32
            %53 = arith.addi %arg5, %c1_i64 : i64
            %54 = arith.sitofp %53 : i64 to f32
            %55 = arith.divf %52, %54 : f32
            scf.yield %53, %55, %55, %44 : i64, f32, f32, f32
          }
          scf.yield %38#0 : f32
        } else {
          scf.yield %35 : f32
        }
        scf.yield %37 : f32
      } else {
        scf.yield %32 : f32
      }
      linalg.yield %36 : f32
    } -> tensor<?xf32>
    %9 = polygeist.submapInverse(%0, %8, %3) {map = #map} : (tensor<?xf32>, tensor<?xf32>, index) -> tensor<?xf32>
    %10 = bufferization.to_memref %9 : memref<?xf32>
    memref.copy %10, %arg3 : memref<?xf32> to memref<?xf32>
    return
  }
}


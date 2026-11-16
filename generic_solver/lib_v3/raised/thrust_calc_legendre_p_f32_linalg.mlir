#map = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @thrust_calc_legendre_p_f32(%arg0: i32, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %false = arith.constant false
    %true = arith.constant true
    %c0_i8 = arith.constant 0 : i8
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant -1.000000e+00 : f32
    %c2_i64 = arith.constant 2 : i64
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %c0_i64 = arith.constant 0 : i64
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = llvm.mlir.undef : f32
    %2 = polygeist.submap(%arg3, %0) {map = #map} : (memref<?xf32>, index) -> memref<?xf32>
    %3 = polygeist.submap(%arg1, %0) {map = #map} : (memref<?xf32>, index) -> memref<?xf32>
    %4 = polygeist.submap(%arg2, %0) {map = #map} : (memref<?xf32>, index) -> memref<?xf32>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%3, %4 : memref<?xf32>, memref<?xf32>) outs(%2 : memref<?xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %5 = arith.fptosi %in_2 : f32 to i64
      %6 = arith.cmpi slt, %5, %c0_i64 : i64
      %7 = arith.cmpi sge, %5, %c0_i64 : i64
      %8 = arith.select %6, %cst_1, %1 : f32
      %9 = math.absf %in : f32
      %10 = arith.cmpf oeq, %9, %cst_0 : f32
      %11 = arith.cmpf ogt, %in, %cst_1 : f32
      %12 = arith.remsi %5, %c2_i64 : i64
      %13 = arith.cmpi eq, %12, %c0_i64 : i64
      %14 = arith.select %11, %true, %13 : i1
      %15 = arith.xori %14, %true : i1
      %16 = arith.select %15, %cst, %cst_0 : f32
      %17 = arith.cmpi eq, %5, %c0_i64 : i64
      %18 = arith.cmpi ne, %5, %c0_i64 : i64
      %19 = arith.select %17, %cst_0, %8 : f32
      %20 = arith.select %10, %false, %18 : i1
      %21 = arith.select %10, %16, %19 : f32
      %22 = arith.select %7, %20, %false : i1
      %23 = arith.select %7, %21, %8 : f32
      %24 = arith.cmpi eq, %5, %c1_i64 : i64
      %25 = arith.cmpi ne, %5, %c1_i64 : i64
      %26 = arith.select %24, %in, %23 : f32
      %27 = scf.if %22 -> (f32) {
        %28 = scf.if %25 -> (f32) {
          %29:4 = scf.while (%arg4 = %c1_i64, %arg5 = %in, %arg6 = %in, %arg7 = %cst_0) : (i64, f32, f32, f32) -> (f32, i64, f32, f32) {
            %30 = arith.cmpi slt, %arg4, %5 : i64
            %31 = arith.cmpf uno, %arg6, %arg6 : f32
            %32 = arith.extui %31 : i1 to i8
            %33 = arith.cmpi eq, %32, %c0_i8 : i8
            %34 = arith.andi %30, %33 : i1
            scf.condition(%34) %arg5, %arg4, %arg6, %arg7 : f32, i64, f32, f32
          } do {
          ^bb0(%arg4: f32, %arg5: i64, %arg6: f32, %arg7: f32):
            %30 = arith.cmpf uno, %arg6, %arg6 : f32
            %31 = arith.extui %30 : i1 to i8
            %32 = arith.cmpi eq, %31, %c0_i8 : i8
            %33 = arith.select %32, %arg6, %arg7 : f32
            %34 = arith.cmpi slt, %arg5, %5 : i64
            %35 = arith.select %34, %33, %arg7 : f32
            %36 = arith.addi %arg5, %arg5 : i64
            %37 = arith.addi %36, %c1_i64 : i64
            %38 = arith.sitofp %37 : i64 to f32
            %39 = arith.mulf %38, %in : f32
            %40 = arith.mulf %39, %arg6 : f32
            %41 = arith.sitofp %arg5 : i64 to f32
            %42 = arith.mulf %41, %arg7 : f32
            %43 = arith.subf %40, %42 : f32
            %44 = arith.addi %arg5, %c1_i64 : i64
            %45 = arith.sitofp %44 : i64 to f32
            %46 = arith.divf %43, %45 : f32
            scf.yield %44, %46, %46, %35 : i64, f32, f32, f32
          }
          scf.yield %29#0 : f32
        } else {
          scf.yield %26 : f32
        }
        scf.yield %28 : f32
      } else {
        scf.yield %23 : f32
      }
      linalg.yield %27 : f32
    }
    return
  }
}


#map = affine_map<()[s0] -> (s0 * 2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @opencv_bilinear2x_f32(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: memref<?x?x?x?xf32>, %arg5: memref<?x?x?x?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c-1 = arith.constant -1 : index
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 5.000000e-01 : f32
    %0 = arith.index_cast %arg2 : i32 to index
    %1 = arith.index_cast %arg3 : i32 to index
    %2 = arith.index_cast %arg0 : i32 to index
    %3 = arith.index_cast %arg1 : i32 to index
    %4 = affine.apply #map()[%1]
    %5 = affine.apply #map()[%0]
    %6 = polygeist.submap(%arg5, %2, %3, %5, %4) {map = #map1} : (memref<?x?x?x?xf32>, index, index, index, index) -> memref<?x?x?x?xf32>
    linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%6 : memref<?x?x?x?xf32>) {
    ^bb0(%out: f32):
      %7 = linalg.index 0 : index
      %8 = linalg.index 1 : index
      %9 = linalg.index 2 : index
      %10 = arith.index_cast %9 : index to i32
      %11 = arith.divsi %10, %c2_i32 : i32
      %12 = arith.addi %11, %c1_i32 : i32
      %13 = arith.cmpi slt, %12, %arg2 : i32
      %14 = arith.select %13, %12, %11 : i32
      %15 = arith.remsi %10, %c2_i32 : i32
      %16 = arith.sitofp %15 : i32 to f32
      %17 = arith.mulf %16, %cst : f32
      %18 = arith.index_cast %11 : i32 to index
      %19 = arith.index_cast %14 : i32 to index
      %20 = linalg.index 3 : index
      %21 = arith.index_cast %20 : index to i32
      %22 = arith.divsi %21, %c2_i32 : i32
      %23 = arith.addi %22, %c1_i32 : i32
      %24 = arith.cmpi slt, %23, %arg3 : i32
      %25 = arith.select %24, %23, %22 : i32
      %26 = arith.remsi %21, %c2_i32 : i32
      %27 = arith.sitofp %26 : i32 to f32
      %28 = arith.mulf %27, %cst : f32
      %29 = arith.index_cast %22 : i32 to index
      %30 = arith.cmpi slt, %9, %c0 : index
      %31 = arith.subi %c-1, %9 : index
      %32 = arith.select %30, %31, %9 : index
      %33 = arith.divsi %32, %c2 : index
      %34 = arith.subi %c-1, %33 : index
      %35 = arith.select %30, %34, %33 : index
      %36 = arith.cmpi slt, %20, %c0 : index
      %37 = arith.subi %c-1, %20 : index
      %38 = arith.select %36, %37, %20 : index
      %39 = arith.divsi %38, %c2 : index
      %40 = arith.subi %c-1, %39 : index
      %41 = arith.select %36, %40, %39 : index
      %42 = memref.load %arg4[%7, %8, %35, %41] : memref<?x?x?x?xf32>
      %43 = arith.index_cast %25 : i32 to index
      %44 = memref.load %arg4[%7, %8, %19, %43] : memref<?x?x?x?xf32>
      %45 = memref.load %arg4[%7, %8, %19, %29] : memref<?x?x?x?xf32>
      %46 = memref.load %arg4[%7, %8, %18, %43] : memref<?x?x?x?xf32>
      %47 = arith.subf %45, %42 : f32
      %48 = arith.mulf %17, %47 : f32
      %49 = arith.addf %42, %48 : f32
      %50 = arith.subf %46, %42 : f32
      %51 = arith.subf %44, %46 : f32
      %52 = arith.subf %51, %45 : f32
      %53 = arith.addf %52, %42 : f32
      %54 = arith.mulf %17, %53 : f32
      %55 = arith.addf %50, %54 : f32
      %56 = arith.mulf %28, %55 : f32
      %57 = arith.addf %49, %56 : f32
      linalg.yield %57 : f32
    }
    return
  }
}


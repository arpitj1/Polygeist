#map = affine_map<()[s0] -> (s0 * 2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @opencv_bilinear2x_f32(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: memref<?x?x?x?xf32>, %arg5: memref<?x?x?x?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 5.000000e-01 : f32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c-1 = arith.constant -1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %0 = bufferization.to_tensor %arg5 : memref<?x?x?x?xf32>
    %1 = bufferization.to_tensor %arg4 : memref<?x?x?x?xf32>
    %2 = arith.index_cast %arg2 : i32 to index
    %3 = arith.index_cast %arg3 : i32 to index
    %4 = arith.index_cast %arg0 : i32 to index
    %5 = arith.index_cast %arg1 : i32 to index
    %6 = affine.apply #map()[%3]
    %7 = affine.apply #map()[%2]
    %8 = polygeist.submap(%0, %4, %5, %7, %6) {map = #map1} : (tensor<?x?x?x?xf32>, index, index, index, index) -> tensor<?x?x?x?xf32>
    %9 = linalg.generic {doc = "", indexing_maps = [#map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%8 : tensor<?x?x?x?xf32>) {
    ^bb0(%out: f32):
      %12 = linalg.index 0 : index
      %13 = linalg.index 1 : index
      %14 = linalg.index 2 : index
      %15 = arith.index_cast %14 : index to i32
      %16 = arith.divsi %15, %c2_i32 : i32
      %17 = arith.addi %16, %c1_i32 : i32
      %18 = arith.cmpi slt, %17, %arg2 : i32
      %19 = arith.select %18, %17, %16 : i32
      %20 = arith.remsi %15, %c2_i32 : i32
      %21 = arith.sitofp %20 : i32 to f32
      %22 = arith.mulf %21, %cst : f32
      %23 = arith.index_cast %16 : i32 to index
      %24 = arith.index_cast %19 : i32 to index
      %25 = linalg.index 3 : index
      %26 = arith.index_cast %25 : index to i32
      %27 = arith.divsi %26, %c2_i32 : i32
      %28 = arith.addi %27, %c1_i32 : i32
      %29 = arith.cmpi slt, %28, %arg3 : i32
      %30 = arith.select %29, %28, %27 : i32
      %31 = arith.remsi %26, %c2_i32 : i32
      %32 = arith.sitofp %31 : i32 to f32
      %33 = arith.mulf %32, %cst : f32
      %34 = arith.index_cast %27 : i32 to index
      %35 = arith.cmpi slt, %14, %c0 : index
      %36 = arith.subi %c-1, %14 : index
      %37 = arith.select %35, %36, %14 : index
      %38 = arith.divsi %37, %c2 : index
      %39 = arith.subi %c-1, %38 : index
      %40 = arith.select %35, %39, %38 : index
      %41 = arith.cmpi slt, %25, %c0 : index
      %42 = arith.subi %c-1, %25 : index
      %43 = arith.select %41, %42, %25 : index
      %44 = arith.divsi %43, %c2 : index
      %45 = arith.subi %c-1, %44 : index
      %46 = arith.select %41, %45, %44 : index
      %extracted = tensor.extract %1[%12, %13, %40, %46] : tensor<?x?x?x?xf32>
      %47 = arith.index_cast %30 : i32 to index
      %extracted_0 = tensor.extract %1[%12, %13, %24, %47] : tensor<?x?x?x?xf32>
      %extracted_1 = tensor.extract %1[%12, %13, %24, %34] : tensor<?x?x?x?xf32>
      %extracted_2 = tensor.extract %1[%12, %13, %23, %47] : tensor<?x?x?x?xf32>
      %48 = arith.subf %extracted_1, %extracted : f32
      %49 = arith.mulf %22, %48 : f32
      %50 = arith.addf %extracted, %49 : f32
      %51 = arith.subf %extracted_2, %extracted : f32
      %52 = arith.subf %extracted_0, %extracted_2 : f32
      %53 = arith.subf %52, %extracted_1 : f32
      %54 = arith.addf %53, %extracted : f32
      %55 = arith.mulf %22, %54 : f32
      %56 = arith.addf %51, %55 : f32
      %57 = arith.mulf %33, %56 : f32
      %58 = arith.addf %50, %57 : f32
      linalg.yield %58 : f32
    } -> tensor<?x?x?x?xf32>
    %10 = polygeist.submapInverse(%0, %9, %4, %5, %7, %6) {map = #map1} : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>, index, index, index, index) -> tensor<?x?x?x?xf32>
    %11 = bufferization.to_memref %10 : memref<?x?x?x?xf32>
    memref.copy %11, %arg5 : memref<?x?x?x?xf32> to memref<?x?x?x?xf32>
    return
  }
}


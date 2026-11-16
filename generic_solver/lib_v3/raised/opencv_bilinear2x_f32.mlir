#map = affine_map<()[s0] -> (s0 * 2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @opencv_bilinear2x_f32(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: memref<?x?x?x?xf32>, %arg5: memref<?x?x?x?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c-1 = arith.constant -1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 5.000000e-01 : f32
    %0 = arith.index_cast %arg2 : i32 to index
    %1 = arith.index_cast %arg3 : i32 to index
    %2 = arith.index_cast %arg0 : i32 to index
    %3 = arith.index_cast %arg1 : i32 to index
    affine.for %arg6 = 0 to %2 {
      affine.for %arg7 = 0 to %3 {
        affine.for %arg8 = 0 to #map()[%0] {
          %4 = arith.index_cast %arg8 : index to i32
          %5 = arith.divsi %4, %c2_i32 : i32
          %6 = arith.addi %5, %c1_i32 : i32
          %7 = arith.cmpi slt, %6, %arg2 : i32
          %8 = arith.select %7, %6, %5 : i32
          %9 = arith.remsi %4, %c2_i32 : i32
          %10 = arith.sitofp %9 : i32 to f32
          %11 = arith.mulf %10, %cst : f32
          %12 = arith.index_cast %5 : i32 to index
          %13 = arith.index_cast %8 : i32 to index
          %14 = arith.cmpi slt, %arg8, %c0 : index
          %15 = arith.subi %c-1, %arg8 : index
          %16 = arith.select %14, %15, %arg8 : index
          %17 = arith.divsi %16, %c2 : index
          %18 = arith.subi %c-1, %17 : index
          %19 = arith.select %14, %18, %17 : index
          affine.for %arg9 = 0 to #map()[%1] {
            %20 = arith.index_cast %arg9 : index to i32
            %21 = arith.divsi %20, %c2_i32 : i32
            %22 = arith.addi %21, %c1_i32 : i32
            %23 = arith.cmpi slt, %22, %arg3 : i32
            %24 = arith.select %23, %22, %21 : i32
            %25 = arith.remsi %20, %c2_i32 : i32
            %26 = arith.sitofp %25 : i32 to f32
            %27 = arith.mulf %26, %cst : f32
            %28 = arith.index_cast %21 : i32 to index
            %29 = arith.cmpi slt, %arg9, %c0 : index
            %30 = arith.subi %c-1, %arg9 : index
            %31 = arith.select %29, %30, %arg9 : index
            %32 = arith.divsi %31, %c2 : index
            %33 = arith.subi %c-1, %32 : index
            %34 = arith.select %29, %33, %32 : index
            %35 = memref.load %arg4[%arg6, %arg7, %19, %34] : memref<?x?x?x?xf32>
            %36 = arith.index_cast %24 : i32 to index
            %37 = memref.load %arg4[%arg6, %arg7, %13, %36] : memref<?x?x?x?xf32>
            %38 = memref.load %arg4[%arg6, %arg7, %13, %28] : memref<?x?x?x?xf32>
            %39 = memref.load %arg4[%arg6, %arg7, %12, %36] : memref<?x?x?x?xf32>
            %40 = arith.subf %38, %35 : f32
            %41 = arith.mulf %11, %40 : f32
            %42 = arith.addf %35, %41 : f32
            %43 = arith.subf %39, %35 : f32
            %44 = arith.subf %37, %39 : f32
            %45 = arith.subf %44, %38 : f32
            %46 = arith.addf %45, %35 : f32
            %47 = arith.mulf %11, %46 : f32
            %48 = arith.addf %43, %47 : f32
            %49 = arith.mulf %27, %48 : f32
            %50 = arith.addf %42, %49 : f32
            affine.store %50, %arg5[%arg6, %arg7, %arg8, %arg9] : memref<?x?x?x?xf32>
          }
        }
      }
    }
    return
  }
}

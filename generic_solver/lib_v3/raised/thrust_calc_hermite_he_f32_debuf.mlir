module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @thrust_calc_hermite_he_f32(%arg0: i32, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c0_i64 = arith.constant 0 : i64
    %cst_1 = arith.constant 0x7FC00000 : f32
    %c128_i64 = arith.constant 128 : i64
    %0 = bufferization.to_tensor %arg3 : memref<?xf32>
    %1 = bufferization.to_tensor %arg2 : memref<?xf32>
    %2 = bufferization.to_tensor %arg1 : memref<?xf32>
    %3 = arith.index_cast %arg0 : i32 to index
    %4 = llvm.mlir.undef : f32
    %5 = affine.for %arg4 = 0 to %3 iter_args(%arg5 = %0) -> (tensor<?xf32>) {
      %extracted = tensor.extract %2[%arg4] : tensor<?xf32>
      %extracted_2 = tensor.extract %1[%arg4] : tensor<?xf32>
      %7 = arith.fptosi %extracted_2 : f32 to i64
      %8 = arith.index_cast %7 : i64 to index
      %9 = arith.cmpi slt, %7, %c0_i64 : i64
      %10 = arith.cmpi sge, %7, %c0_i64 : i64
      %11 = arith.select %9, %cst_0, %4 : f32
      %12 = arith.cmpi eq, %7, %c0_i64 : i64
      %13 = arith.cmpi ne, %7, %c0_i64 : i64
      %14 = arith.select %12, %cst, %11 : f32
      %15 = arith.andi %10, %13 : i1
      %16 = arith.select %10, %14, %11 : f32
      %17 = arith.cmpi eq, %7, %c1_i64 : i64
      %18 = arith.cmpi ne, %7, %c1_i64 : i64
      %19 = arith.select %17, %extracted, %16 : f32
      %20 = arith.cmpi sgt, %7, %c128_i64 : i64
      %21 = arith.cmpi sle, %7, %c128_i64 : i64
      %22 = scf.if %15 -> (f32) {
        %23 = scf.if %18 -> (f32) {
          %24 = scf.if %21 -> (f32) {
            %25:3 = scf.for %arg6 = %c1 to %8 step %c1 iter_args(%arg7 = %extracted, %arg8 = %cst, %arg9 = %4) -> (f32, f32, f32) {
              %26 = arith.index_cast %arg6 : index to i64
              %27 = arith.mulf %extracted, %arg7 : f32
              %28 = arith.sitofp %26 : i64 to f32
              %29 = arith.mulf %28, %arg8 : f32
              %30 = arith.subf %27, %29 : f32
              scf.yield %30, %arg7, %30 : f32, f32, f32
            }
            scf.yield %25#2 : f32
          } else {
            %25 = arith.select %20, %cst_1, %19 : f32
            scf.yield %25 : f32
          }
          scf.yield %24 : f32
        } else {
          scf.yield %19 : f32
        }
        scf.yield %23 : f32
      } else {
        scf.yield %16 : f32
      }
      %inserted = tensor.insert %22 into %arg5[%arg4] : tensor<?xf32>
      affine.yield %inserted : tensor<?xf32>
    }
    %6 = bufferization.to_memref %5 : memref<?xf32>
    memref.copy %6, %arg3 : memref<?xf32> to memref<?xf32>
    return
  }
}


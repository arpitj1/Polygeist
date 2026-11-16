module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @thrust_calc_hermite_h_f32(%arg0: i32, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i64 = arith.constant 1 : i64
    %true = arith.constant true
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c0_i64 = arith.constant 0 : i64
    %c2 = arith.constant 2 : index
    %cst_1 = arith.constant 0x7FC00000 : f32
    %c128_i64 = arith.constant 128 : i64
    %c-1_i64 = arith.constant -1 : i64
    %cst_2 = arith.constant 0x7F800000 : f32
    %0 = bufferization.to_tensor %arg3 : memref<?xf32>
    %1 = bufferization.to_tensor %arg2 : memref<?xf32>
    %2 = bufferization.to_tensor %arg1 : memref<?xf32>
    %3 = arith.index_cast %arg0 : i32 to index
    %4 = llvm.mlir.undef : f32
    %5 = affine.for %arg4 = 0 to %3 iter_args(%arg5 = %0) -> (tensor<?xf32>) {
      %extracted = tensor.extract %2[%arg4] : tensor<?xf32>
      %extracted_3 = tensor.extract %1[%arg4] : tensor<?xf32>
      %7 = math.absf %extracted_3 : f32
      %8 = arith.cmpf oeq, %7, %cst_2 : f32
      %9 = arith.cmpf uno, %extracted_3, %extracted_3 : f32
      %10 = arith.fptosi %extracted_3 : f32 to i64
      %11 = arith.cmpi slt, %10, %c0_i64 : i64
      %12 = arith.cmpi sge, %10, %c0_i64 : i64
      %13 = arith.cmpi eq, %10, %c0_i64 : i64
      %14 = arith.cmpi ne, %10, %c0_i64 : i64
      %15 = arith.cmpi eq, %10, %c1_i64 : i64
      %16 = arith.cmpi sgt, %10, %c128_i64 : i64
      %17 = arith.cmpi sle, %10, %c128_i64 : i64
      %18 = arith.select %9, %c-1_i64, %10 : i64
      %19 = arith.xori %9, %true : i1
      %20 = arith.andi %19, %11 : i1
      %21 = arith.ori %9, %20 : i1
      %22 = arith.xori %9, %true : i1
      %23 = arith.andi %22, %12 : i1
      %24 = arith.xori %9, %true : i1
      %25 = arith.andi %24, %13 : i1
      %26 = arith.xori %9, %true : i1
      %27 = arith.andi %26, %14 : i1
      %28 = arith.ori %9, %27 : i1
      %29 = arith.xori %9, %true : i1
      %30 = arith.andi %29, %15 : i1
      %31 = arith.xori %9, %true : i1
      %32 = arith.andi %31, %16 : i1
      %33 = arith.xori %9, %true : i1
      %34 = arith.andi %33, %17 : i1
      %35 = arith.ori %9, %34 : i1
      %36 = arith.select %8, %c-1_i64, %18 : i64
      %37 = arith.xori %8, %true : i1
      %38 = arith.andi %37, %21 : i1
      %39 = arith.ori %8, %38 : i1
      %40 = arith.xori %8, %true : i1
      %41 = arith.andi %40, %23 : i1
      %42 = arith.xori %8, %true : i1
      %43 = arith.andi %42, %25 : i1
      %44 = arith.xori %8, %true : i1
      %45 = arith.andi %44, %28 : i1
      %46 = arith.ori %8, %45 : i1
      %47 = arith.xori %8, %true : i1
      %48 = arith.andi %47, %30 : i1
      %49 = arith.xori %8, %true : i1
      %50 = arith.andi %49, %32 : i1
      %51 = arith.xori %8, %true : i1
      %52 = arith.andi %51, %35 : i1
      %53 = arith.ori %8, %52 : i1
      %54 = arith.index_cast %36 : i64 to index
      %55 = arith.select %39, %cst_0, %4 : f32
      %56 = arith.andi %41, %46 : i1
      %57 = arith.select %43, %cst, %55 : f32
      %58 = arith.select %41, %57, %55 : f32
      %59 = arith.andi %56, %48 : i1
      %60 = arith.xori %59, %true : i1
      %61 = arith.andi %60, %56 : i1
      %62 = arith.addf %extracted, %extracted : f32
      %63 = arith.select %59, %62, %58 : f32
      %64 = scf.if %61 -> (f32) {
        %65 = scf.if %53 -> (f32) {
          %66 = arith.addf %extracted, %extracted : f32
          %67 = arith.muli %54, %c2 : index
          %68:3 = scf.for %arg6 = %c2 to %67 step %c2 iter_args(%arg7 = %66, %arg8 = %cst, %arg9 = %cst_0) -> (f32, f32, f32) {
            %69 = arith.index_cast %arg6 : index to i64
            %70 = arith.mulf %66, %arg7 : f32
            %71 = arith.sitofp %69 : i64 to f32
            %72 = arith.mulf %71, %arg8 : f32
            %73 = arith.subf %70, %72 : f32
            scf.yield %73, %arg7, %73 : f32, f32, f32
          }
          scf.yield %68#2 : f32
        } else {
          %66 = arith.select %50, %cst_1, %63 : f32
          scf.yield %66 : f32
        }
        scf.yield %65 : f32
      } else {
        scf.yield %63 : f32
      }
      %inserted = tensor.insert %64 into %arg5[%arg4] : tensor<?xf32>
      affine.yield %inserted : tensor<?xf32>
    }
    %6 = bufferization.to_memref %5 : memref<?xf32>
    memref.copy %6, %arg3 : memref<?xf32> to memref<?xf32>
    return
  }
}


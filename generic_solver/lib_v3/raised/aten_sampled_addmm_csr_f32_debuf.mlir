#map = affine_map<(d0) -> (d0 + 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_sampled_addmm_csr_f32(%arg0: i32, %arg1: i32, %arg2: memref<?xi32>, %arg3: i32, %arg4: memref<?xi32>, %arg5: i32, %arg6: memref<?xf32>, %arg7: i32, %arg8: i32, %arg9: memref<?x?xf32>, %arg10: i32, %arg11: i32, %arg12: memref<?x?xf32>, %arg13: f32, %arg14: f32, %arg15: i32, %arg16: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %0 = bufferization.to_tensor %arg16 : memref<?xf32>
    %1 = bufferization.to_tensor %arg12 : memref<?x?xf32>
    %2 = bufferization.to_tensor %arg9 : memref<?x?xf32>
    %3 = bufferization.to_tensor %arg6 : memref<?xf32>
    %4 = bufferization.to_tensor %arg4 : memref<?xi32>
    %5 = bufferization.to_tensor %arg2 : memref<?xi32>
    %6 = arith.index_cast %arg0 : i32 to index
    %7 = arith.index_cast %arg8 : i32 to index
    %8 = tensor.empty() : tensor<f32>
    %9:2 = affine.for %arg17 = 0 to %6 iter_args(%arg18 = %8, %arg19 = %0) -> (tensor<f32>, tensor<?xf32>) {
      %extracted = tensor.extract %5[%arg17] : tensor<?xi32>
      %11 = affine.apply #map(%arg17)
      %extracted_0 = tensor.extract %5[%11] : tensor<?xi32>
      %12 = arith.index_cast %extracted_0 : i32 to index
      %13 = arith.index_cast %extracted : i32 to index
      %14:2 = scf.for %arg20 = %13 to %12 step %c1 iter_args(%arg21 = %arg18, %arg22 = %arg19) -> (tensor<f32>, tensor<?xf32>) {
        %extracted_1 = tensor.extract %4[%arg20] : tensor<?xi32>
        %extracted_2 = tensor.extract %3[%arg20] : tensor<?xf32>
        %15 = arith.index_cast %extracted_1 : i32 to index
        %inserted = tensor.insert %cst into %arg21[] : tensor<f32>
        %16 = affine.for %arg23 = 0 to %7 iter_args(%arg24 = %inserted) -> (tensor<f32>) {
          %extracted_5 = tensor.extract %arg24[] : tensor<f32>
          %extracted_6 = tensor.extract %2[%arg17, %arg23] : tensor<?x?xf32>
          %extracted_7 = tensor.extract %1[%arg23, %15] : tensor<?x?xf32>
          %20 = arith.mulf %extracted_6, %extracted_7 : f32
          %21 = arith.addf %extracted_5, %20 : f32
          %inserted_8 = tensor.insert %21 into %arg24[] : tensor<f32>
          affine.yield %inserted_8 : tensor<f32>
        }
        %extracted_3 = tensor.extract %16[] : tensor<f32>
        %17 = arith.mulf %arg13, %extracted_3 : f32
        %18 = arith.mulf %arg14, %extracted_2 : f32
        %19 = arith.addf %17, %18 : f32
        %inserted_4 = tensor.insert %19 into %arg22[%arg20] : tensor<?xf32>
        scf.yield %16, %inserted_4 : tensor<f32>, tensor<?xf32>
      }
      affine.yield %14#0, %14#1 : tensor<f32>, tensor<?xf32>
    }
    %10 = bufferization.to_memref %9#1 : memref<?xf32>
    memref.copy %10, %arg16 : memref<?xf32> to memref<?xf32>
    return
  }
}


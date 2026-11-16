#map = affine_map<()[s0] -> (s0 - 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @opencv_bilateral3x3_square_u8(%arg0: i32, %arg1: i32, %arg2: memref<?xi8>, %arg3: memref<?xi8>, %arg4: f32, %arg5: f32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c255_i32 = arith.constant 255 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 5.000000e-01 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant -5.000000e-01 : f32
    %0 = arith.index_cast %arg1 : i32 to index
    %1 = arith.index_cast %arg0 : i32 to index
    %2 = arith.mulf %arg4, %arg4 : f32
    %3 = arith.divf %cst_1, %2 : f32
    %4 = arith.mulf %arg5, %arg5 : f32
    %5 = arith.divf %cst_1, %4 : f32
    affine.for %arg6 = 1 to #map()[%1] {
      affine.for %arg7 = 1 to #map()[%0] {
        %6 = affine.load %arg2[%arg7 + %arg6 * symbol(%0)] : memref<?xi8>
        %7 = arith.uitofp %6 : i8 to f32
        %8:2 = affine.for %arg8 = -1 to 2 iter_args(%arg9 = %cst_0, %arg10 = %cst_0) -> (f32, f32) {
          %15 = arith.index_cast %arg8 : index to i32
          %16 = arith.muli %15, %15 : i32
          %17:2 = affine.for %arg11 = -1 to 2 iter_args(%arg12 = %arg9, %arg13 = %arg10) -> (f32, f32) {
            %18 = arith.index_cast %arg11 : index to i32
            %19 = affine.load %arg2[%arg11 + %arg7 + (%arg6 + %arg8) * symbol(%0)] : memref<?xi8>
            %20 = arith.uitofp %19 : i8 to f32
            %21 = arith.subf %20, %7 : f32
            %22 = arith.muli %18, %18 : i32
            %23 = arith.addi %22, %16 : i32
            %24 = arith.sitofp %23 : i32 to f32
            %25 = arith.mulf %24, %5 : f32
            %26 = arith.mulf %21, %21 : f32
            %27 = arith.mulf %26, %3 : f32
            %28 = arith.addf %25, %27 : f32
            %29 = math.exp %28 : f32
            %30 = arith.addf %arg13, %29 : f32
            %31 = arith.mulf %29, %20 : f32
            %32 = arith.addf %arg12, %31 : f32
            affine.yield %32, %30 : f32, f32
          }
          affine.yield %17#0, %17#1 : f32, f32
        }
        %9 = arith.divf %8#0, %8#1 : f32
        %10 = arith.addf %9, %cst : f32
        %11 = arith.fptosi %10 : f32 to i32
        %12 = arith.cmpi slt, %11, %c0_i32 : i32
        %13 = scf.if %12 -> (i32) {
          scf.yield %c0_i32 : i32
        } else {
          %15 = arith.cmpi sgt, %11, %c255_i32 : i32
          %16 = arith.select %15, %c255_i32, %11 : i32
          scf.yield %16 : i32
        }
        %14 = arith.trunci %13 : i32 to i8
        affine.store %14, %arg3[%arg7 + %arg6 * symbol(%0)] : memref<?xi8>
      }
    }
    return
  }
}

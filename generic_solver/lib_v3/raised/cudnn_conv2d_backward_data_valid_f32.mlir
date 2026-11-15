#map = affine_map<()[s0] -> (s0 - 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @cudnn_conv2d_backward_data_valid_f32(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: memref<?x?x?x?xf32>, %arg8: memref<?x?x?x?xf32>, %arg9: memref<?x?x?x?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = arith.index_cast %arg2 : i32 to index
    %2 = arith.addi %arg3, %arg5 : i32
    %3 = arith.index_cast %2 : i32 to index
    %4 = arith.addi %arg4, %arg6 : i32
    %5 = arith.index_cast %4 : i32 to index
    affine.for %arg10 = 0 to %0 {
      affine.for %arg11 = 0 to %1 {
        affine.for %arg12 = 0 to #map()[%3] {
          affine.for %arg13 = 0 to #map()[%5] {
            affine.store %cst, %arg9[%arg10, %arg11, %arg12, %arg13] : memref<?x?x?x?xf32>
          }
        }
      }
    }
    %6 = arith.index_cast %arg1 : i32 to index
    %7 = arith.index_cast %arg3 : i32 to index
    %8 = arith.index_cast %arg4 : i32 to index
    %9 = arith.index_cast %arg5 : i32 to index
    %10 = arith.index_cast %arg6 : i32 to index
    affine.for %arg10 = 0 to %0 {
      affine.for %arg11 = 0 to %6 {
        affine.for %arg12 = 0 to %7 {
          affine.for %arg13 = 0 to %8 {
            affine.for %arg14 = 0 to %1 {
              affine.for %arg15 = 0 to %9 {
                affine.for %arg16 = 0 to %10 {
                  %11 = affine.load %arg8[%arg11, %arg14, %arg15, %arg16] : memref<?x?x?x?xf32>
                  %12 = affine.load %arg7[%arg10, %arg11, %arg12, %arg13] : memref<?x?x?x?xf32>
                  %13 = arith.mulf %11, %12 : f32
                  %14 = affine.load %arg9[%arg10, %arg14, %arg12 + %arg15, %arg13 + %arg16] : memref<?x?x?x?xf32>
                  %15 = arith.addf %14, %13 : f32
                  affine.store %15, %arg9[%arg10, %arg14, %arg12 + %arg15, %arg13 + %arg16] : memref<?x?x?x?xf32>
                }
              }
            }
          }
        }
      }
    }
    return
  }
}

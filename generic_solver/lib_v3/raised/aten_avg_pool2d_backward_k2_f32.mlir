#map = affine_map<(d0) -> (d0 * 2)>
#map1 = affine_map<(d0) -> (d0 * 2 + 2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_avg_pool2d_backward_k2_f32(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: memref<?x?x?xf32>, %arg6: memref<?x?x?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 4.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c2_i32 = arith.constant 2 : i32
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = arith.muli %arg1, %c2_i32 : i32
    %2 = arith.addi %1, %arg3 : i32
    %3 = arith.index_cast %2 : i32 to index
    %4 = arith.muli %arg2, %c2_i32 : i32
    %5 = arith.addi %4, %arg4 : i32
    %6 = arith.index_cast %5 : i32 to index
    %7 = arith.index_cast %arg1 : i32 to index
    %8 = arith.index_cast %arg2 : i32 to index
    affine.for %arg7 = 0 to %0 {
      affine.for %arg8 = 0 to %3 {
        affine.for %arg9 = 0 to %6 {
          affine.store %cst_0, %arg6[%arg7, %arg8, %arg9] : memref<?x?x?xf32>
        }
      }
      affine.for %arg8 = 0 to %7 {
        affine.for %arg9 = 0 to %8 {
          %9 = affine.load %arg5[%arg7, %arg8, %arg9] : memref<?x?x?xf32>
          %10 = arith.divf %9, %cst : f32
          affine.for %arg10 = #map(%arg8) to #map1(%arg8) {
            affine.for %arg11 = #map(%arg9) to #map1(%arg9) {
              %11 = affine.load %arg6[%arg7, %arg10, %arg11] : memref<?x?x?xf32>
              %12 = arith.addf %11, %10 : f32
              affine.store %12, %arg6[%arg7, %arg10, %arg11] : memref<?x?x?xf32>
            }
          }
        }
      }
    }
    return
  }
}

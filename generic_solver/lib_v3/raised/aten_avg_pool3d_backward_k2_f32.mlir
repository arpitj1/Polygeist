#map = affine_map<(d0) -> (d0 * 2)>
#map1 = affine_map<(d0) -> (d0 * 2 + 2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_avg_pool3d_backward_k2_f32(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: memref<?x?x?x?xf32>, %arg8: memref<?x?x?x?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 8.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c2_i32 = arith.constant 2 : i32
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = arith.muli %arg1, %c2_i32 : i32
    %2 = arith.addi %1, %arg4 : i32
    %3 = arith.index_cast %2 : i32 to index
    %4 = arith.muli %arg2, %c2_i32 : i32
    %5 = arith.addi %4, %arg5 : i32
    %6 = arith.index_cast %5 : i32 to index
    %7 = arith.muli %arg3, %c2_i32 : i32
    %8 = arith.addi %7, %arg6 : i32
    %9 = arith.index_cast %8 : i32 to index
    %10 = arith.index_cast %arg1 : i32 to index
    %11 = arith.index_cast %arg2 : i32 to index
    %12 = arith.index_cast %arg3 : i32 to index
    affine.for %arg9 = 0 to %0 {
      affine.for %arg10 = 0 to %3 {
        affine.for %arg11 = 0 to %6 {
          affine.for %arg12 = 0 to %9 {
            affine.store %cst_0, %arg8[%arg9, %arg10, %arg11, %arg12] : memref<?x?x?x?xf32>
          }
        }
      }
      affine.for %arg10 = 0 to %10 {
        affine.for %arg11 = 0 to %11 {
          affine.for %arg12 = 0 to %12 {
            %13 = affine.load %arg7[%arg9, %arg10, %arg11, %arg12] : memref<?x?x?x?xf32>
            %14 = arith.divf %13, %cst : f32
            affine.for %arg13 = #map(%arg10) to #map1(%arg10) {
              affine.for %arg14 = #map(%arg11) to #map1(%arg11) {
                affine.for %arg15 = #map(%arg12) to #map1(%arg12) {
                  %15 = affine.load %arg8[%arg9, %arg13, %arg14, %arg15] : memref<?x?x?x?xf32>
                  %16 = arith.addf %15, %14 : f32
                  affine.store %16, %arg8[%arg9, %arg13, %arg14, %arg15] : memref<?x?x?x?xf32>
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

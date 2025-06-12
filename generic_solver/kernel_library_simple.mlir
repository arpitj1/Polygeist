// Kernel Library - Reusable kernel definitions
// This file contains a collection of kernel definitions that can be loaded
// by the linalg-to-kernel pass and applied to different MLIR modules.

module {
  // Collection of kernel operation definitions
  kernel.defn_collection {
    
    // Simple GEMM operation definition with linalg.generic representation
    kernel.defn @simple_gemm_linalg(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>) -> tensor<?x?xf32> {
      // Simple matrix multiplication: C = A * B + C
      %result = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2) -> (d0, d2)>,
          affine_map<(d0, d1, d2) -> (d2, d1)>,
          affine_map<(d0, d1, d2) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel", "reduction"]
      } ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>) 
        outs(%C : tensor<?x?xf32>) {
        ^bb0(%a: f32, %b: f32, %c: f32):
          %product = arith.mulf %a, %b : f32
          %result = arith.addf %product, %c : f32
          linalg.yield %result : f32
      } -> tensor<?x?xf32>
      kernel.yield %result : tensor<?x?xf32>
    }

    // Scaled GEMM operation definition with alpha and beta coefficients
    kernel.defn @gemm_linalg(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>) -> tensor<?x?xf32> {
      %alpha = arith.constant 1.0 : f32
      %beta = arith.constant 0.0 : f32
      
      // GEMM with scaling: C = alpha * A * B + beta * C
      %result = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2) -> (d0, d2)>,
          affine_map<(d0, d1, d2) -> (d2, d1)>,
          affine_map<(d0, d1, d2) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel", "reduction"]
      } ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>) 
        outs(%C : tensor<?x?xf32>) {
        ^bb0(%a: f32, %b: f32, %c: f32):
          %product = arith.mulf %a, %b : f32
          %scaled = arith.mulf %product, %alpha : f32
          %scaled_c = arith.mulf %c, %beta : f32
          %result = arith.addf %scaled, %scaled_c : f32
          linalg.yield %result : f32
      } -> tensor<?x?xf32>
      kernel.yield %result : tensor<?x?xf32>
    }

    // Sum of absolute values operation (ASUM)
    kernel.defn @asum_linalg(%X: tensor<?xf32>) -> tensor<f32> {
      %c0 = arith.constant 0.0 : f32
      %init = tensor.empty() : tensor<f32>
      %fill = linalg.fill ins(%c0 : f32) outs(%init : tensor<f32>) -> tensor<f32>
      
      // Sum of absolute values: result = sum_i |x_i|
      %result = linalg.generic {
        indexing_maps = [
          affine_map<(d0) -> (d0)>,
          affine_map<(d0) -> ()>
        ],
        iterator_types = ["reduction"]
      } ins(%X : tensor<?xf32>) 
        outs(%fill : tensor<f32>) {
        ^bb0(%in: f32, %out: f32):
          %abs_val = math.absf %in : f32
          %result = arith.addf %abs_val, %out : f32
          linalg.yield %result : f32
      } -> tensor<f32>
      kernel.yield %result : tensor<f32>
    }

    // Vector dot product
    kernel.defn @dot_linalg(%X: tensor<?xf32>, %Y: tensor<?xf32>) -> tensor<f32> {
      %c0 = arith.constant 0.0 : f32
      %init = tensor.empty() : tensor<f32>
      %fill = linalg.fill ins(%c0 : f32) outs(%init : tensor<f32>) -> tensor<f32>
      
      // Dot product: result = sum_i x_i * y_i
      %result = linalg.generic {
        indexing_maps = [
          affine_map<(d0) -> (d0)>,
          affine_map<(d0) -> (d0)>,
          affine_map<(d0) -> ()>
        ],
        iterator_types = ["reduction"]
      } ins(%X, %Y : tensor<?xf32>, tensor<?xf32>) 
        outs(%fill : tensor<f32>) {
        ^bb0(%x: f32, %y: f32, %out: f32):
          %product = arith.mulf %x, %y : f32
          %result = arith.addf %product, %out : f32
          linalg.yield %result : f32
      } -> tensor<f32>
      kernel.yield %result : tensor<f32>
    }
  }
} 
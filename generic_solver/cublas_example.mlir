// Example MLIR module demonstrating kernel operations and their linalg.generic representations
module {
  // Define a collection of kernel operation definitions
  kernel.defn_collection {
    
    // GEMM operation definition with arbitrary code implementation
    kernel.defn @gemm(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>) {
      // This could include arbitrary code to implement the GEMM operation
      // For example, calling into the actual kernel library
      "some.custom_code"() : () -> ()
      kernel.yield
    }

    // GEMM operation definition with linalg.generic representation
    kernel.defn @gemm_linalg(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>) -> tensor<?x?xf32> {
      //TODO: move to function arg
      //TODO: We can do const prop for alpha and beta for simple matmul match
      %alpha = arith.constant 1.0 : f32
      %beta = arith.constant 0.0 : f32
      
      // Implementation using linalg.generic
      %result = linalg.generic {
        indexing_maps = [
          affine_map<(i, j, k) -> (i, k)>,  // A(i,k)
          affine_map<(i, j, k) -> (k, j)>,  // B(k,j)
          affine_map<(i, j, k) -> (i, j)>   // C(i,j)
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

    // Batched GEMM operation definition with arbitrary code
    kernel.defn @batched_gemm(%A2: tensor<?x?x?xf32>, %B2: tensor<?x?x?xf32>, %C2: tensor<?x?x?xf32>) {
      // This could include arbitrary code to implement the batched GEMM operation
      "some.custom_code"() : () -> ()
      kernel.yield
    }

    // Batched GEMM operation definition with linalg.generic representation
    kernel.defn @batched_gemm_linalg(%A2: tensor<?x?x?xf32>, %B2: tensor<?x?x?xf32>, %C2: tensor<?x?x?xf32>) {
      %alpha = arith.constant 1.0 : f32
      %beta = arith.constant 0.0 : f32
      
      // Implementation using linalg.generic
      %result = linalg.generic {
        indexing_maps = [
          affine_map<(b, i, j, k) -> (b, i, k)>,  // A(b,i,k)
          affine_map<(b, i, j, k) -> (b, k, j)>,  // B(b,k,j)
          affine_map<(b, i, j, k) -> (b, i, j)>   // C(b,i,j)
        ],
        iterator_types = ["parallel", "parallel", "parallel", "reduction"]
      } ins(%A2, %B2 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) 
        outs(%C2 : tensor<?x?x?xf32>) {
        ^bb0(%a: f32, %b: f32, %c: f32):
          %product = arith.mulf %a, %b : f32
          %scaled = arith.mulf %product, %alpha : f32
          %scaled_c = arith.mulf %c, %beta : f32
          %result = arith.addf %scaled, %scaled_c : f32
          linalg.yield %result : f32
      } -> tensor<?x?x?xf32>
      kernel.yield
    }
    
    // GEMM operation definition with linalg.generic representation
    kernel.defn @simple_gemm_linalg(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>) -> tensor<?x?xf32> {
      // Implementation using linalg.generic
      %result = linalg.generic {
        indexing_maps = [
          affine_map<(i, j, k) -> (i, k)>,  // A(i,k)
          affine_map<(i, j, k) -> (k, j)>,  // B(k,j)
          affine_map<(i, j, k) -> (i, j)>   // C(i,j)
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


    // Index of maximum absolute value operation definition with arbitrary code
    kernel.defn @iamax(%X: tensor<?xf32>) -> tensor<i32> {
      // This could include arbitrary code to find the index of max absolute value
      %result = "some.custom_code"() : () -> tensor<i32>
      kernel.yield %result : tensor<i32>
    }

    // Index of maximum absolute value operation definition with linalg.generic representation
    kernel.defn @iamax_linalg(%X: tensor<?xf32>) -> tensor<i32> {
      // Create an initial tensor to store the result index
      %c0 = arith.constant 0 : i32
      %init = tensor.empty() : tensor<i32>
      %fill = linalg.fill ins(%c0 : i32) outs(%init : tensor<i32>) -> tensor<i32>
      
      // Implementation using linalg.generic
      %result = linalg.generic {
        indexing_maps = [
          affine_map<(i) -> (i)>,      // Input vector
          affine_map<(i) -> ()>        // Result scalar (index)
        ],
        iterator_types = ["reduction"]
      } ins(%X : tensor<?xf32>) 
        outs(%fill : tensor<i32>) {
        ^bb0(%in: f32, %out: i32):
          %idx = linalg.index 0 : index
          %abs_val = math.absf %in : f32
          %curr_max_idx = arith.index_cast %out : i32 to index
          %curr_max = tensor.extract %X[%curr_max_idx] : tensor<?xf32>
          %curr_max_abs = math.absf %curr_max : f32
          %cmp = arith.cmpf ogt, %abs_val, %curr_max_abs : f32
          %new_idx = arith.select %cmp, %idx, %curr_max_idx : index
          %result = arith.index_cast %new_idx : index to i32
          linalg.yield %result : i32
      } -> tensor<i32>
      kernel.yield %result : tensor<i32>
    }

    // Index of minimum absolute value operation definition with arbitrary code
    kernel.defn @iamin(%X: tensor<?xf32>) -> tensor<i32> {
      // This could include arbitrary code to find the index of min absolute value
      %result = "some.custom_code"() : () -> tensor<i32>
      kernel.yield %result : tensor<i32>
    }

    // Index of minimum absolute value operation definition with linalg.generic representation
    kernel.defn @iamin_linalg(%X: tensor<?xf32>) -> tensor<i32> {
      // Create an initial tensor to store the result index
      %c0 = arith.constant 0 : i32
      %init = tensor.empty() : tensor<i32>
      %fill = linalg.fill ins(%c0 : i32) outs(%init : tensor<i32>) -> tensor<i32>
      
      // Implementation using linalg.generic
      %result = linalg.generic {
        indexing_maps = [
          affine_map<(i) -> (i)>,      // Input vector
          affine_map<(i) -> ()>        // Result scalar (index)
        ],
        iterator_types = ["reduction"]
      } ins(%X : tensor<?xf32>) 
        outs(%fill : tensor<i32>) {
        ^bb0(%in: f32, %out: i32):
          %idx = linalg.index 0 : index
          %abs_val = math.absf %in : f32
          %curr_min_idx = arith.index_cast %out : i32 to index
          %curr_min = tensor.extract %X[%curr_min_idx] : tensor<?xf32>
          %curr_min_abs = math.absf %curr_min : f32
          %cmp = arith.cmpf olt, %abs_val, %curr_min_abs : f32
          %new_idx = arith.select %cmp, %idx, %curr_min_idx : index
          %result = arith.index_cast %new_idx : index to i32
          linalg.yield %result : i32
      } -> tensor<i32>
      kernel.yield %result : tensor<i32>
    }

    // Sum of absolute values operation definition with arbitrary code
    kernel.defn @asum(%X: tensor<?xf32>) -> tensor<f32> {
      // This could include arbitrary code to compute the sum of absolute values
      %result = "some.custom_code"() : () -> tensor<f32>
      kernel.yield %result : tensor<f32>
    }

    // Sum of absolute values operation definition with linalg.generic representation
    kernel.defn @asum_linalg(%X: tensor<?xf32>) -> tensor<f32> {
      // Create an initial tensor to store the result sum
      %c0 = arith.constant 0.0 : f32
      %init = tensor.empty() : tensor<f32>
      %fill = linalg.fill ins(%c0 : f32) outs(%init : tensor<f32>) -> tensor<f32>
      
      // Implementation using linalg.generic
      %result = linalg.generic {
        indexing_maps = [
          affine_map<(i) -> (i)>,      // Input vector
          affine_map<(i) -> ()>        // Result scalar (sum)
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
  
    // Mathematical definitions (commented, for reference)
    // kernel.defn @gemm(...) {
    //   C(i,j) += alpha * A(i,k) * B(k,j);
    // }
    
    // kernel.defn @batched_gemm(...) {
    //   C(b,i,j) += alpha * A(b,i,k) * B(b,k,j);
    // }
    
    // kernel.defn @iamax(...) {
    //   result = argmax_i |x_i|;
    // }
    
    // kernel.defn @iamin(...) {
    //   result = argmin_i |x_i|;
    // }
    
    // kernel.defn @asum(...) {
    //   result = sum_i |x_i|;
    // }
  }
    
    //Func that uses simple gemm
    func.func @simple_gemm(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>) -> tensor<?x?xf32> {
            // Implementation using linalg.generic
      %result = linalg.generic {
        indexing_maps = [
          affine_map<(i, j, k) -> (i, k)>,  // A(i,k)
          affine_map<(i, j, k) -> (k, j)>,  // B(k,j)
          affine_map<(i, j, k) -> (i, j)>   // C(i,j)
        ],
        iterator_types = ["parallel", "parallel", "reduction"]
      } ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>) 
        outs(%C : tensor<?x?xf32>) {
        ^bb0(%a: f32, %b: f32, %c: f32):
          %product = arith.mulf %a, %b : f32
          %result = arith.addf %product, %c : f32
          linalg.yield %result : f32
      } -> tensor<?x?xf32>
      return %result : tensor<?x?xf32>
    }

} 
//RUN: polygeist-opt --linalg-to-kernel="kernel-library-path=/home/arjaiswal/Polygeist/generic_solver/kernel_library_simple.mlir" -allow-unregistered-dialect generic_solver/example.mlir
// Example MLIR module demonstrating kernel operations and their linalg.generic representations
module {
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
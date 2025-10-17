// RUN: polygeist-opt --remove-iter-args --split-input-file %s | FileCheck %s

// ============================================================================
// AFFINE.FOR TEST CASES
// ============================================================================

// Test case 1: Simple direct store (should work with original implementation)
// CHECK-LABEL: func.func @test_direct_store
// CHECK-NOT: iter_args
// CHECK: affine.for
// CHECK: %[[LOADED:.*]] = affine.load %{{.*}}[] : memref<f64>
// CHECK: %[[VAL:.*]] = affine.load
// CHECK: %[[SUM:.*]] = arith.addf %[[LOADED]], %[[VAL]]
// CHECK: affine.store %[[SUM]], %{{.*}}[] : memref<f64>
// CHECK-NOT: affine.yield {{.*}} : f64
func.func @test_direct_store(%A: memref<?xf64>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0.0 : f64
  %result_mem = memref.alloc() : memref<f64>
  
  %sum = affine.for %i = %c0 to %n iter_args(%acc = %init) -> (f64) {
    %val = affine.load %A[%i] : memref<?xf64>
    %new_acc = arith.addf %acc, %val : f64
    affine.yield %new_acc : f64
  }
  affine.store %sum, %result_mem[] : memref<f64>
  
  return
}

// -----

// Test case 2: Multiply after reduction (distributivity)
// Pattern: result = alpha * sum → sum = acc + (alpha * value)
// CHECK-LABEL: func.func @test_multiply_after_add
// CHECK-NOT: iter_args
// CHECK: affine.for
// CHECK: %[[LOADED:.*]] = affine.load %{{.*}}[] : memref<f64>
// CHECK: %[[VAL:.*]] = affine.load %{{.*}}[%{{.*}}]
// CHECK: %[[PROD:.*]] = arith.mulf %{{.*}}, %[[VAL]]
// CHECK: %[[SUM:.*]] = arith.addf %[[LOADED]], %[[PROD]]
// CHECK: affine.store %[[SUM]], %{{.*}}[] : memref<f64>
func.func @test_multiply_after_add(%A: memref<?xf64>, %n: index, %alpha: f64) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0.0 : f64
  %result_mem = memref.alloc() : memref<f64>
  
  %sum = affine.for %i = %c0 to %n iter_args(%acc = %init) -> (f64) {
    %val = affine.load %A[%i] : memref<?xf64>
    %new_acc = arith.addf %acc, %val : f64
    affine.yield %new_acc : f64
  }
  %scaled = arith.mulf %alpha, %sum : f64
  affine.store %scaled, %result_mem[] : memref<f64>
  
  return
}

// -----

// Test case 3: Addition with loop-invariant load (init adjustment)
// Pattern: result = C + sum → init = C, then direct store
// CHECK-LABEL: func.func @test_add_with_invariant_load
// CHECK-NOT: iter_args
// CHECK: affine.for
// CHECK: %[[LOADED:.*]] = affine.load %{{.*}}[] : memref<f64>
// CHECK: %[[VAL:.*]] = affine.load %{{.*}}[%{{.*}}]
// CHECK: %[[SUM:.*]] = arith.addf %[[LOADED]], %[[VAL]]
// CHECK: affine.store %[[SUM]], %{{.*}}[] : memref<f64>
// CHECK-NOT: affine.load %{{.*}}[] : memref<f64>
func.func @test_add_with_invariant_load(%A: memref<?xf64>, %C: memref<f64>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0.0 : f64
  
  %sum = affine.for %i = %c0 to %n iter_args(%acc = %init) -> (f64) {
    %val = affine.load %A[%i] : memref<?xf64>
    %new_acc = arith.addf %acc, %val : f64
    affine.yield %new_acc : f64
  }
  %old_c = affine.load %C[] : memref<f64>
  %new_c = arith.addf %old_c, %sum : f64
  affine.store %new_c, %C[] : memref<f64>
  
  return
}

// -----

// Test case 4: Full GEMM pattern (multiply + add with load)
// Pattern: C = C + alpha * sum (most complex case)
// CHECK-LABEL: func.func @test_gemm_pattern
// CHECK-NOT: iter_args
// CHECK: affine.for
// CHECK: %[[LOADED:.*]] = affine.load %{{.*}}[] : memref<f64>
// CHECK: %[[VAL:.*]] = affine.load %{{.*}}[%{{.*}}]
// CHECK: %[[PROD:.*]] = arith.mulf %{{.*}}, %[[VAL]]
// CHECK: %[[SUM:.*]] = arith.addf %[[LOADED]], %[[PROD]]
// CHECK: affine.store %[[SUM]], %{{.*}}[] : memref<f64>
func.func @test_gemm_pattern(%A: memref<?xf64>, %C: memref<f64>, %n: index, %alpha: f64) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0.0 : f64
  
  %sum = affine.for %i = %c0 to %n iter_args(%acc = %init) -> (f64) {
    %val = affine.load %A[%i] : memref<?xf64>
    %new_acc = arith.addf %acc, %val : f64
    affine.yield %new_acc : f64
  }
  
  %scaled = arith.mulf %alpha, %sum : f64
  %old_c = affine.load %C[] : memref<f64>
  %new_c = arith.addf %old_c, %scaled : f64
  affine.store %new_c, %C[] : memref<f64>
  
  return
}

// -----

// Test case 5: Realistic GEMM inner loop
// C[i,j] += alpha * sum_k(A[i,k] * B[k,j])
// CHECK-LABEL: func.func @test_gemm_inner_loop
// CHECK-NOT: iter_args
// CHECK: affine.for
// CHECK: %[[C_LOADED:.*]] = affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf64>
// CHECK: %[[A_VAL:.*]] = affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf64>
// CHECK: %[[B_VAL:.*]] = affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf64>
// CHECK: %[[PROD1:.*]] = arith.mulf %[[A_VAL]], %[[B_VAL]]
// CHECK: %[[PROD2:.*]] = arith.mulf %{{.*}}, %[[PROD1]]
// CHECK: %[[SUM:.*]] = arith.addf %[[C_LOADED]], %[[PROD2]]
// CHECK: affine.store %[[SUM]], %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf64>
func.func @test_gemm_inner_loop(
    %A: memref<?x?xf64>, %B: memref<?x?xf64>, %C: memref<?x?xf64>,
    %i: index, %j: index, %K: index, %lda: index, %ldb: index, %ldc: index,
    %alpha: f64) {
  %c0 = arith.constant 0 : index
  %init = arith.constant 0.0 : f64
  
  %dot_product = affine.for %k = %c0 to %K iter_args(%acc = %init) -> (f64) {
    %a_ik = affine.load %A[%i, %k] : memref<?x?xf64>
    %b_kj = affine.load %B[%k, %j] : memref<?x?xf64>
    %prod = arith.mulf %a_ik, %b_kj : f64
    %new_acc = arith.addf %acc, %prod : f64
    affine.yield %new_acc : f64
  }
  
  %scaled = arith.mulf %alpha, %dot_product : f64
  %old_c = affine.load %C[%i, %j] : memref<?x?xf64>
  %new_c = arith.addf %old_c, %scaled : f64
  affine.store %new_c, %C[%i, %j] : memref<?x?xf64>
  
  return
}

// -----

// Test case 6: Multiply after multiply reduction (should NOT transform)
// This requires different algebraic properties (not addition)
// CHECK-LABEL: func.func @test_multiply_after_multiply
// CHECK: iter_args
// CHECK: arith.mulf %{{.*}}, %{{.*}} : f64
// CHECK: affine.yield
func.func @test_multiply_after_multiply(%A: memref<?xf64>, %n: index, %alpha: f64) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 1.0 : f64
  %result_mem = memref.alloc() : memref<f64>
  
  %product = affine.for %i = %c0 to %n iter_args(%acc = %init) -> (f64) {
    %val = affine.load %A[%i] : memref<?xf64>
    %new_acc = arith.mulf %acc, %val : f64
    affine.yield %new_acc : f64
  }
  %scaled = arith.mulf %alpha, %product : f64
  affine.store %scaled, %result_mem[] : memref<f64>
  
  return
}

// -----

// Test case 7: Multiple uses of result (should NOT transform)
// CHECK-LABEL: func.func @test_multiple_uses
// CHECK: iter_args
// CHECK: affine.yield
// CHECK: affine.store %{{.*}}, %{{.*}}[] : memref<f64>
// CHECK: affine.store %{{.*}}, %{{.*}}[] : memref<f64>
func.func @test_multiple_uses(%A: memref<?xf64>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0.0 : f64
  %result1 = memref.alloc() : memref<f64>
  %result2 = memref.alloc() : memref<f64>
  
  %sum = affine.for %i = %c0 to %n iter_args(%acc = %init) -> (f64) {
    %val = affine.load %A[%i] : memref<?xf64>
    %new_acc = arith.addf %acc, %val : f64
    affine.yield %new_acc : f64
  }
  
  affine.store %sum, %result1[] : memref<f64>
  affine.store %sum, %result2[] : memref<f64>
  
  return
}

// -----

// ============================================================================
// INTEGER TESTS (AFFINE)
// ============================================================================

// Test case 8: Integer addition - direct store
// CHECK-LABEL: func.func @test_integer_direct_store
// CHECK-NOT: iter_args
// CHECK: affine.for
// CHECK: arith.addi
// CHECK: affine.store
func.func @test_integer_direct_store(%A: memref<?xi32>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0 : i32
  %result_mem = memref.alloc() : memref<i32>
  
  %sum = affine.for %i = %c0 to %n iter_args(%acc = %init) -> (i32) {
    %val = affine.load %A[%i] : memref<?xi32>
    %new_acc = arith.addi %acc, %val : i32
    affine.yield %new_acc : i32
  }
  affine.store %sum, %result_mem[] : memref<i32>
  
  return
}

// -----

// Test case 9: Integer multiply after reduction
// CHECK-LABEL: func.func @test_integer_multiply_after_add
// CHECK-NOT: iter_args
// CHECK: affine.for
// CHECK: arith.muli
// CHECK: arith.addi
// CHECK: affine.store
func.func @test_integer_multiply_after_add(%A: memref<?xi32>, %n: index, %alpha: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0 : i32
  %result_mem = memref.alloc() : memref<i32>
  
  %sum = affine.for %i = %c0 to %n iter_args(%acc = %init) -> (i32) {
    %val = affine.load %A[%i] : memref<?xi32>
    %new_acc = arith.addi %acc, %val : i32
    affine.yield %new_acc : i32
  }
  %scaled = arith.muli %alpha, %sum : i32
  affine.store %scaled, %result_mem[] : memref<i32>
  
  return
}

// -----

// Test case 10: Integer addition with loop-invariant load
// CHECK-LABEL: func.func @test_integer_add_with_invariant_load
// CHECK-NOT: iter_args
// CHECK: affine.for
// CHECK: arith.addi
// CHECK: affine.store
func.func @test_integer_add_with_invariant_load(%A: memref<?xi32>, %C: memref<i32>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0 : i32
  
  %sum = affine.for %i = %c0 to %n iter_args(%acc = %init) -> (i32) {
    %val = affine.load %A[%i] : memref<?xi32>
    %new_acc = arith.addi %acc, %val : i32
    affine.yield %new_acc : i32
  }
  %old_c = affine.load %C[] : memref<i32>
  %new_c = arith.addi %old_c, %sum : i32
  affine.store %new_c, %C[] : memref<i32>
  
  return
}

// -----

// Test case 11: Full integer GEMM-like pattern
// CHECK-LABEL: func.func @test_integer_gemm_pattern
// CHECK-NOT: iter_args
// CHECK: affine.for
// CHECK: arith.muli
// CHECK: arith.addi
// CHECK: affine.store
func.func @test_integer_gemm_pattern(%A: memref<?xi32>, %C: memref<i32>, %n: index, %alpha: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0 : i32
  
  %sum = affine.for %i = %c0 to %n iter_args(%acc = %init) -> (i32) {
    %val = affine.load %A[%i] : memref<?xi32>
    %new_acc = arith.addi %acc, %val : i32
    affine.yield %new_acc : i32
  }
  
  %scaled = arith.muli %alpha, %sum : i32
  %old_c = affine.load %C[] : memref<i32>
  %new_c = arith.addi %old_c, %scaled : i32
  affine.store %new_c, %C[] : memref<i32>
  
  return
}

// -----

// Test case 12: Integer matrix multiply inner loop
// CHECK-LABEL: func.func @test_integer_gemm_inner_loop
// CHECK-NOT: iter_args
// CHECK: affine.for
// CHECK: affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xi32>
// CHECK: arith.muli
// CHECK: arith.muli
// CHECK: arith.addi
// CHECK: affine.store
func.func @test_integer_gemm_inner_loop(
    %A: memref<?x?xi32>, %B: memref<?x?xi32>, %C: memref<?x?xi32>,
    %i: index, %j: index, %K: index, %lda: index, %ldb: index, %ldc: index,
    %alpha: i32) {
  %c0 = arith.constant 0 : index
  %init = arith.constant 0 : i32
  
  %dot_product = affine.for %k = %c0 to %K iter_args(%acc = %init) -> (i32) {
    %a_ik = affine.load %A[%i, %k] : memref<?x?xi32>
    %b_kj = affine.load %B[%k, %j] : memref<?x?xi32>
    %prod = arith.muli %a_ik, %b_kj : i32
    %new_acc = arith.addi %acc, %prod : i32
    affine.yield %new_acc : i32
  }
  
  %scaled = arith.muli %alpha, %dot_product : i32
  %old_c = affine.load %C[%i, %j] : memref<?x?xi32>
  %new_c = arith.addi %old_c, %scaled : i32
  affine.store %new_c, %C[%i, %j] : memref<?x?xi32>
  
  return
}

// -----

// ============================================================================
// SCF.FOR TEST CASES
// ============================================================================

// Test case 13: SCF simple direct store
// CHECK-LABEL: func.func @test_scf_direct_store
// CHECK-NOT: iter_args
// CHECK: scf.for
// CHECK: memref.load %{{.*}}[] : memref<f64>
// CHECK: arith.addf
// CHECK: memref.store
func.func @test_scf_direct_store(%A: memref<?xf64>, %result: memref<f64>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0.0 : f64
  
  %sum = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> (f64) {
    %val = memref.load %A[%i] : memref<?xf64>
    %new_acc = arith.addf %acc, %val : f64
    scf.yield %new_acc : f64
  }
  memref.store %sum, %result[] : memref<f64>
  
  return
}

// -----

// Test case 14: SCF multiply after loop
// CHECK-LABEL: func.func @test_scf_multiply_after
// CHECK-NOT: iter_args
// CHECK: scf.for
// CHECK: memref.load %{{.*}}[] : memref<f64>
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: memref.store
func.func @test_scf_multiply_after(%A: memref<?xf64>, %C: memref<f64>, %n: index, %alpha: f64) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0.0 : f64
  
  %sum = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> (f64) {
    %val = memref.load %A[%i] : memref<?xf64>
    %new_acc = arith.addf %acc, %val : f64
    scf.yield %new_acc : f64
  }
  
  %scaled = arith.mulf %alpha, %sum : f64
  memref.store %scaled, %C[] : memref<f64>
  
  return
}

// -----

// Test case 15: SCF add with invariant load
// CHECK-LABEL: func.func @test_scf_add_with_load
// CHECK-NOT: iter_args
// CHECK: scf.for
// CHECK: memref.load %{{.*}}[] : memref<f64>
// CHECK: arith.addf
// CHECK: memref.store
func.func @test_scf_add_with_load(%A: memref<?xf64>, %C: memref<f64>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0.0 : f64
  
  %sum = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> (f64) {
    %val = memref.load %A[%i] : memref<?xf64>
    %new_acc = arith.addf %acc, %val : f64
    scf.yield %new_acc : f64
  }
  
  %old_c = memref.load %C[] : memref<f64>
  %new_c = arith.addf %old_c, %sum : f64
  memref.store %new_c, %C[] : memref<f64>
  
  return
}

// -----

// Test case 16: SCF full GEMM pattern
// CHECK-LABEL: func.func @test_scf_gemm_pattern
// CHECK-NOT: iter_args
// CHECK: scf.for
// CHECK: memref.load %{{.*}}[] : memref<f64>
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: memref.store
func.func @test_scf_gemm_pattern(%A: memref<?xf64>, %C: memref<f64>, %n: index, %alpha: f64) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0.0 : f64
  
  %sum = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> (f64) {
    %val = memref.load %A[%i] : memref<?xf64>
    %new_acc = arith.addf %acc, %val : f64
    scf.yield %new_acc : f64
  }
  
  %scaled = arith.mulf %alpha, %sum : f64
  %old_c = memref.load %C[] : memref<f64>
  %new_c = arith.addf %old_c, %scaled : f64
  memref.store %new_c, %C[] : memref<f64>
  
  return
}

// -----

// Test case 17: SCF integer operations
// CHECK-LABEL: func.func @test_scf_integer_gemm
// CHECK-NOT: iter_args
// CHECK: scf.for
// CHECK: memref.load %{{.*}}[] : memref<i32>
// CHECK: arith.muli
// CHECK: arith.addi
// CHECK: memref.store
func.func @test_scf_integer_gemm(%A: memref<?xi32>, %C: memref<i32>, %n: index, %alpha: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0 : i32
  
  %sum = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> (i32) {
    %val = memref.load %A[%i] : memref<?xi32>
    %new_acc = arith.addi %acc, %val : i32
    scf.yield %new_acc : i32
  }
  
  %scaled = arith.muli %alpha, %sum : i32
  %old_c = memref.load %C[] : memref<i32>
  %new_c = arith.addi %old_c, %scaled : i32
  memref.store %new_c, %C[] : memref<i32>
  
  return
}


//===- KernelOps.td - Kernel dialect operation definitions -*-- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef KERNEL_OPS
#define KERNEL_OPS

include "polygeist/Kernel/KernelDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

//===----------------------------------------------------------------------===//
// Kernel operation definitions
//===----------------------------------------------------------------------===//

def Kernel_DefnCollectionOp : Kernel_Op<"defn_collection", []> {
  let summary = "Collection of kernel operation definitions";
  let description = [{
    A collection of operation definitions that can be referenced elsewhere.
    This operation serves as a container for multiple kernel operation definitions,
    enabling modular organization of kernel implementations.
  }];
  
  let regions = (region SizedRegion<1>:$defns);
  
  let assemblyFormat = [{
    $defns attr-dict
  }];
}

def Kernel_DefnOp : Kernel_Op<"defn", [SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = "Definition of a kernel operation";
  let description = [{
    A definition of a kernel operation with inputs and arbitrary body code.
    Can contain either literal CUDA/HIP code or a linalg.generic representation
    for high-performance linear algebra operations.
    
    This operation is particularly useful for defining custom GEMM variants,
    batched operations, and other specialized linear algebra kernels.
    
    Example:
    ```mlir
    %result = kernel.defn {
    ^bb0(%A: memref<?x?xf32>, %B: memref<?x?xf32>, 
         %C: memref<?x?xf32>, %alpha: f32):
      // Kernel implementation
      kernel.yield %some_result : tensor<f32>
    } {name = "custom_gemm"} -> tensor<f32>
    ```
  }];

  // TODO: can look into gpu call op (separte namespace) 
  let arguments = (ins
    StrAttr:$name,
    TypeAttrOf<FunctionType>:$function_type
  );

  let results = (outs Variadic<AnyType>:$results);
  
  let regions = (region SizedRegion<1>:$body);
  
  let assemblyFormat = [{
    $body attr-dict `->` type($results)
  }];
  
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    TypeRange getFunctionResultTypes() {
      auto fType = getFunctionType();
      return fType.getResults();
    }
  }];
}

def Kernel_YieldOp : Kernel_Op<"yield", [Pure, Terminator, 
                                          ParentOneOf<["DefnOp"]>]> {
  let summary = "Terminator for kernel.defn operation";
  let description = [{
    The `kernel.yield` operation terminates regions within kernel operations.
    It optionally returns values from the kernel definition.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let assemblyFormat = [{
    ($operands^ `:` type($operands))? attr-dict
  }];

  let builders = [
    OpBuilder<(ins), [{
      build($_builder, $_state, std::nullopt);
    }]>
  ];

  let hasVerifier = 1;
}

#endif // KERNEL_OPS 
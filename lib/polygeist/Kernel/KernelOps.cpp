//===- KernelOps.cpp - Kernel dialect operations ----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "polygeist/Kernel/KernelOps.h"
#include "polygeist/Kernel/KernelDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::polygeist;
using namespace mlir::polygeist::kernel;

//===----------------------------------------------------------------------===//
// DefnOp
//===----------------------------------------------------------------------===//

LogicalResult DefnOp::verify() {
  // Check that the body region has exactly one block
  if (!getBody().hasOneBlock())
    return emitOpError("body region must have exactly one block");
  
  // The block can have any number of arguments
  // No special verification needed for block arguments
  
  return success();
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

LogicalResult YieldOp::verify() {
  // Get the parent DefnOp
  auto defnOp = getParentOp();
  if (!defnOp)
    return emitOpError("must be nested within a kernel.defn operation");
  
  // Get expected result types from the DefnOp's function type
  auto functionType = defnOp.getFunctionType();
  auto expectedTypes = functionType.getResults();
  
  // Check that the number of operands matches expected results
  if (getOperands().size() != expectedTypes.size()) {
    return emitOpError("number of yielded values (")
           << getOperands().size() << ") does not match expected number of results ("
           << expectedTypes.size() << ")";
  }
  
  // Check that operand types match expected types
  for (auto [idx, operand, expectedType] : 
       llvm::enumerate(getOperands(), expectedTypes)) {
    if (operand.getType() != expectedType) {
      return emitOpError("yielded value ") << idx << " has type " 
             << operand.getType() << " but expected " << expectedType;
    }
  }
  
  return success();
}

//===----------------------------------------------------------------------===//
// TableGen'd op definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "polygeist/Kernel/KernelOps.cpp.inc" 
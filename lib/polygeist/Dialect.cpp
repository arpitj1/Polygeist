//===- PolygeistDialect.cpp - Polygeist dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "polygeist/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "polygeist/Ops.h"

using namespace mlir;
using namespace mlir::polygeist;

namespace {
struct PolygeistInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Operation *op, Region *, bool,
                       IRMapping &) const final {
    // These value/view operations have no function-local or GPU execution
    // context. Preserve them when the standard inliner expands C++ helpers.
    // Execution regions, barriers and device-specific operations stay subject
    // to their existing lowering rules.
    return isa<SubIndexOp, Memref2PointerOp, Pointer2MemrefOp, MemrefCastOp,
               SubmapOp, SubmapInverseOp, TypeSizeOp, TypeAlignOp, GetFuncOp,
               TrivialUseOp, NoopOp, UndefOp>(op);
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Polygeist dialect.
//===----------------------------------------------------------------------===//

void PolygeistDialect::initialize() {
  addInterfaces<PolygeistInlinerInterface>();
  addOperations<
#define GET_OP_LIST
#include "polygeist/PolygeistOps.cpp.inc"
      >();
}

#include "polygeist/PolygeistOpsDialect.cpp.inc"

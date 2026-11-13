//===- LowerSupportedMathToLibm.cpp
//-----------------------------------------===//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/MathToLibm/MathToLibm.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "polygeist/Passes/Passes.h"

using namespace mlir;

namespace {
struct LowerSupportedMathToLibmPass
    : PassWrapper<LowerSupportedMathToLibmPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerSupportedMathToLibmPass)

  StringRef getArgument() const final { return "lower-supported-math-to-libm"; }
  StringRef getDescription() const final {
    return "Apply upstream libm lowering before approximate LLVM expansions";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect, LLVM::LLVMDialect,
                    math::MathDialect, vector::VectorDialect>();
  }
  void runOnOperation() override {
    // The upstream conversion pass marks the entire math dialect illegal,
    // although it only supports a subset of operations. Apply its patterns
    // greedily so supported functions (notably expm1/log1p) retain their libm
    // semantics, leaving the remaining operations to MathToLLVM.
    RewritePatternSet patterns(&getContext());
    populateMathToLibmConversionPatterns(patterns);
    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::polygeist::createLowerSupportedMathToLibmPass() {
  return std::make_unique<LowerSupportedMathToLibmPass>();
}

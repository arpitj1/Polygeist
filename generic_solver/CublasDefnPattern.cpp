//===- KernelDefnPattern.cpp - Pattern to match linalg.generic with kernel.defn ------===//
//
// This file implements a pattern to rewrite linalg.generic operations to kernel
// operations by matching against patterns defined in kernel.defn_collection.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/TypeSwitch.h"
#include "KernelOps.h"

using namespace mlir;
using namespace mlir::linalg;

namespace {

// Helper function to check if two regions are structurally equivalent
bool areRegionsEquivalent(Region &first, Region &second) {
  // Compare number of blocks
  if (first.getBlocks().size() != second.getBlocks().size())
    return false;

  // Compare corresponding blocks
  for (auto blockPair : llvm::zip(first.getBlocks(), second.getBlocks())) {
    Block &firstBlock = std::get<0>(blockPair);
    Block &secondBlock = std::get<1>(blockPair);

    // Compare number of arguments
    if (firstBlock.getNumArguments() != secondBlock.getNumArguments())
      return false;

    // Compare argument types
    for (auto argPair : llvm::zip(firstBlock.getArguments(), 
                                  secondBlock.getArguments())) {
      if (std::get<0>(argPair).getType() != std::get<1>(argPair).getType())
        return false;
    }

    // Compare operations (simplified - real implementation would be more complex)
    if (firstBlock.getOperations().size() != secondBlock.getOperations().size())
      return false;

    // For a full implementation, you'd need more sophisticated operation comparison
    // based on operands, attributes, and result types
  }

  return true;
}

// Helper to check if indexing maps are equivalent
bool areIndexingMapsEquivalent(ArrayAttr firstMaps, ArrayAttr secondMaps) {
  if (firstMaps.size() != secondMaps.size())
    return false;

  for (auto mapPair : llvm::zip(firstMaps, secondMaps)) {
    auto firstMap = std::get<0>(mapPair).cast<AffineMapAttr>().getValue();
    auto secondMap = std::get<1>(mapPair).cast<AffineMapAttr>().getValue();
    
    if (firstMap != secondMap)
      return false;
  }

  return true;
}

// Helper to check if iterator types are equivalent
bool areIteratorTypesEquivalent(ArrayAttr firstTypes, ArrayAttr secondTypes) {
  if (firstTypes.size() != secondTypes.size())
    return false;

  for (auto typePair : llvm::zip(firstTypes, secondTypes)) {
    auto firstType = std::get<0>(typePair).cast<StringAttr>().getValue();
    auto secondType = std::get<1>(typePair).cast<StringAttr>().getValue();
    
    if (firstType != secondType)
      return false;
  }

  return true;
}

// Check if a linalg.generic operation matches a kernel.defn in a collection
FailureOr<std::string> matchGenericWithDefn(
    GenericOp genericOp, 
    kernel::DefnCollectionOp collectionOp) {
  
  // Get attributes from the generic operation
  ArrayAttr indexingMaps = genericOp.getIndexingMapsAttr();
  ArrayAttr iteratorTypes = genericOp.getIteratorTypesAttr();
  unsigned numInputs = genericOp.getNumDpsInputs();
  unsigned numOutputs = genericOp.getNumDpsInits();
  
  // Walk through each defn in the collection
  for (Operation &op : collectionOp.getDefns()) {
    auto defnOp = cast<kernel::DefnOp>(op);
    StringAttr opName = defnOp.getNameAttr();
    
    // Check for linalg.generic in the defn's body
    bool foundMatch = false;
    defnOp.getBody().walk([&](GenericOp candidateOp) {
      // Skip if already found a match
      if (foundMatch)
        return;
      
      // Check if this linalg.generic matches our target
      if (candidateOp.getNumDpsInputs() == numInputs &&
          candidateOp.getNumDpsInits() == numOutputs &&
          //TODO: Generalize to a single dialect, with no special ops
          //TODO: Indexing maps and orders might differ
          //TODO: More complex case- where extra loops exists around the ops we have
          //TODO: Custom cost model ?
          //TODO: Constants might require special handling such as bounds
          //IDEA: Descheduling / removing tiles
          areIndexingMapsEquivalent(candidateOp.getIndexingMapsAttr(), indexingMaps) &&
          areIteratorTypesEquivalent(candidateOp.getIteratorTypesAttr(), iteratorTypes) &&
          areRegionsEquivalent(candidateOp.getRegion(), genericOp.getRegion())) {
        foundMatch = true;
      }
    });
    
    if (foundMatch)
      return opName.str();
  }
  
  return failure();
}

// Rewrite pattern to convert linalg.generic to kernel ops
class LinalgGenericToKernelPattern : public OpRewritePattern<GenericOp> {
public:
  LinalgGenericToKernelPattern(MLIRContext *context, 
                              kernel::DefnCollectionOp collectionOp)
      : OpRewritePattern<GenericOp>(context), collectionOp(collectionOp) {}

  LogicalResult matchAndRewrite(GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    // Try to match with a defn in the collection
    auto matchResult = matchGenericWithDefn(genericOp, collectionOp);
    if (failed(matchResult))
      return failure();
    
    std::string opName = *matchResult;
    
    // Create the appropriate kernel operation based on the matched pattern
    if (opName == "Kernel_gemm") {
      // Get inputs and outputs
      Value outputTensor = genericOp.getDpsInitOperand(0)->get();
      Value inputA = genericOp.getDpsInputOperand(0)->get();
      Value inputB = genericOp.getDpsInputOperand(1)->get();
      
      // Default alpha and beta values (could be extracted from pattern)
      FloatAttr alpha = rewriter.getF32FloatAttr(1.0);
      FloatAttr beta = rewriter.getF32FloatAttr(0.0);
      
      // Create the kernel.gemm operation
      rewriter.replaceOpWithNewOp<kernel::GemmOp>(
          genericOp, genericOp.getResultTypes(), 
          outputTensor, inputA, inputB, alpha, beta);
      
      return success();
    } 
    else if (opName == "Kernel_batched_gemm") {
      // Get inputs and outputs
      Value outputTensor = genericOp.getDpsInitOperand(0)->get();
      Value inputA = genericOp.getDpsInputOperand(0)->get();
      Value inputB = genericOp.getDpsInputOperand(1)->get();
      
      // Default alpha and beta values
      FloatAttr alpha = rewriter.getF32FloatAttr(1.0);
      FloatAttr beta = rewriter.getF32FloatAttr(0.0);
      
      // Create the kernel.batched_gemm operation
      rewriter.replaceOpWithNewOp<kernel::BatchedGemmOp>(
          genericOp, genericOp.getResultTypes(), 
          outputTensor, inputA, inputB, alpha, beta);
      
      return success();
    } 
    else if (opName == "Kernel_iamax") {
      // Get input
      Value input = genericOp.getDpsInputOperand(0)->get();
      
      // Create the kernel.iamax operation
      rewriter.replaceOpWithNewOp<kernel::IndexMaxAbsOp>(
          genericOp, genericOp.getResultTypes(), input);
      
      return success();
    }
    else if (opName == "Kernel_iamin") {
      // Get input
      Value input = genericOp.getDpsInputOperand(0)->get();
      
      // Create the kernel.iamin operation
      rewriter.replaceOpWithNewOp<kernel::IndexMinAbsOp>(
          genericOp, genericOp.getResultTypes(), input);
      
      return success();
    }
    else if (opName == "Kernel_asum") {
      // Get input
      Value input = genericOp.getDpsInputOperand(0)->get();
      
      // Create the kernel.asum operation
      rewriter.replaceOpWithNewOp<kernel::AbsSumOp>(
          genericOp, genericOp.getResultTypes(), input);
      
      return success();
    }
    
    return failure();
  }

private:
  kernel::DefnCollectionOp collectionOp;
};

// Pass to apply the rewrite pattern
class LinalgToKernelPass
    : public PassWrapper<LinalgToKernelPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgToKernelPass)

  void runOnOperation() override {
    ModuleOp module = getOperation();
    
    // Find the kernel.defn_collection in the module
    kernel::DefnCollectionOp collectionOp;
    module.walk([&](kernel::DefnCollectionOp op) {
      collectionOp = op;
      return WalkResult::interrupt();
    });
    
    if (!collectionOp) {
      module.emitError("No kernel.defn_collection found in module");
      return signalPassFailure();
    }
    
    // Apply the rewrite pattern
    RewritePatternSet patterns(&getContext());
    patterns.add<LinalgGenericToKernelPattern>(&getContext(), collectionOp);
    
    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

// Create a pass to convert linalg.generic to kernel
std::unique_ptr<Pass> createLinalgToKernelPass() {
  return std::make_unique<LinalgToKernelPass>();
}

// Register the pass
void registerLinalgToKernelPasses() {
  PassRegistration<LinalgToKernelPass>("linalg-to-kernel",
                                      "Convert linalg.generic to kernel operations");
} 
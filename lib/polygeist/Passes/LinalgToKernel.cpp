//===- LinalgToKernel.cpp - Pattern to match linalg.generic with kernel.defn ------===//
//
// This file implements a pattern to rewrite linalg.generic operations to kernel
// operations by matching against patterns defined in kernel.defn_collection.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "polygeist/Kernel/KernelDialect.h"
#include "polygeist/Kernel/KernelOps.h"
#include "polygeist/Passes/Passes.h"

#include <stack>
#include <set>

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::polygeist;
using namespace mlir::polygeist::kernel;

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
    auto firstType = std::get<0>(typePair).cast<linalg::IteratorTypeAttr>().getValue();
    auto secondType = std::get<1>(typePair).cast<linalg::IteratorTypeAttr>().getValue();
    
    if (firstType != secondType)
      return false;
  }

  return true;
}

// Check if a linalg.generic operation matches a kernel.defn in a collection
FailureOr<StringRef> matchGenericWithDefn(
    GenericOp genericOp, 
    kernel::DefnCollectionOp collectionOp) {
  
  // Get attributes from the generic operation
  ArrayAttr indexingMaps = genericOp.getIndexingMapsAttr();
  ArrayAttr iteratorTypes = genericOp.getIteratorTypesAttr();
  unsigned numInputs = genericOp.getNumDpsInputs();
  unsigned numOutputs = genericOp.getNumDpsInits();
  
  // Variables to capture the match result
  StringRef matchedOpName;
  
  SmallVector<kernel::DefnOp> defnOps;

  collectionOp.walk([&](kernel::DefnOp defnOp) {
      defnOps.push_back(defnOp);
  });

  bool foundMatch = false;
  
  // Walk through each defn in the collection
  for (auto defnOp : defnOps) {
    
    StringRef opName = defnOp.getSymName();
    // Check for linalg.generic in the defn's body
    GenericOp candidateOp;

    defnOp.walk([&](GenericOp genericOp) {
        candidateOp = genericOp; //TODO: Add checks to make sure there is only single linalg.generic in the defn
    });

    if(!candidateOp) {
      continue;
    }
    
    // Check if this linalg.generic matches our target
    if (candidateOp.getNumDpsInputs() == numInputs &&
        candidateOp.getNumDpsInits() == numOutputs &&
        areIndexingMapsEquivalent(candidateOp.getIndexingMapsAttr(), indexingMaps) &&
        areIteratorTypesEquivalent(candidateOp.getIteratorTypesAttr(), iteratorTypes) &&
        areRegionsEquivalent(candidateOp.getRegion(), genericOp.getRegion())) {
      foundMatch = true;
      matchedOpName = opName;
    }
    
    if (foundMatch) {
      return matchedOpName;
    }
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
    
    auto module = genericOp->getParentOfType<ModuleOp>();
    //Check if the parent of the generic op is a kernel.defn
    if (auto parentOp = genericOp->getParentOp()) {
      if (isa<kernel::DefnOp>(parentOp)) {
        return failure();
      }
    }
    
    // Try to match with a defn in the collection
    auto matchResult = matchGenericWithDefn(genericOp, collectionOp);
    if (failed(matchResult))
      return failure();
    
    StringRef opName = *matchResult;
    
    // Find the matched kernel.defn operation
    kernel::DefnOp matchedDefnOp;
    // Use const_cast to work around the const issue
    const_cast<kernel::DefnCollectionOp&>(collectionOp).walk([&](kernel::DefnOp defnOp) {
      if (defnOp.getSymName() == opName) {
        matchedDefnOp = defnOp;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    
    if (!matchedDefnOp) {
      return failure();
    }
    
    // Check if the kernel.defn already exists in the target module
    kernel::DefnOp existingDefn;
    module.walk([&](kernel::DefnOp defnOp) {
      if (defnOp.getSymName() == opName) {
        // Check if this defn is inside a defn_collection (template) or at module level (callable)
        if (!defnOp->getParentOfType<kernel::DefnCollectionOp>()) {
          existingDefn = defnOp;
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    
    // If the kernel.defn doesn't exist in the module, copy it
    if (!existingDefn) {
      // Clone the matched kernel.defn operation
      rewriter.setInsertionPointToStart(module.getBody());
      auto clonedDefn = rewriter.clone(*matchedDefnOp.getOperation());
      (void)clonedDefn; // Suppress unused variable warning
    }
    
    // Create kernel.launch operation to replace the genericOp
    Location loc = genericOp.getLoc();
    
    // Set insertion point to the genericOp location
    rewriter.setInsertionPoint(genericOp);
    
    // Get operands from the generic operation (inputs and outputs)
    SmallVector<Value> operands;
    operands.append(genericOp.getInputs().begin(), genericOp.getInputs().end());
    operands.append(genericOp.getOutputs().begin(), genericOp.getOutputs().end());
    
    // Get result types from the generic operation
    TypeRange resultTypes = genericOp.getResultTypes();
    
    // Create the kernel.launch operation
    auto launchOp = rewriter.create<kernel::LaunchOp>(
        loc, 
        resultTypes,
        opName,
        operands
    );
    
    // Replace the generic operation with the launch operation
    rewriter.replaceOp(genericOp, launchOp.getResults());
    
    return success();
  }

private:
  kernel::DefnCollectionOp collectionOp;
};

// Pass to apply the rewrite pattern
struct LinalgToKernelPass : public LinalgToKernelBase<LinalgToKernelPass> {
  using LinalgToKernelBase::LinalgToKernelBase;
  
  // Constructor that allows setting the kernel library path
  LinalgToKernelPass() = default;
  LinalgToKernelPass(const std::string& libraryPath) : externalLibraryPath(libraryPath) {}
  
  void runOnOperation() override {
    ModuleOp module = getOperation();
    
    kernel::DefnCollectionOp collectionOp;
    
    // Determine which path to use for kernel library
    std::string effectiveLibraryPath = externalLibraryPath;
    // If no external path was provided via constructor, try the command line option
    if (effectiveLibraryPath.empty()) {
      effectiveLibraryPath = std::string(kernelLibraryPath);
    }
    
    // Debug output
    llvm::errs() << "DEBUG: externalLibraryPath = '" << externalLibraryPath << "'\n";
    llvm::errs() << "DEBUG: kernelLibraryPath = '" << std::string(kernelLibraryPath) << "'\n";
    llvm::errs() << "DEBUG: effectiveLibraryPath = '" << effectiveLibraryPath << "'\n";
    
    // Check if we should load kernel definitions from an external file
    if (!effectiveLibraryPath.empty()) {
      //llvm::errs() << "DEBUG: Loading kernel definitions from external file: " << effectiveLibraryPath << "\n";
      // Load kernel definitions from external file
      std::string errorMessage;
      auto memoryBuffer = mlir::openInputFile(effectiveLibraryPath, &errorMessage);
      if (!memoryBuffer) {
        module.emitError("Failed to open kernel library file: ") << effectiveLibraryPath 
                         << " - " << errorMessage;
        return signalPassFailure();
      }
      
      // Parse the external file
      llvm::SourceMgr sourceMgr;
      sourceMgr.AddNewSourceBuffer(std::move(memoryBuffer), llvm::SMLoc());
      
      auto externalModule = mlir::parseSourceFile<ModuleOp>(sourceMgr, &getContext());
      if (!externalModule) {
        module.emitError("Failed to parse kernel library file: ") << effectiveLibraryPath;
        return signalPassFailure();
      }
      
      // Debug: Print the loaded external module
      //llvm::errs() << "DEBUG: Successfully loaded external module:\n";
      //externalModule->print(llvm::errs());
      //llvm::errs() << "\n";
      
      // Find the kernel.defn_collection in the external module
      externalModule->walk([&](kernel::DefnCollectionOp op) {
        collectionOp = op;
        llvm::errs() << "DEBUG: Found kernel.defn_collection in external module\n";
        return WalkResult::interrupt();
      });
      
      if (!collectionOp) {
        module.emitError("No kernel.defn_collection found in external kernel library: ") 
                         << effectiveLibraryPath;
        return signalPassFailure();
      }
      
      // Debug: Print the found collection
      //llvm::errs() << "DEBUG: kernel.defn_collection contents:\n";
      //collectionOp.print(llvm::errs());
      //llvm::errs() << "\n";
    } else {
      // Find the kernel.defn_collection in the current module (original behavior)
      module.walk([&](kernel::DefnCollectionOp op) {
        collectionOp = op;
        return WalkResult::interrupt();
      });
      
      if (!collectionOp) {
        module.emitError("No kernel.defn_collection found in module. "
                         "Either include one in the input module or specify "
                         "--kernel-library-path to load from external file.");
        return signalPassFailure();
      }
    }
    
    // Apply the rewrite pattern
    RewritePatternSet patterns(&getContext());
    patterns.add<LinalgGenericToKernelPattern>(&getContext(), collectionOp);
    
    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
      return signalPassFailure();
  }

private:
  std::string externalLibraryPath;
};

} // namespace

namespace mlir::polygeist {

// Create a pass to convert linalg.generic to kernel
std::unique_ptr<Pass> createLinalgToKernelPass() {
  return std::make_unique<LinalgToKernelPass>();
}

// Create a pass to convert linalg.generic to kernel with kernel library path
std::unique_ptr<Pass> createLinalgToKernelPass(const std::string& kernelLibraryPath) {
  return std::make_unique<LinalgToKernelPass>(kernelLibraryPath);
}

} // namespace mlir::polygeist 
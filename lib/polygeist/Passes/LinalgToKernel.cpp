//===- LinalgToKernel.cpp - Pattern to match linalg.generic with kernel.defn ------===//
//
// This file implements a pattern to rewrite linalg.generic operations to kernel
// operations by matching against patterns defined in kernel.defn_collection.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "polygeist/Kernel/KernelDialect.h"
#include "polygeist/Kernel/KernelOps.h"
#include "polygeist/Passes/Passes.h"

#include <stack>
#include <set>
#include <functional>

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::polygeist;
using namespace mlir::polygeist::kernel;

namespace {

// Structure to represent an operation node in the dependency graph
struct OpNode {
  Operation *op;
  StringRef opName;
  SmallVector<Type> operandTypes;
  SmallVector<Type> resultTypes;
  SmallVector<OpNode*> dependencies;  // Operations this depends on
  SmallVector<OpNode*> dependents;    // Operations that depend on this
  
  OpNode(Operation *operation) : op(operation) {
    if (operation) {
      // Regular operation node
      opName = operation->getName().getStringRef();
      for (Value operand : operation->getOperands()) {
        operandTypes.push_back(operand.getType());
      }
      for (Value result : operation->getResults()) {
        resultTypes.push_back(result.getType());
      }
    } else {
      // Special node for block arguments - will be set later
      opName = "block_arg";
    }
  }
  
  // Check if two nodes are structurally equivalent (same operation type and types)
  bool isEquivalentTo(const OpNode &other) const {
    return opName == other.opName && 
           operandTypes == other.operandTypes && 
           resultTypes == other.resultTypes;
  }
};

// Structure to represent a dependency graph for a region
struct DependencyGraph {
  SmallVector<std::unique_ptr<OpNode>> nodes;
  DenseMap<Operation*, OpNode*> opToNode;
  SmallVector<OpNode*> blockArgNodes;  // Special nodes for block arguments
  
  void buildFromRegion(Region &region) {
    // Process each block in the region
    for (Block &block : region.getBlocks()) {
      
      // Create pseudo-nodes for block arguments
      for (BlockArgument arg : block.getArguments()) {
        // Block arguments are represented as special nodes
        auto argNode = std::make_unique<OpNode>(nullptr);
        argNode->resultTypes.push_back(arg.getType());
        blockArgNodes.push_back(argNode.get());
        
        // Map the block argument value to this node for dependency tracking
        // We'll use a separate map for this
        nodes.push_back(std::move(argNode));
      }
      
      // Create nodes for each operation
      for (Operation &op : block.getOperations()) {
        auto node = std::make_unique<OpNode>(&op);
        OpNode *nodePtr = node.get();
        opToNode[&op] = nodePtr;
        nodes.push_back(std::move(node));
      }
      
      // Build dependency edges
      for (Operation &op : block.getOperations()) {
        OpNode *currentNode = opToNode[&op];
        
        // For each operand, find what it depends on
        for (Value operand : op.getOperands()) {
          if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
            // Depends on a block argument
            size_t argIndex = blockArg.getArgNumber();
            if (argIndex < blockArgNodes.size()) {
              OpNode *argNode = blockArgNodes[argIndex];
              currentNode->dependencies.push_back(argNode);
              argNode->dependents.push_back(currentNode);
            }
          } else if (Operation *definingOp = operand.getDefiningOp()) {
            // Depends on another operation
            if (opToNode.count(definingOp)) {
              OpNode *depNode = opToNode[definingOp];
              currentNode->dependencies.push_back(depNode);
              depNode->dependents.push_back(currentNode);
            }
          }
        }
      }
    }
  }
  
  // Get nodes in topological order (dependencies first)
  SmallVector<OpNode*> getTopologicalOrder() const {
    SmallVector<OpNode*> result;
    DenseSet<OpNode*> visited;
    
    std::function<void(OpNode*)> dfs = [&](OpNode* node) {
      if (visited.contains(node)) return;
      visited.insert(node);
      
      // Visit all dependencies first
      for (OpNode* dep : node->dependencies) {
        dfs(dep);
      }
      
      result.push_back(node);
    };
    
    // Start DFS from all nodes
    for (const auto &node : nodes) {
      dfs(node.get());
    }
    
    return result;
  }
};

// Enhanced region equivalence check using dependency graphs
bool areRegionsEquivalent(Region &first, Region &second) {
  // Fast early checks before expensive graph construction
  
  // Check number of blocks
  if (first.getBlocks().size() != second.getBlocks().size()) {
    return false;
  }
  
  // Check each block's basic properties
  for (auto blockPair : llvm::zip(first.getBlocks(), second.getBlocks())) {
    Block &firstBlock = std::get<0>(blockPair);
    Block &secondBlock = std::get<1>(blockPair);
    
    // Check number of arguments
    if (firstBlock.getNumArguments() != secondBlock.getNumArguments()) {
      return false;
    }
    
    // Check argument types
    for (auto argPair : llvm::zip(firstBlock.getArguments(), secondBlock.getArguments())) {
      if (std::get<0>(argPair).getType() != std::get<1>(argPair).getType()) {
        return false;
      }
    }
    
    // Check number of operations
    if (firstBlock.getOperations().size() != secondBlock.getOperations().size()) {
      return false;
    }
  }
  
  // If basic checks pass, proceed with detailed graph-based analysis
  // Build dependency graphs for both regions
  DependencyGraph firstGraph, secondGraph;
  firstGraph.buildFromRegion(first);
  secondGraph.buildFromRegion(second);
  
  // Quick structural checks
  if (firstGraph.nodes.size() != secondGraph.nodes.size()) {
    return false;
  }
  
  if (firstGraph.blockArgNodes.size() != secondGraph.blockArgNodes.size()) {
    return false;
  }
  
  // Get topological orderings
  auto firstOrder = firstGraph.getTopologicalOrder();
  auto secondOrder = secondGraph.getTopologicalOrder();
  
  if (firstOrder.size() != secondOrder.size()) {
    return false;
  }
  
  // Compare nodes in topological order
  DenseMap<OpNode*, OpNode*> nodeMapping;
  
  for (size_t i = 0; i < firstOrder.size(); ++i) {
    OpNode *firstNode = firstOrder[i];
    OpNode *secondNode = secondOrder[i];
    
    // Check if the nodes are structurally equivalent
    if (!firstNode->isEquivalentTo(*secondNode)) {
      return false;
    }
    
    // Check if dependency structure matches
    if (firstNode->dependencies.size() != secondNode->dependencies.size()) {
      return false;
    }
    
    // Verify that dependencies map correctly
    for (size_t j = 0; j < firstNode->dependencies.size(); ++j) {
      OpNode *firstDep = firstNode->dependencies[j];
      OpNode *secondDep = secondNode->dependencies[j];
      
      // Check if we've established a mapping for these dependencies
      auto it = nodeMapping.find(firstDep);
      if (it != nodeMapping.end()) {
        if (it->second != secondDep) {
          return false; // Inconsistent mapping
        }
      } else {
        nodeMapping[firstDep] = secondDep;
      }
    }
    
    // Establish mapping for current nodes
    nodeMapping[firstNode] = secondNode;
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

      //llvm::errs() << "DEBUG: kernel.defn_collection contents:\n";
      //llvm::errs() << collectionOp;
      //llvm::errs() << collectionOp.getOperation();
      //llvm::errs() << "\n";
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
    
    kernel::DefnCollectionOp collectionOp = nullptr;
    OwningOpRef<ModuleOp> externalModule;
    // Determine which path to use for kernel library
    std::string effectiveLibraryPath = externalLibraryPath;
    // If no external path was provided via constructor, try the command line option
    if (effectiveLibraryPath.empty()) {
      effectiveLibraryPath = std::string(kernelLibraryPath);
    }
    
    //// Debug output
    //llvm::errs() << "DEBUG: externalLibraryPath = '" << externalLibraryPath << "'\n";
    //llvm::errs() << "DEBUG: kernelLibraryPath = '" << std::string(kernelLibraryPath) << "'\n";
    //llvm::errs() << "DEBUG: effectiveLibraryPath = '" << effectiveLibraryPath << "'\n";
    
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
      
      externalModule = mlir::parseSourceFile<ModuleOp>(sourceMgr, &getContext());
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
      //llvm::errs() << collectionOp;
      //llvm::errs() << collectionOp.getOperation();
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
      
      //llvm::errs() << "DEBUG: kernel.defn_collection contents:\n";
      //llvm::errs() << collectionOp.getOperation();
      //llvm::errs() << "\n";
      //llvm::errs() << collectionOp;
      //llvm::errs() << "\n";
    
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
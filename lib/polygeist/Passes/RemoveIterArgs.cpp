#include "PassDetails.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "polygeist/Passes/Passes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "remove-scf-iter-args"

using namespace mlir;
using namespace mlir::arith;
using namespace polygeist;
using namespace scf;
using namespace affine;

struct RemoveSCFIterArgs : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {

    LLVM_DEBUG(llvm::dbgs() << "\n=== RemoveSCFIterArgs::matchAndRewrite ===\n");
    LLVM_DEBUG(llvm::dbgs() << "Processing scf.for loop:\n" << forOp << "\n");

    ModuleOp module = forOp->getParentOfType<ModuleOp>();
    if (!forOp.getRegion().hasOneBlock()) {
      LLVM_DEBUG(llvm::dbgs() << "REJECTED: Loop doesn't have exactly one block\n");
      return failure();
    }
    unsigned numIterArgs = forOp.getNumRegionIterArgs();
    LLVM_DEBUG(llvm::dbgs() << "Number of iter_args: " << numIterArgs << "\n");
    
    auto loc = forOp->getLoc();
    bool changed = false;
    llvm::SetVector<unsigned> removed;
    llvm::MapVector<unsigned, Value> steps;
    auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    
    for (unsigned i = 0; i < numIterArgs; i++) {
      LLVM_DEBUG(llvm::dbgs() << "\n--- Processing iter_arg #" << i << " ---\n");
      auto ba = forOp.getRegionIterArgs()[i];
      auto init = forOp.getInits()[i];
      auto lastOp = yieldOp->getOperand(i);
      LLVM_DEBUG(llvm::dbgs() << "  iter_arg type: " << ba.getType() << "\n");

      // General Case(TODO):
      // ALGo:
      //  1. Create an alloca(stack) variable
      //     How to know it's dims? It should be based on number of reduction
      //     loops
      //  2. Initialize it with init value just outside the for loop if init
      //  value is non-zero
      //  3. memref.load that value in the for loop
      //  4. Replace all the uses of the iter_arg with the loaded value
      //  5. Add a memref.store for the value to be yielded
      //  6. Replace all uses of for-loops yielded value with a single inserted
      //  memref.load
      // Special case:
      // ALGo:
      // Optimize away memref.store and memref.load, if the only users of
      // memref.load are memref.store (can use affine-scalrep pass for that ? No
      // it does store to load forwarding) What we need is forwarding of local
      // store to final store and deleting the intermediate alloca created. This
      // is only possible if the user of alloca is a storeOp.
      //  1. Identify the single store of the for loop result
      //  2. Initialize it with iter arg init, outside the for loop. (TODO)
      //  3. Do a load from the memref
      //  4. move the store to memref inside the loop.

      auto result = forOp.getResult(i);
      LLVM_DEBUG(llvm::dbgs() << "  Loop result has " << std::distance(result.user_begin(), result.user_end()) << " use(s)\n");
      
      if (result.hasOneUse()) {
        LLVM_DEBUG(llvm::dbgs() << "  Result has exactly one use\n");
        auto storeOp = dyn_cast<memref::StoreOp>(*result.getUsers().begin());
        if (storeOp) {
          LLVM_DEBUG(llvm::dbgs() << "  ✓ User is memref.store - can remove iter_arg!\n");
          LLVM_DEBUG(llvm::dbgs() << "  Store operation: " << *storeOp << "\n");
          {
            rewriter.setInsertionPointToStart(forOp.getBody());
            auto memrefLoad = rewriter.create<memref::LoadOp>(
                forOp.getLoc(), storeOp.getMemref(), storeOp.getIndices());
            LLVM_DEBUG(llvm::dbgs() << "  Created memref.load at loop start: " << memrefLoad << "\n");
            rewriter.replaceAllUsesWith(ba, memrefLoad.getResult());
          }
          {
            rewriter.setInsertionPoint(yieldOp);
            auto newStore = rewriter.create<memref::StoreOp>(forOp.getLoc(), lastOp,
                                             storeOp.getMemref(),
                                             storeOp.getIndices());
            LLVM_DEBUG(llvm::dbgs() << "  Created memref.store before yield: " << newStore << "\n");
            storeOp.erase();
            LLVM_DEBUG(llvm::dbgs() << "  Erased original store outside loop\n");
          }
        } else {
          LLVM_DEBUG(llvm::dbgs() << "  ✗ User is NOT memref.store: " << **result.getUsers().begin() << "\n");
          return failure();
        }
      } else {
        LLVM_DEBUG(llvm::dbgs() << "  ✗ Result has multiple uses or no uses\n");
        for (auto user : result.getUsers()) {
          LLVM_DEBUG(llvm::dbgs() << "    User: " << *user << "\n");
        }
      }
      // else{
      //   alloca = rewriter.create<memref::AllocaOp>(
      //         forOp.getLoc(), MemRefType::get(ArrayRef<int64_t>(),
      //         forOp.getType()), ValueRange());
      //   //Skipping init for now

      //  auto memrefLoad = rewriter.create<memref::LoadOp>(
      //      forOp.getLoc(), alloca.getMemref(), op.getIndices());
      //  rewriter.replaceOp(op, memrefLoad.getResult());

      //  rewriter.create<memref::StoreOp>(forOp.getLoc(), lastOp, alloca,
      //                                   forOp.getBody()->getArguments());

      //  rewriter.replaceAllUsesWith(result,)
      //}

      rewriter.setInsertionPointToStart(forOp.getBody());
      // rewriter.replaceAllUsesWith(ba, replacementIV);
      changed = true;
    }

    if (!changed) {
      LLVM_DEBUG(llvm::dbgs() << "\nNo iter_args were transformed - REJECTED\n");
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs() << "\n✓ All iter_args successfully transformed!\n");
    LLVM_DEBUG(llvm::dbgs() << "Creating new scf.for without iter_args...\n");

    rewriter.setInsertionPoint(forOp);
    auto newForOp = rewriter.create<scf::ForOp>(
        loc, forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep());
    if (!newForOp.getRegion().empty())
      newForOp.getRegion().front().erase();
    assert(newForOp.getRegion().empty());
    rewriter.inlineRegionBefore(forOp.getRegion(), newForOp.getRegion(),
                                newForOp.getRegion().begin());

    LLVM_DEBUG(llvm::dbgs() << "Deleting " << numIterArgs << " region arguments...\n");
    // Delete region args
    llvm::BitVector toDelete(numIterArgs + 1);
    for (unsigned i = 0; i < numIterArgs; i++)
      toDelete[i + 1] = true;
    newForOp.getBody()->eraseArguments(toDelete);

    SmallVector<Value> newYields;
    {
      ValueRange empty;
      rewriter.setInsertionPoint(yieldOp);
      auto newYieldOp = rewriter.create<scf::YieldOp>(loc);
      LLVM_DEBUG(llvm::dbgs() << "Replacing yield with empty yield\n");
      // rewriter.replaceOpWithNewOp<scf::YieldOp>(yieldOp, newYieldOp);
      rewriter.eraseOp(yieldOp);
    }

    rewriter.setInsertionPoint(newForOp);
    rewriter.eraseOp(forOp);

    LLVM_DEBUG(llvm::dbgs() << "=== RemoveSCFIterArgs SUCCESS ===\n\n");
    return success();
  }
};

// General Case(TODO):
// ALGo:
//  1. Create an alloca(stack) variable
//     How to know it's dims? It should be based on number of reduction
//     loops
//  2. Initialize it with init value just outside the for loop if init
//  value is non-zero
//  3. memref.load that value in the for loop
//  4. Replace all the uses of the iter_arg with the loaded value
//  5. Add a memref.store for the value to be yielded
//  6. Replace all uses of for-loops yielded value with a single inserted
//  memref.load
// Special case:
// ALGo:
// Optimize away memref.store and memref.load, if the only users of
// memref.load are memref.store (can use affine-scalrep pass for that ? No
// it does store to load forwarding) What we need is forwarding of local
// store to final store and deleting the intermediate alloca created. This
// is only possible if the user of alloca is a storeOp.
//  1. Identify the single store of the for loop result
//  2. Initialize it with iter arg init, outside the for loop. (TODO)
//  3. Do a load from the memref
//  4. move the store to memref inside the loop.

struct RemoveAffineIterArgs : public OpRewritePattern<affine::AffineForOp> {
  using OpRewritePattern<affine::AffineForOp>::OpRewritePattern;
  
  // Helper: Check if a value is loop-invariant w.r.t. the given loop
  bool isLoopInvariant(Value val, affine::AffineForOp forOp) const {
    // Check if the value is defined outside the loop
    if (auto defOp = val.getDefiningOp()) {
      return !forOp->isAncestor(defOp);
    }
    // Block arguments from parent regions are invariant
    if (auto blockArg = dyn_cast<BlockArgument>(val)) {
      return blockArg.getOwner()->getParentOp() != forOp.getOperation();
    }
    return true;
  }
  
  LogicalResult matchAndRewrite(affine::AffineForOp forOp,
                                PatternRewriter &rewriter) const override {

    LLVM_DEBUG(llvm::dbgs() << "\n=== RemoveAffineIterArgs::matchAndRewrite ===\n");
    LLVM_DEBUG(llvm::dbgs() << "Processing affine.for loop:\n" << forOp << "\n");

    ModuleOp module = forOp->getParentOfType<ModuleOp>();
    rewriter.setInsertionPoint(forOp);
    
    unsigned numIterArgs = forOp.getNumRegionIterArgs();
    LLVM_DEBUG(llvm::dbgs() << "Number of iter_args: " << numIterArgs << "\n");
    
    if (numIterArgs == 0) {
      LLVM_DEBUG(llvm::dbgs() << "REJECTED: No iter_args to remove\n");
      return failure();
    }
   
    LLVM_DEBUG(llvm::dbgs() << "Processing last iter_arg (index " << (numIterArgs - 1) << ")\n");
    
    auto loc = forOp->getLoc();
    auto yieldOp =
        cast<affine::AffineYieldOp>(forOp.getBody()->getTerminator());

    auto ba = forOp.getRegionIterArgs()[numIterArgs - 1];
    auto init = forOp.getInits()[numIterArgs - 1];
    auto lastOp = yieldOp->getOperand(numIterArgs - 1);

    LLVM_DEBUG(llvm::dbgs() << "  iter_arg type: " << ba.getType() << "\n");
    LLVM_DEBUG(llvm::dbgs() << "  yielded value: " << lastOp << "\n");

    auto result = forOp.getResult(numIterArgs - 1);
    LLVM_DEBUG(llvm::dbgs() << "  Loop result has " << std::distance(result.user_begin(), result.user_end()) << " use(s)\n");
    
    if (!result.hasOneUse()) {
      LLVM_DEBUG(llvm::dbgs() << "  ✗ Result has multiple uses or no uses\n");
      for (auto user : result.getUsers()) {
        LLVM_DEBUG(llvm::dbgs() << "    User: " << *user << "\n");
      }
      return failure();
    }
    
    LLVM_DEBUG(llvm::dbgs() << "  Result has exactly one use\n");
    
    // Try to find a store by traversing the use chain and pulling operations into the loop
    Value currentValue = result;
    SmallVector<std::pair<Operation*, Value>, 4> opsChain; // (op, invariant_operand)
    affine::AffineStoreOp storeOp = nullptr;
    affine::AffineLoadOp initLoad = nullptr;
    
    LLVM_DEBUG(llvm::dbgs() << "  Traversing use chain to find store...\n");
    
    // Check if yield is an addition (required for distributivity transformations)
    auto yieldedAddOp = dyn_cast_or_null<arith::AddFOp>(lastOp.getDefiningOp());
    bool yieldIsAddition = (yieldedAddOp != nullptr);
    LLVM_DEBUG(llvm::dbgs() << "  Yielded operation is addition: " << (yieldIsAddition ? "YES" : "NO") << "\n");
    
    int traverseLimit = 10; // Prevent infinite loops
    while (currentValue.hasOneUse() && traverseLimit-- > 0) {
      Operation *user = *currentValue.getUsers().begin();
      LLVM_DEBUG(llvm::dbgs() << "    Checking user: " << *user << "\n");
      
      // Check if we reached a store
      if (auto store = dyn_cast<affine::AffineStoreOp>(user)) {
        storeOp = store;
        LLVM_DEBUG(llvm::dbgs() << "    ✓ Found affine.store!\n");
        break;
      }
      
      // Check if this is a multiply that can distribute over addition
      if (auto mulOp = dyn_cast<arith::MulFOp>(user)) {
        if (!yieldIsAddition) {
          LLVM_DEBUG(llvm::dbgs() << "    ✗ Cannot pull multiply: yield is not addition\n");
          return failure();
        }
        
        // Check that one operand is the loop result and the other is loop-invariant
        Value lhs = mulOp.getLhs();
        Value rhs = mulOp.getRhs();
        Value invariantOp;
        
        if (lhs == currentValue && isLoopInvariant(rhs, forOp)) {
          invariantOp = rhs;
        } else if (rhs == currentValue && isLoopInvariant(lhs, forOp)) {
          invariantOp = lhs;
        } else {
          LLVM_DEBUG(llvm::dbgs() << "    ✗ Multiply operands don't match pattern\n");
          return failure();
        }
        
        LLVM_DEBUG(llvm::dbgs() << "    ✓ Can pull multiply into loop (distributivity)\n");
        opsChain.push_back({mulOp, invariantOp});
        currentValue = mulOp.getResult();
        continue;
      }
      
      // Check if this is an addition with a loop-invariant load
      if (auto addOp = dyn_cast<arith::AddFOp>(user)) {
        if (!yieldIsAddition) {
          LLVM_DEBUG(llvm::dbgs() << "    ✗ Cannot merge addition: yield is not addition\n");
          return failure();
        }
        
        // Get the other operand (not the loop result)
        Value otherOperand = (addOp.getLhs() == currentValue) ? addOp.getRhs() : addOp.getLhs();
        
        // Check if it's a loop-invariant load
        if (auto loadOp = dyn_cast<affine::AffineLoadOp>(otherOperand.getDefiningOp())) {
          bool allInvariant = true;
          for (Value operand : loadOp.getMapOperands()) {
            if (!isLoopInvariant(operand, forOp)) {
              allInvariant = false;
              break;
            }
          }
          
          if (allInvariant) {
            LLVM_DEBUG(llvm::dbgs() << "    ✓ Found loop-invariant load, will merge into init\n");
            initLoad = loadOp;
            opsChain.push_back({addOp, otherOperand});
            currentValue = addOp.getResult();
            continue;
          }
        }
        
        LLVM_DEBUG(llvm::dbgs() << "    ✗ Addition doesn't match pattern\n");
        return failure();
      }
      
      // Unknown operation
      LLVM_DEBUG(llvm::dbgs() << "    ✗ Unknown operation type: " << user->getName() << "\n");
      return failure();
    }
    
    if (!storeOp) {
      LLVM_DEBUG(llvm::dbgs() << "  ✗ Could not find affine.store in use chain\n");
      return failure();
    }
    
    LLVM_DEBUG(llvm::dbgs() << "  ✓ Successfully traced to store!\n");
    LLVM_DEBUG(llvm::dbgs() << "  Operations in chain: " << opsChain.size() << "\n");
    
    // Now perform the transformation using IRMapping:
    // 1. Create new loop with correct signature
    // 2. Clone loop body using IRMapping
    // 3. Pull operations from outside into loop (using mapper)
    // 4. Create load/store pattern
    // 5. Fix yield and cleanup
    
    Value newInit = init;
    
    // Step 1: Adjust initialization if we have a loop-invariant load
    if (initLoad) {
      LLVM_DEBUG(llvm::dbgs() << "  Step 1: Using loop-invariant load as init\n");
      newInit = initLoad.getResult();
    }
    
    LLVM_DEBUG(llvm::dbgs() << "  Step 2: Creating new affine.for with " << (numIterArgs - 1) << " iter_args...\n");
    
    // Prepare new iter_args (drop the last one we're removing)
    SmallVector<Value> newIterArgs(forOp.getInits());
    if (!newIterArgs.empty()) {
      newIterArgs[numIterArgs - 1] = newInit; // Use the adjusted init
      newIterArgs.pop_back(); // Remove last iter_arg
    }
    
    // Create new loop with correct signature (fewer iter_args)
    auto newForOp = rewriter.create<affine::AffineForOp>(
        loc, forOp.getLowerBoundOperands(), forOp.getLowerBoundMap(),
        forOp.getUpperBoundOperands(), forOp.getUpperBoundMap(),
        forOp.getStep(), newIterArgs);

    LLVM_DEBUG(llvm::dbgs() << "  Step 3: Cloning loop body using IRMapping\n");
    
    // Create IRMapping for value remapping
    IRMapping mapper;
    
    // Map the induction variable
    mapper.map(forOp.getInductionVar(), newForOp.getInductionVar());
    
    // Map the iter_args (except the last one we're removing)
    for (unsigned i = 0; i < numIterArgs - 1; i++) {
      mapper.map(forOp.getRegionIterArgs()[i], newForOp.getRegionIterArgs()[i]);
    }
    
    // For the iter_arg we're removing (ba), we'll create a load and map it
    BlockArgument oldBa = ba;
    
    // Create load at the beginning that will replace the iter_arg
    Block *oldBody = forOp.getBody();
    Block *newBody = newForOp.getBody();
    rewriter.setInsertionPointToStart(newBody);
    
    auto memrefLoad = rewriter.create<affine::AffineLoadOp>(
        loc, storeOp.getMemref(), storeOp.getMap(),
        storeOp.getMapOperands());
    LLVM_DEBUG(llvm::dbgs() << "    Created affine.load at loop start: " << memrefLoad << "\n");
    
    // Map the old iter_arg to the loaded value
    mapper.map(oldBa, memrefLoad.getResult());
    
    // Now clone all operations - they'll automatically use the mapped load value
    for (Operation &op : oldBody->without_terminator()) {
      rewriter.clone(op, mapper);
    }
    
    LLVM_DEBUG(llvm::dbgs() << "  Step 4: Pulling operations from outside into loop\n");
    
    // Get the yielded value (mapped to new loop)
    Value oldYieldedValue = yieldOp.getOperand(numIterArgs - 1);
    Value currentAccum = mapper.lookupOrDefault(oldYieldedValue);
    if (!currentAccum) currentAccum = oldYieldedValue;
    
    // Pull multiply operations into the loop
    for (auto &[op, invariantOp] : opsChain) {
      if (auto mulOp = dyn_cast<arith::MulFOp>(op)) {
        LLVM_DEBUG(llvm::dbgs() << "    Pulling multiply into loop: " << *mulOp << "\n");
        
        // We need to insert the multiply before the operation that produces currentAccum
        if (auto defOp = currentAccum.getDefiningOp()) {
          rewriter.setInsertionPointAfter(defOp);
        } else {
          rewriter.setInsertionPointToStart(newBody);
        }
        
        // The multiply scales the accumulated value
        // If currentAccum is the result of an AddFOp, we need to modify it
        if (auto addOp = currentAccum.getDefiningOp<arith::AddFOp>()) {
          // Find which operand is the accumulator vs the value being added
          Value lhs = addOp.getLhs();
          Value rhs = addOp.getRhs();
          
          // Check if either operand references the old iter_arg
          bool lhsIsAccum = false;
          bool rhsIsAccum = false;
          
          // Walk back through mapper to check
          for (auto arg : forOp.getRegionIterArgs()) {
            Value mappedArg = mapper.lookupOrDefault(arg);
            if (mappedArg && mappedArg == lhs) lhsIsAccum = true;
            if (mappedArg && mappedArg == rhs) rhsIsAccum = true;
          }
          
          Value valueToScale = rhsIsAccum ? lhs : rhs;
          Value accumValue = rhsIsAccum ? rhs : lhs;
          
          // Create new multiply
          rewriter.setInsertionPoint(addOp);
          auto newMul = rewriter.create<arith::MulFOp>(loc, invariantOp, valueToScale);
          
          // Create new addition
          auto newAdd = rewriter.create<arith::AddFOp>(loc, accumValue, newMul.getResult());
          
          // Replace the old add
          rewriter.replaceOp(addOp, newAdd.getResult());
          currentAccum = newAdd.getResult();
          
          LLVM_DEBUG(llvm::dbgs() << "      Created: " << newMul << "\n");
          LLVM_DEBUG(llvm::dbgs() << "      Created: " << newAdd << "\n");
        }
      }
    }
    
    LLVM_DEBUG(llvm::dbgs() << "  Step 5: Creating store at end of loop\n");
    
    // Create store before the yield (load was already created and mapped earlier)
    rewriter.setInsertionPoint(newBody->getTerminator());
    auto newStore = rewriter.create<affine::AffineStoreOp>(
        loc, currentAccum, storeOp.getMemref(), storeOp.getMap(),
        storeOp.getMapOperands());
    LLVM_DEBUG(llvm::dbgs() << "    Created affine.store before yield: " << newStore << "\n");
    
    LLVM_DEBUG(llvm::dbgs() << "  Step 6: Fixing yield operation\n");
    
    // Create new yield with mapped operands (excluding the iter_arg we removed)
    SmallVector<Value> newYieldOperands;
    for (unsigned i = 0; i < numIterArgs - 1; i++) {
      Value oldOperand = yieldOp.getOperand(i);
      Value newOperand = mapper.lookupOrDefault(oldOperand);
      if (!newOperand) newOperand = oldOperand;
      newYieldOperands.push_back(newOperand);
    }
    
    rewriter.setInsertionPoint(newBody->getTerminator());
    rewriter.replaceOpWithNewOp<affine::AffineYieldOp>(
        newBody->getTerminator(), newYieldOperands);
    
    LLVM_DEBUG(llvm::dbgs() << "  Step 7: Erasing old operations outside loop\n");
    
    // Erase the external store
    LLVM_DEBUG(llvm::dbgs() << "    Erasing store: " << *storeOp << "\n");
    storeOp.erase();
    
    // Erase operations in reverse order
    for (auto it = opsChain.rbegin(); it != opsChain.rend(); ++it) {
      auto &[op, _] = *it;
      LLVM_DEBUG(llvm::dbgs() << "    Erasing: " << *op << "\n");
      rewriter.eraseOp(op);
    }
    
    // Erase the init load if it exists
    if (initLoad) {
      LLVM_DEBUG(llvm::dbgs() << "    Erasing init load: " << *initLoad << "\n");
      rewriter.eraseOp(initLoad);
    }

    LLVM_DEBUG(llvm::dbgs() << "  Step 8: Replacing uses of old loop results with new loop\n");
    for(unsigned i = 0; i < numIterArgs - 1; i++){
      rewriter.replaceAllUsesWith(forOp.getResult(i), newForOp.getResult(i));
    }

    LLVM_DEBUG(llvm::dbgs() << "  Step 9: Erasing old loop\n");
    rewriter.eraseOp(forOp);
    LLVM_DEBUG(llvm::dbgs() << "=== RemoveAffineIterArgs SUCCESS ===\n\n");
    return success();
  }
};

namespace {
struct RemoveIterArgs : public RemoveIterArgsBase<RemoveIterArgs> {

  void runOnOperation() override {
    LLVM_DEBUG(llvm::dbgs() << "\n\n");
    LLVM_DEBUG(llvm::dbgs() << "===================================================\n");
    LLVM_DEBUG(llvm::dbgs() << "=== STARTING RemoveIterArgs PASS ===\n");
    LLVM_DEBUG(llvm::dbgs() << "===================================================\n");
    
    GreedyRewriteConfig config;
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    patterns.insert<RemoveSCFIterArgs>(patterns.getContext());
    patterns.insert<RemoveAffineIterArgs>(patterns.getContext());

    LLVM_DEBUG(llvm::dbgs() << "Registered patterns: RemoveSCFIterArgs, RemoveAffineIterArgs\n");
    LLVM_DEBUG(llvm::dbgs() << "Applying patterns greedily...\n\n");

    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      LLVM_DEBUG(llvm::dbgs() << "\n!!! RemoveIterArgs PASS FAILED !!!\n");
      signalPassFailure();
      return;
    }
    
    LLVM_DEBUG(llvm::dbgs() << "\n");
    LLVM_DEBUG(llvm::dbgs() << "===================================================\n");
    LLVM_DEBUG(llvm::dbgs() << "=== RemoveIterArgs PASS COMPLETED SUCCESSFULLY ===\n");
    LLVM_DEBUG(llvm::dbgs() << "===================================================\n\n");
  }
};
} // namespace

namespace mlir {
namespace polygeist {
std::unique_ptr<Pass> createRemoveIterArgsPass() {
  return std::make_unique<RemoveIterArgs>();
}
} // namespace polygeist
} // namespace mlir

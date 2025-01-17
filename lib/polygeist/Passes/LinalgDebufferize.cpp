#include "PassDetails.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "polygeist/Ops.h"
#include "polygeist/Passes/Passes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "linalg-debufferize"

using namespace mlir;
using namespace mlir::arith;
using namespace polygeist;
using namespace affine;
using namespace linalg;
using namespace tensor;
using namespace bufferization;


bool isCaptured(Value v, Operation *potentialUser = nullptr,
                bool *seenuse = nullptr);

std::vector<Operation *> getSortedUsers(Value val) {
   std::vector<Operation*> users;
  for (Operation *user : val.getUsers()) {
    users.push_back(user);
  }

  // Sort the users based on their topological order
  std::sort(users.begin(), users.end(), [](Operation *a, Operation *b) {
    return a->isBeforeInBlock(b);
  });

  return users;
}

std::vector<Operation *> getSortedUsers(Operation *op) {
  // Find the parent function
  auto funcOp = op->getParentOfType<func::FuncOp>();
  if (!funcOp)
    return {};

  // Map to store order of operations
  llvm::DenseMap<Operation *, size_t> opOrder;
  size_t order = 0;

  funcOp.walk([&](Operation *curOp) { opOrder[curOp] = order++; });

  std::vector<Operation *> sortedUsers(op->getUsers().begin(),
                                       op->getUsers().end());

  std::sort(
      sortedUsers.begin(), sortedUsers.end(),
      [&](Operation *a, Operation *b) { return opOrder[a] < opOrder[b]; });

  return sortedUsers;
}

struct debufferizationAllocaRemoval : public OpRewritePattern<memref::AllocaOp> {
  using OpRewritePattern<memref::AllocaOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocaOp allocaOp,
                                PatternRewriter &rewriter) const final {
    Value allocaResult = allocaOp.getResult();
    bool userToTensorOp = false;
    bool userCopyOp = false;
    bool userOtherOp = false;
    Value copyOp;
    Value toTensorOp;
    for (Operation *user : allocaResult.getUsers()) {
      if (isa<bufferization::ToTensorOp>(user)) {
        userToTensorOp = true;
        toTensorOp = user->getResult(0);
      }
      else if (isa<memref::CopyOp>(user)) {
        userCopyOp = true;
        copyOp = user->getResult(0);
      }
      else
        userOtherOp = true;
    }
    
    if(!(!userOtherOp&&userCopyOp&&userToTensorOp))
      return failure();

    auto emptyTensor =
    rewriter.create<tensor::EmptyOp>(allocaOp.getLoc(),allocaOp.getType().getShape(),
    allocaOp.getType().getElementType());

    rewriter.replaceAllUsesWith(toTensorOp, emptyTensor.getResult());
    rewriter.eraseOp(copyOp.getDefiningOp());
    rewriter.eraseOp(toTensorOp.getDefiningOp());
    return success();
  }
}; 

struct LinalgDebufferization : public OpRewritePattern<func::FuncOp> {
  using OpRewritePattern<func::FuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::FuncOp funcOp,
                                PatternRewriter &rewriter) const final {

    auto module = funcOp->getParentOfType<ModuleOp>();

    SmallVector<Operation *> opsToDelete;
    llvm::SmallPtrSet<Operation *, 16> opsToDeleteSet;
    // Tracks both old linalg.generics and linalg.generics with repeated values
    // in ins and outs
    llvm::SmallPtrSet<Operation *, 16> processedGenericOps;

    LogicalResult passResult = failure();

    auto handleMemref = [&](Value memVal) -> LogicalResult {
      auto module = memVal.getParentRegion()->getParentOfType<ModuleOp>();
      
      if (!memVal.getType().isa<MemRefType>()) {
        return failure(); 
      }

      bool isNoalias = false;
      if (auto mem = memVal.getDefiningOp<MemoryEffectOpInterface>()) {
        if (auto defOp = memVal.getDefiningOp()) {//if (mem has allocation like) {
          if (isa<memref::AllocaOp>(defOp)) {
            isNoalias = true;
          }
        }
      } else if (auto ba = dyn_cast<BlockArgument>(memVal)) {
        if (auto fn = dyn_cast<FunctionOpInterface>(ba.getOwner()->getParentOp())) {
          if (fn.getArgAttr(ba.getArgNumber(), LLVM::LLVMDialect::getNoAliasAttrName())) {
            isNoalias = true;
          }
        }
      } else if (memVal.getDefiningOp<memref::GetGlobalOp>() ||
                memVal.getDefiningOp<LLVM::AddressOfOp>()) {
        isNoalias = true; //TODO: is this correct?
      }

      // if we are no alias we can just look at all users of the value
      // if we are not noalias, or we are captured, then we have to look at all users that
      // could read or write
      if (!isNoalias) { //|| isCaptured(memVal)) { TODO: need to improve isCaptured to include linalg.generic 
        return failure(); //|| isCaptured(memVal)) { TODO: need to improve isCaptured to include linalg.generic
      }
      
      MemRefType memrefType;
      if (auto blockArg = memVal.dyn_cast<BlockArgument>()) {
        memrefType = blockArg.getType().dyn_cast<MemRefType>();
      } else if (auto allocaOp = memVal.getDefiningOp<memref::AllocaOp>()) {
        memrefType = allocaOp.getType();
      } else {
        return failure();
      } 
      
      
      rewriter.setInsertionPointAfterValue(memVal);
      auto tensorType = RankedTensorType::get(
          memrefType.getShape(), memrefType.getElementType());

      // Check to see if only linalg.generic are users of the Value op for now.
      // TODO: Extend this
      if (!llvm::all_of(memVal.getUsers(), [](Operation *op) {
            return isa<linalg::GenericOp>(op);
          })) {
        return failure();
      }

      // auto emptyTensor =
      // rewriter.create<tensor::EmptyOp>(allocaOp.getLoc(),allocaOp.getType().getShape(),
      // allocaOp.getType().getElementType());
      auto toTensorOp = rewriter.create<bufferization::ToTensorOp>(
          memVal.getLoc(), tensorType, memVal);
      Value currentTensor = toTensorOp;

      auto sortedUsers = getSortedUsers(memVal);

      // Check if allocaOp is an output in current genericOp
      for (auto user : sortedUsers) {
        if (auto genericOp = dyn_cast<linalg::GenericOp>(user)) {

          // auto genericOp = cast<linalg::GenericOp>(user);
          if (processedGenericOps.count(genericOp) > 0)
            continue;
          rewriter.setInsertionPointAfter(genericOp);

          SmallVector<Value, 4> newInputs;
          SmallVector<Value, 4> newOutputs;
          SmallVector<Type> resultTypes;
          // Create a new linalg.generic in Destination Style Passing format

          ArrayAttr indexingMaps = genericOp.getIndexingMaps();
          for (auto input : genericOp.getInputs()) {
            newInputs.push_back(input == memVal ? currentTensor : input);
          }

          // ArrayRef<Type> resultTypes;
          int newCurrentTensorIndex = -1;
          int index = 0;
          for (auto output : genericOp.getOutputs()) {
            newOutputs.push_back(output == memVal ? currentTensor : output);
            resultTypes.push_back(output == memVal ? currentTensor.getType()
                                                     : output.getType());
            if (output == memVal) {
              newCurrentTensorIndex = index;
            }
            index++;
          }

          StringAttr empty = StringAttr::get(genericOp.getContext());
          ArrayRef<Type> resultTypesRef(resultTypes);
          auto newGenericOp = rewriter.create<linalg::GenericOp>(
              genericOp.getLoc(), resultTypesRef, newInputs, newOutputs,
              genericOp.getIndexingMaps(), genericOp.getIteratorTypes(), empty,
              empty);

          Region &opRegion = newGenericOp.getRegion();
          rewriter.cloneRegionBefore(genericOp.getRegion(),
                                     newGenericOp.getRegion(),
                                     newGenericOp.getRegion().end());

          // Replace all uses of original generic op with the new one
          for (unsigned i = 0; i < genericOp->getNumResults(); ++i) {
            genericOp->getResult(i).replaceAllUsesWith(
                newGenericOp->getResult(i));
          }

          // Delete the original genericOp
          if (newCurrentTensorIndex != -1)
            currentTensor = newGenericOp.getResult(newCurrentTensorIndex);

          processedGenericOps.insert(genericOp.getOperation());
          // Delete the original genericOp
          //genericOp.erase();
          //WalkResult::interrupt();
          opsToDelete.push_back(genericOp.getOperation());
        }
      }

      auto toMemrefOp = rewriter.create<bufferization::ToMemrefOp>(
          memVal.getLoc(), memrefType, currentTensor);
      rewriter.create<memref::CopyOp>(memVal.getLoc(), toMemrefOp, memVal);
      // opsToDelete.push_back(allocaOp.getOperation());
      return success();
    };

    
    bool changed;
    do {
      changed = funcOp.walk([&](memref::AllocaOp alloca) {
        //if (handleMemref(alloca.getResult()).succeeded())
        //  return WalkResult::advance();
        //return WalkResult::interrupt();
        handleMemref(alloca.getResult()).succeeded();
        return WalkResult::advance();
      }).wasInterrupted();
      
      if (changed)
        passResult = success();
    } while (changed);
    
    if (llvm::any_of(llvm::map_range(funcOp.getArguments(), handleMemref), [](LogicalResult res) {return res.succeeded();}))
    
    passResult = success();
    for (Operation *op : opsToDelete) {
      op->erase();
    }
    opsToDelete.clear();

    return passResult;
  }
};

namespace {
struct LinalgDebufferize : public LinalgDebufferizeBase<LinalgDebufferize> {
  void runOnOperation() override;
};
} // namespace

void LinalgDebufferize::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.insert<LinalgDebufferization>(&getContext());
  //patterns.insert<debufferizationAllocaRemoval>(&getContext());
  GreedyRewriteConfig config;
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                     config);
}

namespace mlir {
namespace polygeist {
std::unique_ptr<Pass> createLinalgDebufferizePass() {
  return std::make_unique<LinalgDebufferize>();
}
} // namespace polygeist
} // namespace mlir

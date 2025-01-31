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

bool isAncestor(Operation *potentialAncestor, Operation *op) {
    Operation *current = op->getParentOp();
    while (current != nullptr) {
        if (current == potentialAncestor)
            return true;
        current = current->getParentOp();
    }
    return false;
}

//Checks if a comes before b
bool comesBefore(Operation *a, Operation *b) {
    if (a == b) return false;
    
    if (isAncestor(a, b)) return true;
    if (isAncestor(b, a)) return false;

    //Block *aBlock = a->getBlock();
    //Block *bBlock = b->getBlock();
    
    //// Same block: compare operation order
    //if (aBlock == bBlock) {
    //    for (Operation &op : aBlock->getOperations()) {
    //        if (&op == a) return true;
    //        if (&op == b) return false;
    //    }
    //    llvm_unreachable("Operations not found in their parent block");
    //}

    //// Different blocks: compare region hierarchy
    //Region *aRegion = aBlock->getParent();
    //Region *bRegion = bBlock->getParent();
    
    //// Same region: compare block order
    //if (aRegion == bRegion) {
    //    //auto aBlockIt = std::find(aRegion->begin(), aRegion->end(), aBlock);
    //    //auto bBlockIt = std::find(aRegion->begin(), aRegion->end(), bBlock);
    //    //return aBlockIt < bBlockIt;
    //    //const int aIndex = std::distance(aRegion->begin(), aRegion->find(aBlock));
    //    //const int bIndex = std::distance(aRegion->begin(), aRegion->find(bBlock));
    //    //return  aIndex < bIndex;
    //    auto get_block_pos = [](Region *region, Block *block) {
    //      auto &blocks = region->getBlocks();
    //      auto it = llvm::find_if(blocks, [block](Block &b) {
    //        return &b == block; // Address comparison
    //      });
    //      assert(it != blocks.end() && "Block not found in region");
    //      return std::distance(blocks.begin(), it);
    //      //return std::distance(region->getBlocks().begin(), 
    //      //                     llvm::find(region->getBlocks(), block));
    //    };
    //    return get_block_pos(aRegion, aBlock) < 
    //           get_block_pos(aRegion, bBlock);
    //}

    //// Different regions: compare parent operations
    //Operation *aParent = aRegion->getParentOp();
    //Operation *bParent = bRegion->getParentOp();
    
    //// Same parent op: compare region order
    //if (aParent == bParent) {
    //    //auto aRegionIt = std::find(aParent->getRegions().begin(), 
    //    //                         aParent->getRegions().end(), aRegion);
    //    //auto bRegionIt = std::find(bParent->getRegions().begin(),
    //    //                         bParent->getRegions().end(), bRegion);
    //    //return aRegionIt < bRegionIt;
    //    //auto get_region_position = [](Operation *parent, Region *target) {
    //    //return std::distance(
    //    //    parent->getRegions.begin(),
    //    //    llvm::find_if(parent->getRegions(), [&](Region &r) {
    //    //        return &r == target; // Compare region addresses
    //    //    })
    //    //  );
    //    //};

    //    auto get_region_position = [](Operation *parent, Region *target) {
    //      auto regions = parent->getRegions(); // Get reference to region list
    //      auto begin = regions.begin();
    //      auto it = llvm::find_if(regions, [&](Region &r) {
    //          return &r == target;
    //      });
    //      return std::distance(begin, it);
    //    };
    //  return get_region_position(aParent, aRegion) < 
    //       get_region_position(aParent, bRegion);
    //}

    Operation *aParent = a->getParentOp();
    Operation *bParent = b->getParentOp();
    // Walk up b's hierarchy until we reach a's level
    Operation *bAncestor = b;
    //We traverse B's ancestors here
    while (Operation *parent = bAncestor->getParentOp()) {
        if (parent == aParent) {
            // Compare positions within aParent's regions/blocks
            Region *aRegion = a->getParentRegion();
            Region *bRegion = bAncestor->getParentRegion();
            
            if (aRegion == bRegion) {
                // Same region: compare block order
                Block *aBlock = a->getBlock();
                Block *bBlock = bAncestor->getBlock();
                if (aBlock != bBlock) {
                  auto get_block_pos = [](Region *region, Block *block) {
                    auto &blocks = region->getBlocks();
                    auto it = llvm::find_if(blocks, [block](Block &b) {
                      return &b == block; // Address comparison
                    });
                    assert(it != blocks.end() && "Block not found in region");
                    return std::distance(blocks.begin(), it);
                  };
                  return get_block_pos(aRegion, aBlock) < 
                         get_block_pos(bRegion, bBlock);
                };
              // Same block: compare operation order
              return a->isBeforeInBlock(bAncestor);
            }
            
            // Different regions: compare region order
            auto compareRegions = [parent](Region *x, Region *y) {
                auto get_region_position = [](Operation *parent, Region *target) {
                  auto regions = parent->getRegions(); // Get reference to region list
                  auto begin = regions.begin();
                  auto it = llvm::find_if(regions, [&](Region &r) {
                      return &r == target;
                  });
                  return std::distance(begin, it);
                };
                return get_region_position(parent, x) < 
                   get_region_position(parent, y);
            };
            return compareRegions(aRegion, bRegion);
        }
        bAncestor = parent;
      }
    
    Operation *aAncestor = a;
    //We traverse A's ancestors here
    while (Operation *parent = aAncestor->getParentOp()) {
        if (parent == bParent) {
            // Compare positions within aParent's regions/blocks
            Region *bRegion = b->getParentRegion();
            Region *aRegion = aAncestor->getParentRegion();
            
            if (aRegion == bRegion) {
                // Same region: compare block order
                Block *bBlock = b->getBlock();
                Block *aBlock = aAncestor->getBlock();
                if (aBlock != bBlock) {
                  auto get_block_pos = [](Region *region, Block *block) {
                    auto &blocks = region->getBlocks();
                    auto it = llvm::find_if(blocks, [block](Block &b) {
                      return &b == block; // Address comparison
                    });
                    assert(it != blocks.end() && "Block not found in region");
                    return std::distance(blocks.begin(), it);
                  };
                  return !(get_block_pos(bRegion, bBlock) < 
                         get_block_pos(aRegion, aBlock));
                };
              // Same block: compare operation order
              return !b->isBeforeInBlock(aAncestor);
            }
            
            // Different regions: compare region order
            auto compareRegions = [parent](Region *x, Region *y) {
                auto get_region_position = [](Operation *parent, Region *target) {
                  auto regions = parent->getRegions(); // Get reference to region list
                  auto begin = regions.begin();
                  auto it = llvm::find_if(regions, [&](Region &r) {
                      return &r == target;
                  });
                  return std::distance(begin, it);
                };
                return get_region_position(parent, x) < 
                   get_region_position(parent, y);
            };
            return !compareRegions(bRegion, aRegion);
        }
        aAncestor = parent;
      }

    llvm_unreachable("Operations do not share a common ancestor");
    //// Recursive case: compare parent operations
    //return comesBefore(aParent, bParent);
}

std::vector<Operation *> getSortedUsers(Value val) {
   std::vector<Operation*> users;
  for (Operation *user : val.getUsers()) {
    users.push_back(user);
  }

  //TODO: problem is this only works for 1 level
  // Sort the users based on their topological order
  std::sort(users.begin(), users.end(), [](Operation *a, Operation *b) {
    return comesBefore(a,b);
    //if (a->getBlock() == b->getBlock()) {
    //  return a->isBeforeInBlock(b);
    //}
    //if (a->getParentRegion() == b->getParentRegion()) {
    //  Block *blockA = a->getBlock();
    //  Block *blockB = b->getBlock();
    //  return std::distance(blockA->getParent()->begin(), blockA->getIterator()) <
    //       std::distance(blockB->getParent()->begin(), blockB->getIterator());
    //}

    //return a->getParentRegion()->isAncestor(b->getParentRegion());
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

Region* findCommonAncestorRegion(Operation* a, Operation* b) {
    DenseMap<Region*, size_t> regionCounts;
    
    // Walk up from operation A
    Operation* currentOp = a;
    while (Region* region = currentOp->getParentRegion()) {
        regionCounts[region]++;
        currentOp = region->getParentOp();
    }

    // Walk up from operation B to find common region
    currentOp = b;
    while (Region* region = currentOp->getParentRegion()) {
        if (regionCounts.count(region))
            return region;
        currentOp = region->getParentOp();
    }
    return nullptr;
}


struct debufferizationAllocaRemoval : public OpRewritePattern<memref::AllocaOp> {
  using OpRewritePattern<memref::AllocaOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocaOp allocaOp,
                                PatternRewriter &rewriter) const final {
    Value allocaResult = allocaOp.getResult();
    bool userToTensorOp = false;
    bool userCopyOp = false;
    bool userOtherOp = false;
    memref::CopyOp copyOp;
    bufferization::ToTensorOp toTensorOp;
    for (Operation *user : allocaResult.getUsers()) {
      if (isa<bufferization::ToTensorOp>(user)) {
        userToTensorOp = true;
        toTensorOp = cast<bufferization::ToTensorOp>(user);
      }
      else if (isa<memref::CopyOp>(user)) {
        userCopyOp = true;
        copyOp = cast<memref::CopyOp>(user);
      }
      else
        userOtherOp = true;
    }
    
    if(!(!userOtherOp&&userCopyOp&&userToTensorOp))
      return failure();

    auto emptyTensor =
    rewriter.create<tensor::EmptyOp>(allocaOp.getLoc(),allocaOp.getType().getShape(),
    allocaOp.getType().getElementType());

    rewriter.replaceAllUsesWith(toTensorOp.getResult(), emptyTensor.getResult());

    rewriter.eraseOp(copyOp);
    rewriter.eraseOp(toTensorOp);
    return success();
  }
}; 

// Problems with this implementation: The way this implementation works is by jumping over users
// of alloca/args. The users we get are not in sorted order. We write a function to sort out the users across
// regions, blocks and ops as long as they lie in the same ancestry.
// Now as we update an op, and use the output tensor to give input to the next op- it works fine for simple cases with no region.
// But things becomes more complicated when we have nested regions like in scf.if and scf.for ops
// Why? Because we need to update scf.if and scf.for ops to yield correct tensors to be used by the next user.
// So how to do it? Well the best way is to traverse all the IR in a walk and and as we encouter a user and it's linalg.generic then we update
// it's params to tensor and generate an output tensor if it can, and move to the next op and repeat this until we encounter an end of region.
// At this point we need to decide if we need to yield the tensor or not? This depends if there is an external user of the original arg/alloca
// still left over. I think this can be done by tracking users of an op, and eliminating the ones which have been used.
// In the current way it's done- we can go the next user and check if the previous user is in the same block if not we need to propagate the previous
// users output tensor through regions with yield.
// How does this work if the user is not actually outputing data, that means it didn't generate an output tensor. In which case the original tensor needs to be continued.
// In current flow, we are tracking updated output tensor, now we can iteratively yield the value until it reaches the same block as next user.
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
      if ((!isNoalias) || isCaptured(memVal)) { //TODO: need to improve isCaptured to include linalg.generic 
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
      Value prevTensor = toTensorOp;

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

          //check_if_current_tensor_is_available_to_user_if_not_propagate_to_scope() {
          //  extract_common_ancestor of curentTensor and userOp.
          //  propagte currentTensor all the way to common ancestor.
          //  Make the propagated value the current tensor.
          //}
          auto commonRegion =  findCommonAncestorRegion(currentTensor.getDefiningOp(), user);
          if (!commonRegion) return failure();
          // Collect regions from source to common ancestor
          SmallVector<Region*> regions;
          for (Region* r = currentTensor.getParentRegion(); r != commonRegion; 
               r = r->getParentOp()->getParentRegion()) {
              regions.push_back(r);
          }

          // Propagate value through each region
          Value currentValue = currentTensor;
          for (Region* region : llvm::reverse(regions)) {
              Block& block = region->front();
              Operation* terminator = block.getTerminator();
              Operation *parentOp = region->getParentOp();

              if( auto prevIf = dyn_cast_or_null<scf::IfOp>(parentOp)) {
                auto prevResults = prevIf.getResults();
                SmallVector<Type> newResultTypes;
                for (auto res : prevResults)
                    newResultTypes.push_back(res.getType());
                newResultTypes.push_back(currentValue.getType());
              
                // Yield original results + new value
                auto thenYieldArgs = prevIf.thenYield().getOperands();
                SmallVector<Value> thenYieldValues;
                for (const auto &it :thenYieldArgs) {
                  thenYieldValues.push_back(it);
                }
                thenYieldValues.push_back(currentValue);
              
                SmallVector<Value> elseYieldValues;
                if(!prevIf.getElseRegion().empty()){
                  auto elseYieldArgs = prevIf.elseYield().getOperands();
                  for (const auto &it :elseYieldArgs) {
                    elseYieldValues.push_back(it);
                  }
                }
                elseYieldValues.push_back(prevTensor);
              
                //Create new Ifop
                rewriter.setInsertionPoint(prevIf);
                auto newIf = rewriter.create<scf::IfOp>(prevIf.getLoc(),
                                                        newResultTypes, // Combined types
                                                        prevIf.getCondition(),      // New condition value
                                                        true
                                                    ); 
                if (newIf.thenBlock())
                  rewriter.eraseBlock(newIf.thenBlock());
              
                newIf.getThenRegion().takeBody(prevIf.getThenRegion());
                if(!prevIf.getElseRegion().empty())
                  newIf.getElseRegion().takeBody(prevIf.getElseRegion());
              

                //Update yield ops 
                rewriter.setInsertionPointToEnd(newIf.thenBlock());
                rewriter.replaceOpWithNewOp<scf::YieldOp>(newIf.thenYield(), thenYieldValues);
                if(!prevIf.getElseRegion().empty()) {
                  rewriter.setInsertionPointToEnd(newIf.elseBlock());
                  rewriter.replaceOpWithNewOp<scf::YieldOp>(newIf.elseYield(), elseYieldValues);
                } else {
                  rewriter.setInsertionPointToEnd(newIf.elseBlock());
                  rewriter.create<scf::YieldOp>(newIf.getLoc(), elseYieldValues);
                }

                currentValue = newIf->getResult(newIf->getNumResults() - 1); 
              }
          }
          currentTensor = currentValue;

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

          rewriter.setInsertionPointAfter(genericOp);
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
          if (newCurrentTensorIndex != -1){
            prevTensor = currentTensor;
            currentTensor = newGenericOp.getResult(newCurrentTensorIndex);
          }

          processedGenericOps.insert(genericOp.getOperation());
          // Delete the original genericOp
          genericOp.erase();
          //WalkResult::interrupt();
          //opsToDelete.push_back(genericOp.getOperation());
        }
      }

      auto toMemrefOp = rewriter.create<bufferization::ToMemrefOp>(
          memVal.getLoc(), memrefType, currentTensor);
      rewriter.create<memref::CopyOp>(memVal.getLoc(), toMemrefOp, memVal);
      // opsToDelete.push_back(allocaOp.getOperation());
      return success();
    };

    
    bool changed;
    //Fix instead of walk, just get the list of allocaOp users, so that you can easily delete ops inside
    SmallVector<memref::AllocaOp> listOfAllocaOps;
    
    funcOp.walk([&](memref::AllocaOp alloca) {
      listOfAllocaOps.push_back(alloca);
    });
    
    for (auto alloca : listOfAllocaOps) {
      handleMemref(alloca);
    }

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
  patterns.insert<debufferizationAllocaRemoval>(&getContext());
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

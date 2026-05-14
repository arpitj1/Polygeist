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

using opTuple = std::tuple<Value, Value>; //First: result, Second: prev_tensor ?

bool isCaptured(Value v, Operation *potentialUser = nullptr,
                bool *seenuse = nullptr);

//===----------------------------------------------------------------------===//
// Region Context Tracking for Correct SSA Threading
//===----------------------------------------------------------------------===//

/// Tracks tensor state per region in a tree structure
/// This prevents sibling if regions from polluting each other's tensor state
struct RegionTensorState {
  Value tensor;
  bool valid = false;
};

/// Tracks pending yield updates for scf.if operations
struct PendingIfInfo {
  scf::IfOp ifOp;
  Value entryTensor;      // Tensor value before entering the if
  Value thenResult;       // Final tensor value from THEN branch (or entryTensor if no users)
  Value elseResult;       // Final tensor value from ELSE branch (or entryTensor if no users)
  bool thenProcessed = false;
  bool elseProcessed = false;
};

/// Check if an operation is inside a specific region (directly or nested)
bool isInRegion(Operation* op, Region* region) {
  return region->isAncestor(op->getParentRegion());
}

/// Check if an operation is inside the THEN branch of an scf.if
bool isInIfThenBranch(Operation* op, scf::IfOp ifOp) {
  bool result = ifOp.getThenRegion().isAncestor(op->getParentRegion());
  LLVM_DEBUG(llvm::dbgs() << "          isInIfThenBranch(" << op->getName() << " at " << op->getLoc() 
                          << ", if at " << ifOp.getLoc() << ") = " << result << "\n");
  return result;
}

/// Check if an operation is inside the ELSE branch of an scf.if
bool isInIfElseBranch(Operation* op, scf::IfOp ifOp) {
  bool result = ifOp.getElseRegion().isAncestor(op->getParentRegion());
  LLVM_DEBUG(llvm::dbgs() << "          isInIfElseBranch(" << op->getName() << " at " << op->getLoc() 
                          << ", if at " << ifOp.getLoc() << ") = " << result << "\n");
  return result;
}

/// Find the innermost scf.if that contains this operation
scf::IfOp findContainingIf(Operation* op) {
  Operation* parent = op->getParentOp();
  while (parent) {
    if (auto ifOp = dyn_cast<scf::IfOp>(parent))
      return ifOp;
    parent = parent->getParentOp();
  }
  return nullptr;
}

/// Get all scf.if ops between an operation and a root region (innermost first)
SmallVector<scf::IfOp> getContainingIfs(Operation* op, Region* rootRegion) {
  SmallVector<scf::IfOp> result;
  Region* current = op->getParentRegion();
  while (current && current != rootRegion) {
    if (auto ifOp = dyn_cast<scf::IfOp>(current->getParentOp())) {
      result.push_back(ifOp);
    }
    current = current->getParentOp()->getParentRegion();
  }
  return result;
}

/// Get the current tensor for a region by tracing up the tree until we find a valid entry
/// This ensures sibling regions don't pollute each other - each inherits from parent only
Value getCurrentTensorForRegion(Region* region, 
                                 llvm::DenseMap<Region*, RegionTensorState>& regionTensorTree,
                                 Value fallbackTensor) {
  Region* current = region;
  while (current) {
    auto it = regionTensorTree.find(current);
    if (it != regionTensorTree.end() && it->second.valid) {
      LLVM_DEBUG(llvm::dbgs() << "      getCurrentTensorForRegion: found valid tensor in region\n");
      return it->second.tensor;
    }
    // Go to parent region
    Operation* parentOp = current->getParentOp();
    if (!parentOp) break;
    current = parentOp->getParentRegion();
  }
  LLVM_DEBUG(llvm::dbgs() << "      getCurrentTensorForRegion: using fallback tensor\n");
  return fallbackTensor;
}

/// Set the tensor state for a region
void setRegionTensor(Region* region, Value tensor,
                     llvm::DenseMap<Region*, RegionTensorState>& regionTensorTree) {
  regionTensorTree[region] = RegionTensorState{tensor, true};
  LLVM_DEBUG(llvm::dbgs() << "      setRegionTensor: set tensor for region\n");
}

/// Record the current tensor value for all containing if branches
/// This should be called after any tensor modification (store, linalg.generic, etc.)
void recordBranchResult(Operation* user, Value newTensor, 
                        llvm::DenseMap<scf::IfOp, PendingIfInfo>& pendingIfs,
                        Region* rootRegion) {
  LLVM_DEBUG(llvm::dbgs() << "      recordBranchResult called for user: " << user->getName() << " at " << user->getLoc() << "\n");
  LLVM_DEBUG(llvm::dbgs() << "        newTensor: " << newTensor << "\n");
  
  // For each containing if, record the tensor in the appropriate branch
  auto containingIfs = getContainingIfs(user, rootRegion);
  LLVM_DEBUG(llvm::dbgs() << "        Found " << containingIfs.size() << " containing ifs\n");
  
  for (scf::IfOp ifOp : containingIfs) {
    auto it = pendingIfs.find(ifOp);
    if (it != pendingIfs.end()) {
      PendingIfInfo& info = it->second;
      if (isInIfThenBranch(user, ifOp)) {
        LLVM_DEBUG(llvm::dbgs() << "        Recording THEN result for if at " << ifOp.getLoc() << "\n");
        info.thenResult = newTensor;
        info.thenProcessed = true;
        LLVM_DEBUG(llvm::dbgs() << "          Set thenResult, thenProcessed=true\n");
      } else if (isInIfElseBranch(user, ifOp)) {
        LLVM_DEBUG(llvm::dbgs() << "        Recording ELSE result for if at " << ifOp.getLoc() << "\n");
        info.elseResult = newTensor;
        info.elseProcessed = true;
        LLVM_DEBUG(llvm::dbgs() << "          Set elseResult, elseProcessed=true\n");
      } else {
        LLVM_DEBUG(llvm::dbgs() << "        WARNING: User not in THEN or ELSE branch of if at " << ifOp.getLoc() << "!\n");
      }
    } else {
      LLVM_DEBUG(llvm::dbgs() << "        No pending info for if at " << ifOp.getLoc() << " (skipping)\n");
    }
  }
}

//===----------------------------------------------------------------------===//
// Subview Chain Tracing and Affine Map Composition
//===----------------------------------------------------------------------===//

/// Structure to hold information about a chain of submaps from a leaf memref
/// back to the root memref (alloca/alloc/function arg)
struct SubmapChainInfo {
  Value rootMemref;                           // The root alloca/alloc/arg
  SmallVector<polygeist::SubmapOp> submaps;   // Chain of polygeist.submap ops (root to leaf)
  
  bool isEmpty() const { return submaps.empty(); }
};

/// Trace from a memref value back through submap operations to find the root
/// Returns the chain info with all operations collected
SubmapChainInfo traceSubmapChainToRoot(Value memref) {
  SubmapChainInfo info;
  Value current = memref;
  
  // Walk up the def-use chain through submaps
  while (auto submapOp = current.getDefiningOp<polygeist::SubmapOp>()) {
    info.submaps.push_back(submapOp);
    current = submapOp.getViewSource();
  }
  
  info.rootMemref = current;
  
  // Reverse so ops are in root-to-leaf order
  std::reverse(info.submaps.begin(), info.submaps.end());
  
  return info;
}

/// Get the tensor type for a submap chain's result
RankedTensorType getSubmapChainTensorType(const SubmapChainInfo &chain) {
  if (chain.isEmpty()) {
    auto memrefType = chain.rootMemref.getType().cast<MemRefType>();
    return RankedTensorType::get(memrefType.getShape(), 
                                  memrefType.getElementType());
  }
  
  // Get type from the last submap
  auto leafSubmap = chain.submaps.back();
  auto resultType = leafSubmap.getType().cast<MemRefType>();
  return RankedTensorType::get(resultType.getShape(), 
                                resultType.getElementType());
}

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

    //llvm_unreachable("Operations do not share a common ancestor");
    //// Recursive case: compare parent operations
    return comesBefore(aParent, bParent);
}

std::vector<Operation *> getSortedUsers(Value val) {
   std::vector<Operation*> users;
  for (Operation *user : val.getUsers()) {
    //This logic is to prevent duplication of users
    auto it = std::find_if(users.begin(), users.end(),
                           [user](const Operation* op) {
                               return op == user;
                           });
    if(it == users.end())
      users.push_back(user);
  }

  std::sort(users.begin(), users.end(), [](Operation *a, Operation *b) {
    return comesBefore(a,b);
  });

  return users;
}

// std::vector<Operation *> getSortedUsers(Operation *op) {
//   // Find the parent function
//   auto funcOp = op->getParentOfType<func::FuncOp>();
//   if (!funcOp)
//     return {};

//   // Map to store order of operations
//   llvm::DenseMap<Operation *, size_t> opOrder;
//   size_t order = 0;

//   funcOp.walk([&](Operation *curOp) { opOrder[curOp] = order++; });

//   std::vector<Operation *> sortedUsers(op->getUsers().begin(),
//                                        op->getUsers().end());

//   std::sort(
//       sortedUsers.begin(), sortedUsers.end(),
//       [&](Operation *a, Operation *b) { return opOrder[a] < opOrder[b]; });

//   return sortedUsers;
// }

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
    allocaOp.getType().getElementType(), allocaOp.getDynamicSizes());

    rewriter.replaceAllUsesWith(toTensorOp.getResult(), emptyTensor.getResult());

    rewriter.eraseOp(copyOp);
    rewriter.eraseOp(toTensorOp);
    return success();
  }
}; 

void findUsersInRegion(
    mlir::Value value, 
    mlir::Region& region, 
    llvm::SmallVectorImpl<mlir::Operation*>& users
) {
    for (mlir::Block& block : region) {
        for (mlir::Operation& op : block) {
            for (mlir::Value operand : op.getOperands()) {
                if (operand == value) {
                    users.push_back(&op);
                    break; // No need to check other operands for this op
                }
            }

            // Recursively check all sub-regions of this operation
            for (mlir::Region& subRegion : op.getRegions()) {
                findUsersInRegion(value, subRegion, users);
            }
        }
    }
}

/// Updated propagateValueThroughRegion that correctly handles both THEN and ELSE branches
/// 
/// Key insight: When we call this function, currentValue is the tensor value computed
/// in some branch. We need to determine which branch it came from and yield correctly:
/// - If currentValue is in THEN branch: THEN yields currentValue, ELSE yields initTensor
/// - If currentValue is in ELSE branch: THEN yields initTensor, ELSE yields currentValue
void propagateValueThroughRegion(Value &currentValue, SmallVector<Region*> regions, 
                                  std::vector<Operation *> expandedUserList, 
                                  llvm::DenseMap<Operation*, opTuple> opResultMap, 
                                  PatternRewriter &rewriter,
                                  llvm::DenseMap<scf::IfOp, PendingIfInfo> &pendingIfs) {
  LLVM_DEBUG(llvm::dbgs() << "      propagateValueThroughRegion: Processing " << regions.size() << " regions\n");
  LLVM_DEBUG(llvm::dbgs() << "      Current pendingIfs state (" << pendingIfs.size() << " entries):\n");
  // Note: We only print locations and processed flags, not the actual Values,
  // because some Values might point to erased operations and crash when printed
  LLVM_DEBUG({
    for (auto& [ifOp, info] : pendingIfs) {
      llvm::dbgs() << "        If at " << ifOp.getLoc() << ": ";
      llvm::dbgs() << "thenProcessed=" << info.thenProcessed << ", ";
      llvm::dbgs() << "elseProcessed=" << info.elseProcessed << "\n";
    }
  });
  
  for (Region* region : regions) {
    LLVM_DEBUG(llvm::dbgs() << "      Processing region in: " << region->getParentOp()->getName() << " at " << region->getParentOp()->getLoc() << "\n");
      Block& block = region->front();
      (void)block; // Silence unused warning
      Operation *parentOp = region->getParentOp();

      //Find init Tensor for the given for loop, i.e first match to expanded user list
      mlir::Value initTensor;
      int insertIdx = 0;
      bool insertIdxFound = false;
      for(auto user: expandedUserList) {
        mlir::Region *opRegion = user->getParentRegion();
        if(region->isAncestor(opRegion)) {
          insertIdxFound = true;
          //Maintain a map data structure for tracking every user and if they have been processed then the corresponding result
          auto it = opResultMap.find(user);
          if(it == opResultMap.end())
            continue;
          auto keys_value = it->second;
          // op_result (std::get<0>) not used currently, only initTensor needed
          initTensor = std::get<1>(keys_value);
          break;
        }
        if(!insertIdxFound)
          insertIdx++; 
      }

      if( auto prevIf = dyn_cast_or_null<scf::IfOp>(parentOp)) {
        LLVM_DEBUG(llvm::dbgs() << "      Processing scf.if at " << prevIf.getLoc() << "\n");
        
        // Check if we have pending info for this if (from branch processing)
        auto pendingIt = pendingIfs.find(prevIf);
        
        Value thenValue, elseValue;
        Value entryTensor = initTensor ? initTensor : currentValue;
        
        if (pendingIt != pendingIfs.end()) {
          // We have recorded branch results - use them directly
          PendingIfInfo& info = pendingIt->second;
          entryTensor = info.entryTensor;
          
          LLVM_DEBUG(llvm::dbgs() << "      PendingIfInfo state: thenProcessed=" << info.thenProcessed 
                                    << ", elseProcessed=" << info.elseProcessed << "\n");
          
          // Use recorded values: if a branch was processed, use its result; otherwise use entry tensor
          thenValue = info.thenProcessed ? info.thenResult : entryTensor;
          elseValue = info.elseProcessed ? info.elseResult : entryTensor;
          
          LLVM_DEBUG(llvm::dbgs() << "      Using recorded values for THEN and ELSE branches\n");
        } else {
          // First time seeing this if - no users processed yet, use entry tensor for both
          thenValue = entryTensor;
          elseValue = entryTensor;
          
          // Record for future reference
          PendingIfInfo info;
          info.ifOp = prevIf;
          info.entryTensor = entryTensor;
          info.thenResult = entryTensor;
          info.elseResult = entryTensor;
          info.thenProcessed = false;
          info.elseProcessed = false;
          pendingIfs[prevIf] = info;
          
          LLVM_DEBUG(llvm::dbgs() << "      First time seeing if, using entry tensor for both branches\n");
        }
        
        initTensor = entryTensor;
        
        LLVM_DEBUG(llvm::dbgs() << "      Building new if with yields for THEN and ELSE branches\n");
        
        auto prevResults = prevIf.getResults();
        SmallVector<Type> newResultTypes;
        for (auto res : prevResults)
            newResultTypes.push_back(res.getType());
        newResultTypes.push_back(currentValue.getType());

        // Build yield values with correct values for each branch
        auto thenYieldArgs = prevIf.thenYield().getOperands();
        SmallVector<Value> thenYieldValues;
        for (const auto &it :thenYieldArgs) {
          thenYieldValues.push_back(it);
        }
        thenYieldValues.push_back(thenValue);

        // Save whether prevIf has else BEFORE takeBody moves it
        bool hadElse = !prevIf.getElseRegion().empty();
        
        SmallVector<Value> elseYieldValues;
        if(hadElse){
          auto elseYieldArgs = prevIf.elseYield().getOperands();
          for (const auto &it :elseYieldArgs) {
            elseYieldValues.push_back(it);
          }
        }
        elseYieldValues.push_back(elseValue);

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
        if(hadElse)
          newIf.getElseRegion().takeBody(prevIf.getElseRegion());


        //Update yield ops 
        rewriter.setInsertionPointToEnd(newIf.thenBlock());
        rewriter.replaceOpWithNewOp<scf::YieldOp>(newIf.thenYield(), thenYieldValues);
        if(hadElse) {
          rewriter.setInsertionPointToEnd(newIf.elseBlock());
          rewriter.replaceOpWithNewOp<scf::YieldOp>(newIf.elseYield(), elseYieldValues);
        } else {
          rewriter.setInsertionPointToEnd(newIf.elseBlock());
          rewriter.create<scf::YieldOp>(newIf.getLoc(), elseYieldValues);
        }
        
        // Replace uses of old if results with new ones and erase old if
        for (auto [oldResult, newResult] : llvm::zip(prevIf.getResults(), newIf.getResults().drop_back())) {
          oldResult.replaceAllUsesWith(newResult);
        }
        rewriter.eraseOp(prevIf);
        
        // Update pending info to reference new if
        if (pendingIt != pendingIfs.end()) {
          pendingIfs.erase(pendingIt);
        }
        pendingIfs[newIf] = PendingIfInfo{newIf, initTensor, thenValue, elseValue, true, true};
        
        opResultMap[newIf] = std::make_tuple(newIf->getResult(newIf->getNumResults() - 1), initTensor);
        currentValue = newIf->getResult(newIf->getNumResults() - 1); 
        
        LLVM_DEBUG(llvm::dbgs() << "      Created new if at " << newIf->getLoc() << " with " << newIf->getNumResults() << " results\n");
        
        // FIX: Update outer ifs to use this if's result instead of raw inner tensor values
        // This is critical for nested ifs - outer ifs should yield the inner if's RESULT,
        // not values defined inside the inner if (which wouldn't dominate the yield)
        for (auto& [outerIfOp, outerInfo] : pendingIfs) {
          if (outerIfOp == newIf) continue;  // Skip self
          
          // Check if newIf is nested inside outerIfOp
          if (outerIfOp.getThenRegion().isAncestor(newIf->getParentRegion())) {
            // newIf is in outer's THEN branch - outer should yield newIf's result
            LLVM_DEBUG(llvm::dbgs() << "      Updating outer if at " << outerIfOp.getLoc() << " THEN result\n");
            outerInfo.thenResult = currentValue;
            outerInfo.thenProcessed = true;
          } else if (outerIfOp.getElseRegion().isAncestor(newIf->getParentRegion())) {
            // newIf is in outer's ELSE branch - outer should yield newIf's result
            LLVM_DEBUG(llvm::dbgs() << "      Updating outer if at " << outerIfOp.getLoc() << " ELSE result\n");
            outerInfo.elseResult = currentValue;
            outerInfo.elseProcessed = true;
          }
        }
        
      }
      else if (auto prevFor = dyn_cast_or_null<scf::ForOp>(parentOp)) {

        //After first match, now find all the users of the init Tensor in a region.
        llvm::SmallVector<mlir::Operation*> initOpUsers;
        findUsersInRegion(initTensor, *region, initOpUsers);
        
        SmallVector<Value> newInitOperands = prevFor.getInitArgs();
        newInitOperands.push_back(initTensor); //Needs to be the earliest use inside the region.
        //TODO: Does this require fix in if as well?

        SmallVector<Type, 5> newResultTypes(prevFor.getResultTypes().begin(), prevFor.getResultTypes().end());
        newResultTypes.push_back(currentValue.getType());

        rewriter.setInsertionPoint(prevFor);
        scf::ForOp newLoop = rewriter.create<scf::ForOp>(
            prevFor.getLoc(),
            prevFor.getLowerBound(),
            prevFor.getUpperBound(),
            prevFor.getStep(),
            newInitOperands
        );
        newLoop->setAttrs(prevFor.getOperation()->getAttrs());

        // Create block with induction variable + original args + new arg
        SmallVector<Type> blockArgTypes;
        blockArgTypes.push_back(newLoop.getInductionVar().getType());  // IV
        llvm::append_range(blockArgTypes, newLoop.getResultTypes());    // Original args

        // Transfer operations from original block to new block
        Block *newBlock = &newLoop.getRegion().front();
        Block *originalBlock = &prevFor.getRegion().front();
        newBlock->getOperations().splice(
            newBlock->end(),
            originalBlock->getOperations()
        );

        // Replace uses of original block arguments with new ones
        for (unsigned i = 0; i < originalBlock->getNumArguments()-1; ++i) {
            originalBlock->getArgument(i + 1)  // +1 for IV
                .replaceAllUsesWith(newBlock->getArgument(i + 1));
        }

        auto yieldOp = cast<scf::YieldOp>(newBlock->getTerminator());
        SmallVector<Value> newYieldValues = yieldOp.getOperands();
        // Add new iteration arg from block arguments
        newYieldValues.push_back(currentValue);
        
        rewriter.setInsertionPoint(yieldOp);
        rewriter.replaceOpWithNewOp<scf::YieldOp>(yieldOp, newYieldValues);
        
        //Update users of initOp to use iterArgs
        for(auto initOpUser: initOpUsers) {
          // Iterate over all operands (both inputs and outputs)
          for (const auto &en : llvm::enumerate(initOpUser->getOperands())) {
            if (en.value() == initTensor) {
              OpOperand &operand = initOpUser->getOpOperand(en.index());
              Value newValue = newLoop.getRegionIterArg(newLoop.getRegion().front().getNumArguments()-2); //-1 for IV
              operand.set(newValue);
            }
          }
        }

        //Update users of prev For loops results 
        for (auto [oldResult, newResult] : llvm::zip(prevFor.getResults(), newLoop.getResults().drop_back())) {
          oldResult.replaceAllUsesWith(newResult);
        }
        rewriter.eraseOp(prevFor);
        currentValue = newLoop.getResults().back(); 
        
        //Store this in the user list for this region, need to create a data structure for users
        opResultMap[newLoop] = std::make_tuple(currentValue, initTensor);
        //Update the user list with the for Loop
        expandedUserList.insert(expandedUserList.begin() + insertIdx, newLoop);
      }
  }
}

bool isDirectUser(Operation *consumer, Operation *producer) {
    for (Value operand : consumer->getOperands()) {
        if (operand.getDefiningOp() == producer)
            return true;
    }
    return false;
}

/// Check if all users of a memref are supported for debufferization
bool areAllUsersSupportedForDebufferization(Value memVal) {
  for (Operation *user : memVal.getUsers()) {
    if (isa<memref::AllocaOp, memref::AllocOp, memref::DeallocOp,
            memref::SubViewOp, memref::LoadOp, memref::StoreOp,
            affine::AffineLoadOp, affine::AffineStoreOp,
            linalg::GenericOp, bufferization::ToTensorOp>(user)) {
      continue;
    }
    // Check if it's a subview that we should also trace
    if (auto subviewOp = dyn_cast<memref::SubViewOp>(user)) {
      // Recursively check subview users
      if (!areAllUsersSupportedForDebufferization(subviewOp.getResult())) {
        return false;
      }
      continue;
    }
    LLVM_DEBUG(llvm::dbgs() << "  Unsupported user: " << user->getName() << " at " << user->getLoc() << "\n");
    return false;
  }
  return true;
}

/// Collect all memory operations (load/store/linalg.generic) on a memref
/// including those that access through subviews
/// Recursively collect all memory operations (load/store/linalg) that use a memref,
/// including through submap chains
void collectMemoryOpsRecursively(Value memVal, 
                                  SmallVectorImpl<Operation*> &memOps,
                                  llvm::SmallPtrSetImpl<Operation*> &visited) {
  for (Operation *user : memVal.getUsers()) {
    // Skip if already visited
    if (visited.count(user))
      continue;
    visited.insert(user);
    
    if (isa<memref::LoadOp, memref::StoreOp, 
            affine::AffineLoadOp, affine::AffineStoreOp,
            linalg::GenericOp>(user)) {
      memOps.push_back(user);
    } else if (auto submapOp = dyn_cast<polygeist::SubmapOp>(user)) {
      // Recursively collect ops on the submap result
      collectMemoryOpsRecursively(submapOp.getResult(), memOps, visited);
    }
  }
}

/// Get all operations that access a memref (directly or through subview/submap)
std::vector<Operation*> getAllMemoryUsers(Value memVal) {
  SmallVector<Operation*> memOps;
  llvm::SmallPtrSet<Operation*, 16> visited;
  collectMemoryOpsRecursively(memVal, memOps, visited);
  
  // Sort by execution order
  std::sort(memOps.begin(), memOps.end(), [](Operation *a, Operation *b) {
    return comesBefore(a, b);
  });
  
  return std::vector<Operation*>(memOps.begin(), memOps.end());
}

//===----------------------------------------------------------------------===//
// Main Debufferization Pattern
//===----------------------------------------------------------------------===//

// Algorithm Overview:
// 1. For a given root memref (alloca/alloc/func arg), create initial tensor
// 2. Maintain CurrentSlices map: root memref -> current tensor state
// 3. For each memory operation in sorted order:
//    - SubViewOp: NOOP (trace chain at load/store time)
//    - LoadOp: trace to root, compose indices, use submap to gather, extract
//    - StoreOp: trace to root, compose indices, insert, submapInverse
//    - LinalgGenericOp: submap for inputs, submapInverse for outputs
// 4. At the end, write back final tensor to original memref

struct LinalgDebufferization : public OpRewritePattern<func::FuncOp> {
  using OpRewritePattern<func::FuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::FuncOp funcOp,
                                PatternRewriter &rewriter) const final {

    LLVM_DEBUG(llvm::dbgs() << "\n=== LinalgDebufferization::matchAndRewrite ===\n");
    LLVM_DEBUG(llvm::dbgs() << "Processing function: " << funcOp.getName() << "\n");

    LogicalResult passResult = failure();

    // The main handler for each root memref
    auto handleMemref = [&](Value memVal) -> LogicalResult {
      LLVM_DEBUG(llvm::dbgs() << "\n--- handleMemref ---\n");
      LLVM_DEBUG(llvm::dbgs() << "Processing memref value: " << memVal << "\n");
      
      if (!memVal.getType().isa<MemRefType>()) {
        LLVM_DEBUG(llvm::dbgs() << "REJECTED: Not a MemRefType\n");
        return failure(); 
      }

      MemRefType memrefType;
      if (auto blockArg = memVal.dyn_cast<BlockArgument>()) {
        LLVM_DEBUG(llvm::dbgs() << "  Getting MemRefType from BlockArgument\n");
        memrefType = blockArg.getType().dyn_cast<MemRefType>();
      } else if (auto allocaOp = memVal.getDefiningOp<memref::AllocaOp>()) {
        LLVM_DEBUG(llvm::dbgs() << "  Getting MemRefType from AllocaOp\n");
        memrefType = allocaOp.getType();
      } else if (auto allocOp = memVal.getDefiningOp<memref::AllocOp>()) {
        LLVM_DEBUG(llvm::dbgs() << "  Getting MemRefType from AllocOp\n");
        memrefType = allocOp.getType();
      } else {
        LLVM_DEBUG(llvm::dbgs() << "REJECTED: Cannot determine MemRefType\n");
        return failure();
      }
      
      LLVM_DEBUG(llvm::dbgs() << "  MemRefType: " << memrefType << "\n");
      
      // Get all memory users (including those through subview/submap chains)
      auto sortedUsers = getAllMemoryUsers(memVal);
      
      LLVM_DEBUG(llvm::dbgs() << "  Found " << sortedUsers.size() << " memory users (including through submap/subview)\n");
      for (size_t i = 0; i < sortedUsers.size(); i++) {
        LLVM_DEBUG(llvm::dbgs() << "    User " << i << ": " << *sortedUsers[i] << "\n");
      }

      // If no memory users found, nothing to debufferize
      if (sortedUsers.empty()) {
        LLVM_DEBUG(llvm::dbgs() << "REJECTED: No memory users found\n");
        return failure();
      }
      
      // Initialize: Create tensor from memref
      rewriter.setInsertionPointAfterValue(memVal);
      auto tensorType = RankedTensorType::get(
          memrefType.getShape(), memrefType.getElementType());
      
      LLVM_DEBUG(llvm::dbgs() << "  Creating bufferization.to_tensor\n");
      auto toTensorOp = rewriter.create<bufferization::ToTensorOp>(
          memVal.getLoc(), tensorType, memVal);
      
      // CurrentSlices: Map from root memref to current tensor state
      // For now we only track one root memref at a time
      llvm::DenseMap<Value, Value> CurrentSlices;
      CurrentSlices[memVal] = toTensorOp.getResult();
      
      LLVM_DEBUG(llvm::dbgs() << "  ToTensorOp created: " << toTensorOp << "\n");
      LLVM_DEBUG(llvm::dbgs() << "  CurrentSlices[" << memVal << "] = " << CurrentSlices[memVal] << "\n");

      // For region propagation (existing logic)
      llvm::DenseMap<Operation*, opTuple> opResultMap;
      llvm::DenseMap<scf::IfOp, PendingIfInfo> pendingIfs;  // Track pending if yields
      std::vector<Operation *> expandedUserList(sortedUsers);
      Value currentTensor = CurrentSlices[memVal];
      
      int userIdx = 0;
      LLVM_DEBUG(llvm::dbgs() << "\n  Processing " << sortedUsers.size() << " users:\n");
      
      // Tree-based tensor tracking: each region has its own tensor state
      // This prevents sibling regions from polluting each other
      llvm::DenseMap<Region*, RegionTensorState> regionTensorTree;
      
      // Initialize the function body region with the initial tensor
      regionTensorTree[&funcOp.getBody()] = RegionTensorState{currentTensor, true};
      
      Region* lastUserRegion = nullptr;
      Operation* lastUser = nullptr;
      
      for (auto user : sortedUsers) {
        LLVM_DEBUG(llvm::dbgs() << "\n  [User " << userIdx << "] Processing: " << user->getName() << " at " << user->getLoc() << "\n");
        
        // Check if we're entering a new region
        Region* userRegion = user->getParentRegion();
        if (lastUserRegion != userRegion) {
          LLVM_DEBUG(llvm::dbgs() << "    Region changed! Using tree-based tensor lookup...\n");
          
          // STEP 1: Detect ifs we're EXITING (to update parent regions)
          if (lastUser) {
            auto oldContainingIfs = getContainingIfs(lastUser, &funcOp.getBody());
            auto newContainingIfs = getContainingIfs(user, &funcOp.getBody());
            
            // Convert new containing ifs to a set for fast lookup
            llvm::DenseSet<scf::IfOp> newIfsSet;
            for (auto ifOp : newContainingIfs) {
              newIfsSet.insert(ifOp);
            }
            
            // Check which ifs we're leaving (in old but not in new)
            // Process innermost first (oldContainingIfs is already innermost-first)
            for (auto oldIf : oldContainingIfs) {
              if (!newIfsSet.contains(oldIf)) {
                // We're exiting this if! Update its parent region
                LLVM_DEBUG(llvm::dbgs() << "    Exiting if at " << oldIf.getLoc() << "\n");
                
                // Get the branch we were in
                Region* oldThenRegion = &oldIf.getThenRegion();
                Region* oldElseRegion = &oldIf.getElseRegion();
                
                // Get the tensor value from the branch we're leaving
                auto thenIt = regionTensorTree.find(oldThenRegion);
                auto elseIt = regionTensorTree.find(oldElseRegion);
                
                if (thenIt != regionTensorTree.end() && thenIt->second.valid) {
                  // We were in THEN branch - update pendingIfs
                  auto pendingIt = pendingIfs.find(oldIf);
                  if (pendingIt != pendingIfs.end()) {
                    pendingIt->second.thenResult = thenIt->second.tensor;
                    pendingIt->second.thenProcessed = true;
                    LLVM_DEBUG(llvm::dbgs() << "      Updated THEN result on exit\n");
                  }
                } else if (elseIt != regionTensorTree.end() && elseIt->second.valid) {
                  // We were in ELSE branch - update pendingIfs  
                  auto pendingIt = pendingIfs.find(oldIf);
                  if (pendingIt != pendingIfs.end()) {
                    pendingIt->second.elseResult = elseIt->second.tensor;
                    pendingIt->second.elseProcessed = true;
                    LLVM_DEBUG(llvm::dbgs() << "      Updated ELSE result on exit\n");
                  }
                }
                
                // MERGE PHASE 2 INTO PHASE 1: If exiting a function-body-level if,
                // rebuild it immediately so sibling ifs get the correct entry tensor
                Region* parentRegion = oldIf->getParentRegion();
                if (parentRegion == &funcOp.getBody()) {
                  LLVM_DEBUG(llvm::dbgs() << "      Function-body if - rebuilding immediately\n");
                  
                  auto pendingIt = pendingIfs.find(oldIf);
                  if (pendingIt != pendingIfs.end()) {
                    // Build regions list containing just the parent region
                    SmallVector<Region*> exitRegions;
                    exitRegions.push_back(parentRegion);
                    
                    // Get entry tensor for this if
                    Value entryTensor = pendingIt->second.entryTensor;
                    
                    // Rebuild the if with yields
                    propagateValueThroughRegion(entryTensor, exitRegions, expandedUserList, opResultMap, rewriter, pendingIfs);
                    
                    // Find the rebuilt if and update currentTensor
                    for (auto& op : funcOp.getBody().front()) {
                      if (auto newIf = dyn_cast<scf::IfOp>(&op)) {
                        if (newIf.getNumResults() > 0 && newIf.getLoc() == oldIf.getLoc()) {
                          currentTensor = newIf.getResult(newIf.getNumResults() - 1);
                          regionTensorTree[parentRegion] = RegionTensorState{currentTensor, true};
                          LLVM_DEBUG(llvm::dbgs() << "      Updated currentTensor from rebuilt if result\n");
                          break;
                        }
                      }
                    }
                  }
                } else {
                  // For nested ifs, just update the parent region tensor
                  auto pendingIt = pendingIfs.find(oldIf);
                  if (pendingIt != pendingIfs.end()) {
                    if (thenIt != regionTensorTree.end() && thenIt->second.valid) {
                      regionTensorTree[parentRegion] = RegionTensorState{thenIt->second.tensor, true};
                    } else if (elseIt != regionTensorTree.end() && elseIt->second.valid) {
                      regionTensorTree[parentRegion] = RegionTensorState{elseIt->second.tensor, true};
                    }
                    LLVM_DEBUG(llvm::dbgs() << "      Updated parent region tensor on exit\n");
                  }
                }
              }
            }
          }
          
          // STEP 2: Get the correct tensor for the new region from the tree
          // This traces up to the parent region, avoiding sibling pollution
          Region* parentRegion = userRegion;
          // Find the parent region that has a valid tensor (go up the tree)
          currentTensor = getCurrentTensorForRegion(parentRegion, regionTensorTree, CurrentSlices[memVal]);
          LLVM_DEBUG(llvm::dbgs() << "    Got tensor from tree for current region: " << currentTensor << "\n");
          
          // STEP 3: Set up entry tensor for any new ifs we're entering
          auto containingIfs = getContainingIfs(user, &funcOp.getBody());
          
          // Process outermost first
          for (auto it = containingIfs.rbegin(); it != containingIfs.rend(); ++it) {
            scf::IfOp ifOp = *it;
            Region* thenRegion = &ifOp.getThenRegion();
            Region* elseRegion = &ifOp.getElseRegion();
            
            // Check if we're entering this if's THEN branch for the first time
            if (thenRegion->isAncestor(userRegion)) {
              auto thenIt = regionTensorTree.find(thenRegion);
              if (thenIt == regionTensorTree.end() || !thenIt->second.valid) {
                // First time entering THEN - get tensor from PARENT region (not currentTensor!)
                Region* ifParentRegion = ifOp->getParentRegion();
                Value entryTensor = getCurrentTensorForRegion(ifParentRegion, regionTensorTree, CurrentSlices[memVal]);
                
                regionTensorTree[thenRegion] = RegionTensorState{entryTensor, true};
                currentTensor = entryTensor;
                
                // Set up PendingIfInfo if not exists
                if (pendingIfs.find(ifOp) == pendingIfs.end()) {
                  PendingIfInfo info;
                  info.ifOp = ifOp;
                  info.entryTensor = entryTensor;
                  info.thenResult = entryTensor;
                  info.elseResult = entryTensor;
                  pendingIfs[ifOp] = info;
                  LLVM_DEBUG(llvm::dbgs() << "    Created PendingIfInfo for if at " << ifOp.getLoc() << " with entry: " << entryTensor << "\n");
                }
                LLVM_DEBUG(llvm::dbgs() << "    Entering THEN branch of if at " << ifOp.getLoc() << " with tensor: " << entryTensor << "\n");
              }
            }
            // Check if we're entering this if's ELSE branch for the first time  
            else if (elseRegion->isAncestor(userRegion)) {
              auto elseIt = regionTensorTree.find(elseRegion);
              if (elseIt == regionTensorTree.end() || !elseIt->second.valid) {
                // First time entering ELSE - get tensor from PARENT region
                Region* ifParentRegion = ifOp->getParentRegion();
                Value entryTensor = getCurrentTensorForRegion(ifParentRegion, regionTensorTree, CurrentSlices[memVal]);
                
                regionTensorTree[elseRegion] = RegionTensorState{entryTensor, true};
                currentTensor = entryTensor;
                
                // Set up PendingIfInfo if not exists
                if (pendingIfs.find(ifOp) == pendingIfs.end()) {
                  PendingIfInfo info;
                  info.ifOp = ifOp;
                  info.entryTensor = entryTensor;
                  info.thenResult = entryTensor;
                  info.elseResult = entryTensor;
                  pendingIfs[ifOp] = info;
                  LLVM_DEBUG(llvm::dbgs() << "    Created PendingIfInfo for if at " << ifOp.getLoc() << " with entry: " << entryTensor << "\n");
                }
                LLVM_DEBUG(llvm::dbgs() << "    Entering ELSE branch of if at " << ifOp.getLoc() << " with tensor: " << entryTensor << "\n");
              }
            }
          }
          
          lastUserRegion = userRegion;
          LLVM_DEBUG(llvm::dbgs() << "    After region transition, currentTensor: " << currentTensor << "\n");
        }
        
        lastUser = user;
        
        //=== SubmapOp: NOOP ===
        if (auto submapOp = dyn_cast<polygeist::SubmapOp>(user)) {
          LLVM_DEBUG(llvm::dbgs() << "    Detected polygeist.submap - NOOP\n");
          LLVM_DEBUG(llvm::dbgs() << "    (Will use submap/submapInverse when we hit linalg.generic)\n");
          userIdx++;
          continue;
        }
        
        //=== LoadOp: direct extract from root tensor ===
        else if (auto loadOp = dyn_cast<memref::LoadOp>(user)) {
          LLVM_DEBUG(llvm::dbgs() << "    Detected memref.load\n");
          
          Value loadMemref = loadOp.getMemRef();
          
          // Only handle direct loads from the root memref
          Value rootTensor = CurrentSlices[loadMemref];
          if (!rootTensor) {
            LLVM_DEBUG(llvm::dbgs() << "    ERROR: No tensor for memref\n");
            userIdx++;
            continue;
          }
          
          rewriter.setInsertionPoint(loadOp);
          
          // Create tensor.extract with the load indices
          auto extractOp = rewriter.create<tensor::ExtractOp>(
              loadOp.getLoc(), rootTensor, loadOp.getIndices());
          
          LLVM_DEBUG(llvm::dbgs() << "    Created tensor.extract: " << extractOp << "\n");
          
          // Replace load result with extract result
          loadOp.getResult().replaceAllUsesWith(extractOp.getResult());
          rewriter.eraseOp(loadOp);
          
          LLVM_DEBUG(llvm::dbgs() << "    Erased original load, load->extract complete\n");
        }
        
        //=== StoreOp: direct insert into root tensor ===
        else if (auto storeOp = dyn_cast<memref::StoreOp>(user)) {
          LLVM_DEBUG(llvm::dbgs() << "    Detected memref.store\n");
          
          Value storeMemref = storeOp.getMemRef();
          Value valueToStore = storeOp.getValueToStore();
          
          // Only handle direct stores to the root memref
          Value rootTensor = CurrentSlices[storeMemref];
          if (!rootTensor) {
            LLVM_DEBUG(llvm::dbgs() << "    ERROR: No tensor for memref\n");
            userIdx++;
            continue;
          }
          
          rewriter.setInsertionPoint(storeOp);
          
          // Create tensor.insert to produce new tensor
          auto insertOp = rewriter.create<tensor::InsertOp>(
              storeOp.getLoc(), valueToStore, rootTensor, storeOp.getIndices());
          
          LLVM_DEBUG(llvm::dbgs() << "    Created tensor.insert: " << insertOp << "\n");
          
          // Update CurrentSlices - this is the key for SSA semantics!
          CurrentSlices[storeMemref] = insertOp.getResult();
          currentTensor = insertOp.getResult();
          
          // Update the region tensor tree for correct scoping
          regionTensorTree[user->getParentRegion()] = RegionTensorState{currentTensor, true};
          
          LLVM_DEBUG(llvm::dbgs() << "    Updated CurrentSlices[root] = " << insertOp.getResult() << "\n");
          
          // Record this tensor for containing if branches
          recordBranchResult(user, currentTensor, pendingIfs, &funcOp.getBody());
          
          rewriter.eraseOp(storeOp);
          
          LLVM_DEBUG(llvm::dbgs() << "    Erased original store, store->insert complete\n");
        }
        
        //=== AffineLoadOp: apply affine map, then extract ===
        else if (auto affineLoadOp = dyn_cast<affine::AffineLoadOp>(user)) {
          LLVM_DEBUG(llvm::dbgs() << "    Detected affine.load\n");
          
          Value loadMemref = affineLoadOp.getMemRef();
          
          // Only handle direct loads from the root memref
          Value rootTensor = CurrentSlices[loadMemref];
          if (!rootTensor) {
            LLVM_DEBUG(llvm::dbgs() << "    ERROR: No tensor for memref\n");
            userIdx++;
            continue;
          }
          
          rewriter.setInsertionPoint(affineLoadOp);
          AffineMap map = affineLoadOp.getAffineMap();
          SmallVector<Value> mapOperands(affineLoadOp.getMapOperands());
          
          // Apply affine map to get actual indices
          SmallVector<Value> affineIndices;
          for (unsigned i = 0; i < map.getNumResults(); ++i) {
            auto applyOp = rewriter.create<affine::AffineApplyOp>(
                affineLoadOp.getLoc(), map.getSubMap({i}), mapOperands);
            affineIndices.push_back(applyOp.getResult());
          }
          
          // Create tensor.extract
          auto extractOp = rewriter.create<tensor::ExtractOp>(
              affineLoadOp.getLoc(), rootTensor, affineIndices);
          
          affineLoadOp.getResult().replaceAllUsesWith(extractOp.getResult());
          rewriter.eraseOp(affineLoadOp);
          
          LLVM_DEBUG(llvm::dbgs() << "    affine.load -> tensor.extract complete\n");
        }
        
        //=== AffineStoreOp: apply affine map, then insert ===
        else if (auto affineStoreOp = dyn_cast<affine::AffineStoreOp>(user)) {
          LLVM_DEBUG(llvm::dbgs() << "    Detected affine.store\n");
          
          Value storeMemref = affineStoreOp.getMemRef();
          Value valueToStore = affineStoreOp.getValueToStore();
          
          // Only handle direct stores to the root memref
          Value rootTensor = CurrentSlices[storeMemref];
          if (!rootTensor) {
            LLVM_DEBUG(llvm::dbgs() << "    ERROR: No tensor for memref\n");
            userIdx++;
            continue;
          }
          
          // Apply affine map to get actual indices
          rewriter.setInsertionPoint(affineStoreOp);
          AffineMap map = affineStoreOp.getAffineMap();
          SmallVector<Value> mapOperands(affineStoreOp.getMapOperands());
          
          SmallVector<Value> affineIndices;
          for (unsigned i = 0; i < map.getNumResults(); ++i) {
            auto applyOp = rewriter.create<affine::AffineApplyOp>(
                affineStoreOp.getLoc(), map.getSubMap({i}), mapOperands);
            affineIndices.push_back(applyOp.getResult());
          }
          
          // Create tensor.insert
          auto insertOp = rewriter.create<tensor::InsertOp>(
              affineStoreOp.getLoc(), valueToStore, rootTensor, affineIndices);
          
          // Update CurrentSlices
          CurrentSlices[storeMemref] = insertOp.getResult();
          currentTensor = insertOp.getResult();
          
          // Update the region tensor tree for correct scoping
          regionTensorTree[user->getParentRegion()] = RegionTensorState{currentTensor, true};
          
          // Record this tensor for containing if branches
          recordBranchResult(user, currentTensor, pendingIfs, &funcOp.getBody());
          
          rewriter.eraseOp(affineStoreOp);
          
          LLVM_DEBUG(llvm::dbgs() << "    affine.store -> tensor.insert complete\n");
        }
        
        //=== LinalgGenericOp: submap for inputs, submapInverse for outputs ===
        else if (auto genericOp = dyn_cast<linalg::GenericOp>(user)) {
          LLVM_DEBUG(llvm::dbgs() << "    Detected linalg.generic\n");

          // Handle region propagation for SSA value availability
          auto commonRegion = findCommonAncestorRegion(currentTensor.getDefiningOp(), user);
          if (!commonRegion) {
            LLVM_DEBUG(llvm::dbgs() << "    ERROR: No common region found\n");
            return failure();
          }
          
          SmallVector<Region*> regions;
          for (Region* r = currentTensor.getParentRegion(); r != commonRegion; 
               r = r->getParentOp()->getParentRegion()) {
              regions.push_back(r);
          }
          
          if (!regions.empty()) {
            propagateValueThroughRegion(currentTensor, regions, expandedUserList, opResultMap, rewriter, pendingIfs);
          }

          SmallVector<Value, 4> newInputs;
          SmallVector<Value, 4> newOutputs;
          SmallVector<Type> resultTypes;
          
          // Set insertion point BEFORE the generic to create submap ops for inputs/outputs
          rewriter.setInsertionPoint(genericOp);
          
          // Process inputs
          for (auto input : genericOp.getInputs()) {
            if (input == memVal) {
              // Direct use of root memref
              newInputs.push_back(currentTensor);
            } else if (auto inputMemref = input.getType().dyn_cast<MemRefType>()) {
              // Check if this input traces back to our root through submap chain
              SubmapChainInfo chain = traceSubmapChainToRoot(input);
              if (chain.rootMemref == memVal && !chain.isEmpty()) {
                // Input is through a submap chain - use submap
                Location loc = genericOp.getLoc();
                auto lastSubmap = chain.submaps.back();
                AffineMap map = lastSubmap.getMap();
                SmallVector<Value> submapOperands(lastSubmap.getIndicesAndSizes());
                
                RankedTensorType sliceTensorType = getSubmapChainTensorType(chain);
                
                auto submapOp = rewriter.create<polygeist::SubmapOp>(
                    loc, sliceTensorType, currentTensor, submapOperands, map);
                
                newInputs.push_back(submapOp.getResult());
                LLVM_DEBUG(llvm::dbgs() << "    Created submap for input: " << submapOp << "\n");
              } else {
                newInputs.push_back(input);
              }
            } else {
              newInputs.push_back(input);
            }
          }

          // Process outputs
          int newCurrentTensorIndex = -1;
          int index = 0;
          SmallVector<SubmapChainInfo, 4> outputChains;
          
          for (auto output : genericOp.getOutputs()) {
            if (output == memVal) {
              // Direct use of root memref
              newOutputs.push_back(currentTensor);
              resultTypes.push_back(currentTensor.getType());
              newCurrentTensorIndex = index;
              outputChains.push_back(SubmapChainInfo{memVal, {}});
            } else if (auto outputMemref = output.getType().dyn_cast<MemRefType>()) {
              // Check if this output traces back to our root through submap chain
              SubmapChainInfo chain = traceSubmapChainToRoot(output);
              if (chain.rootMemref == memVal && !chain.isEmpty()) {
                // Output is through a submap chain - need submap for init value
                Location loc = genericOp.getLoc();
                auto lastSubmap = chain.submaps.back();
                AffineMap map = lastSubmap.getMap();
                SmallVector<Value> submapOperands(lastSubmap.getIndicesAndSizes());
                
                RankedTensorType sliceTensorType = getSubmapChainTensorType(chain);
                
                auto submapOp = rewriter.create<polygeist::SubmapOp>(
                    loc, sliceTensorType, currentTensor, submapOperands, map);
                
                newOutputs.push_back(submapOp.getResult());
                resultTypes.push_back(sliceTensorType);
                newCurrentTensorIndex = index;
                outputChains.push_back(chain);
                LLVM_DEBUG(llvm::dbgs() << "    Created submap for output: " << submapOp << "\n");
              } else {
                newOutputs.push_back(output);
                resultTypes.push_back(output.getType());
                outputChains.push_back(SubmapChainInfo{});
              }
            } else {
              newOutputs.push_back(output);
              resultTypes.push_back(output.getType());
              outputChains.push_back(SubmapChainInfo{});
            }
            index++;
          }

          // Set insertion point AFTER the generic for new linalg.generic and submapInverse
          rewriter.setInsertionPointAfter(genericOp);
          StringAttr empty = StringAttr::get(genericOp.getContext());
          auto newGenericOp = rewriter.create<linalg::GenericOp>(
              genericOp.getLoc(), ArrayRef<Type>(resultTypes), newInputs, newOutputs,
              genericOp.getIndexingMaps(), genericOp.getIteratorTypes(), empty, empty);

          rewriter.cloneRegionBefore(genericOp.getRegion(),
                                     newGenericOp.getRegion(),
                                     newGenericOp.getRegion().end());

          // Handle outputs that need submapInverse
          Value finalTensor = currentTensor;
          for (unsigned i = 0; i < outputChains.size(); ++i) {
            const auto &chain = outputChains[i];
            if (chain.rootMemref && !chain.isEmpty()) {
              // Need to scatter this result back using submapInverse
              Location loc = genericOp.getLoc();
              auto lastSubmap = chain.submaps.back();
              AffineMap map = lastSubmap.getMap();
              SmallVector<Value> submapOperands(lastSubmap.getIndicesAndSizes());
              
              auto inverseOp = rewriter.create<polygeist::SubmapInverseOp>(
                  loc, finalTensor.getType(), finalTensor, 
                  newGenericOp.getResult(i), submapOperands, map);
              
              finalTensor = inverseOp.getResult();
              LLVM_DEBUG(llvm::dbgs() << "    Created submapInverse: " << inverseOp << "\n");
            } else if (chain.rootMemref == memVal) {
              // Direct output to root - use result directly
              finalTensor = newGenericOp.getResult(i);
            }
          }

          // Replace all uses of original generic op
          for (unsigned i = 0; i < genericOp->getNumResults(); ++i) {
            genericOp->getResult(i).replaceAllUsesWith(newGenericOp->getResult(i));
          }

          // Update CurrentSlices
          if (newCurrentTensorIndex != -1) {
            CurrentSlices[memVal] = finalTensor;
            currentTensor = finalTensor;
            opResultMap[newGenericOp] = std::make_tuple(finalTensor, currentTensor);
            
            // Update the region tensor tree for correct scoping
            regionTensorTree[user->getParentRegion()] = RegionTensorState{currentTensor, true};
            
            // Record this tensor for containing if branches
            recordBranchResult(user, currentTensor, pendingIfs, &funcOp.getBody());
          }

          rewriter.eraseOp(genericOp);
          
          // Update expandedUserList: replace old generic with new one
          if (userIdx < expandedUserList.size()) {
            expandedUserList[userIdx] = newGenericOp;
          }
          
          LLVM_DEBUG(llvm::dbgs() << "    linalg.generic transformation complete\n");
        }
        else {
          LLVM_DEBUG(llvm::dbgs() << "    Unknown user type (skipping): " << user->getName() << "\n");
        }
        userIdx++;
      }
      
      // Final propagation for yields
      LLVM_DEBUG(llvm::dbgs() << "\n  Finalizing: Adding yields for last use\n");
      auto commonRegion = findCommonAncestorRegion(currentTensor.getDefiningOp(), toTensorOp);
      if (!commonRegion) {
        LLVM_DEBUG(llvm::dbgs() << "  ERROR: No common region for final propagation\n");
        return failure();
      }
      
      SmallVector<Region*> regions;
      for (Region* r = currentTensor.getParentRegion(); r != commonRegion; 
           r = r->getParentOp()->getParentRegion()) {
          regions.push_back(r);
      }

      LLVM_DEBUG(llvm::dbgs() << "  Final propagation through " << regions.size() << " regions\n");
      propagateValueThroughRegion(currentTensor, regions, expandedUserList, opResultMap, rewriter, pendingIfs);

      // Only insert to_memref and copy if tensor was actually transformed
      if (currentTensor != toTensorOp.getResult()) {
        LLVM_DEBUG(llvm::dbgs() << "  Tensor was transformed, creating to_memref and copy\n");
        rewriter.setInsertionPointAfter(currentTensor.getDefiningOp());
        auto toMemrefOp = rewriter.create<bufferization::ToMemrefOp>(
            memVal.getLoc(), memrefType, currentTensor);
        LLVM_DEBUG(llvm::dbgs() << "  Created to_memref: " << toMemrefOp << "\n");
        auto copyOp = rewriter.create<memref::CopyOp>(memVal.getLoc(), toMemrefOp, memVal);
        LLVM_DEBUG(llvm::dbgs() << "  Created copy: " << copyOp << "\n");
      } else {
        LLVM_DEBUG(llvm::dbgs() << "  Tensor was NOT transformed\n");
      }
      
      LLVM_DEBUG(llvm::dbgs() << "handleMemref SUCCESS\n");
      LLVM_DEBUG(llvm::dbgs() << "=== IR after handleMemref ===\n");
      LLVM_DEBUG(funcOp.print(llvm::dbgs()));
      LLVM_DEBUG(llvm::dbgs() << "\n=== END IR after handleMemref ===\n\n");
      return success();
    };

    
    bool anySuccess = false;
    //Fix instead of walk, just get the list of allocaOp users, so that you can easily delete ops inside
    SmallVector<memref::AllocaOp> listOfAllocaOps;
    SmallVector<memref::AllocOp> listOfAllocOps;
    
    funcOp.walk([&](memref::AllocaOp alloca) {
      listOfAllocaOps.push_back(alloca);
    });
    //TODO: Adding allocOp for now, without alias check
    funcOp.walk([&](memref::AllocOp alloc) {
      listOfAllocOps.push_back(alloc);
    });
    
    LLVM_DEBUG(llvm::dbgs() << "\nProcessing " << listOfAllocaOps.size() << " AllocaOps\n");
    for (auto alloca : listOfAllocaOps) {
      LLVM_DEBUG(llvm::dbgs() << "Processing AllocaOp: " << alloca << "\n");
      anySuccess |= succeeded(handleMemref(alloca));
    }
    
    LLVM_DEBUG(llvm::dbgs() << "\nProcessing " << listOfAllocOps.size() << " AllocOps\n");
    for (auto alloc : listOfAllocOps) {
      LLVM_DEBUG(llvm::dbgs() << "Processing AllocOp: " << alloc << "\n");
      anySuccess |= succeeded(handleMemref(alloc));
    }

    LLVM_DEBUG(llvm::dbgs() << "\nProcessing " << funcOp.getNumArguments() << " function arguments\n");
    for(auto arg: funcOp.getArguments()){
      LLVM_DEBUG(llvm::dbgs() << "Processing argument: " << arg << "\n");
      anySuccess |= succeeded(handleMemref(arg));
    }
    
    passResult = anySuccess ? success() : failure();
    LLVM_DEBUG(llvm::dbgs() << "\n=== LinalgDebufferization " << (anySuccess ? "SUCCESS" : "FAILURE") << " ===\n\n");
    //for (Operation *op : opsToDelete) {
    //  op->erase();
    //}
    //opsToDelete.clear();

    return passResult;
  }
};

//===----------------------------------------------------------------------===//
// V2: Region-recursive debufferization
//===----------------------------------------------------------------------===//
//
// Design (see notes/polygeist_raise_to_linalg/linalg_debufferize_stress_survey.md):
// Per-root walk over the IR. A single SSA `currentTensor` flows through the
// recursion. Region-bearing ops (scf.for so far) are rebuilt with extra
// iter_args / yields when their body modifies the root, and the walk recurses
// inside. No flat user list; no per-region tensor tree; no pendingIfs.
//
// Stage 1: linear function-body scope.
// Stage 2: + scf.for (this commit).
// Future: scf.if, scf.while, affine.for, full submap-inverse chain.

namespace v2 {

// Does `v` transitively come from `root` via a chain of polygeist.submap ops?
static bool tracesToRoot(Value v, Value root) {
  while (true) {
    if (v == root) return true;
    if (auto sm = v.getDefiningOp<polygeist::SubmapOp>()) {
      v = sm.getViewSource();
      continue;
    }
    return false;
  }
}

// True if `op`'s ancestor chain up to a func::FuncOp consists only of
// region-bearing ops we know how to rebuild.
// Stage 5: scf.for + scf.if + affine.for + scf.while.
static bool ancestorsAreHandled(Operation *op) {
  Operation *parent = op->getParentOp();
  while (parent && !isa<func::FuncOp>(parent)) {
    if (!isa<scf::ForOp, scf::IfOp, affine::AffineForOp, scf::WhileOp>(parent))
      return false;
    parent = parent->getParentOp();
  }
  return true;
}

// Precondition: can we safely debufferize `root` end-to-end?
// All transitive memory users (through polygeist.submap) must be
// load/store/linalg.generic, each under only handled region-bearing
// ancestors. There must also be at least one such memory op (otherwise
// there's no work to do and re-firing the pattern would loop forever).
static bool canHandle(Value root) {
  SmallPtrSet<Operation *, 16> visited;
  SmallVector<Value, 4> worklist;
  worklist.push_back(root);
  bool hasMemoryOp = false;
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    for (Operation *user : v.getUsers()) {
      if (!visited.insert(user).second) continue;
      if (isa<memref::DeallocOp, bufferization::ToTensorOp,
              bufferization::ToMemrefOp, memref::CopyOp>(user))
        continue;
      if (isa<memref::LoadOp, memref::StoreOp,
              affine::AffineLoadOp, affine::AffineStoreOp,
              linalg::GenericOp>(user)) {
        if (!ancestorsAreHandled(user)) return false;
        hasMemoryOp = true;
        continue;
      }
      if (auto submap = dyn_cast<polygeist::SubmapOp>(user)) {
        worklist.push_back(submap.getResult());
        continue;
      }
      return false;
    }
  }
  return hasMemoryOp;
}

// Does anything inside `r` *write* to `root` (via store/affine.store/
// linalg.generic with root in outs)?
static bool regionWritesRoot(Region &r, Value root) {
  bool writes = false;
  r.walk([&](Operation *op) {
    if (writes) return WalkResult::interrupt();
    if (auto store = dyn_cast<memref::StoreOp>(op)) {
      if (tracesToRoot(store.getMemRef(), root)) writes = true;
    } else if (auto astore = dyn_cast<affine::AffineStoreOp>(op)) {
      if (tracesToRoot(astore.getMemRef(), root)) writes = true;
    } else if (auto generic = dyn_cast<linalg::GenericOp>(op)) {
      for (Value o : generic.getOutputs())
        if (o.getType().isa<MemRefType>() && tracesToRoot(o, root)) {
          writes = true;
          break;
        }
    }
    return writes ? WalkResult::interrupt() : WalkResult::advance();
  });
  return writes;
}

// Rebuild a submap chain on the tensor side, starting from `baseTensor`.
static Value buildTensorSubmapChain(Value baseTensor,
                                    const SubmapChainInfo &chain,
                                    PatternRewriter &rewriter) {
  Value t = baseTensor;
  for (auto submap : chain.submaps) {
    auto resMemref = submap.getResult().getType().cast<MemRefType>();
    auto resTensor = RankedTensorType::get(resMemref.getShape(),
                                           resMemref.getElementType());
    auto newSubmap = rewriter.create<polygeist::SubmapOp>(
        submap.getLoc(), resTensor, t,
        SmallVector<Value>(submap.getIndicesAndSizes()),
        submap.getMap());
    t = newSubmap.getResult();
  }
  return t;
}

// Scatter `sliceTensor` (at the leaf-view shape) all the way back into
// `baseTensor` (the root). For a chain [sm0, sm1, sm2]:
//   base[i] tensors:  bases[0]=baseTensor  (root)
//                     bases[1]=submap(bases[0], sm0)
//                     bases[2]=submap(bases[1], sm1)
//                     -- (the leaf view at depth 3 is sliceTensor's shape;
//                         we don't need a bases[3])
// Then unwind innermost-first:
//   bases[2]' = submapInverse(bases[2], sliceTensor, sm2.ops, sm2.map)
//   bases[1]' = submapInverse(bases[1], bases[2]',   sm1.ops, sm1.map)
//   bases[0]' = submapInverse(bases[0], bases[1]',   sm0.ops, sm0.map)
// Return bases[0]'.
static Value applySubmapInverseChain(Value baseTensor, Value sliceTensor,
                                     const SubmapChainInfo &chain,
                                     Location loc,
                                     PatternRewriter &rewriter) {
  if (chain.isEmpty()) return sliceTensor;

  // Build intermediate bases by applying chain forward, skipping the leaf
  // (whose "base output" is sliceTensor's domain).
  SmallVector<Value> bases;
  bases.push_back(baseTensor);
  for (size_t i = 0; i + 1 < chain.submaps.size(); ++i) {
    auto sm = chain.submaps[i];
    auto resMemref = sm.getResult().getType().cast<MemRefType>();
    auto resTensor = RankedTensorType::get(resMemref.getShape(),
                                           resMemref.getElementType());
    auto fwd = rewriter.create<polygeist::SubmapOp>(
        sm.getLoc(), resTensor, bases.back(),
        SmallVector<Value>(sm.getIndicesAndSizes()), sm.getMap());
    bases.push_back(fwd.getResult());
  }

  // Unwind: leaf first.
  Value current = sliceTensor;
  for (int i = static_cast<int>(chain.submaps.size()) - 1; i >= 0; --i) {
    auto sm = chain.submaps[i];
    Value base = bases[i];
    auto inv = rewriter.create<polygeist::SubmapInverseOp>(
        sm.getLoc(), base.getType(), base, current,
        SmallVector<Value>(sm.getIndicesAndSizes()), sm.getMap());
    current = inv.getResult();
  }
  return current;
}

// Forward declarations
struct WalkCtx;
static void walkBlock(WalkCtx &ctx, Block &block);
static void handleScfFor(WalkCtx &ctx, scf::ForOp forOp);
static void handleScfIf(WalkCtx &ctx, scf::IfOp ifOp);
static void handleAffineFor(WalkCtx &ctx, affine::AffineForOp forOp);
static void handleScfWhile(WalkCtx &ctx, scf::WhileOp whileOp);
static void rewriteLinalgGenericForRoot(WalkCtx &ctx, linalg::GenericOp generic);

// Per-root walk context. `didRewrite` flips true as soon as we mutate the IR
// (rewriting a load, store, or generic). It distinguishes the "we did
// something" case from the "current tensor reverted to entry" case, which
// matters for multi-root linalg.generics where we rewrite inputs but the
// output tensor flow stays unchanged.
struct WalkCtx {
  Value root;
  Value currentTensor;
  PatternRewriter *rewriter;
  bool didRewrite = false;
};

static void rewriteLinalgGenericForRoot(WalkCtx &ctx, linalg::GenericOp generic) {
  Value root = ctx.root;
  PatternRewriter &rewriter = *ctx.rewriter;
  rewriter.setInsertionPoint(generic);
  SmallVector<Value> newInputs, newOutputs;
  SmallVector<Type> resultTypes;
  int outRootIdx = -1;
  SubmapChainInfo outRootChain;

  auto routeOperand = [&](Value v) -> std::pair<Value, std::optional<SubmapChainInfo>> {
    if (v == root) return {ctx.currentTensor, SubmapChainInfo{root, {}}};
    if (!v.getType().isa<MemRefType>()) return {v, std::nullopt};
    SubmapChainInfo chain = traceSubmapChainToRoot(v);
    if (chain.rootMemref != root) return {v, std::nullopt};
    if (chain.isEmpty()) return {ctx.currentTensor, chain};
    return {buildTensorSubmapChain(ctx.currentTensor, chain, rewriter), chain};
  };

  for (Value in : generic.getInputs()) {
    auto [nv, _] = routeOperand(in);
    newInputs.push_back(nv);
  }
  int idx = 0;
  for (Value out : generic.getOutputs()) {
    auto [nv, chainOpt] = routeOperand(out);
    newOutputs.push_back(nv);
    resultTypes.push_back(nv.getType());
    if (chainOpt.has_value()) {
      outRootIdx = idx;
      outRootChain = *chainOpt;
    }
    ++idx;
  }

  rewriter.setInsertionPointAfter(generic);
  StringAttr empty = StringAttr::get(generic.getContext());
  auto newGeneric = rewriter.create<linalg::GenericOp>(
      generic.getLoc(), ArrayRef<Type>(resultTypes), newInputs, newOutputs,
      generic.getIndexingMaps(), generic.getIteratorTypes(), empty, empty);
  rewriter.cloneRegionBefore(generic.getRegion(), newGeneric.getRegion(),
                             newGeneric.getRegion().end());

  if (outRootIdx >= 0) {
    Value resultSlice = newGeneric.getResult(outRootIdx);
    if (outRootChain.isEmpty()) {
      ctx.currentTensor = resultSlice;
    } else {
      ctx.currentTensor = applySubmapInverseChain(
          ctx.currentTensor, resultSlice, outRootChain, generic.getLoc(), rewriter);
    }
  }

  for (auto [oldR, newR] : llvm::zip(generic.getResults(), newGeneric.getResults()))
    oldR.replaceAllUsesWith(newR);
  rewriter.eraseOp(generic);
}

static void handleScfFor(WalkCtx &ctx, scf::ForOp forOp) {
  PatternRewriter &rewriter = *ctx.rewriter;

  // Body only READS root → walk inline; currentTensor unchanged outside.
  // We still recurse to rewrite reads/sub-ops; the outer-scope tensor
  // dominates the body and is the right SSA value for them.
  if (!regionWritesRoot(forOp.getRegion(), ctx.root)) {
    Value saved = ctx.currentTensor;
    walkBlock(ctx, forOp.getRegion().front());
    ctx.currentTensor = saved;
    return;
  }

  // Body WRITES root → rebuild scf.for with one extra iter_arg carrying
  // the tensor for this root.
  rewriter.setInsertionPoint(forOp);
  SmallVector<Value> newInits(forOp.getInitArgs());
  newInits.push_back(ctx.currentTensor);

  auto newFor = rewriter.create<scf::ForOp>(
      forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
      forOp.getStep(), newInits);

  Block *oldBody = forOp.getBody();
  Block *newBody = newFor.getBody();

  // The newly-built scf.for body has a default terminator that the builder
  // inserted. Remove it so mergeBlocks can append the old body cleanly.
  if (!newBody->empty()) {
    Operation *term = newBody->getTerminator();
    rewriter.eraseOp(term);
  }
  // Map oldBody's [IV, iter_args...] block-args onto newBody's first N+1
  // arguments (everything except the trailing new tensor iter_arg).
  rewriter.mergeBlocks(oldBody, newBody, newBody->getArguments().drop_back());

  // Now walk the new body with currentTensor = the appended tensor iter_arg.
  Value entryTensor = newBody->getArguments().back();
  ctx.currentTensor = entryTensor;
  walkBlock(ctx, *newBody);

  // Append the inner-final tensor to the yield's operand list.
  auto yield = cast<scf::YieldOp>(newBody->getTerminator());
  SmallVector<Value> newYields(yield.getOperands());
  newYields.push_back(ctx.currentTensor);
  rewriter.setInsertionPoint(yield);
  rewriter.replaceOpWithNewOp<scf::YieldOp>(yield, newYields);

  // Rewire users of the old for's results to the new for's matching results.
  for (auto [oldR, newR] :
       llvm::zip(forOp.getResults(), newFor.getResults().drop_back()))
    oldR.replaceAllUsesWith(newR);
  rewriter.eraseOp(forOp);

  // The outer continuation should now see the new for's last result.
  ctx.currentTensor = newFor.getResults().back();
  ctx.didRewrite = true;
}

static void handleScfIf(WalkCtx &ctx, scf::IfOp ifOp) {
  PatternRewriter &rewriter = *ctx.rewriter;

  bool thenWrites = regionWritesRoot(ifOp.getThenRegion(), ctx.root);
  bool elseWrites = !ifOp.getElseRegion().empty() &&
                    regionWritesRoot(ifOp.getElseRegion(), ctx.root);

  // Neither branch writes → walk inline for reads only; currentTensor
  // unchanged because the outer-scope tensor dominates both branch bodies.
  if (!thenWrites && !elseWrites) {
    Value saved = ctx.currentTensor;
    if (!ifOp.getThenRegion().empty())
      walkBlock(ctx, ifOp.getThenRegion().front());
    ctx.currentTensor = saved;
    if (!ifOp.getElseRegion().empty())
      walkBlock(ctx, ifOp.getElseRegion().front());
    ctx.currentTensor = saved;
    return;
  }

  // Rebuild scf.if with one extra tensor result for the root.
  Value entryTensor = ctx.currentTensor;
  SmallVector<Type> newResultTypes(ifOp.getResultTypes().begin(),
                                   ifOp.getResultTypes().end());
  newResultTypes.push_back(entryTensor.getType());

  rewriter.setInsertionPoint(ifOp);
  auto newIf = rewriter.create<scf::IfOp>(
      ifOp.getLoc(), newResultTypes, ifOp.getCondition(),
      /*withElseRegion=*/true);

  // THEN branch: splice old's contents into new's then block, then walk.
  Block *oldThen = &ifOp.getThenRegion().front();
  Block *newThen = &newIf.getThenRegion().front();
  if (!newThen->empty()) rewriter.eraseOp(newThen->getTerminator());
  rewriter.mergeBlocks(oldThen, newThen, /*argValues=*/{});

  ctx.currentTensor = entryTensor;
  walkBlock(ctx, *newThen);
  Value thenFinal = ctx.currentTensor;

  {
    auto thenYield = cast<scf::YieldOp>(newThen->getTerminator());
    SmallVector<Value> thenYields(thenYield.getOperands());
    thenYields.push_back(thenFinal);
    rewriter.setInsertionPoint(thenYield);
    rewriter.replaceOpWithNewOp<scf::YieldOp>(thenYield, thenYields);
  }

  // ELSE branch: either splice old's contents or synthesize "yield entry".
  Block *newElse = &newIf.getElseRegion().front();
  if (!ifOp.getElseRegion().empty()) {
    Block *oldElse = &ifOp.getElseRegion().front();
    if (!newElse->empty()) rewriter.eraseOp(newElse->getTerminator());
    rewriter.mergeBlocks(oldElse, newElse, /*argValues=*/{});

    ctx.currentTensor = entryTensor;
    walkBlock(ctx, *newElse);
    Value elseFinal = ctx.currentTensor;

    auto elseYield = cast<scf::YieldOp>(newElse->getTerminator());
    SmallVector<Value> elseYields(elseYield.getOperands());
    elseYields.push_back(elseFinal);
    rewriter.setInsertionPoint(elseYield);
    rewriter.replaceOpWithNewOp<scf::YieldOp>(elseYield, elseYields);
  } else {
    // Original had no else. Synthesize: yield the entry tensor unchanged.
    // newElse is non-empty: it contains a default empty yield op the
    // builder inserted. Replace it with one that yields entryTensor.
    SmallVector<Value> elseYields{entryTensor};
    if (!newElse->empty()) {
      auto elseYield = cast<scf::YieldOp>(newElse->getTerminator());
      rewriter.setInsertionPoint(elseYield);
      rewriter.replaceOpWithNewOp<scf::YieldOp>(elseYield, elseYields);
    } else {
      rewriter.setInsertionPointToEnd(newElse);
      rewriter.create<scf::YieldOp>(ifOp.getLoc(), elseYields);
    }
  }

  // Rewire old if's pre-existing results to the new if's matching ones.
  for (auto [oldR, newR] :
       llvm::zip(ifOp.getResults(), newIf.getResults().drop_back()))
    oldR.replaceAllUsesWith(newR);
  rewriter.eraseOp(ifOp);

  ctx.currentTensor = newIf.getResults().back();
  ctx.didRewrite = true;
}

static void handleAffineFor(WalkCtx &ctx, affine::AffineForOp forOp) {
  PatternRewriter &rewriter = *ctx.rewriter;

  if (!regionWritesRoot(forOp.getRegion(), ctx.root)) {
    Value saved = ctx.currentTensor;
    walkBlock(ctx, forOp.getRegion().front());
    ctx.currentTensor = saved;
    return;
  }

  rewriter.setInsertionPoint(forOp);
  SmallVector<Value> newInits(forOp.getInits());
  newInits.push_back(ctx.currentTensor);

  auto newFor = rewriter.create<affine::AffineForOp>(
      forOp.getLoc(), forOp.getLowerBoundOperands(), forOp.getLowerBoundMap(),
      forOp.getUpperBoundOperands(), forOp.getUpperBoundMap(),
      forOp.getStep(), newInits);

  Block *oldBody = forOp.getBody();
  Block *newBody = newFor.getBody();

  if (!newBody->empty()) {
    Operation *term = newBody->getTerminator();
    rewriter.eraseOp(term);
  }
  rewriter.mergeBlocks(oldBody, newBody, newBody->getArguments().drop_back());

  Value entryTensor = newBody->getArguments().back();
  ctx.currentTensor = entryTensor;
  walkBlock(ctx, *newBody);

  auto yield = cast<affine::AffineYieldOp>(newBody->getTerminator());
  SmallVector<Value> newYields(yield.getOperands());
  newYields.push_back(ctx.currentTensor);
  rewriter.setInsertionPoint(yield);
  rewriter.replaceOpWithNewOp<affine::AffineYieldOp>(yield, newYields);

  for (auto [oldR, newR] :
       llvm::zip(forOp.getResults(), newFor.getResults().drop_back()))
    oldR.replaceAllUsesWith(newR);
  rewriter.eraseOp(forOp);

  ctx.currentTensor = newFor.getResults().back();
  ctx.didRewrite = true;
}

static void handleScfWhile(WalkCtx &ctx, scf::WhileOp whileOp) {
  PatternRewriter &rewriter = *ctx.rewriter;

  bool beforeWrites = regionWritesRoot(whileOp.getBefore(), ctx.root);
  bool afterWrites = regionWritesRoot(whileOp.getAfter(), ctx.root);

  // Neither region writes → walk inline (just for reads).
  if (!beforeWrites && !afterWrites) {
    Value saved = ctx.currentTensor;
    if (!whileOp.getBefore().empty())
      walkBlock(ctx, whileOp.getBefore().front());
    ctx.currentTensor = saved;
    if (!whileOp.getAfter().empty())
      walkBlock(ctx, whileOp.getAfter().front());
    ctx.currentTensor = saved;
    return;
  }

  // Rebuild scf.while with one extra tensor iter_arg threaded through both
  // regions:
  //   - extra `before` block arg     (init = currentTensor)
  //   - extra scf.condition operand  (latest tensor in before)
  //   - extra `after` block arg      (carried from condition)
  //   - extra scf.yield operand      (latest tensor in after — feeds next iter)
  //   - extra scf.while result       (final tensor after loop exits)
  Value entryTensor = ctx.currentTensor;
  Type tensorType = entryTensor.getType();

  SmallVector<Value> newOperands(whileOp.getOperands());
  newOperands.push_back(entryTensor);

  SmallVector<Type> newResultTypes(whileOp.getResultTypes().begin(),
                                   whileOp.getResultTypes().end());
  newResultTypes.push_back(tensorType);

  rewriter.setInsertionPoint(whileOp);
  auto newWhile =
      rewriter.create<scf::WhileOp>(whileOp.getLoc(), newResultTypes,
                                    newOperands);

  // Build the before block manually (with the extra tensor arg appended).
  SmallVector<Type> beforeArgTypes(
      whileOp.getBefore().front().getArgumentTypes());
  beforeArgTypes.push_back(tensorType);
  SmallVector<Location> beforeArgLocs(beforeArgTypes.size(), whileOp.getLoc());
  Block *newBefore =
      rewriter.createBlock(&newWhile.getBefore(), {}, beforeArgTypes,
                           beforeArgLocs);

  Block *oldBefore = &whileOp.getBefore().front();
  rewriter.mergeBlocks(oldBefore, newBefore, newBefore->getArguments().drop_back());

  ctx.currentTensor = newBefore->getArguments().back();
  walkBlock(ctx, *newBefore);
  Value beforeFinal = ctx.currentTensor;

  // Replace scf.condition with one that carries the tensor too.
  auto cond = cast<scf::ConditionOp>(newBefore->getTerminator());
  SmallVector<Value> newCondArgs(cond.getArgs());
  newCondArgs.push_back(beforeFinal);
  rewriter.setInsertionPoint(cond);
  rewriter.replaceOpWithNewOp<scf::ConditionOp>(cond, cond.getCondition(),
                                                newCondArgs);

  // Build the after block manually too.
  SmallVector<Type> afterArgTypes(
      whileOp.getAfter().front().getArgumentTypes());
  afterArgTypes.push_back(tensorType);
  SmallVector<Location> afterArgLocs(afterArgTypes.size(), whileOp.getLoc());
  Block *newAfter =
      rewriter.createBlock(&newWhile.getAfter(), {}, afterArgTypes,
                           afterArgLocs);

  Block *oldAfter = &whileOp.getAfter().front();
  rewriter.mergeBlocks(oldAfter, newAfter, newAfter->getArguments().drop_back());

  ctx.currentTensor = newAfter->getArguments().back();
  walkBlock(ctx, *newAfter);
  Value afterFinal = ctx.currentTensor;

  // Replace scf.yield with one that yields the tensor too.
  auto yield = cast<scf::YieldOp>(newAfter->getTerminator());
  SmallVector<Value> newYields(yield.getOperands());
  newYields.push_back(afterFinal);
  rewriter.setInsertionPoint(yield);
  rewriter.replaceOpWithNewOp<scf::YieldOp>(yield, newYields);

  for (auto [oldR, newR] :
       llvm::zip(whileOp.getResults(), newWhile.getResults().drop_back()))
    oldR.replaceAllUsesWith(newR);
  rewriter.eraseOp(whileOp);

  ctx.currentTensor = newWhile.getResults().back();
  ctx.didRewrite = true;
}

static void walkBlock(WalkCtx &ctx, Block &block) {
  for (auto it = block.begin(), end = block.end(); it != end;) {
    Operation &op = *it++;

    if (auto load = dyn_cast<memref::LoadOp>(&op)) {
      if (load.getMemRef() == ctx.root) {
        ctx.rewriter->setInsertionPoint(load);
        auto extract = ctx.rewriter->create<tensor::ExtractOp>(
            load.getLoc(), ctx.currentTensor, load.getIndices());
        load.getResult().replaceAllUsesWith(extract.getResult());
        ctx.rewriter->eraseOp(load);
        ctx.didRewrite = true;
      }
    } else if (auto store = dyn_cast<memref::StoreOp>(&op)) {
      if (store.getMemRef() == ctx.root) {
        ctx.rewriter->setInsertionPoint(store);
        auto insert = ctx.rewriter->create<tensor::InsertOp>(
            store.getLoc(), store.getValueToStore(), ctx.currentTensor,
            store.getIndices());
        ctx.currentTensor = insert.getResult();
        ctx.rewriter->eraseOp(store);
        ctx.didRewrite = true;
      }
    } else if (auto aload = dyn_cast<affine::AffineLoadOp>(&op)) {
      if (aload.getMemRef() == ctx.root) {
        ctx.rewriter->setInsertionPoint(aload);
        AffineMap map = aload.getAffineMap();
        SmallVector<Value> mapOperands(aload.getMapOperands());
        SmallVector<Value> idx;
        for (unsigned i = 0; i < map.getNumResults(); ++i) {
          auto apply = ctx.rewriter->create<affine::AffineApplyOp>(
              aload.getLoc(), map.getSubMap({i}), mapOperands);
          idx.push_back(apply.getResult());
        }
        auto extract = ctx.rewriter->create<tensor::ExtractOp>(
            aload.getLoc(), ctx.currentTensor, idx);
        aload.getResult().replaceAllUsesWith(extract.getResult());
        ctx.rewriter->eraseOp(aload);
        ctx.didRewrite = true;
      }
    } else if (auto astore = dyn_cast<affine::AffineStoreOp>(&op)) {
      if (astore.getMemRef() == ctx.root) {
        ctx.rewriter->setInsertionPoint(astore);
        AffineMap map = astore.getAffineMap();
        SmallVector<Value> mapOperands(astore.getMapOperands());
        SmallVector<Value> idx;
        for (unsigned i = 0; i < map.getNumResults(); ++i) {
          auto apply = ctx.rewriter->create<affine::AffineApplyOp>(
              astore.getLoc(), map.getSubMap({i}), mapOperands);
          idx.push_back(apply.getResult());
        }
        auto insert = ctx.rewriter->create<tensor::InsertOp>(
            astore.getLoc(), astore.getValueToStore(), ctx.currentTensor, idx);
        ctx.currentTensor = insert.getResult();
        ctx.rewriter->eraseOp(astore);
        ctx.didRewrite = true;
      }
    } else if (auto generic = dyn_cast<linalg::GenericOp>(&op)) {
      // Rewrite only if this generic touches our root via in/out operands.
      bool touches = false;
      for (Value v : generic.getInputs()) {
        if (v.getType().isa<MemRefType>() &&
            traceSubmapChainToRoot(v).rootMemref == ctx.root) {
          touches = true;
          break;
        }
      }
      if (!touches) {
        for (Value v : generic.getOutputs()) {
          if (v.getType().isa<MemRefType>() &&
              traceSubmapChainToRoot(v).rootMemref == ctx.root) {
            touches = true;
            break;
          }
        }
      }
      if (touches) {
        rewriteLinalgGenericForRoot(ctx, generic);
        ctx.didRewrite = true;
      }
    } else if (isa<polygeist::SubmapOp>(&op)) {
      // NOOP — re-emitted at linalg.generic time.
    } else if (auto forOp = dyn_cast<scf::ForOp>(&op)) {
      handleScfFor(ctx, forOp);
    } else if (auto ifOp = dyn_cast<scf::IfOp>(&op)) {
      handleScfIf(ctx, ifOp);
    } else if (auto affFor = dyn_cast<affine::AffineForOp>(&op)) {
      handleAffineFor(ctx, affFor);
    } else if (auto whileOp = dyn_cast<scf::WhileOp>(&op)) {
      handleScfWhile(ctx, whileOp);
    }
    // Anything else: leave alone. canHandle has ensured no unsupported
    // op touches our root.
  }
}

static LogicalResult handleRoot(Value root, Block *body,
                                PatternRewriter &rewriter) {
  auto memrefType = root.getType().dyn_cast<MemRefType>();
  if (!memrefType) return failure();
  if (!canHandle(root)) return failure();

  rewriter.setInsertionPointAfterValue(root);
  auto tensorType = RankedTensorType::get(memrefType.getShape(),
                                          memrefType.getElementType());
  auto initT = rewriter.create<bufferization::ToTensorOp>(
      root.getLoc(), tensorType, root);
  Value initTensor = initT.getResult();

  WalkCtx ctx{root, initTensor, &rewriter};
  walkBlock(ctx, *body);

  if (!ctx.didRewrite) {
    // Nothing actually changed. Undo the speculative to_tensor — but only
    // if it has no uses (e.g. an input-only rewrite of a generic would
    // have wired tensor submaps to it, in which case didRewrite is true).
    if (initT.getResult().use_empty()) rewriter.eraseOp(initT);
    return failure();
  }

  // Write back if the current tensor diverged from the entry tensor.
  // If only reads (loads) or input-only generic rewrites happened, the
  // outer memref hasn't been logically modified — no copy needed.
  if (ctx.currentTensor != initTensor) {
    rewriter.setInsertionPointAfterValue(ctx.currentTensor);
    auto toMemref = rewriter.create<bufferization::ToMemrefOp>(
        root.getLoc(), memrefType, ctx.currentTensor);
    rewriter.create<memref::CopyOp>(root.getLoc(), toMemref, root);
  }
  return success();
}

} // namespace v2

struct LinalgDebufferizationRecursive : public OpRewritePattern<func::FuncOp> {
  using OpRewritePattern<func::FuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::FuncOp funcOp,
                                PatternRewriter &rewriter) const final {
    if (funcOp.isExternal() || funcOp.empty()) return failure();
    // Multi-block CFG isn't supported yet; future stages will follow cf.br.
    if (!llvm::hasSingleElement(funcOp.getBody())) return failure();
    Block *body = &funcOp.getBody().front();
    bool anyChanged = false;

    SmallVector<Value> roots;
    funcOp.walk([&](memref::AllocaOp op) { roots.push_back(op.getResult()); });
    funcOp.walk([&](memref::AllocOp op) { roots.push_back(op.getResult()); });
    for (auto arg : funcOp.getArguments())
      if (arg.getType().isa<MemRefType>()) roots.push_back(arg);

    for (Value root : roots) {
      if (succeeded(v2::handleRoot(root, body, rewriter)))
        anyChanged = true;
    }
    return anyChanged ? success() : failure();
  }
};

namespace {
struct LinalgDebufferize : public LinalgDebufferizeBase<LinalgDebufferize> {
  void runOnOperation() override;
};
} // namespace

void LinalgDebufferize::runOnOperation() {
  auto module = getOperation()->getParentOfType<ModuleOp>();
  RewritePatternSet patterns(&getContext());
  if (useRecursive) {
    patterns.insert<LinalgDebufferizationRecursive>(&getContext());
  } else {
    patterns.insert<LinalgDebufferization>(&getContext());
  }
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

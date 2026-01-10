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
  LLVM_DEBUG(llvm::dbgs() << "      recordBranchResult called for user: " << *user << "\n");
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
        LLVM_DEBUG(llvm::dbgs() << "          Old thenResult: " << info.thenResult << "\n");
        info.thenResult = newTensor;
        info.thenProcessed = true;
        LLVM_DEBUG(llvm::dbgs() << "          New thenResult: " << info.thenResult << "\n");
      } else if (isInIfElseBranch(user, ifOp)) {
        LLVM_DEBUG(llvm::dbgs() << "        Recording ELSE result for if at " << ifOp.getLoc() << "\n");
        LLVM_DEBUG(llvm::dbgs() << "          Old elseResult: " << info.elseResult << "\n");
        info.elseResult = newTensor;
        info.elseProcessed = true;
        LLVM_DEBUG(llvm::dbgs() << "          New elseResult: " << info.elseResult << "\n");
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
    allocaOp.getType().getElementType());

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
    LLVM_DEBUG(llvm::dbgs() << "      Processing region in: " << *region->getParentOp() << "\n");
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
          
          LLVM_DEBUG(llvm::dbgs() << "      PendingIfInfo state:\n");
          LLVM_DEBUG(llvm::dbgs() << "        entryTensor: " << info.entryTensor << "\n");
          LLVM_DEBUG(llvm::dbgs() << "        thenResult: " << info.thenResult << " (processed=" << info.thenProcessed << ")\n");
          LLVM_DEBUG(llvm::dbgs() << "        elseResult: " << info.elseResult << " (processed=" << info.elseProcessed << ")\n");
          
          // Use recorded values: if a branch was processed, use its result; otherwise use entry tensor
          thenValue = info.thenProcessed ? info.thenResult : entryTensor;
          elseValue = info.elseProcessed ? info.elseResult : entryTensor;
          
          LLVM_DEBUG(llvm::dbgs() << "      Final values - THEN: " << thenValue << ", ELSE: " << elseValue << "\n");
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
          
          LLVM_DEBUG(llvm::dbgs() << "      First time seeing if, using entry tensor for both: " << entryTensor << "\n");
        }
        
        initTensor = entryTensor;
        
        LLVM_DEBUG(llvm::dbgs() << "      Building new if with yields:\n");
        LLVM_DEBUG(llvm::dbgs() << "        THEN will yield: " << thenValue << "\n");
        LLVM_DEBUG(llvm::dbgs() << "        ELSE will yield: " << elseValue << "\n");
        LLVM_DEBUG(llvm::dbgs() << "        Entry tensor: " << entryTensor << "\n");
        
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
        
        LLVM_DEBUG(llvm::dbgs() << "      Created new if with result: " << currentValue << "\n");
        LLVM_DEBUG(llvm::dbgs() << "      New if: " << *newIf << "\n");
        
        // FIX: Update outer ifs to use this if's result instead of raw inner tensor values
        // This is critical for nested ifs - outer ifs should yield the inner if's RESULT,
        // not values defined inside the inner if (which wouldn't dominate the yield)
        for (auto& [outerIfOp, outerInfo] : pendingIfs) {
          if (outerIfOp == newIf) continue;  // Skip self
          
          // Check if newIf is nested inside outerIfOp
          if (outerIfOp.getThenRegion().isAncestor(newIf->getParentRegion())) {
            // newIf is in outer's THEN branch - outer should yield newIf's result
            LLVM_DEBUG(llvm::dbgs() << "      Updating outer if's THEN result to use inner if result\n");
            LLVM_DEBUG(llvm::dbgs() << "        Outer if at: " << outerIfOp.getLoc() << "\n");
            // Note: Don't print old thenResult - it might be a deleted Value
            outerInfo.thenResult = currentValue;
            outerInfo.thenProcessed = true;
            LLVM_DEBUG(llvm::dbgs() << "        New thenResult: " << outerInfo.thenResult << "\n");
          } else if (outerIfOp.getElseRegion().isAncestor(newIf->getParentRegion())) {
            // newIf is in outer's ELSE branch - outer should yield newIf's result
            LLVM_DEBUG(llvm::dbgs() << "      Updating outer if's ELSE result to use inner if result\n");
            LLVM_DEBUG(llvm::dbgs() << "        Outer if at: " << outerIfOp.getLoc() << "\n");
            // Note: Don't print old elseResult - it might be a deleted Value
            outerInfo.elseResult = currentValue;
            outerInfo.elseProcessed = true;
            LLVM_DEBUG(llvm::dbgs() << "        New elseResult: " << outerInfo.elseResult << "\n");
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
    LLVM_DEBUG(llvm::dbgs() << "  Unsupported user: " << *user << "\n");
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
        LLVM_DEBUG(llvm::dbgs() << "\n  [User " << userIdx << "] Processing: " << *user << "\n");
        
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
            for (auto oldIf : oldContainingIfs) {
              if (!newIfsSet.contains(oldIf)) {
                // We're exiting this if! Update its parent region
                LLVM_DEBUG(llvm::dbgs() << "    Exiting if at " << oldIf.getLoc() << "\n");
                
                // Get the branch we were in
                Region* oldThenRegion = &oldIf.getThenRegion();
                Region* oldElseRegion = &oldIf.getElseRegion();
                
                // Get the tensor value from the branch we're leaving
                Value exitTensor = currentTensor;
                auto thenIt = regionTensorTree.find(oldThenRegion);
                auto elseIt = regionTensorTree.find(oldElseRegion);
                
                if (thenIt != regionTensorTree.end() && thenIt->second.valid) {
                  // We were in THEN branch - update pendingIfs
                  auto pendingIt = pendingIfs.find(oldIf);
                  if (pendingIt != pendingIfs.end()) {
                    pendingIt->second.thenResult = thenIt->second.tensor;
                    pendingIt->second.thenProcessed = true;
                    LLVM_DEBUG(llvm::dbgs() << "      Updated THEN result on exit: " << thenIt->second.tensor << "\n");
                  }
                } else if (elseIt != regionTensorTree.end() && elseIt->second.valid) {
                  // We were in ELSE branch - update pendingIfs  
                  auto pendingIt = pendingIfs.find(oldIf);
                  if (pendingIt != pendingIfs.end()) {
                    pendingIt->second.elseResult = elseIt->second.tensor;
                    pendingIt->second.elseProcessed = true;
                    LLVM_DEBUG(llvm::dbgs() << "      Updated ELSE result on exit: " << elseIt->second.tensor << "\n");
                  }
                }
                
                // Update parent region's tensor to reflect we processed this if
                // For now, use the exit tensor value from the branch we left
                Region* parentRegion = oldIf->getParentRegion();
                auto pendingIt = pendingIfs.find(oldIf);
                if (pendingIt != pendingIfs.end()) {
                  // Use the appropriate branch result
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

namespace {
struct LinalgDebufferize : public LinalgDebufferizeBase<LinalgDebufferize> {
  void runOnOperation() override;
};
} // namespace

void LinalgDebufferize::runOnOperation() {
  auto module = getOperation()->getParentOfType<ModuleOp>();
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

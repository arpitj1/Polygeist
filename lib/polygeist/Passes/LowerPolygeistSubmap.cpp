#include "PassDetails.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "polygeist/Ops.h"
#include "polygeist/Passes/Passes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "lower-polygeist-submap"

using namespace mlir;
using namespace polygeist;

namespace {

// Compose pure-dim-bearing polygeist.submap operands of a linalg.generic into
// the linalg's indexing_maps and switch the operands to the submap bases.
// This is done per-linalg.generic (rather than per-submap) so we can verify
// the resulting indexing_maps collectively cover every iter dim — otherwise
// linalg's shape-to-loops inference becomes ill-defined.
//
// Eligible submaps: numSymbols == 0 AND every result expression contains at
// least one DimExpr (allows `d0`, `d0 + const`, etc.; rejects pure-symbol or
// pure-constant slots). Symbol-bearing or constant-only forms are handled by
// the Subview/ExtractSlice patterns separately.
struct ComposeSubmapIntoLinalgGeneric
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  static bool isComposable(SubmapOp s) {
    if (s.getMap().getNumSymbols() != 0) return false;
    for (AffineExpr e : s.getMap().getResults()) {
      bool foundDim = false;
      e.walk([&](AffineExpr sub) { if (sub.isa<AffineDimExpr>()) foundDim = true; });
      if (!foundDim) return false;
    }
    return true;
  }

  LogicalResult matchAndRewrite(linalg::GenericOp genOp,
                                PatternRewriter &rewriter) const final {
    // Identify operands defined by composable submaps.
    SmallVector<AffineMap> newIndexingMaps(genOp.getIndexingMapsArray());
    SmallVector<std::pair<unsigned, SubmapOp>> toRewrite;
    for (OpOperand &opd : genOp->getOpOperands()) {
      auto submap = opd.get().getDefiningOp<SubmapOp>();
      if (!submap) continue;
      if (!isComposable(submap)) continue;
      unsigned mapIdx = opd.getOperandNumber();
      newIndexingMaps[mapIdx] = submap.getMap().compose(newIndexingMaps[mapIdx]);
      toRewrite.emplace_back(mapIdx, submap);
    }
    if (toRewrite.empty()) return failure();

    // Check the new collective indexing_maps still cover every iter dim
    // (otherwise the linalg becomes ill-defined).
    unsigned numIterDims = genOp.getNumLoops();
    SmallVector<bool, 8> dimCovered(numIterDims, false);
    for (AffineMap m : newIndexingMaps) {
      for (AffineExpr e : m.getResults()) {
        e.walk([&](AffineExpr sub) {
          if (auto d = sub.dyn_cast<AffineDimExpr>())
            if (d.getPosition() < numIterDims)
              dimCovered[d.getPosition()] = true;
        });
      }
    }
    for (bool b : dimCovered) {
      if (!b) return failure();
    }

    // Apply: switch operands to bases, install new indexing_maps.
    for (auto &p : toRewrite) {
      genOp->setOperand(p.first, p.second.getBase());
    }
    genOp.setIndexingMapsAttr(rewriter.getAffineMapArrayAttr(newIndexingMaps));
    return success();
  }
};

// Lower polygeist.submap on a memref result, when the affine map has symbols,
// to an equivalent memref.subview. Each map result expression must be of one
// of the supported shapes:
//   - a pure DimExpr `d_k`            (identity slice on that view-dim)
//   - a pure SymbolExpr `s_k`         (fixed offset, rank-reduced dim)
//   - `s_k + d_j` (or `d_j + s_k`)    (offset + identity stride along view-dim j)
//
// More complex expressions (multiplications by constants, multiple symbols in
// one expression, etc.) are unsupported and the pattern fails. The current
// raise pass produces only these shapes for symbol-bearing submaps.
struct LowerSymbolBearingSubmapToSubview : public OpRewritePattern<SubmapOp> {
  using OpRewritePattern<SubmapOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SubmapOp submap,
                                PatternRewriter &rewriter) const final {
    AffineMap submapMap = submap.getMap();
    auto outTy = dyn_cast<MemRefType>(submap.getResult().getType());
    auto baseTy = dyn_cast<MemRefType>(submap.getBase().getType());
    if (!outTy || !baseTy) return failure();
    if (submapMap.getNumResults() != (unsigned)baseTy.getRank())
      return failure();
    // Skip cases ComposeSubmapIntoLinalgGeneric handles (pure DimExpr results
    // with no symbols). Anything with symbols, constants, or dim+constant
    // shifts falls here.
    bool anyNonPureDim = false;
    for (AffineExpr e : submapMap.getResults()) {
      if (!e.isa<AffineDimExpr>()) { anyNonPureDim = true; break; }
    }
    if (submapMap.getNumSymbols() == 0 && !anyNonPureDim) return failure();

    Location loc = submap.getLoc();
    ValueRange symbols = submap.getSymbols();
    ValueRange sizes = submap.getSizes();
    unsigned numViewDims = submapMap.getNumDims();

    // Parse each result expression of the submap's map. For each base-dim k,
    // determine (offset_k, size_k, stride_k) AND whether this base-dim is
    // contributed by a view-dim (i.e., it must appear in the output of the
    // subview) or is symbol-fixed (rank-reduced).
    SmallVector<OpFoldResult> offsets, subSizes, strides;
    // Track, for each view-dim, which base-dim it maps to (or -1).
    SmallVector<int64_t> viewDimToBaseDim(numViewDims, -1);

    OpFoldResult zeroAttr = rewriter.getIndexAttr(0);
    OpFoldResult oneAttr = rewriter.getIndexAttr(1);

    // Helper: classify each result expr into (offset, has-view-dim?, view-dim-idx).
    auto classify = [&](AffineExpr e, OpFoldResult &offset, bool &hasViewDim,
                        unsigned &viewDim) -> bool {
      // Pure SymbolExpr: fixed offset, no view-dim.
      if (auto s = e.dyn_cast<AffineSymbolExpr>()) {
        unsigned si = s.getPosition();
        if (si >= symbols.size()) return false;
        offset = symbols[si];
        hasViewDim = false;
        return true;
      }
      // Pure ConstantExpr: static offset, no view-dim.
      if (auto c = e.dyn_cast<AffineConstantExpr>()) {
        offset = rewriter.getIndexAttr(c.getValue());
        hasViewDim = false;
        return true;
      }
      // Pure DimExpr: identity slice, view-dim present, offset 0.
      if (auto d = e.dyn_cast<AffineDimExpr>()) {
        unsigned di = d.getPosition();
        if (di >= numViewDims) return false;
        offset = zeroAttr;
        hasViewDim = true;
        viewDim = di;
        return true;
      }
      // AffineBinaryOp Add: combinations of (Symbol|Constant) + Dim.
      if (auto add = e.dyn_cast<AffineBinaryOpExpr>()) {
        if (add.getKind() != AffineExprKind::Add) return false;
        AffineExpr lhs = add.getLHS();
        AffineExpr rhs = add.getRHS();
        AffineExpr dimSide;
        AffineExpr offExpr;
        if (lhs.isa<AffineDimExpr>()) {
          dimSide = lhs; offExpr = rhs;
        } else if (rhs.isa<AffineDimExpr>()) {
          dimSide = rhs; offExpr = lhs;
        } else {
          return false;
        }
        unsigned di = dimSide.cast<AffineDimExpr>().getPosition();
        if (di >= numViewDims) return false;
        // Offset side: must be a SymbolExpr or a ConstantExpr.
        if (auto s = offExpr.dyn_cast<AffineSymbolExpr>()) {
          unsigned si = s.getPosition();
          if (si >= symbols.size()) return false;
          offset = symbols[si];
        } else if (auto c = offExpr.dyn_cast<AffineConstantExpr>()) {
          offset = rewriter.getIndexAttr(c.getValue());
        } else {
          return false;
        }
        hasViewDim = true;
        viewDim = di;
        return true;
      }
      return false;
    };

    for (unsigned k = 0; k < submapMap.getNumResults(); ++k) {
      AffineExpr e = submapMap.getResult(k);
      OpFoldResult offset;
      bool hasViewDim;
      unsigned viewDim = 0;
      if (!classify(e, offset, hasViewDim, viewDim)) return failure();
      offsets.push_back(offset);
      if (hasViewDim) {
        if (viewDim >= sizes.size()) return failure();
        subSizes.push_back(sizes[viewDim]);
        strides.push_back(oneAttr);
        viewDimToBaseDim[viewDim] = k;
      } else {
        subSizes.push_back(oneAttr);
        strides.push_back(oneAttr);
      }
    }

    // Verify every view-dim is represented exactly once. If a view-dim isn't
    // represented in any output expression, this is a broadcast — handle in
    // a separate pass.
    for (unsigned j = 0; j < numViewDims; ++j)
      if (viewDimToBaseDim[j] == -1) return failure();

    // The output rank must equal the count of view-dim-bearing base-dims.
    // Otherwise the shape can't be expressed via a single rank-reducing
    // subview — bail.
    unsigned dimBearingBaseDims = 0;
    for (int64_t bk : viewDimToBaseDim)
      if (bk != -1) ++dimBearingBaseDims;
    if (dimBearingBaseDims != numViewDims) return failure();

    SmallVector<int64_t> resultShape(numViewDims, ShapedType::kDynamic);

    MemRefType inferredTy = cast<MemRefType>(
        memref::SubViewOp::inferRankReducedResultType(
            resultShape, baseTy, offsets, subSizes, strides));
    Value sub = rewriter.create<memref::SubViewOp>(
        loc, inferredTy, submap.getBase(), offsets, subSizes, strides);

    // If the inferred type matches the submap's result type exactly, we can
    // RAUW. Otherwise we need a cast.
    if (sub.getType() == outTy) {
      rewriter.replaceOp(submap, sub);
      return success();
    }
    Value casted = rewriter.create<memref::CastOp>(loc, outTy, sub);
    rewriter.replaceOp(submap, casted);
    return success();
  }
};

// Tensor variant of polygeist.submap is handled by replacing with
// tensor.extract_slice (analogous to memref.subview).
struct LowerSymbolBearingSubmapToExtractSlice
    : public OpRewritePattern<SubmapOp> {
  using OpRewritePattern<SubmapOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SubmapOp submap,
                                PatternRewriter &rewriter) const final {
    AffineMap submapMap = submap.getMap();
    auto outTy = dyn_cast<RankedTensorType>(submap.getResult().getType());
    auto baseTy = dyn_cast<RankedTensorType>(submap.getBase().getType());
    if (!outTy || !baseTy) return failure();
    if (submapMap.getNumResults() != (unsigned)baseTy.getRank())
      return failure();
    bool anyNonPureDim = false;
    for (AffineExpr e : submapMap.getResults()) {
      if (!e.isa<AffineDimExpr>()) { anyNonPureDim = true; break; }
    }
    if (submapMap.getNumSymbols() == 0 && !anyNonPureDim) return failure();

    Location loc = submap.getLoc();
    ValueRange symbols = submap.getSymbols();
    ValueRange sizes = submap.getSizes();
    unsigned numViewDims = submapMap.getNumDims();

    SmallVector<OpFoldResult> offsets, subSizes, strides;
    SmallVector<int64_t> viewDimToBaseDim(numViewDims, -1);
    OpFoldResult zeroAttr = rewriter.getIndexAttr(0);
    OpFoldResult oneAttr = rewriter.getIndexAttr(1);

    auto classify = [&](AffineExpr e, OpFoldResult &offset, bool &hasViewDim,
                        unsigned &viewDim) -> bool {
      if (auto s = e.dyn_cast<AffineSymbolExpr>()) {
        unsigned si = s.getPosition();
        if (si >= symbols.size()) return false;
        offset = symbols[si];
        hasViewDim = false;
        return true;
      }
      if (auto c = e.dyn_cast<AffineConstantExpr>()) {
        offset = rewriter.getIndexAttr(c.getValue());
        hasViewDim = false;
        return true;
      }
      if (auto d = e.dyn_cast<AffineDimExpr>()) {
        unsigned di = d.getPosition();
        if (di >= numViewDims) return false;
        offset = zeroAttr;
        hasViewDim = true;
        viewDim = di;
        return true;
      }
      if (auto add = e.dyn_cast<AffineBinaryOpExpr>()) {
        if (add.getKind() != AffineExprKind::Add) return false;
        AffineExpr lhs = add.getLHS(), rhs = add.getRHS();
        AffineExpr dimSide;
        AffineExpr offExpr;
        if (lhs.isa<AffineDimExpr>()) {
          dimSide = lhs; offExpr = rhs;
        } else if (rhs.isa<AffineDimExpr>()) {
          dimSide = rhs; offExpr = lhs;
        } else {
          return false;
        }
        unsigned di = dimSide.cast<AffineDimExpr>().getPosition();
        if (di >= numViewDims) return false;
        if (auto s = offExpr.dyn_cast<AffineSymbolExpr>()) {
          unsigned si = s.getPosition();
          if (si >= symbols.size()) return false;
          offset = symbols[si];
        } else if (auto c = offExpr.dyn_cast<AffineConstantExpr>()) {
          offset = rewriter.getIndexAttr(c.getValue());
        } else {
          return false;
        }
        hasViewDim = true;
        viewDim = di;
        return true;
      }
      return false;
    };

    for (unsigned k = 0; k < submapMap.getNumResults(); ++k) {
      AffineExpr e = submapMap.getResult(k);
      OpFoldResult offset;
      bool hasViewDim;
      unsigned viewDim = 0;
      if (!classify(e, offset, hasViewDim, viewDim)) return failure();
      offsets.push_back(offset);
      if (hasViewDim) {
        if (viewDim >= sizes.size()) return failure();
        subSizes.push_back(sizes[viewDim]);
        strides.push_back(oneAttr);
        viewDimToBaseDim[viewDim] = k;
      } else {
        subSizes.push_back(oneAttr);
        strides.push_back(oneAttr);
      }
    }
    for (unsigned j = 0; j < numViewDims; ++j)
      if (viewDimToBaseDim[j] == -1) return failure();
    unsigned dimBearingBaseDims = 0;
    for (int64_t bk : viewDimToBaseDim)
      if (bk != -1) ++dimBearingBaseDims;
    if (dimBearingBaseDims != numViewDims) return failure();

    SmallVector<int64_t> resultShape(numViewDims, ShapedType::kDynamic);
    auto inferredTy = RankedTensorType::get(resultShape, baseTy.getElementType());
    Value sliced = rewriter.create<tensor::ExtractSliceOp>(
        loc, inferredTy, submap.getBase(), offsets, subSizes, strides);
    if (sliced.getType() == outTy) {
      rewriter.replaceOp(submap, sliced);
      return success();
    }
    Value casted = rewriter.create<tensor::CastOp>(loc, outTy, sliced);
    rewriter.replaceOp(submap, casted);
    return success();
  }
};

// Lower polygeist.submapInverse on tensors to tensor.insert_slice.
// For memref form, submapInverse is conceptually a no-op (modifications are
// already in place via the view) — we replace it with its base operand.
struct LowerSubmapInverse : public OpRewritePattern<SubmapInverseOp> {
  using OpRewritePattern<SubmapInverseOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SubmapInverseOp inv,
                                PatternRewriter &rewriter) const final {
    Value base = inv.getBaseOriginal();
    Value view = inv.getViewModified();

    if (isa<MemRefType>(inv.getType())) {
      // For memref, the view's writes have already mutated the base. The
      // submapInverse simply returns the base.
      rewriter.replaceOp(inv, base);
      return success();
    }

    auto outTy = dyn_cast<RankedTensorType>(inv.getType());
    auto baseTy = dyn_cast<RankedTensorType>(base.getType());
    auto viewTy = dyn_cast<RankedTensorType>(view.getType());
    if (!outTy || !baseTy || !viewTy) return failure();

    AffineMap m = inv.getMap();
    if (m.getNumResults() != (unsigned)baseTy.getRank()) return failure();

    Location loc = inv.getLoc();
    ValueRange symbols = inv.getSymbols();
    ValueRange sizes = inv.getSizes();
    unsigned numViewDims = m.getNumDims();

    SmallVector<OpFoldResult> offsets, subSizes, strides;
    SmallVector<int64_t> viewDimSeen(numViewDims, 0);
    OpFoldResult zeroAttr = rewriter.getIndexAttr(0);
    OpFoldResult oneAttr = rewriter.getIndexAttr(1);

    auto classify = [&](AffineExpr e, OpFoldResult &offset, bool &hasViewDim,
                        unsigned &viewDim) -> bool {
      if (auto s = e.dyn_cast<AffineSymbolExpr>()) {
        unsigned si = s.getPosition();
        if (si >= symbols.size()) return false;
        offset = symbols[si];
        hasViewDim = false;
        return true;
      }
      if (auto c = e.dyn_cast<AffineConstantExpr>()) {
        offset = rewriter.getIndexAttr(c.getValue());
        hasViewDim = false;
        return true;
      }
      if (auto d = e.dyn_cast<AffineDimExpr>()) {
        unsigned di = d.getPosition();
        if (di >= numViewDims) return false;
        offset = zeroAttr;
        hasViewDim = true;
        viewDim = di;
        return true;
      }
      if (auto add = e.dyn_cast<AffineBinaryOpExpr>()) {
        if (add.getKind() != AffineExprKind::Add) return false;
        AffineExpr lhs = add.getLHS(), rhs = add.getRHS();
        AffineExpr dimSide;
        AffineExpr offExpr;
        if (lhs.isa<AffineDimExpr>()) {
          dimSide = lhs; offExpr = rhs;
        } else if (rhs.isa<AffineDimExpr>()) {
          dimSide = rhs; offExpr = lhs;
        } else {
          return false;
        }
        unsigned di = dimSide.cast<AffineDimExpr>().getPosition();
        if (di >= numViewDims) return false;
        if (auto s = offExpr.dyn_cast<AffineSymbolExpr>()) {
          unsigned si = s.getPosition();
          if (si >= symbols.size()) return false;
          offset = symbols[si];
        } else if (auto c = offExpr.dyn_cast<AffineConstantExpr>()) {
          offset = rewriter.getIndexAttr(c.getValue());
        } else {
          return false;
        }
        hasViewDim = true;
        viewDim = di;
        return true;
      }
      return false;
    };

    for (unsigned k = 0; k < m.getNumResults(); ++k) {
      AffineExpr e = m.getResult(k);
      OpFoldResult offset;
      bool hasViewDim;
      unsigned viewDim = 0;
      if (!classify(e, offset, hasViewDim, viewDim)) return failure();
      offsets.push_back(offset);
      if (hasViewDim) {
        if (viewDim >= sizes.size()) return failure();
        subSizes.push_back(sizes[viewDim]);
        strides.push_back(oneAttr);
        viewDimSeen[viewDim] = 1;
      } else {
        subSizes.push_back(oneAttr);
        strides.push_back(oneAttr);
      }
    }
    for (unsigned j = 0; j < numViewDims; ++j)
      if (!viewDimSeen[j]) return failure();

    // If the view's rank differs from the slice's rank (because of symbol-
    // only base-dims that rank-reduced on the way in), we need to reshape
    // the view to match. For now we only support the case where view's rank
    // equals the count of dim-bearing base-dims.
    unsigned numDimBearingBaseDims = 0;
    for (unsigned k = 0; k < m.getNumResults(); ++k)
      if (!m.getResult(k).isa<AffineSymbolExpr>())
        ++numDimBearingBaseDims;
    if (numDimBearingBaseDims != (unsigned)viewTy.getRank())
      return failure();

    Value result = rewriter.create<tensor::InsertSliceOp>(
        loc, view, base, offsets, subSizes, strides);
    rewriter.replaceOp(inv, result);
    return success();
  }
};

struct LowerPolygeistSubmapPass
    : public mlir::polygeist::LowerPolygeistSubmapBase<
          LowerPolygeistSubmapPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ComposeSubmapIntoLinalgGeneric,
                 LowerSymbolBearingSubmapToSubview,
                 LowerSymbolBearingSubmapToExtractSlice,
                 LowerSubmapInverse>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                             std::move(patterns)))) {
      // Some submaps remain — caller may want to know but it's not fatal.
    }
  }
};

} // anonymous namespace

namespace mlir {
namespace polygeist {
std::unique_ptr<Pass> createLowerPolygeistSubmapPass() {
  return std::make_unique<LowerPolygeistSubmapPass>();
}
} // namespace polygeist
} // namespace mlir

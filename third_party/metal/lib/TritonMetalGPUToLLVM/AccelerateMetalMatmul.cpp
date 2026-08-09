/// TritonMetalGPUAccelerateMatmul.cpp
/// Rewrites DotOp with BlockedEncoding to use MetalMfmaEncodingAttr,
/// enabling codegen via air.simdgroup_matrix_8x8_* intrinsics.
///
/// Each simdgroup (warp) handles one 8x8 output tile.  warpsPerCTA is
/// chosen so that warps_m * warps_n == numWarps and every tile fits inside
/// the block shape.

#include "TritonMetalGPUToLLVM/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include <utility>

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace mlir::triton {
#define GEN_PASS_DEF_TRITONMETALGPUACCELERATEMATMUL
#include "TritonMetalGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton

namespace mlir {

namespace {

SmallVector<unsigned, 3> planWarps(Operation *dotOp, ArrayRef<int64_t> shape,
                                   int numWarps) {
  std::pair<int64_t, int64_t> instrShape = {8, 8};
  auto rank = shape.size();
  // early exit for batched matmul
  if (rank == 3)
    return {static_cast<unsigned>(numWarps), 1, 1};

  // TODO add back special cases for chained dot ops

  // regular cases
  SmallVector<int64_t, 2> tensorShape = {shape[0], shape[1]};
  SmallVector<unsigned, 3> ret = {1, 1};
  do {
    if (ret[0] * ret[1] >= numWarps)
      break;
    if (tensorShape[0] / (instrShape.first * 2) / ret[0] >=
        tensorShape[1] / instrShape.second / ret[1]) {
      if (ret[0] < tensorShape[0] / instrShape.first) {
        ret[0] *= 2;
      } else {
        ret[1] *= 2;
      }
    } else {
      ret[1] *= 2;
    }
  } while (true);

  if (ret[1] * instrShape.second > tensorShape[1]) {
    return {ret[1], ret[0]};
  }

  return ret;
}

// kWidth: consecutive K elements each thread holds for one k-tile in the
// simdgroup_matrix_8x8 scheme.  With 32 threads per simdgroup and an 8×8
// instruction tile: 8*8/32 = 2 elements per thread per k-tile.
static constexpr unsigned kSimdgroupKWidth = 2;

static Value convertLayout(PatternRewriter &rewriter, Value value,
                           Attribute newEncoding) {
  auto oldType = cast<RankedTensorType>(value.getType());
  auto newType = RankedTensorType::get(oldType.getShape(),
                                       oldType.getElementType(), newEncoding);
  return ttg::ConvertLayoutOp::create(rewriter, value.getLoc(), newType, value);
}

class BlockedToMetalMFMA : public OpRewritePattern<tt::DotOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tt::DotOp dotOp,
                                PatternRewriter &rewriter) const override {
    RankedTensorType oldRetType = dotOp.getType();

    // only rewrite if result is BlockedEncoding
    if (!isa_and_nonnull<ttg::BlockedEncodingAttr>(oldRetType.getEncoding()))
      return rewriter.notifyMatchFailure(dotOp,
                                         "result is not BlockedEncoding");

    // support f16 or f32 for simdgroup_matrix_8x8
    Type elemType = oldRetType.getElementType();
    if (!elemType.isF16() && !elemType.isF32())
      return rewriter.notifyMatchFailure(
          dotOp, "element type must be f16 or f32 for Metal simdgroup matrix");

    // get encoding for given number of warps
    auto retShape = oldRetType.getShape(); // [M, N]
    if (retShape.size() != 2) {
      return rewriter.notifyMatchFailure(dotOp, "only 2D dot ops supported");
    }
    int numWarps = ttg::lookupNumWarps(dotOp);

    int64_t blockM = retShape[0];
    int64_t blockN = retShape[1];

    // require M and N divisible by 8
    // TODO need to relax this restriction later on?
    if (blockM % 8 != 0 || blockN % 8 != 0)
      return rewriter.notifyMatchFailure(
          dotOp, "BLOCK_M and BLOCK_N must be divisible by 8");

    auto aTensorTy = cast<RankedTensorType>(dotOp.getA().getType());
    int64_t blockK = aTensorTy.getShape()[1];
    if (blockK % 8 != 0) {
      return rewriter.notifyMatchFailure(dotOp,
                                         "BLOCK_K must be divisible by 8");
    }

    auto warpsPerTile = planWarps(dotOp, retShape, numWarps);

    auto *ctx = rewriter.getContext();
    auto CGALayout = ttg::getCGALayout(oldRetType.getEncoding());

    SmallVector<unsigned, 3> instrShape = {8u, 8u, 8u};
    auto mfmaEnc = ttg::MetalMfmaEncodingAttr::get(ctx, warpsPerTile, CGALayout,
                                                   instrShape);

    // BlockedEncoding -> MetalMfmaEncoding.
    auto oldAcc = dotOp.getC();
    auto newAcc = convertLayout(rewriter, oldAcc, mfmaEnc);

    // BlockedEncoding -> DotOperandEncoding
    auto newAEnc = ttg::DotOperandEncodingAttr::get(ctx, /*opIdx=*/0, mfmaEnc,
                                                    kSimdgroupKWidth);
    Value newA = convertLayout(rewriter, dotOp.getA(), newAEnc);
    auto newBEnc = ttg::DotOperandEncodingAttr::get(ctx, /*opIdx=*/1, mfmaEnc,
                                                    kSimdgroupKWidth);
    Value newB = convertLayout(rewriter, dotOp.getB(), newBEnc);

    auto newRetType = RankedTensorType::get(retShape, elemType, mfmaEnc);
    Value newDot = tt::DotOp::create(rewriter, dotOp.getLoc(), newRetType, newA,
                                     newB, newAcc, dotOp.getInputPrecision(),
                                     dotOp.getMaxNumImpreciseAcc());

    // convert back to the original BlockedEncoding
    Value dotOutput = convertLayout(rewriter, newDot, oldRetType.getEncoding());

    rewriter.replaceOp(dotOp, dotOutput);

    return success();
  }
};

struct TritonMetalGPUAccelerateMatmulPass
    : triton::impl::TritonMetalGPUAccelerateMatmulBase<TritonMetalGPUAccelerateMatmulPass> {
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<BlockedToMetalMFMA>(context);
    if (applyPatternsGreedily(getOperation(), std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // namespace

namespace triton {
std::unique_ptr<OperationPass<ModuleOp>> createTritonMetalGPUAccelerateMatmul() {
  return std::make_unique<TritonMetalGPUAccelerateMatmulPass>();
}
} // namespace triton

} // namespace mlir

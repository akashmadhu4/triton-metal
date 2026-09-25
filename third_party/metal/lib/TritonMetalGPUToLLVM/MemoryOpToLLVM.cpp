#include "PatternTritonGPUOpToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::triton::gpu;

namespace {

// air.wg.barrier(mem_flags, scope)
// mem_flags = 2 (mem_threadgroup), scope = 1 (threadgroup)
constexpr int kAirMemFlagsThreadgroup = 2;
constexpr int kAirScopeThreadgroup = 1;

struct BarrierOpConversion
    : public ConvertOpToLLVMPattern<mlir::gpu::BarrierOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(mlir::gpu::BarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    auto voidTy = LLVM::LLVMVoidType::get(ctx);
    auto i32Ty = IntegerType::get(ctx, 32);
    auto funcTy = LLVM::LLVMFunctionType::get(voidTy, {i32Ty, i32Ty});

    Operation *parentOp = rewriter.getInsertionBlock()->getParentOp();
    LLVM::LLVMFuncOp funcOp =
        appendOrGetExternFuncOp(rewriter, parentOp, "air.wg.barrier", funcTy);
    funcOp.setCConv(LLVM::CConv::C);

    Value memFlags =
        LLVM::createConstantI32(loc, rewriter, kAirMemFlagsThreadgroup);
    Value scope = LLVM::createConstantI32(loc, rewriter, kAirScopeThreadgroup);
    LLVM::createLLVMCallOp(rewriter, loc, funcOp, ValueRange{memFlags, scope});

    rewriter.eraseOp(op);
    return success();
  }
};

struct TritonBarrierOpConversion
    : public ConvertOpToLLVMPattern<mlir::triton::gpu::BarrierOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(mlir::triton::gpu::BarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    auto voidTy = LLVM::LLVMVoidType::get(ctx);
    auto i32Ty = IntegerType::get(ctx, 32);
    auto funcTy = LLVM::LLVMFunctionType::get(voidTy, {i32Ty, i32Ty});

    Operation *parentOp = rewriter.getInsertionBlock()->getParentOp();
    LLVM::LLVMFuncOp funcOp =
        appendOrGetExternFuncOp(rewriter, parentOp, "air.wg.barrier", funcTy);
    funcOp.setCConv(LLVM::CConv::C);

    Value memFlags =
        LLVM::createConstantI32(loc, rewriter, kAirMemFlagsThreadgroup);
    Value scope = LLVM::createConstantI32(loc, rewriter, kAirScopeThreadgroup);
    LLVM::createLLVMCallOp(rewriter, loc, funcOp, ValueRange{memFlags, scope});

    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

namespace mlir::triton::metal {

void populateBarrierOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     PatternBenefit benefit) {
  patterns.add<BarrierOpConversion>(typeConverter, benefit);
  patterns.add<TritonBarrierOpConversion>(typeConverter, benefit);
}

} // namespace mlir::triton::metal

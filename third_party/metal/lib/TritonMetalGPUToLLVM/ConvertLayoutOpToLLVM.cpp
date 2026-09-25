#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir::triton::metal {

namespace {

static Type getMatTy(MLIRContext *ctx, Type elemTy) {
  return VectorType::get({64}, elemTy);
}

static Type getCoordTy(MLIRContext *ctx) {
  auto i64Ty = IntegerType::get(ctx, 64);
  return VectorType::get({2}, i64Ty);
}

static std::string getElemSuffix(Type elemTy) {
  if (elemTy.isF32())
    return "f32";
  if (elemTy.isF16())
    return "f16";
  if (elemTy.isBF16())
    return "bf16";
  llvm_unreachable("unsupported element type for simdgroup_matrix");
}

static LLVM::LLVMFuncOp
getOrCreateSimdgroupFunc(ConversionPatternRewriter &rewriter,
                         Operation *parentOp, StringRef funcName,
                         Type funcType) {
  return triton::gpu::appendOrGetExternFuncOp(rewriter, parentOp, funcName,
                                              funcType);
}

static Value emitLoad(ConversionPatternRewriter &rewriter, Location loc,
                      Operation *parentOp, Value ptrBase, Value stride,
                      Value col, Value row, Type elemTy, bool transpose) {
  MLIRContext *ctx = rewriter.getContext();
  auto matTy = getMatTy(ctx, elemTy);
  auto coordTy = getCoordTy(ctx);
  auto ptrTy = LLVM::LLVMPointerType::get(ctx, 3);
  auto i64Ty = IntegerType::get(ctx, 64);
  auto funcType =
      LLVM::LLVMFunctionType::get(matTy, {ptrTy, coordTy, coordTy, coordTy});
  std::string suffix = getElemSuffix(elemTy);
  auto funcOp = getOrCreateSimdgroupFunc(
      rewriter, parentOp,
      "air.simdgroup_matrix_8x8_load.v64" + suffix + ".p3" + suffix, funcType);

  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value stride64 = LLVM::ZExtOp::create(rewriter, loc, i64Ty, stride);

  Value strideVec = b.undef(coordTy);
  strideVec = LLVM::InsertElementOp::create(rewriter, loc, coordTy, strideVec,
                                            stride64, b.i32_val(0));
  strideVec = LLVM::InsertElementOp::create(rewriter, loc, coordTy, strideVec,
                                            stride64, b.i32_val(1));

  Value layoutVec = b.undef(coordTy);
  Value valX = transpose ? stride64 : b.i64_val(1);
  Value valY = transpose ? b.i64_val(1) : stride64;
  layoutVec = LLVM::InsertElementOp::create(rewriter, loc, coordTy, layoutVec,
                                            valX, b.i32_val(0));
  layoutVec = LLVM::InsertElementOp::create(rewriter, loc, coordTy, layoutVec,
                                            valY, b.i32_val(1));

  Value originVec = b.undef(coordTy);
  originVec = LLVM::InsertElementOp::create(rewriter, loc, coordTy, originVec,
                                            b.i64_val(0), b.i32_val(0));
  originVec = LLVM::InsertElementOp::create(rewriter, loc, coordTy, originVec,
                                            b.i64_val(0), b.i32_val(1));

  Value offset = b.add(b.mul(row, stride), col);
  Value ptr = b.gep(ptrTy, elemTy, ptrBase, offset);

  return LLVM::createLLVMCallOp(
             rewriter, loc, funcOp,
             ValueRange{ptr, strideVec, layoutVec, originVec})
      .getResult();
}

static void emitStore(ConversionPatternRewriter &rewriter, Location loc,
                      Operation *parentOp, Value ptrBase, Value stride,
                      Value col, Value row, Type elemTy, Value vec) {
  MLIRContext *ctx = rewriter.getContext();
  auto matTy = getMatTy(ctx, elemTy);
  auto coordTy = getCoordTy(ctx);
  auto ptrTy = LLVM::LLVMPointerType::get(ctx, 3);
  auto i64Ty = IntegerType::get(ctx, 64);
  auto i1Ty = IntegerType::get(ctx, 1);
  auto voidTy = LLVM::LLVMVoidType::get(ctx);
  auto funcType = LLVM::LLVMFunctionType::get(
      voidTy, {vec.getType(), ptrTy, coordTy, coordTy, coordTy});
  std::string suffix = getElemSuffix(elemTy);
  auto funcOp = getOrCreateSimdgroupFunc(
      rewriter, parentOp,
      "air.simdgroup_matrix_8x8_store.v64" + suffix + ".p3" + suffix, funcType);

  auto b = TritonLLVMOpBuilder(loc, rewriter);

  Value offset = b.add(b.mul(row, stride), col);
  Value ptr = b.gep(ptrTy, elemTy, ptrBase, offset);

  Value stride64 = LLVM::ZExtOp::create(rewriter, loc, i64Ty, stride);

  Value strideVec = b.undef(coordTy);
  strideVec = LLVM::InsertElementOp::create(rewriter, loc, coordTy, strideVec,
                                            stride64, b.i32_val(0));
  strideVec = LLVM::InsertElementOp::create(rewriter, loc, coordTy, strideVec,
                                            stride64, b.i32_val(1));

  Value layoutVec = b.undef(coordTy);
  layoutVec = LLVM::InsertElementOp::create(rewriter, loc, coordTy, layoutVec,
                                            b.i64_val(1), b.i32_val(0));
  layoutVec = LLVM::InsertElementOp::create(rewriter, loc, coordTy, layoutVec,
                                            stride64, b.i32_val(1));

  Value originVec = b.undef(coordTy);
  originVec = LLVM::InsertElementOp::create(rewriter, loc, coordTy, originVec,
                                            b.i64_val(0), b.i32_val(0));
  originVec = LLVM::InsertElementOp::create(rewriter, loc, coordTy, originVec,
                                            b.i64_val(0), b.i32_val(1));

  LLVM::createLLVMCallOp(rewriter, loc, funcOp,
                         ValueRange{vec, ptr, strideVec, layoutVec, originVec});
}

struct ConvertSharedToMetalMfma
    : public ConvertOpToLLVMPattern<triton::gpu::LocalLoadOp> {
  ConvertSharedToMetalMfma(const LLVMTypeConverter &converter,
                           PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::gpu::LocalLoadOp>(converter, benefit) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalLoadOp op,
                  triton::gpu::LocalLoadOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcTy = cast<triton::gpu::MemDescType>(op.getSrc().getType());
    auto dstTy = cast<RankedTensorType>(op.getResult().getType());

    if (!isa<triton::gpu::SharedLinearEncodingAttr>(srcTy.getEncoding()))
      return failure();

    auto dstEnc = dstTy.getEncoding();
    triton::gpu::MetalMfmaEncodingAttr mma;
    int opIdx = -1;

    if (auto dotEnc = dyn_cast<triton::gpu::DotOperandEncodingAttr>(dstEnc)) {
      mma = dyn_cast<triton::gpu::MetalMfmaEncodingAttr>(dotEnc.getParent());
      opIdx = dotEnc.getOpIdx();
    } else {
      mma = dyn_cast<triton::gpu::MetalMfmaEncodingAttr>(dstEnc);
      opIdx = 2; // Treat C as opIdx=2
    }

    if (!mma)
      return failure();

    auto loc = op.getLoc();
    auto *ctx = op.getContext();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto p3Ty = LLVM::LLVMPointerType::get(ctx, 3);
    auto elemTy = srcTy.getElementType();

    auto typeConverter = getTypeConverter();
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getSrc(), typeConverter->convertType(elemTy), rewriter);
    Value basePtr = smemObj.getBase();
    if (basePtr.getType() != p3Ty)
      basePtr = LLVM::AddrSpaceCastOp::create(rewriter, loc, p3Ty, basePtr);

    // Add subview offset for pipelining
    Value smemOffset = smemObj.getShmemOffset(loc, rewriter, srcTy);
    basePtr = b.gep(p3Ty, typeConverter->convertType(elemTy), basePtr, smemOffset);

    // Stride for simdgroup_load = number of columns (leading dimension for
    // row-major)
    Value stride = b.i32_val(srcTy.getShape()[1]);

    auto warpsPerCTA = mma.getWarpsPerCTA();

    int numWarps0 =
        (opIdx == 0) ? warpsPerCTA[0] : ((opIdx == 1) ? 1 : warpsPerCTA[0]);
    int numWarps1 =
        (opIdx == 0) ? 1 : ((opIdx == 1) ? warpsPerCTA[1] : warpsPerCTA[1]);

    int shapePerWarp0 = srcTy.getShape()[0] / numWarps0;
    int shapePerWarp1 = srcTy.getShape()[1] / numWarps1;

    int reps0 = std::max(1, shapePerWarp0 / 8);
    int reps1 = std::max(1, shapePerWarp1 / 8);

    Value warpId = getLaneAndWarpId(rewriter, loc).second;

    Value numWarps0Val = b.i32_val(numWarps0);
    Value warpId0;
    if (opIdx == 1) {
      warpId0 = b.i32_val(0);
    } else {
      warpId0 = b.urem(warpId, b.i32_val(warpsPerCTA[0]));
    }
    
    Value warpId1;
    if (opIdx == 0) {
      warpId1 = b.i32_val(0);
    } else {
      warpId1 = b.udiv(warpId, b.i32_val(warpsPerCTA[0]));
    }

    SmallVector<Value> loadedVectors;
    Operation *parentOp = rewriter.getInsertionBlock()->getParentOp();

    for (int r0 = 0; r0 < reps0; r0++) {
      for (int r1 = 0; r1 < reps1; r1++) {
        Value row =
            b.add(b.i32_val(r0 * 8), b.mul(warpId0, b.i32_val(shapePerWarp0)));
        Value col =
            b.add(b.i32_val(r1 * 8), b.mul(warpId1, b.i32_val(shapePerWarp1)));

        bool transpose = false;
        Value vec = emitLoad(rewriter, loc, parentOp, basePtr, stride, col, row,
                             elemTy, transpose);
        loadedVectors.push_back(vec);
      }
    }

    auto vecTy = getMatTy(ctx, elemTy);
    auto structTy = LLVM::LLVMStructType::getLiteral(
        ctx, SmallVector<Type>(loadedVectors.size(), vecTy));
    Value result = b.undef(structTy);
    int structIdx = 0;
    for (size_t i = 0; i < loadedVectors.size(); ++i) {
      result = LLVM::InsertValueOp::create(rewriter, loc, result,
                                           loadedVectors[i], structIdx++);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ConvertMetalMfmaToShared
    : public ConvertOpToLLVMPattern<triton::gpu::LocalAllocOp> {
  ConvertMetalMfmaToShared(const LLVMTypeConverter &converter,
                           const TargetInfoBase &targetInfo,
                           PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::gpu::LocalAllocOp>(converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalAllocOp op,
                  triton::gpu::LocalAllocOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.getSrc())
      return failure();
    if (!op.isSharedMemoryAlloc())
      return failure();

    auto srcTy = cast<RankedTensorType>(op.getSrc().getType());
    auto dstTy = cast<triton::gpu::MemDescType>(op.getResult().getType());

    if (!isa<triton::gpu::MetalMfmaEncodingAttr>(srcTy.getEncoding()))
      return failure();

    if (!isa<triton::gpu::SharedLinearEncodingAttr>(dstTy.getEncoding()))
      return failure();

    auto loc = op.getLoc();
    auto *ctx = op.getContext();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto p3Ty = LLVM::LLVMPointerType::get(ctx, 3);
    auto elemTy = dstTy.getElementType();
    auto typeConverter = getTypeConverter();

    SmallVector<Value> smemBases = LLVM::getSharedMemoryBases(
        loc, rewriter, targetInfo, op.getOperation());
    Value basePtr = smemBases[0];
    if (basePtr.getType() != p3Ty)
      basePtr = LLVM::AddrSpaceCastOp::create(rewriter, loc, p3Ty, basePtr);

    auto llvmElemTy = typeConverter->convertType(elemTy);
    auto smemObj = SharedMemoryObject(smemBases, llvmElemTy, dstTy.getRank(),
                                      loc, rewriter);
    // Stride for simdgroup_store = number of columns (leading dimension for
    // row-major)
    Value stride = b.i32_val(dstTy.getShape()[1]);

    auto mma = cast<triton::gpu::MetalMfmaEncodingAttr>(srcTy.getEncoding());
    auto warpsPerCTA = mma.getWarpsPerCTA();
    int numWarps0 = warpsPerCTA[0];
    int numWarps1 = warpsPerCTA[1];

    int shapePerWarp0 = srcTy.getShape()[0] / numWarps0;
    int shapePerWarp1 = srcTy.getShape()[1] / numWarps1;

    int reps0 = std::max(1, shapePerWarp0 / 8);
    int reps1 = std::max(1, shapePerWarp1 / 8);

    Value warpId = getLaneAndWarpId(rewriter, loc).second;
    Value numWarps0Val = b.i32_val(numWarps0);
    Value warpId0 = b.urem(warpId, numWarps0Val);
    Value warpId1 = b.udiv(warpId, numWarps0Val);

    Operation *parentOp = rewriter.getInsertionBlock()->getParentOp();

    int structIdx = 0;
    for (int r0 = 0; r0 < reps0; r0++) {
      for (int r1 = 0; r1 < reps1; r1++) {
        Value row =
            b.add(b.i32_val(r0 * 8), b.mul(warpId0, b.i32_val(shapePerWarp0)));
        Value col =
            b.add(b.i32_val(r1 * 8), b.mul(warpId1, b.i32_val(shapePerWarp1)));

        Value vec = LLVM::ExtractValueOp::create(rewriter, loc,
                                                 adaptor.getSrc(), structIdx++);

        emitStore(rewriter, loc, parentOp, basePtr, stride, col, row, elemTy,
                  vec);
      }
    }

    Value smemStruct =
        LLVM::getStructFromSharedMemoryObject(loc, smemObj, rewriter);
    rewriter.replaceOp(op, smemStruct);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

} // namespace

struct ConvertLayoutOpToMetalMfma
    : public ConvertOpToLLVMPattern<triton::gpu::ConvertLayoutOp> {
  ConvertLayoutOpToMetalMfma(const LLVMTypeConverter &converter,
                             const TargetInfoBase &targetInfo,
                             PatternBenefit benefit = 2)
      : ConvertOpToLLVMPattern<triton::gpu::ConvertLayoutOp>(converter,
                                                             benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::ConvertLayoutOp op,
                  triton::gpu::ConvertLayoutOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcTy = cast<RankedTensorType>(op.getSrc().getType());
    auto dstTy = cast<RankedTensorType>(op.getResult().getType());

    auto srcEnc = srcTy.getEncoding();
    auto dstEnc = dstTy.getEncoding();

    bool srcIsMfma =
        isa<triton::gpu::MetalMfmaEncodingAttr>(srcEnc) ||
        (isa<triton::gpu::DotOperandEncodingAttr>(srcEnc) &&
         isa<triton::gpu::MetalMfmaEncodingAttr>(
             cast<triton::gpu::DotOperandEncodingAttr>(srcEnc).getParent()));
    bool dstIsMfma =
        isa<triton::gpu::MetalMfmaEncodingAttr>(dstEnc) ||
        (isa<triton::gpu::DotOperandEncodingAttr>(dstEnc) &&
         isa<triton::gpu::MetalMfmaEncodingAttr>(
             cast<triton::gpu::DotOperandEncodingAttr>(dstEnc).getParent()));

    if (!srcIsMfma && !dstIsMfma)
      return failure();

    if (srcIsMfma && dstIsMfma)
      return failure(); // TODO: handle register-to-register conversion

    auto loc = op.getLoc();
    auto *ctx = op.getContext();

    // Create SharedLinearEncoding
    int64_t M = srcTy.getShape()[0], N = srcTy.getShape()[1];
    unsigned alignment = srcTy.getElementType().getIntOrFloatBitWidth() / 8;
    std::vector<std::vector<int32_t>> offsetBases;
    for (int64_t i = 1; i < N; i *= 2)
      offsetBases.push_back({0, (int32_t)i});
    for (int64_t i = 1; i < M; i *= 2)
      offsetBases.push_back({(int32_t)i, 0});
    std::vector<std::vector<int32_t>> blockBases = {};
    StringAttr kOffset = StringAttr::get(ctx, "offset");
    StringAttr kBlock = StringAttr::get(ctx, "block");
    StringAttr dim0 = StringAttr::get(ctx, "dim0");
    StringAttr dim1 = StringAttr::get(ctx, "dim1");
    triton::LinearLayout ll({{kOffset, offsetBases}, {kBlock, blockBases}},
                            {{dim0, (int32_t)M}, {dim1, (int32_t)N}}, true);
    auto sharedMemSpace = triton::gpu::SharedMemorySpaceAttr::get(ctx);
    auto sharedEnc =
        triton::gpu::SharedLinearEncodingAttr::get(ctx, ll, alignment);
    auto sharedTy =
        triton::gpu::MemDescType::get(srcTy.getShape(), srcTy.getElementType(),
                                      sharedEnc, sharedMemSpace, true);

    // Insert a barrier before alloc to ensure previous loads from shared memory have finished.
    triton::gpu::BarrierOp::create(rewriter, loc, triton::gpu::AddrSpace::Local);
    auto alloc =
        triton::gpu::LocalAllocOp::create(rewriter, loc, sharedTy, op.getSrc());
    if (op->hasAttr("allocation.offset")) {
      alloc->setAttr("allocation.offset", op->getAttr("allocation.offset"));
    }
    
    // MembarAnalysis misses ConvertLayoutOp because it is Pure. We must manually
    // insert a barrier to prevent race conditions during layout conversion through shared memory.
    triton::gpu::BarrierOp::create(rewriter, loc, triton::gpu::AddrSpace::Local);
    
    auto load = triton::gpu::LocalLoadOp::create(rewriter, loc, dstTy, alloc);

    rewriter.replaceOp(op, load.getResult());
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

void populateConvertLayoutOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                           const TargetInfo &targetInfo,
                                           RewritePatternSet &patterns,
                                           PatternBenefit benefit) {
  patterns.add<ConvertSharedToMetalMfma>(typeConverter, benefit);
  patterns.add<ConvertMetalMfmaToShared>(typeConverter, targetInfo, benefit);
  patterns.add<ConvertLayoutOpToMetalMfma>(typeConverter, targetInfo, benefit);
}

} // namespace mlir::triton::metal

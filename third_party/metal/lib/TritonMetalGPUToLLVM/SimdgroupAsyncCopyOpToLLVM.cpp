#include "Dialect/TritonMetalGPU/IR/Dialect.h"
#include "TargetInfo.h"
#include "TritonMetalGPUToLLVM/MetalKernelArgs.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton::gpu;

namespace ttmetalgpu = mlir::triton::metalgpu;

namespace mlir::triton::metal {

namespace {

static Value
emitAirSimdgroupAsyncCopy2D(ConversionPatternRewriter &rewriter, Location loc,
                            Operation *parentOp,
                            Value dest,   // addrspace(3) ptr
                            Value src,    // addrspace(1) ptr
                            Value stride, // i32, row stride of src
                            ArrayRef<int64_t> tileShape, // [rows, cols]
                            Type elemTy) {
  MLIRContext *ctx = rewriter.getContext();
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  auto i32Ty = IntegerType::get(ctx, 32);
  auto i64Ty = IntegerType::get(ctx, 64);
  auto long2Ty = VectorType::get({2}, i64Ty);
  auto p0Ty = LLVM::LLVMPointerType::get(ctx, 0); // thread (default)
  auto p1Ty = LLVM::LLVMPointerType::get(ctx, 1); // device
  auto p3Ty = LLVM::LLVMPointerType::get(ctx, 3); // threadgroup

  unsigned elemBytes = elemTy.getIntOrFloatBitWidth() / 8;
  int64_t rows = tileShape[0];
  int64_t cols = tileShape[1];

  // returns thread ptr _simdgroup_event_t* (addrspace 0, opaque ptr)
  auto funcType = LLVM::LLVMFunctionType::get(
      p0Ty, {
                i64Ty, i64Ty, // elemSize, elemAlign
                p3Ty,         // dest
                i64Ty, i64Ty, // destElemsPerRow, destElemsPerCol (stride)
                long2Ty,      // destTileSize [cols, rows]
                p1Ty,         // src
                i64Ty, i64Ty, // srcElemsPerRow, srcElemsPerCol (stride)
                long2Ty,      // srcTileSize [cols, rows]
                long2Ty,      // matrixOrigin [col, row]
                i32Ty         // transposeFlag
            });

  auto funcOp = appendOrGetExternFuncOp(
      rewriter, parentOp, "air.simdgroup_async_copy_2d.p3i8.p1i8", funcType);

  // cast ptrs to untyped i8* in the right addr space
  Value destI8 = LLVM::BitcastOp::create(rewriter, loc, p3Ty, dest);
  Value srcI8 = LLVM::BitcastOp::create(rewriter, loc, p1Ty, src);

  // src row stride as i64
  Value strideI64 = LLVM::ZExtOp::create(rewriter, loc, i64Ty, stride);

  // tile size vector
  Value tileVec = b.undef(long2Ty);
  tileVec = LLVM::InsertElementOp::create(rewriter, loc, long2Ty, tileVec,
                                          b.i64_val(cols), b.i32_val(0));
  tileVec = LLVM::InsertElementOp::create(rewriter, loc, long2Ty, tileVec,
                                          b.i64_val(rows), b.i32_val(1));

  // origin = {0, 0}: copy starts at [0, 0] in src
  Value originVec = b.undef(long2Ty);
  originVec = LLVM::InsertElementOp::create(rewriter, loc, long2Ty, originVec,
                                            b.i64_val(0), b.i32_val(0));
  originVec = LLVM::InsertElementOp::create(rewriter, loc, long2Ty, originVec,
                                            b.i64_val(0), b.i32_val(1));

  return LLVM::createLLVMCallOp(
             rewriter, loc, funcOp,
             ValueRange{
                 b.i64_val(elemBytes),
                 b.i64_val(elemBytes), // elemSize, elemAlign
                 destI8, b.i64_val(cols),
                 b.i64_val(1), // dest: elemsPerRow=tile_cols, elemsPerCol=1
                               // (row-major)
                 tileVec, srcI8, strideI64,
                 b.i64_val(1), // src: elemsPerRow=matrix stride, elemsPerCol=1
                               // (row-major)
                 tileVec, originVec,
                 b.i32_val(0) // no transpose
             })
      .getResult();
}

#if 0 // Legacy custom op, no longer used
struct SimdgroupAsyncCopyOpConversion
    : public ConvertOpToLLVMPattern<ttmetalgpu::SimdgroupAsyncCopyOp> {
  explicit SimdgroupAsyncCopyOpConversion(
      LLVMTypeConverter &typeConverter,
      const mlir::triton::metal::TargetInfo &targetInfo, PatternBenefit benefit)
      : ConvertOpToLLVMPattern<ttmetalgpu::SimdgroupAsyncCopyOp>(typeConverter,
                                                                 benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(ttmetalgpu::SimdgroupAsyncCopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = op.getContext();

    // src: tensor<MxNx!tt.ptr<elemTy>>
    auto srcTensorTy = cast<RankedTensorType>(op.getSrc().getType());
    auto ptrTy = cast<mlir::triton::PointerType>(srcTensorTy.getElementType());
    Type elemTy = ptrTy.getPointeeType();

    // tile shape from dst MemDesc
    auto dstMemDescTy = cast<MemDescType>(op.getDst().getType());
    ArrayRef<int64_t> tileShape = dstMemDescTy.getShape();

    // recover base ptr of src block
    //
    // srcPtrs[0] is ptr to element owned by this thread in the distributed
    // tensor, simdgroup_async_copy_2d needs top left ptr of tile
    auto srcPtrs = unpackLLElements(loc, adaptor.getSrc(), rewriter);

    // get layout from src tensor encoding
    auto srcEnc =
        cast<triton::gpu::BlockedEncodingAttr>(srcTensorTy.getEncoding());

    // use emitIndices to get thread's element indices, handling all encoding
    // layouts (warpsPerCTA distribution, order, etc.) correctly
    auto indices = emitIndices(loc, rewriter, targetInfo, srcEnc, srcTensorTy,
                               /*withCTAOffset=*/false);
    // indices[0] gives the (row, col) of this thread's first element
    Value threadRow = indices[0][0];
    Value threadCol = indices[0][1];

    auto func = rewriter.getInsertionBlock()
                    ->getParent()
                    ->getParentOfType<LLVM::LLVMFuncOp>();
    unsigned numArgs = func.getNumArguments();
    Value simdgroupIdxInThreadgroupVal =
        func.getArgument(numArgs - mlir::triton::metal::kSimdgroupIdxFromEnd);

    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto i32Ty = IntegerType::get(ctx, 32);
    Value simdgroupIdInThreadgroup = LLVM::TruncOp::create(
        rewriter, loc, i32Ty, simdgroupIdxInThreadgroupVal);

    Value stride = adaptor.getStride(); // i32, elements per row of src matrix
    Value threadOffset = b.add(b.mul(threadRow, stride), threadCol);
    Value negOffset = b.sub(b.i32_val(0), threadOffset);

    // walk back to base ptr
    Value srcThreadPtr = srcPtrs[0];
    auto p1Ty = LLVM::LLVMPointerType::get(ctx, 1);
    if (srcThreadPtr.getType() != p1Ty)
      srcThreadPtr =
          LLVM::AddrSpaceCastOp::create(rewriter, loc, p1Ty, srcThreadPtr);
    Value srcBase = LLVM::GEPOp::create(rewriter, loc, p1Ty, elemTy,
                                        srcThreadPtr, ValueRange{negOffset});

    // dst smem base pointer
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getDst(), typeConverter->convertType(elemTy), rewriter);
    Value dstBase = smemObj.getBase();
    auto p3Ty = LLVM::LLVMPointerType::get(ctx, 3);
    if (dstBase.getType() != p3Ty)
      dstBase = LLVM::AddrSpaceCastOp::create(rewriter, loc, p3Ty, dstBase);

    // Guard: only simdgroup 0 performs async copy to avoid redundant
    // transfers. Phi node merges real event (simdgroup 0) with a null
    // placeholder (other simdgroups) so wait op receives an event.
    auto p0Ty = LLVM::LLVMPointerType::get(ctx, 0);
    Value isSimdgroup0 = b.icmp_eq(simdgroupIdInThreadgroup, b.i32_val(0));
    auto *curBlock = rewriter.getInsertionBlock();
    auto *afterBlock = curBlock->splitBlock(rewriter.getInsertionPoint());
    afterBlock->addArgument(p0Ty, loc);
    auto *thenBlock = rewriter.createBlock(afterBlock);
    auto *elseBlock = rewriter.createBlock(afterBlock);

    // branch based on simdgroup index
    rewriter.setInsertionPointToEnd(curBlock);
    LLVM::CondBrOp::create(rewriter, loc, isSimdgroup0, thenBlock, elseBlock);

    // then block (simdgroup 0): perform async copy
    rewriter.setInsertionPointToStart(thenBlock);
    Operation *parentOp = rewriter.getInsertionBlock()->getParentOp();
    Value event =
        emitAirSimdgroupAsyncCopy2D(rewriter, loc, parentOp, dstBase, srcBase,
                                    adaptor.getStride(), tileShape, elemTy);
    LLVM::BrOp::create(rewriter, loc, ValueRange{event}, afterBlock);

    // else block (other simdgroups): pass null event
    rewriter.setInsertionPointToStart(elseBlock);
    Value nullEvent = LLVM::ZeroOp::create(rewriter, loc, p0Ty);
    LLVM::BrOp::create(rewriter, loc, ValueRange{nullEvent}, afterBlock);

    // after block: phi merges real event and null
    rewriter.setInsertionPointToStart(afterBlock);
    rewriter.replaceOp(op, afterBlock->getArgument(0));
    return success();
  }

protected:
  const mlir::triton::metal::TargetInfo &targetInfo;
};
#endif

} // namespace

// Helper: get-or-create a named alloca in the kernel entry block.
// Initializes scalar allocas (size==1) to zero on first creation.
static Value getOrCreateQueueAlloca(ConversionPatternRewriter &rewriter,
                                    Location loc, LLVM::LLVMFuncOp func,
                                    StringRef name, Type allocaTy, Type elemTy,
                                    int size) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  for (auto &op : func.getBody().front()) {
    if (auto alloca = dyn_cast<LLVM::AllocaOp>(op)) {
      if (alloca->hasAttr("queue_name") &&
          alloca->getAttrOfType<StringAttr>("queue_name").getValue() == name)
        return alloca.getResult();
    }
  }
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(&func.getBody().front());
  auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext(), 0);
  auto alloca =
      LLVM::AllocaOp::create(rewriter, loc, ptrTy, allocaTy, b.i32_val(1), 8);
  alloca->setAttr("queue_name", rewriter.getStringAttr(name));
  if (size == 1) {
    LLVM::StoreOp::create(rewriter, loc, b.i32_val(0), alloca);
  }
  return alloca;
}

namespace {

// Lower ttg.async_copy_global_to_local using air.simdgroup_async_copy_2d.
// Only simdgroup 0 issues the copy; the event pointer is stored in a
// per-kernel circular software queue so AsyncWaitOp can retrieve it later.
struct AsyncCopyGlobalToLocalOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::AsyncCopyGlobalToLocalOp> {
  explicit AsyncCopyGlobalToLocalOpConversion(
      LLVMTypeConverter &typeConverter,
      const mlir::triton::metal::TargetInfo &targetInfo, PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::gpu::AsyncCopyGlobalToLocalOp>(
            typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::AsyncCopyGlobalToLocalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = op.getContext();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    auto srcTensorTy = cast<RankedTensorType>(op.getSrc().getType());
    auto ptrTy = cast<mlir::triton::PointerType>(srcTensorTy.getElementType());
    Type elemTy = ptrTy.getPointeeType();

    auto dstMemDescTy = cast<MemDescType>(op.getResult().getType());
    ArrayRef<int64_t> tileShape = dstMemDescTy.getShape();

    auto srcPtrs = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    auto srcEnc =
        cast<triton::gpu::BlockedEncodingAttr>(srcTensorTy.getEncoding());
    auto indices = emitIndices(loc, rewriter, targetInfo, srcEnc, srcTensorTy,
                               /*withCTAOffset=*/false);
    Value threadRow = indices[0][0];
    Value threadCol = indices[0][1];

    auto func = rewriter.getInsertionBlock()
                    ->getParent()
                    ->getParentOfType<LLVM::LLVMFuncOp>();
    unsigned numArgs = func.getNumArguments();
    Value simdgroupIdxInThreadgroupVal =
        func.getArgument(numArgs - mlir::triton::metal::kSimdgroupIdxFromEnd);

    auto i32Ty = IntegerType::get(ctx, 32);
    Value simdgroupIdInThreadgroup = LLVM::TruncOp::create(
        rewriter, loc, i32Ty, simdgroupIdxInThreadgroupVal);

    // Use metal.stride_arg_idx attribute set by InjectTensorStrideArgs
    auto attr = op->getAttrOfType<IntegerAttr>("metal.stride_arg_idx");
    if (!attr)
      return rewriter.notifyMatchFailure(
          op, "missing metal.stride_arg_idx attribute");
    unsigned argIdx = attr.getInt();
    auto i64Ty = IntegerType::get(ctx, 64);
    Value stridePtr = func.getArgument(argIdx + 1);
    Value strideI64 = LLVM::LoadOp::create(rewriter, loc, i64Ty, stridePtr);
    Value stride = LLVM::TruncOp::create(rewriter, loc, i32Ty, strideI64);

    Value threadOffset = b.add(b.mul(threadRow, stride), threadCol);
    Value negOffset = b.sub(b.i32_val(0), threadOffset);

    Value srcThreadPtr = srcPtrs[0];
    auto p1Ty = LLVM::LLVMPointerType::get(ctx, 1);
    if (srcThreadPtr.getType() != p1Ty)
      srcThreadPtr =
          LLVM::AddrSpaceCastOp::create(rewriter, loc, p1Ty, srcThreadPtr);
    Value srcBase = LLVM::GEPOp::create(rewriter, loc, p1Ty, elemTy,
                                        srcThreadPtr, ValueRange{negOffset});

    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getResult(), typeConverter->convertType(elemTy), rewriter);
    Value dstBase = smemObj.getBase();
    auto p3Ty = LLVM::LLVMPointerType::get(ctx, 3);
    if (dstBase.getType() != p3Ty)
      dstBase = LLVM::AddrSpaceCastOp::create(rewriter, loc, p3Ty, dstBase);

    // Add subview offset for pipelining
    Value dstOffset = smemObj.getShmemOffset(loc, rewriter, dstMemDescTy);
    dstBase = b.gep(p3Ty, typeConverter->convertType(elemTy), dstBase, dstOffset);

    auto p0Ty = LLVM::LLVMPointerType::get(ctx, 0);
    Value isSimdgroup0 = b.icmp_eq(simdgroupIdInThreadgroup, b.i32_val(0));
    auto *curBlock = rewriter.getInsertionBlock();
    auto *afterBlock = curBlock->splitBlock(rewriter.getInsertionPoint());
    auto *thenBlock = rewriter.createBlock(afterBlock);
    auto *elseBlock = rewriter.createBlock(afterBlock);

    rewriter.setInsertionPointToEnd(curBlock);
    LLVM::CondBrOp::create(rewriter, loc, isSimdgroup0, thenBlock, elseBlock);

    // then block (simdgroup 0): perform async copy and push event to queue
    rewriter.setInsertionPointToStart(thenBlock);
    Operation *parentOp = rewriter.getInsertionBlock()->getParentOp();
    Value event = emitAirSimdgroupAsyncCopy2D(rewriter, loc, parentOp, dstBase,
                                              srcBase, stride, tileShape, elemTy);

    auto arrTy = LLVM::LLVMArrayType::get(p0Ty, 64);
    Value eventQueue = getOrCreateQueueAlloca(rewriter, loc, func, "eventQueue",
                                             arrTy, p0Ty, 64);
    Value headPtr = getOrCreateQueueAlloca(rewriter, loc, func, "headPtr",
                                          i32Ty, i32Ty, 1);
    Value currentGroupSizePtr = getOrCreateQueueAlloca(
        rewriter, loc, func, "currentGroupSizePtr", i32Ty, i32Ty, 1);

    Value head = LLVM::LoadOp::create(rewriter, loc, i32Ty, headPtr);
    Value headMod = b.urem(head, b.i32_val(64));
    Value eventPtr = LLVM::GEPOp::create(rewriter, loc, p0Ty, arrTy, eventQueue,
                                         ArrayRef<LLVM::GEPArg>{0, headMod});
    LLVM::StoreOp::create(rewriter, loc, event, eventPtr);

    Value newHead = b.add(head, b.i32_val(1));
    LLVM::StoreOp::create(rewriter, loc, newHead, headPtr);

    Value curSize = LLVM::LoadOp::create(rewriter, loc, i32Ty, currentGroupSizePtr);
    Value newCurSize = b.add(curSize, b.i32_val(1));
    LLVM::StoreOp::create(rewriter, loc, newCurSize, currentGroupSizePtr);

    LLVM::BrOp::create(rewriter, loc, ValueRange{}, afterBlock);

    // else block: do nothing
    rewriter.setInsertionPointToStart(elseBlock);
    LLVM::BrOp::create(rewriter, loc, ValueRange{}, afterBlock);

    rewriter.setInsertionPointToStart(afterBlock);
    rewriter.replaceOp(op, ValueRange{b.i32_val(0)});
    return success();
  }

protected:
  const mlir::triton::metal::TargetInfo &targetInfo;
};

// Commit a group: record how many async copies are in this group.
struct AsyncCommitGroupOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::AsyncCommitGroupOp> {
  explicit AsyncCommitGroupOpConversion(
      LLVMTypeConverter &typeConverter,
      const mlir::triton::metal::TargetInfo &targetInfo, PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::gpu::AsyncCommitGroupOp>(typeConverter,
                                                                benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::AsyncCommitGroupOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto i32Ty = IntegerType::get(ctx, 32);

    Operation *parentOp = rewriter.getInsertionBlock()->getParentOp();
    LLVM::LLVMFuncOp func = parentOp->getParentOfType<LLVM::LLVMFuncOp>();
    auto arrTy = LLVM::LLVMArrayType::get(i32Ty, 16);
    Value groupSizeQueue = getOrCreateQueueAlloca(rewriter, loc, func,
                                                  "groupSizeQueue", arrTy, i32Ty, 16);
    Value groupHeadPtr = getOrCreateQueueAlloca(rewriter, loc, func,
                                               "groupHeadPtr", i32Ty, i32Ty, 1);
    Value currentGroupSizePtr = getOrCreateQueueAlloca(
        rewriter, loc, func, "currentGroupSizePtr", i32Ty, i32Ty, 1);

    Value groupHead = LLVM::LoadOp::create(rewriter, loc, i32Ty, groupHeadPtr);
    Value groupHeadMod = b.urem(groupHead, b.i32_val(16));
    auto p0Ty = LLVM::LLVMPointerType::get(ctx, 0);
    Value sizePtr = LLVM::GEPOp::create(rewriter, loc, p0Ty, arrTy,
                                        groupSizeQueue,
                                        ArrayRef<LLVM::GEPArg>{0, groupHeadMod});

    Value curSize = LLVM::LoadOp::create(rewriter, loc, i32Ty, currentGroupSizePtr);
    LLVM::StoreOp::create(rewriter, loc, curSize, sizePtr);

    Value newGroupHead = b.add(groupHead, b.i32_val(1));
    LLVM::StoreOp::create(rewriter, loc, newGroupHead, groupHeadPtr);
    LLVM::StoreOp::create(rewriter, loc, b.i32_val(0), currentGroupSizePtr);

    rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(op, i32Ty,
                                                   rewriter.getI32IntegerAttr(0));
    return success();
  }

protected:
  const mlir::triton::metal::TargetInfo &targetInfo;
};

// Wait: pop (groupHead - groupTail - num) groups from the queue and call
// air.wait_simdgroup_events for each group's events. Only simdgroup 0 waits.
struct AsyncWaitOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::AsyncWaitOp> {
  explicit AsyncWaitOpConversion(
      LLVMTypeConverter &typeConverter,
      const mlir::triton::metal::TargetInfo &targetInfo, PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::gpu::AsyncWaitOp>(typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::AsyncWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto i32Ty = IntegerType::get(ctx, 32);
    auto voidTy = LLVM::LLVMVoidType::get(ctx);
    auto p0Ty = LLVM::LLVMPointerType::get(ctx, 0);

    auto func = rewriter.getInsertionBlock()
                    ->getParent()
                    ->getParentOfType<LLVM::LLVMFuncOp>();
    unsigned numArgs = func.getNumArguments();
    Value simdgroupIdxInThreadgroupVal =
        func.getArgument(numArgs - mlir::triton::metal::kSimdgroupIdxFromEnd);
    Value simdgroupIdInThreadgroup = LLVM::TruncOp::create(
        rewriter, loc, i32Ty, simdgroupIdxInThreadgroupVal);
    Value isSimdgroup0 = b.icmp_eq(simdgroupIdInThreadgroup, b.i32_val(0));

    auto *curBlock = rewriter.getInsertionBlock();
    auto *afterBlock = curBlock->splitBlock(rewriter.getInsertionPoint());
    auto *thenBlock = rewriter.createBlock(afterBlock);

    rewriter.setInsertionPointToEnd(curBlock);
    LLVM::CondBrOp::create(rewriter, loc, isSimdgroup0, thenBlock, afterBlock);

    rewriter.setInsertionPointToStart(thenBlock);

    auto arrTyEvents = LLVM::LLVMArrayType::get(p0Ty, 64);
    auto arrTySizes = LLVM::LLVMArrayType::get(i32Ty, 16);
    Value eventQueue = getOrCreateQueueAlloca(rewriter, loc, func, "eventQueue",
                                             arrTyEvents, p0Ty, 64);
    Value sizeQueue = getOrCreateQueueAlloca(rewriter, loc, func, "groupSizeQueue",
                                            arrTySizes, i32Ty, 16);
    Value tailPtr = getOrCreateQueueAlloca(rewriter, loc, func, "tailPtr",
                                          i32Ty, i32Ty, 1);
    Value groupHeadPtr = getOrCreateQueueAlloca(rewriter, loc, func,
                                               "groupHeadPtr", i32Ty, i32Ty, 1);
    Value groupTailPtr = getOrCreateQueueAlloca(rewriter, loc, func,
                                               "groupTailPtr", i32Ty, i32Ty, 1);

    // Loop: while (groupHead - groupTail) > num, pop one group and wait
    auto *condBlock = rewriter.createBlock(afterBlock);
    auto *loopBodyBlock = rewriter.createBlock(afterBlock);
    auto *doWaitBlock = rewriter.createBlock(afterBlock);
    auto *loopEndBlock = rewriter.createBlock(afterBlock);

    LLVM::BrOp::create(rewriter, loc, condBlock);

    rewriter.setInsertionPointToStart(condBlock);
    Value curGroupTail = LLVM::LoadOp::create(rewriter, loc, i32Ty, groupTailPtr);
    Value curGroupHead = LLVM::LoadOp::create(rewriter, loc, i32Ty, groupHeadPtr);
    Value diff = b.sub(curGroupHead, curGroupTail);
    Value numVal = b.i32_val(op.getNum());
    Value cmp = b.icmp_sgt(diff, numVal);
    LLVM::CondBrOp::create(rewriter, loc, cmp, loopBodyBlock, afterBlock);

    rewriter.setInsertionPointToStart(loopBodyBlock);
    Value gtMod = b.urem(curGroupTail, b.i32_val(16));
    Value sizePtr = LLVM::GEPOp::create(rewriter, loc, p0Ty, arrTySizes,
                                        sizeQueue,
                                        ArrayRef<LLVM::GEPArg>{0, gtMod});
    Value size = LLVM::LoadOp::create(rewriter, loc, i32Ty, sizePtr);
    Value sizeCmp = b.icmp_sgt(size, b.i32_val(0));
    LLVM::CondBrOp::create(rewriter, loc, sizeCmp, doWaitBlock, loopEndBlock);

    rewriter.setInsertionPointToStart(doWaitBlock);
    Value curTail = LLVM::LoadOp::create(rewriter, loc, i32Ty, tailPtr);
    Value tMod = b.urem(curTail, b.i32_val(64));
    Value evPtr = LLVM::GEPOp::create(rewriter, loc, p0Ty, arrTyEvents,
                                      eventQueue,
                                      ArrayRef<LLVM::GEPArg>{0, tMod});
    auto waitFuncType = LLVM::LLVMFunctionType::get(voidTy, {i32Ty, p0Ty});
    Operation *parentOp = rewriter.getInsertionBlock()->getParentOp();
    auto waitFuncOp = appendOrGetExternFuncOp(rewriter, parentOp,
                                              "air.wait_simdgroup_events",
                                              waitFuncType);
    LLVM::createLLVMCallOp(rewriter, loc, waitFuncOp, ValueRange{size, evPtr});
    Value newTail = b.add(curTail, size);
    LLVM::StoreOp::create(rewriter, loc, newTail, tailPtr);
    LLVM::BrOp::create(rewriter, loc, loopEndBlock);

    rewriter.setInsertionPointToStart(loopEndBlock);
    Value newGroupTail = b.add(curGroupTail, b.i32_val(1));
    LLVM::StoreOp::create(rewriter, loc, newGroupTail, groupTailPtr);
    LLVM::BrOp::create(rewriter, loc, condBlock);

    rewriter.setInsertionPointToStart(afterBlock);
    if (op->getNumResults() > 0) {
      rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(op, i32Ty,
                                                     rewriter.getI32IntegerAttr(0));
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }

protected:
  const mlir::triton::metal::TargetInfo &targetInfo;
};

} // namespace

void populateSimdgroupAsyncCopyOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const mlir::triton::metal::TargetInfo &targetInfo, PatternBenefit benefit) {
#if 0
  patterns.add<SimdgroupAsyncCopyOpConversion>(typeConverter, targetInfo,
                                               benefit);
#endif
  patterns.add<AsyncCopyGlobalToLocalOpConversion>(typeConverter, targetInfo,
                                                   benefit);
  patterns.add<AsyncCommitGroupOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<AsyncWaitOpConversion>(typeConverter, targetInfo, benefit);
}
} // namespace mlir::triton::metal

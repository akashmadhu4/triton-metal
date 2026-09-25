#include "TargetInfo.h"
#include "TritonMetalGPUToLLVM/MetalKernelArgs.h"
#include "Utility.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Value.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

namespace {

static LLVM::LLVMFuncOp
getOrCreateSimdgroupFunc(ConversionPatternRewriter &rewriter, Operation *parent,
                         StringRef name, LLVM::LLVMFunctionType funcType) {
  return appendOrGetExternFuncOp(rewriter, parent, name, funcType);
}

// get <64 x elemTy> simdgroup matrix type (elemTy is f32 or f16)
static VectorType getMatTy(MLIRContext *ctx, Type elemTy) {
  return VectorType::get({64}, elemTy);
}

// returns f32 or f16 for AIR intrinsic names
static std::string getElemSuffix(Type elemTy) {
  if (isa<Float16Type>(elemTy))
    return "f16";
  assert(isa<Float32Type>(elemTy) &&
         "unsupported simdgroup matrix element type");
  return "f32";
}

// get <2 x i64> coordinate type
static VectorType getCoordTy(MLIRContext *ctx) {
  return VectorType::get({2}, IntegerType::get(ctx, 64));
}

struct DotOpSimdgroupMatmulConversionHelper {
  BlockedEncodingAttr resultLayout;
  ConversionPatternRewriter &rewriter;
  const LLVMTypeConverter *typeConverter;
  Location loc;
  MLIRContext *ctx{};
  const DenseMap<int, std::array<Operation *, 3>> &dotAllocOps;
  const mlir::triton::metal::TargetInfo &targetInfo;

  virtual ~DotOpSimdgroupMatmulConversionHelper() = default;

  explicit DotOpSimdgroupMatmulConversionHelper(
      BlockedEncodingAttr resultLayout, ConversionPatternRewriter &rewriter,
      const LLVMTypeConverter *typeConverter, Location loc,
      const DenseMap<int, std::array<Operation *, 3>> &dotAllocOps,
      const mlir::triton::metal::TargetInfo &targetInfo)
      : resultLayout(resultLayout), rewriter(rewriter),
        typeConverter(typeConverter), loc(loc), ctx(resultLayout.getContext()),
        dotAllocOps(dotAllocOps), targetInfo(targetInfo) {}

  // -----------------------------------------------------------------------
  // emit air.wg.barrier(2, 1) (threadgroup barrier)
  // -----------------------------------------------------------------------
  void emitBarrier() const {
    auto voidTy = LLVM::LLVMVoidType::get(ctx);
    auto i32Ty = IntegerType::get(ctx, 32);
    auto funcType = LLVM::LLVMFunctionType::get(voidTy, {i32Ty, i32Ty});
    Operation *parentOp = rewriter.getInsertionBlock()->getParentOp();
    auto funcOp =
        appendOrGetExternFuncOp(rewriter, parentOp, "air.wg.barrier", funcType);
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    LLVM::createLLVMCallOp(rewriter, loc, funcOp,
                           ValueRange{b.i32_val(2), b.i32_val(1)});
  }

  // -----------------------------------------------------------------------
  // get thread_idx and simdgroup_idx from kernel args
  // -----------------------------------------------------------------------
  std::pair<Value, Value> getThreadAndWarpIdxs() const {
    auto func = rewriter.getInsertionBlock()
                    ->getParent()
                    ->getParentOfType<LLVM::LLVMFuncOp>();
    unsigned numArgs = func.getNumArguments();
    Value threadIdx =
        func.getArgument(numArgs - mlir::triton::metal::kThreadIdxFromEnd);
    Value simdgroupIdx =
        func.getArgument(numArgs - mlir::triton::metal::kSimdgroupIdxFromEnd);
    return {threadIdx, simdgroupIdx};
  }

  // -----------------------------------------------------------------------
  // init <64 x elemTy> accumulator (elemTy is f32 or f16)
  // -----------------------------------------------------------------------
  Value emitInitFilled(float fillVal, Type elemTy) const {
    auto matTy = getMatTy(ctx, elemTy);
    auto funcType = LLVM::LLVMFunctionType::get(matTy, {elemTy});
    Operation *parentOp = rewriter.getInsertionBlock()->getParentOp();
    std::string suffix = getElemSuffix(elemTy);
    auto funcOp = getOrCreateSimdgroupFunc(
        rewriter, parentOp,
        "air.simdgroup_matrix_8x8_init_filled.v64" + suffix + "." + suffix,
        funcType);
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value fillConst;
    if (isa<Float16Type>(elemTy))
      fillConst = LLVM::ConstantOp::create(rewriter, loc, elemTy,
                                           rewriter.getF16FloatAttr(fillVal));
    else
      fillConst = mlir::LLVM::createConstantF32(loc, rewriter, fillVal);
    auto op =
        LLVM::createLLVMCallOp(rewriter, loc, funcOp, ValueRange{fillConst});
    return op.getResult();
  }

  // -----------------------------------------------------------------------
  // simdgroup_matrix_8x8_load from addrspace(3) (threadgroup shared memory)
  // Return: <64 x elemTy> (f32 or f16)
  // stride = number of cols in row-major buffer
  // -----------------------------------------------------------------------
  Value emitLoad(Value ptrBase, Value stride, Value col, Value row,
                 Type elemTy) const {
    auto matTy = getMatTy(ctx, elemTy);
    auto coordTy = getCoordTy(ctx);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx, 3);
    auto i64Ty = IntegerType::get(ctx, 64);
    auto i1Ty = IntegerType::get(ctx, 1);
    auto funcType =
        LLVM::LLVMFunctionType::get(matTy, {ptrTy, i64Ty, coordTy, i1Ty});
    Operation *parentOp = rewriter.getInsertionBlock()->getParentOp();
    std::string suffix = getElemSuffix(elemTy);
    auto funcOp = getOrCreateSimdgroupFunc(rewriter, parentOp,
                                           "air.simdgroup_matrix_8x8_load.v64" +
                                               suffix + ".p3" + suffix,
                                           funcType);

    auto b = TritonLLVMOpBuilder(loc, rewriter);
    // <2 x i64> [col, row]
    Value coordVec = b.undef(coordTy);
    coordVec = LLVM::InsertElementOp::create(rewriter, loc, coordTy, coordVec,
                                             col, b.i64_val(0));
    coordVec = LLVM::InsertElementOp::create(rewriter, loc, coordTy, coordVec,
                                             row, b.i64_val(1));
    Value falseVal = LLVM::ConstantOp::create(rewriter, loc, i1Ty, (int64_t)0);
    auto op =
        LLVM::createLLVMCallOp(rewriter, loc, funcOp,
                               ValueRange{ptrBase, stride, coordVec, falseVal})
            .getResult();
    return op;
  }

  // -----------------------------------------------------------------------
  // simdgroup_matrix_8x8_multiply_accumulate
  // Supports mixed precision: inputElemTy (A, B) may differ from
  // outputElemTy (C, D). E.g. f16 inputs with f32 accumulator.
  // -----------------------------------------------------------------------
  Value emitMAC(Value a, Value b_mat, Value c, Type inputElemTy,
                Type outputElemTy) const {
    auto matInTy = getMatTy(ctx, inputElemTy);
    auto matOutTy = getMatTy(ctx, outputElemTy);
    auto funcType =
        LLVM::LLVMFunctionType::get(matOutTy, {matInTy, matInTy, matOutTy});
    Operation *parentOp = rewriter.getInsertionBlock()->getParentOp();
    std::string inSuffix = getElemSuffix(inputElemTy);
    std::string outSuffix = getElemSuffix(outputElemTy);
    auto funcName = "air.simdgroup_matrix_8x8_multiply_accumulate.v64" +
                    outSuffix + ".v64" + inSuffix + ".v64" + inSuffix + ".v64" +
                    outSuffix;
    auto funcOp =
        getOrCreateSimdgroupFunc(rewriter, parentOp, funcName, funcType);
    return LLVM::createLLVMCallOp(rewriter, loc, funcOp,
                                  ValueRange{a, b_mat, c})
        .getResult();
  }

  // -----------------------------------------------------------------------
  // simdgroup_matrix_8x8_store to addrspace(3) (threadgroup)
  // -----------------------------------------------------------------------
  void emitStore(Value mat, Value ptrBase, Value stride, Value col, Value row,
                 Type elemTy) const {
    auto matTy = getMatTy(ctx, elemTy);
    auto coordTy = getCoordTy(ctx);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx, /*addrspace=*/3);
    auto i64Ty = IntegerType::get(ctx, 64);
    auto i1Ty = IntegerType::get(ctx, 1);
    auto voidTy = LLVM::LLVMVoidType::get(ctx);
    auto funcType = LLVM::LLVMFunctionType::get(
        voidTy, {matTy, ptrTy, i64Ty, coordTy, i1Ty});
    Operation *parentOp = rewriter.getInsertionBlock()->getParentOp();
    std::string suffix = getElemSuffix(elemTy);
    auto funcOp = getOrCreateSimdgroupFunc(
        rewriter, parentOp,
        "air.simdgroup_matrix_8x8_store.v64" + suffix + ".p3" + suffix,
        funcType);

    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value coordVec = b.undef(coordTy);
    coordVec = LLVM::InsertElementOp::create(rewriter, loc, coordTy, coordVec,
                                             col, b.i64_val(0));
    coordVec = LLVM::InsertElementOp::create(rewriter, loc, coordTy, coordVec,
                                             row, b.i64_val(1));
    Value falseVal = LLVM::ConstantOp::create(rewriter, loc, i1Ty, (int64_t)0);
    LLVM::createLLVMCallOp(
        rewriter, loc, funcOp,
        ValueRange{mat, ptrBase, stride, coordVec, falseVal});
  }

  Value getSmemPtr(Operation *allocOp) const {
    auto offsetAttr = allocOp->getAttrOfType<IntegerAttr>("allocation.offset");
    assert(offsetAttr && "LocalAllocOp missing allocation.offset");
    int64_t offset = offsetAttr.getValue().getZExtValue();

    // get global_smem from module
    auto func = rewriter.getInsertionBlock()
                    ->getParent()
                    ->getParentOfType<LLVM::LLVMFuncOp>();
    auto mod = func->getParentOfType<ModuleOp>();
    auto globalBase = dyn_cast<LLVM::GlobalOp>(mod.lookupSymbol("global_smem"));
    assert(globalBase && "global_smem not found in module");

    auto ptrTy =
        LLVM::LLVMPointerType::get(ctx, targetInfo.getSharedAddressSpace());
    auto i8Ty = IntegerType::get(ctx, 8);
    Value stackPtr = LLVM::AddressOfOp::create(rewriter, loc, globalBase);
    auto gepOp = LLVM::GEPOp::create(rewriter, loc, ptrTy, i8Ty, stackPtr,
                                     ArrayRef<LLVM::GEPArg>{(int32_t)offset});
    return gepOp;
  }

  LogicalResult convertDot(DotOp op, DotOp::Adaptor adaptor) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    // emit barrier at beginning to ensure A and B are stored in SMEM
    emitBarrier();

    // get thread and warp ids
    // thread idx is within entire grid
    // simdgroup idx is within threadgroup
    auto [threadIdx, sgIdx] = getThreadAndWarpIdxs();
    sgIdx = b.zext(i64_ty, sgIdx);

    // smem should already be allocated for A and B by GPU transform pass
    // LocalAllocOps have been erased from the IR, so cannot call
    // getSharedMemoryBases. Instead, read allocation.offset from erased op
    // and compute smem ptr manually
    auto dotIdx = op->getAttrOfType<IntegerAttr>("metal.dot_idx").getInt();
    auto it = dotAllocOps.find(dotIdx);
    assert(it != dotAllocOps.end() && "expected smem alloc ops for dot op");
    auto &allocOps = it->second;

    // there is likely not enough shared memory for entire output tile
    // base ptrs for A and B in smem
    // ptr for CTA iteration region of C
    Value smemAPtr = getSmemPtr(allocOps[0]);
    Value smemBPtr = getSmemPtr(allocOps[1]);
    Value smemCPtr = getSmemPtr(allocOps[2]);

    // each warp handles 8x8 output tile
    // based on simdgroup idx, need to determine which output tiles this thread
    // is responsible for
    auto dTensorTy = cast<RankedTensorType>(op.getD().getType());
    auto warpsPerCTA = getWarpsPerCTA(dTensorTy);
    auto numWarps = product(warpsPerCTA);

    // block shape
    auto dShapePerCTA = getShapePerCTA(dTensorTy);
    BlockedEncodingAttr dLayout =
        cast<BlockedEncodingAttr>(dTensorTy.getEncoding());
    llvm::SmallVector<unsigned> sizePerThread{dLayout.getSizePerThread()};
    unsigned elemsPerThread = product(sizePerThread);

    // input shape
    auto aTensorTy = cast<RankedTensorType>(op.getA().getType());
    auto aShapePerCTA = getShapePerCTA(aTensorTy);
    Type inputElemTy = aTensorTy.getElementType();
    Type outputElemTy = dTensorTy.getElementType();

    // each CTA iteration handles shapePerCTATile and should fill up C region in
    // smem
    SmallVector<unsigned> shapePerCTATile;
    for (auto [reg, thread, warp] :
         llvm::zip(sizePerThread, dLayout.getThreadsPerWarp(),
                   dLayout.getWarpsPerCTA())) {
      unsigned ctaTileDim = reg * thread * warp;
      assert(ctaTileDim % 8 == 0 &&
             "tile dim per CTA iteration should be divisible by 8 to "
             "accomodate 8x8 air intrinsics");
      shapePerCTATile.push_back(ctaTileDim);
    }

    // now divide shapePerCTATile over warps
    SmallVector<unsigned> warpCoverage{8, 8};
    SmallVector<unsigned> warpItersNeededPerDim;
    for (auto [dim, dimWarpCov] : llvm::zip(shapePerCTATile, warpCoverage)) {
      warpItersNeededPerDim.push_back(dim / dimWarpCov);
    }
    unsigned totalWarpItersNeeded = product(warpItersNeededPerDim);

    // require 8x8 tiles divide evenly across warps in each dim
    // so all warps handle same number of tiles with no runtime conditionals
    assert(totalWarpItersNeeded % numWarps == 0 &&
           "total number of 8x8 tiles per CTA iteration should divide evenly "
           "among warps to avoid runtime conditionals");
    unsigned tilesPerWarp = totalWarpItersNeeded / numWarps;

    // thread-local struct
    // put elements in here based on output encoding, this struct is returned by
    // the lowering of tt.dot and later values are stored into global mem
    auto cc = unpackLLElements(loc, adaptor.getC(), rewriter);
    SmallVector<Value> acc = cc;

    // outer loop over CTA iterations
    // need this loop in case CTA can't cover entire output block
    SmallVector<unsigned> ctaReps;
    for (auto [blockDim, dimPerCTATile] :
         llvm::zip(dShapePerCTA, shapePerCTATile)) {
      // TODO probably should relax this restriction, but keep now for
      // simplicity
      assert(blockDim % dimPerCTATile == 0 &&
             "output block should be evenly divided by CTA iterations");
      ctaReps.push_back(blockDim / dimPerCTATile);
    }

    unsigned totalCtaReps = product(ctaReps);

    // later, threads need to put values from smem into thread-local acc
    // based on output encoding, so need to determine which output idxs this
    // thread owns
    auto idxs =
        emitIndices(loc, rewriter, targetInfo, resultLayout, dTensorTy, true);

    for (unsigned ctaRepIdx = 0; ctaRepIdx < totalCtaReps; ++ctaRepIdx) {
      unsigned ctaRepRow = (ctaRepIdx / ctaReps[1]) * shapePerCTATile[0];
      unsigned ctaRepCol = (ctaRepIdx % ctaReps[1]) * shapePerCTATile[1];

      // assign 8x8 tiles within shapePerCTATile to warps with round robin
      for (unsigned tileIdx = 0; tileIdx < tilesPerWarp; ++tileIdx) {
        Value flatTileIdx = b.add(sgIdx, b.i64_val(numWarps * tileIdx));

        // emit code to compute output row and column for this 8x8 tile
        Value tileRow =
            b.udiv(flatTileIdx, b.i64_val(warpItersNeededPerDim[1]));
        Value tileCol =
            b.urem(flatTileIdx, b.i64_val(warpItersNeededPerDim[1]));
        tileRow = b.mul(tileRow, b.i64_val(warpCoverage[0]));
        tileCol = b.mul(tileCol, b.i64_val(warpCoverage[1]));

        // tileRow and tileCol are within shapePerCTATile, now need to get row
        // and col within overall output block
        tileRow = b.add(tileRow, b.i64_val(ctaRepRow));
        tileCol = b.add(tileCol, b.i64_val(ctaRepCol));

        // emit code to init empty accumulator
        Value acc = emitInitFilled(0.0f, outputElemTy);

        // number of loops to cover K dimension with 8x8 tiles
        unsigned K = aShapePerCTA[1];
        assert(K % 8 == 0 &&
               "K dimension per CTA iteration should be divisible by 8 to "
               "accomodate 8x8 air intrinsics");

        // inner loop iterates over reduction dim one 8x8 tile at a time
        // accumulates result in acc
        unsigned numKTiles = K / 8;
        for (unsigned kTileIdx = 0; kTileIdx < numKTiles; ++kTileIdx) {
          // emit code to load 8x8 tile from A in smem
          Value kTileOffset = b.i64_val(kTileIdx * 8);
          Value aTile = emitLoad(smemAPtr, b.i64_val(aShapePerCTA[1]),
                                 kTileOffset, tileRow, inputElemTy);

          // load 8x8 tile from B in smem
          Value bTile = emitLoad(smemBPtr, b.i64_val(dShapePerCTA[1]), tileCol,
                                 kTileOffset, inputElemTy);

          // call matmul intrinsic
          acc = emitMAC(aTile, bTile, acc, inputElemTy, outputElemTy);
        }

        // after accumulating over reduction dim, store in smem
        emitStore(acc, smemCPtr, b.i64_val(dShapePerCTA[1]), tileCol, tileRow,
                  outputElemTy);
      }
      // barrier to ensure this CTA iteration has been stored in smem
      emitBarrier();

      // only load for this repetition
      auto idxsToLoad = llvm::ArrayRef<SmallVector<Value>>(idxs).slice(
          ctaRepIdx * elemsPerThread, elemsPerThread);
      auto i64Ty = IntegerType::get(ctx, 64);
      for (auto [i, idxToLoad] : llvm::enumerate(idxsToLoad)) {
        Value row = b.zext(i64Ty, idxToLoad[0]);
        Value col = b.zext(i64Ty, idxToLoad[1]);
        Value offset = b.add(b.mul(row, b.i64_val(dShapePerCTA[1])), col);
        Type smemCPtrTy =
            LLVM::LLVMPointerType::get(ctx, targetInfo.getSharedAddressSpace());
        Value offsetPtr = b.gep(smemCPtrTy, outputElemTy, smemCPtr, offset);
        auto prevAcc = acc[i + ctaRepIdx * elemsPerThread];
        auto amountToAdd = b.load(outputElemTy, offsetPtr);
        acc[i + ctaRepIdx * elemsPerThread] = b.fadd(prevAcc, amountToAdd);
      }

      // prevent next iteration from overwriting smem
      emitBarrier();
    }

    auto res = packLLElements(loc, typeConverter, acc, rewriter, dTensorTy);
    rewriter.replaceOp(op, res);

    return success();
  }
};

} // namespace

namespace mlir::triton::metal {
LogicalResult convertSimdgroupMatmul(
    triton::DotOp op, triton::DotOp::Adaptor adaptor,
    const LLVMTypeConverter *typeConverter, ConversionPatternRewriter &rewriter,
    const DenseMap<int, std::array<Operation *, 3>> &dotAllocOps,
    const mlir::triton::metal::TargetInfo &targetInfo) {
  auto rankedTType = [](Value tensor) {
    return cast<RankedTensorType>(tensor.getType());
  };

  assert(isa<DotOperandEncodingAttr>(rankedTType(op.getA()).getEncoding()) &&
         isa<DotOperandEncodingAttr>(rankedTType(op.getB()).getEncoding()) &&
         "A and B should have DotOperand encoding.");

  auto cTensorTy = rankedTType(op.getC());
  auto dTensorTy = rankedTType(op.getD());
  assert(isa<BlockedEncodingAttr>(cTensorTy.getEncoding()) &&
         "Only support C with blocked layout");

  assert(cTensorTy.getShape()[0] == dTensorTy.getShape()[0] &&
         cTensorTy.getShape()[1] == dTensorTy.getShape()[1] &&
         "DotOp C operand should have same shape as D");

  auto loc = op.getLoc();
  auto resultLayout = cast<BlockedEncodingAttr>(
      cast<RankedTensorType>(op.getResult().getType()).getEncoding());

  DotOpSimdgroupMatmulConversionHelper helper(
      resultLayout, rewriter, typeConverter, loc, dotAllocOps, targetInfo);

  return helper.convertDot(op, adaptor);
}

LogicalResult convertMetalMfmaDot(triton::DotOp op,
                                  triton::DotOp::Adaptor adaptor,
                                  const LLVMTypeConverter *typeConverter,
                                  ConversionPatternRewriter &rewriter) {
  auto loc = op.getLoc();
  auto *ctx = op.getContext();
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  auto aTensorTy = cast<RankedTensorType>(op.getA().getType());
  auto dTensorTy = cast<RankedTensorType>(op.getResult().getType());

  auto aElems = unpackLLElements(loc, adaptor.getA(), rewriter);
  auto bElems = unpackLLElements(loc, adaptor.getB(), rewriter);
  auto cElems = unpackLLElements(loc, adaptor.getC(), rewriter);

  auto dEncoding = cast<MetalMfmaEncodingAttr>(dTensorTy.getEncoding());

  auto warpsPerCTA = dEncoding.getWarpsPerCTA();
  int repsM = std::max<int>(1, dTensorTy.getShape()[0] / (warpsPerCTA[0] * 8));
  int repsN = std::max<int>(1, dTensorTy.getShape()[1] / (warpsPerCTA[1] * 8));
  int repsK = std::max<int>(1, aTensorTy.getShape()[1] / 8);

  Type aElemTy = aTensorTy.getElementType();
  Type bElemTy = cast<RankedTensorType>(op.getB().getType()).getElementType();
  Type dElemTy = dTensorTy.getElementType();

  VectorType vec64Ty = VectorType::get({64}, dElemTy);
  VectorType vec64TyA = VectorType::get({64}, aElemTy);
  VectorType vec64TyB = VectorType::get({64}, bElemTy);

  std::string inSuffix = getElemSuffix(aElemTy);
  std::string outSuffix = getElemSuffix(dElemTy);
  std::string funcName = "air.simdgroup_matrix_8x8_multiply_accumulate.v64" +
                         outSuffix + ".v64" + inSuffix + ".v64" + inSuffix +
                         ".v64" + outSuffix;

  auto funcType =
      LLVM::LLVMFunctionType::get(typeConverter->convertType(vec64Ty),
                                  {typeConverter->convertType(vec64TyA),
                                   typeConverter->convertType(vec64TyB),
                                   typeConverter->convertType(vec64Ty)});
  Operation *parentOp = rewriter.getInsertionBlock()->getParentOp();
  auto funcOp =
      getOrCreateSimdgroupFunc(rewriter, parentOp, funcName, funcType);

  SmallVector<Value> dElems(cElems.size());

  for (int m = 0; m < repsM; ++m) {
    for (int n = 0; n < repsN; ++n) {
      int cIdx = n + repsN * m;
      Value accVec = cElems[cIdx];

      for (int k = 0; k < repsK; ++k) {
        int aIdx = k + repsK * m;
        Value aVec = aElems[aIdx];

        int bIdx = n + repsN * k;
        Value bVec = bElems[bIdx];

        accVec = LLVM::createLLVMCallOp(rewriter, loc, funcOp,
                                        ValueRange{aVec, bVec, accVec})
                     .getResult();
      }

      dElems[cIdx] = accVec;
    }
  }

  Value result =
      packLLElements(loc, typeConverter, dElems, rewriter, dTensorTy);
  rewriter.replaceOp(op, result);
  return success();
}

} // namespace mlir::triton::metal

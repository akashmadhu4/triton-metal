#include "PatternTritonGPUOpToLLVM.h"
#include "TargetInfo.h"
#include "Utility.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

using namespace mlir;
using namespace mlir::triton::gpu;
using ::mlir::LLVM::getSharedMemoryBase;
using ::mlir::LLVM::metal::getVectorSize;
using ::mlir::LLVM::metal::llLoad;
using ::mlir::LLVM::metal::llStore;

namespace {

// TODO make sure these are correct, copied from AMD
std::optional<const char *> getMetalGPUMemScopeStr(MemSyncScope scope) {
  switch (scope) {
  case MemSyncScope::GPU:
    return "agent";
  case MemSyncScope::CTA:
    return "workgroup";
  case MemSyncScope::SYSTEM:
  default:
    return "";
  }
}

Value emitRedundantThreadPredicateNonNull(
    const llvm::MapVector<StringAttr, int32_t> &freeVarMasks,
    ConversionPatternRewriter &rewriter, Location loc,
    const metal::TargetInfo &targetInfo) {
  auto res =
      emitRedundantThreadPredicate(freeVarMasks, rewriter, loc, targetInfo);
  if (!res) {
    TritonLLVMOpBuilder b(loc, rewriter);
    return b.i1_val(true);
  }
  return res;
}

struct LoadStoreConversionBase {
  explicit LoadStoreConversionBase(
      const triton::metal::TargetInfo &targetInfo,
      mlir::triton::ModuleAxisInfoAnalysis &axisAnalysisPass)
      : targetInfo(targetInfo), axisAnalysisPass(axisAnalysisPass) {}

  // Create a LLVM vector of type `vecTy` containing all zeros
  Value createZeroVector(OpBuilder &builder, Location loc,
                         VectorType vecTy) const {
    mlir::Attribute zeroAttr = builder.getZeroAttr(vecTy.getElementType());
    auto denseValue =
        DenseElementsAttr::get(cast<mlir::ShapedType>(vecTy), zeroAttr);
    Value zeroVal = LLVM::ConstantOp::create(builder, loc, vecTy, denseValue);
    return zeroVal;
  }

  // Given a vector of values `elems` and a starting point `start`, create a
  // LLVM vector of length `vec` whose elements are `elems[start, ...,
  // elems+vec-1]`
  Value packElementRangeIntoVector(RewriterBase &rewriter,
                                   const LLVMTypeConverter *typeConverter,
                                   Location loc, VectorType vecTy,
                                   ArrayRef<Value> elems, int64_t start) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    int64_t vec = vecTy.getNumElements();
    // If we need to mask the loaded value with other elements
    Value v = b.undef(vecTy);
    for (size_t s = 0; s < vec; ++s) {
      Value otherElem = elems[start + s];
      Value indexVal =
          LLVM::createIndexConstant(rewriter, loc, typeConverter, s);
      v = b.insert_element(vecTy, v, otherElem, indexVal);
    }
    return v;
  }

  // Unpack the elements contained in a `llvmStruct` into a `SmallVector` of
  // `Value`s. While you do that, check also the alignment of the mask and
  // update the vector length `vec` accordingly
  SmallVector<Value>
  getMaskElemsAndUpdateVeclen(ConversionPatternRewriter &rewriter, Location loc,
                              Value llMask, Value mask, unsigned &vec) const {
    SmallVector<Value> maskElems;
    if (llMask) {
      vec = std::min<size_t>(vec, getMaskAlignment(mask));
      maskElems = unpackLLElements(loc, llMask, rewriter);
    }
    return maskElems;
  }

  unsigned getMaskAlignment(Value mask) const {
    return axisAnalysisPass.getMaskAlignment(mask);
  }

protected:
  const triton::metal::TargetInfo &targetInfo;
  mlir::triton::ModuleAxisInfoAnalysis &axisAnalysisPass;
};

struct LoadOpConversion : public ConvertOpToLLVMPattern<triton::LoadOp>,
                          public LoadStoreConversionBase {
  LoadOpConversion(LLVMTypeConverter &converter,
                   const metal::TargetInfo &targetInfo,
                   mlir::triton::ModuleAxisInfoAnalysis &axisAnalysisPass,
                   PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto b = triton::TritonLLVMOpBuilder(loc, rewriter);

    // original values
    Value ptr = op.getPtr();
    Value mask = op.getMask();
    Value other = op.getOther();

    // adaptor values
    Value llPtr = adaptor.getPtr();
    Value llMask = adaptor.getMask();
    Value llOther = adaptor.getOther();

    // Determine the vectorization size
    Type valueTy = op.getType();
    Type valueElemTy =
        typeConverter->convertType(getElementTypeOrSelf(valueTy));
    unsigned vec = getVectorSize(ptr, axisAnalysisPass);
    unsigned numElems = getTotalElemsPerThread(ptr.getType());

    // Get the LLVM values for pointers
    auto ptrElems = unpackLLElements(loc, llPtr, rewriter);
    assert(ptrElems.size() == numElems);

    // Get the LLVM values for mask
    SmallVector<Value> maskElems =
        getMaskElemsAndUpdateVeclen(rewriter, loc, llMask, mask, vec);

    SmallVector<Value> otherElems;
    if (other) {
      otherElems = unpackLLElements(loc, llOther, rewriter);
    }

    // vectorized iteration through all the pointer/mask/other elements
    const int valueElemNBits =
        std::max(8u, valueElemTy.getIntOrFloatBitWidth());
    const int numVecs = numElems / vec;

    auto cacheMod = op.getCache();
    SmallVector<Value> loadedVals;
    Type vecTy = LLVM::getVectorType(valueElemTy, vec);
    for (size_t vecStart = 0; vecStart < numElems; vecStart += vec) {
      const size_t maxWordWidth = std::max<size_t>(32, valueElemNBits);
      const size_t totalWidth = valueElemNBits * vec;
      const size_t width = std::min(totalWidth, maxWordWidth);
      const size_t nWords = std::max<size_t>(1, totalWidth / width);
      const size_t wordNElems = width / valueElemNBits;
      const size_t movWidth = width < 16 ? 16 : width;
      assert(wordNElems * nWords * numVecs == numElems);

      // mask should be the same for elems in this vector
      Value pred = mask ? maskElems[vecStart] : b.int_val(1, 1);
      Value ptr = ptrElems[vecStart];

      Value falseVal = createZeroVector(rewriter, loc, cast<VectorType>(vecTy));
      // If we need to mask the loaded value with other elements
      if (otherElems.size() != 0) {
        falseVal = packElementRangeIntoVector(
            rewriter, this->getTypeConverter(), loc, cast<VectorType>(vecTy),
            otherElems, vecStart);
      }

      Value loadVal = llLoad(rewriter, loc, ptr, vecTy, pred, falseVal);
      for (size_t ii = 0; ii < vec; ++ii) {
        Value vecIdx = createIndexAttrConstant(
            rewriter, loc, getTypeConverter()->getIndexType(), ii);
        Value loaded = b.extract_element(valueElemTy, loadVal, vecIdx);
        loadedVals.push_back(loaded);
      }

    } // end vec
    Type llvmResultStructTy = getTypeConverter()->convertType(valueTy);
    Value resultStruct = packLLElements(loc, getTypeConverter(), loadedVals,
                                        rewriter, llvmResultStructTy);

    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

struct StoreOpConversion : public ConvertOpToLLVMPattern<triton::StoreOp>,
                           public LoadStoreConversionBase {
  StoreOpConversion(LLVMTypeConverter &converter,
                    const metal::TargetInfo &targetInfo,
                    ModuleAxisInfoAnalysis &axisAnalysisPass,
                    PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value ptr = op.getPtr();
    Value value = op.getValue();
    Value mask = op.getMask();

    Value llPtr = adaptor.getPtr();
    Value llMask = adaptor.getMask();
    Value llValue = adaptor.getValue();

    auto loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    MLIRContext *ctx = rewriter.getContext();
    auto moduleOp = op->getParentOfType<ModuleOp>();

    auto valueTy = value.getType();
    Type valueElemTy =
        typeConverter->convertType(getElementTypeOrSelf(valueTy));

    // Determine the vectorization size
    unsigned vec = getVectorSize(ptr, axisAnalysisPass);
    unsigned elemsPerThread = getTotalElemsPerThread(ptr.getType());

    auto ptrElems = unpackLLElements(loc, llPtr, rewriter);
    auto valueElems = unpackLLElements(loc, llValue, rewriter);
    assert(ptrElems.size() == valueElems.size());

    SmallVector<Value> maskElems =
        getMaskElemsAndUpdateVeclen(rewriter, loc, llMask, mask, vec);

    const size_t valueElemNBits =
        std::max<int>(8, valueElemTy.getIntOrFloatBitWidth());
    const size_t valueElemNBytes = valueElemNBits / 8;

    auto cacheMod = op.getCache();
    const int numVecs = elemsPerThread / vec;
    auto freeVarMasks = getFreeVariableMasks(valueTy);
    Value threadPred = emitRedundantThreadPredicateNonNull(
        freeVarMasks, rewriter, loc, targetInfo);
    uint32_t regMask = freeVarMasks[str_attr("reg")];
    for (size_t vecStart = 0; vecStart < elemsPerThread; vecStart += vec) {
      if (!isCanonicalIndex(vecStart, regMask)) {
        // Don't emit store ops for redundant elements within a thread
        continue;
      }

      Value pred =
          llMask ? b.and_(threadPred, maskElems[vecStart]) : threadPred;

      auto vecTy = LLVM::getVectorType(valueElemTy, vec);

      const size_t maxWordWidth = std::max<size_t>(32, valueElemNBits);
      const size_t totalWidth = valueElemNBits * vec;
      const size_t width = std::min(totalWidth, maxWordWidth);
      const size_t nWords = std::max<size_t>(1, totalWidth / width);
      const size_t wordNElems = width / valueElemNBits;
      assert(wordNElems * nWords * numVecs == elemsPerThread);

      SmallVector<std::pair<Value, std::string>> asmArgs;
      Value elem = valueElems[vecStart];
      Value ptr = ptrElems[vecStart];

      // Create the store val
      Value storeVal = packElementRangeIntoVector(
          rewriter, this->getTypeConverter(), loc, cast<VectorType>(vecTy),
          valueElems, vecStart);
      llStore(rewriter, loc, ptr, storeVal, pred);
    } // end vec
    rewriter.eraseOp(op);
    return success();
  }
};

struct AtomicCASOpConversion
    : public ConvertOpToLLVMPattern<triton::AtomicCASOp>,
      public LoadStoreConversionBase {
  AtomicCASOpConversion(LLVMTypeConverter &converter,
                        const metal::TargetInfo &targetInfo,
                        ModuleAxisInfoAnalysis &axisAnalysisPass,
                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::AtomicCASOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    MLIRContext *ctx = rewriter.getContext();

    Value llPtr = adaptor.getPtr();
    Value llCmp = adaptor.getCmp();
    Value llVal = adaptor.getVal();

    // prep data by unpacking to get data ready
    auto ptrElements = unpackLLElements(loc, llPtr, rewriter);
    auto cmpElements = unpackLLElements(loc, llCmp, rewriter);
    auto valElements = unpackLLElements(loc, llVal, rewriter);

    auto memOrdering = op.getSem();
    auto atomicMemOrdering = getMemoryOrdering(memOrdering);
    if (!atomicMemOrdering)
      return rewriter.notifyMatchFailure(op,
                                         "Unknown Metal GPU memory ordering");
    auto scope = getMetalGPUMemScopeStr(op.getScope());
    if (!scope)
      return rewriter.notifyMatchFailure(op, "Unknown Metal GPU memory scope");

    // deal with tensor or scalar
    auto valueTy = op.getResult().getType();
    auto tensorTy = dyn_cast<RankedTensorType>(valueTy);
    Type valueElemTy =
        tensorTy ? getTypeConverter()->convertType(tensorTy.getElementType())
                 : valueTy;
    auto valueElemNBits = valueElemTy.getIntOrFloatBitWidth();
    Type valueElemIntTy{};
    if (!valueElemTy.isSignlessInteger()) {
      valueElemIntTy = rewriter.getIntegerType(valueElemNBits);
    }
    auto elemsPerThread = getTotalElemsPerThread(op.getVal().getType());
    SmallVector<Value> resultVals(elemsPerThread);

    auto successOrdering = *atomicMemOrdering;
    auto failureOrdering = LLVM::AtomicOrdering::monotonic;
    auto scopeStr = StringRef(scope.value());

    auto freeVarMasks = getFreeVariableMasks(op.getPtr().getType());
    Value threadPred = emitRedundantThreadPredicateNonNull(
        freeVarMasks, rewriter, loc, targetInfo);
    uint32_t regMask = freeVarMasks[str_attr("reg")];

    // atomic ops
    for (size_t i = 0; i < elemsPerThread; i += 1) {
      if (tensorTy && (i & ~regMask) != i) {
        resultVals[i] = resultVals[i & ~regMask];
        continue;
      }

      Value casVal = valElements[i];
      Value casCmp = cmpElements[i];
      Value casPtr = ptrElements[i];
      if (valueElemIntTy) {
        casVal = LLVM::BitcastOp::create(rewriter, loc, valueElemIntTy, casVal);
        casCmp = LLVM::BitcastOp::create(rewriter, loc, valueElemIntTy, casCmp);
      }
      // use op
      if (tensorTy) { // for tensor
        Value undefVal = b.undef(valueElemTy);
        auto *curBlock = rewriter.getInsertionBlock();
        auto *endBlock = curBlock->splitBlock(rewriter.getInsertionPoint());
        auto *atomicBlock = rewriter.createBlock(
            curBlock->getParent(), std::next(Region::iterator(curBlock)));
        endBlock->addArgument({valueElemTy}, {loc});

        rewriter.setInsertionPointToEnd(curBlock);
        LLVM::CondBrOp::create(rewriter, loc, threadPred, atomicBlock, endBlock,
                               undefVal);

        rewriter.setInsertionPointToEnd(atomicBlock);

        auto cmpxchg = LLVM::AtomicCmpXchgOp::create(
            rewriter, loc, casPtr, casCmp, casVal, successOrdering,
            failureOrdering, scopeStr);

        // Extract the new_loaded value from the pair.
        Value ret;
        if (valueElemIntTy) {
          ret = b.extract_val(valueElemIntTy, cmpxchg, 0);
          ret = LLVM::BitcastOp::create(rewriter, loc, valueElemTy, ret);
        } else {
          ret = b.extract_val(valueElemTy, cmpxchg, 0);
        }

        LLVM::BrOp::create(rewriter, loc, ret, endBlock);
        rewriter.setInsertionPointToStart(endBlock);
        resultVals[i] = endBlock->getArgument(0);
      } else { // for scalar
        // Build blocks to bypass the atomic instruction for ~rmwMask.
        auto *curBlock = rewriter.getInsertionBlock();
        auto *endBlock = curBlock->splitBlock(rewriter.getInsertionPoint());
        auto *atomicBlock = rewriter.createBlock(
            curBlock->getParent(), std::next(Region::iterator(curBlock)));

        // Fill entry block with global memory barrier and conditional branch.
        rewriter.setInsertionPointToEnd(curBlock);
        auto tid = getThreadId(rewriter, loc);
        Value pred = b.icmp_eq(tid, b.i32_val(i));
        LLVM::CondBrOp::create(rewriter, loc, pred, atomicBlock, endBlock);

        // Build main block with atomic_cmpxchg.
        rewriter.setInsertionPointToEnd(atomicBlock);

        auto cmpxchg = LLVM::AtomicCmpXchgOp::create(
            rewriter, loc, casPtr, casCmp, casVal, successOrdering,
            failureOrdering, scopeStr);

        if (!op.getResult().use_empty()) {
          // Extract the new_loaded value from the pair.
          Value newLoaded;
          if (valueElemIntTy) {
            newLoaded = b.extract_val(valueElemIntTy, cmpxchg, 0);
            newLoaded =
                LLVM::BitcastOp::create(rewriter, loc, valueElemTy, newLoaded);
          } else {
            newLoaded = b.extract_val(valueElemTy, cmpxchg, 0);
          }
          Value atomPtr =
              getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());
          b.store(newLoaded, atomPtr);
        }

        LLVM::BrOp::create(rewriter, loc, ValueRange(), endBlock);

        // Build the last block: synced load from shared memory, exit.
        rewriter.setInsertionPointToStart(endBlock);

        if (op.getResult().use_empty()) {
          rewriter.eraseOp(op);
          return success();
        }

        b.barrier(triton::gpu::AddrSpace::Local);
        Value atomPtr =
            getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());
        Value ret = b.load(valueElemTy, atomPtr);
        rewriter.replaceOp(op, {ret});
        return success();
      }
    }

    finalizeTensorAtomicResults(op, tensorTy, rewriter, resultVals, valueElemTy,
                                b, threadPred, targetInfo, getTypeConverter());
    return success();
  }
};

// similar implementation to AtomicCasOpConversion
struct AtomicRMWOpConversion
    : public ConvertOpToLLVMPattern<triton::AtomicRMWOp>,
      public LoadStoreConversionBase {
  AtomicRMWOpConversion(LLVMTypeConverter &converter,
                        const metal::TargetInfo &targetInfo,
                        ModuleAxisInfoAnalysis &axisAnalysisPass,
                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::AtomicRMWOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    MLIRContext *ctx = rewriter.getContext();

    auto binOp = matchAtomicOp(op.getAtomicRmwOp());
    if (!binOp)
      return rewriter.notifyMatchFailure(op, "Unsupported RMW operation");

    auto memOrder = getMemoryOrdering(op.getSem());
    if (!memOrder)
      return rewriter.notifyMatchFailure(op, "Unsupported RMW memory order");

    auto scopeStr = getMetalGPUMemScopeStr(op.getScope());
    if (!scopeStr)
      return rewriter.notifyMatchFailure(op, "Unsupported RMW scope");

    Value val = op.getVal();
    Value opResult = op.getResult();

    Value llPtr = adaptor.getPtr();
    Value llVal = adaptor.getVal();
    Value llMask = adaptor.getMask();

    auto valElements = unpackLLElements(loc, llVal, rewriter);
    auto ptrElements = unpackLLElements(loc, llPtr, rewriter);
    SmallVector<Value> maskElements;
    if (llMask)
      maskElements = unpackLLElements(loc, llMask, rewriter);

    auto tensorTy = dyn_cast<RankedTensorType>(opResult.getType());
    Type valueElemTy =
        tensorTy ? getTypeConverter()->convertType(tensorTy.getElementType())
                 : opResult.getType();

    auto elemsPerThread = getTotalElemsPerThread(val.getType());

    auto freeVarMasks = getFreeVariableMasks(op.getPtr().getType());
    Value threadPred = emitRedundantThreadPredicateNonNull(
        freeVarMasks, rewriter, loc, targetInfo);
    uint32_t regMask = freeVarMasks[str_attr("reg")];

    auto scopeAttr = StringAttr::get(ctx, scopeStr.value());

    SmallVector<Value> resultVals(elemsPerThread);
    for (size_t i = 0; i < elemsPerThread; i += 1) {
      if (tensorTy && (i & ~regMask) != i) {
        resultVals[i] = resultVals[i & ~regMask];
        continue;
      }

      Value rmwMask = llMask ? b.and_(threadPred, maskElements[i]) : threadPred;
      Value rmwVal = valElements[i];
      Value rmwPtr = ptrElements[i];

      if (tensorTy) {
        Value undefVal = b.undef(valueElemTy);
        auto *curBlock = rewriter.getInsertionBlock();
        auto *endBlock = curBlock->splitBlock(rewriter.getInsertionPoint());
        auto *atomicBlock = rewriter.createBlock(
            curBlock->getParent(), std::next(Region::iterator(curBlock)));
        endBlock->addArgument({valueElemTy}, {loc});

        rewriter.setInsertionPointToEnd(curBlock);
        LLVM::CondBrOp::create(rewriter, loc, rmwMask, atomicBlock, endBlock,
                               undefVal);

        rewriter.setInsertionPointToEnd(atomicBlock);
        Value retVal = LLVM::AtomicRMWOp::create(rewriter, loc, *binOp, rmwPtr,
                                                 rmwVal, *memOrder, scopeAttr);

        LLVM::BrOp::create(rewriter, loc, retVal, endBlock);
        rewriter.setInsertionPointToStart(endBlock);
        resultVals[i] = endBlock->getArgument(0);
      } else {
        auto *curBlock = rewriter.getInsertionBlock();
        auto *endBlock = curBlock->splitBlock(rewriter.getInsertionPoint());
        auto *atomicBlock = rewriter.createBlock(
            curBlock->getParent(), std::next(Region::iterator(curBlock)));

        rewriter.setInsertionPointToEnd(curBlock);
        auto tid = getThreadId(rewriter, loc);
        Value pred = b.icmp_eq(tid, b.i32_val(i));
        LLVM::CondBrOp::create(rewriter, loc, pred, atomicBlock, endBlock);

        rewriter.setInsertionPointToEnd(atomicBlock);
        Value retVal = LLVM::AtomicRMWOp::create(rewriter, loc, *binOp, rmwPtr,
                                                 rmwVal, *memOrder, scopeAttr);

        if (!opResult.use_empty()) {
          Value atomPtr =
              getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());
          b.store(retVal, atomPtr);
        }

        LLVM::BrOp::create(rewriter, loc, ValueRange(), endBlock);

        rewriter.setInsertionPointToStart(endBlock);

        if (opResult.use_empty()) {
          rewriter.eraseOp(op);
          return success();
        }

        b.barrier(triton::gpu::AddrSpace::Local);
        Value atomPtr =
            getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());
        Value ret = b.load(valueElemTy, atomPtr);
        rewriter.replaceOp(op, {ret});
        return success();
      }
    }

    finalizeTensorAtomicResults(op, tensorTy, rewriter, resultVals, valueElemTy,
                                b, threadPred, targetInfo, getTypeConverter());
    return success();
  }
};



} // namespace

namespace mlir::triton::metal {
void populateLoadStoreOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                       const TargetInfo &targetInfo,
                                       RewritePatternSet &patterns,
                                       ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                       PatternBenefit benefit) {
  patterns.add<LoadOpConversion, StoreOpConversion, AtomicCASOpConversion,
               AtomicRMWOpConversion>(
      typeConverter, targetInfo, axisInfoAnalysis, benefit);
}


} // namespace mlir::triton::metal

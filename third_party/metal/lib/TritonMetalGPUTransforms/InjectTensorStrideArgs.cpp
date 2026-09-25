#include "TritonMetalGPUTransforms/Passes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"

#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

#undef DEBUG_TYPE
#define DEBUG_TYPE "tritonmetal-inject-tensor-stride-args"

namespace mlir {

static std::optional<unsigned> findTensorInputArg(Value ptr) {
  Value cur = ptr;
  
  while (true) {
    if (auto blockArg = dyn_cast<BlockArgument>(cur)) {
      if (auto forOp = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp())) {
        unsigned iterArgIdx = blockArg.getArgNumber() - forOp.getNumInductionVars();
        cur = forOp.getInitArgs()[iterArgIdx];
        continue;
      }
    }

    if (auto addptr = cur.getDefiningOp<tt::AddPtrOp>()) {
      cur = addptr.getPtr();
      continue;
    }
    if (auto bcast = cur.getDefiningOp<tt::BroadcastOp>()) {
      cur = bcast.getSrc();
      continue;
    }
    if (auto expand = cur.getDefiningOp<tt::ExpandDimsOp>()) {
      cur = expand.getSrc();
      continue;
    }
    
    break;
  }

  auto splatOp = cur.getDefiningOp<tt::SplatOp>();
  if (!splatOp)
    return std::nullopt;

  Value scalarPtr = splatOp.getSrc();
  while (scalarPtr) {
    if (auto addptr = scalarPtr.getDefiningOp<tt::AddPtrOp>()) {
      scalarPtr = addptr.getPtr();
      continue;
    }
    if (auto bitcast = scalarPtr.getDefiningOp<tt::BitcastOp>()) {
      scalarPtr = bitcast.getSrc();
      continue;
    }
    break;
  }

  auto ptrArg = dyn_cast<BlockArgument>(scalarPtr);
  if (!ptrArg) {
    return std::nullopt;
  }

  return ptrArg.getArgNumber();
}

#define GEN_PASS_DEF_TRITONMETALGPUINJECTTENSORSTRIDEARGS
#include "TritonMetalGPUTransforms/Passes.h.inc"

struct TritonMetalGPUInjectTensorStrideArgsPass
    : impl::TritonMetalGPUInjectTensorStrideArgsBase<
          TritonMetalGPUInjectTensorStrideArgsPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    mod.walk([&](tt::FuncOp funcOp) {
      if (!triton::isKernel(funcOp))
        return;

      auto *ctx = funcOp.getContext();
      // stride arg is a pointer to a packed array of i64 strides
      auto stridePtrTy =
          tt::PointerType::get(IntegerType::get(ctx, 64), /*addressSpace=*/1);

      // collect ptr arg indices before modification
      SmallVector<unsigned> ptrArgIndices;
      for (unsigned i = 0; i < funcOp.getNumArguments(); ++i) {
        if (isa<tt::PointerType>(funcOp.getArgument(i).getType()))
          ptrArgIndices.push_back(i);
      }

      // insert stride arg immediately after each tensor ptr arg
      // track insertOffset since insertion shifts later indices
      unsigned insertOffset = 0;
      for (unsigned origIdx : ptrArgIndices) {
        unsigned insertIdx = origIdx + 1 + insertOffset;
        (void)funcOp.insertArgument(insertIdx, stridePtrTy,
                                    DictionaryAttr::get(ctx), funcOp.getLoc());
        // Record which original ptr arg this stride belongs to.
        funcOp.setArgAttr(insertIdx, "metal.implicit_stride_for",
                          IntegerAttr::get(IntegerType::get(ctx, 32),
                                           origIdx + insertOffset));
        insertOffset++;
      }

      // Pre-compute the stride arg index for all async copies before lowering ruins the IR
      funcOp.walk([&](ttg::AsyncCopyGlobalToLocalOp copyOp) {
        auto argIdx = findTensorInputArg(copyOp.getSrc());
        if (argIdx) {
          copyOp->setAttr("metal.stride_arg_idx",
                          IntegerAttr::get(IntegerType::get(ctx, 32), *argIdx));
        }
      });
    });
  }
};

} // namespace mlir

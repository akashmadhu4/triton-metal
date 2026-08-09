#ifndef TRITON_THIRD_PARTY_METAL_INCLUDE_TRITONMETALGPUTOLLVM_PASSES_H_
#define TRITON_THIRD_PARTY_METAL_INCLUDE_TRITONMETALGPUTOLLVM_PASSES_H_

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/IR/Function.h"

#include <memory>

namespace mlir {
class ModuleOp;
template <typename T> class OperationPass;

} // namespace mlir

namespace mlir::triton {

#define GEN_PASS_DECL
#include "TritonMetalGPUToLLVM/Passes.h.inc"

} // namespace mlir::triton

namespace mlir::triton {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonMetalGPUToLLVMPass(StringRef targetArch);

std::unique_ptr<OperationPass<ModuleOp>>
createTritonMetalGPUAccelerateMatmul();


#define GEN_PASS_REGISTRATION
#include "TritonMetalGPUToLLVM/Passes.h.inc"

} // namespace mlir::triton

#endif // TRITON_THIRD_PARTY_METAL_INCLUDE_TRITONMETALGPUTOLLVM_PASSES_H_
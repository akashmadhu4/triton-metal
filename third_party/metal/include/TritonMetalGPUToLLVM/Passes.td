#ifndef TRITONMETALGPU_CONVERSION_PASSES
#define TRITONMETALGPU_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTritonMetalGPUToLLVM : Pass<"convert-triton-metalgpu-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert TritonGPU to LLVM";
    let constructor = "mlir::triton::createConvertTritonMetalGPUToLLVMPass(\"\")";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::math::MathDialect",
                             "mlir::gpu::GPUDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::LLVM::LLVMDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::gpu::TritonGPUDialect"];

    let options = [
        Option<"arch", "arch", "std::string", /*default*/"\"\"",
               "target device architecture">,
    ];
}

def TritonMetalGPUAccelerateMatmul : Pass<"accelerate-metal-matmul", "mlir::ModuleOp"> {
    let summary = "accelerate matmul";
    let constructor = "mlir::triton::createTritonMetalGPUAccelerateMatmul()";

    let dependentDialects = ["mlir::triton::TritonDialect",
                             "mlir::triton::gpu::TritonGPUDialect"];
}

#endif

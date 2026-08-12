// #include "TritonMetalGPUTransforms/Passes.h"
// #include "mlir/Dialect/SCF/IR/SCF.h"
// #include "mlir/IR/TypeUtilities.h"
// #include "mlir/Transforms/GreedyPatternRewriteDriver.h"
// #include "triton/Dialect/Triton/IR/Dialect.h"
// #include "triton/Dialect/TritonGPU/IR/Dialect.h"
// 
// namespace tt = mlir::triton;
// namespace ttg = mlir::triton::gpu;
// 
// #undef DEBUG_TYPE
// #define DEBUG_TYPE "tritonmetal-accelerate-matmul"
// 
// namespace mlir {
// namespace {
// Value findBasePtr(PatternRewriter &rewriter, Value val, tt::DotOp dotOp) {
//   auto convertLayoutOp = val.getDefiningOp<ttg::ConvertLayoutOp>();
//   if (!convertLayoutOp) {
//     llvm::errs() << "failed to find convert layout op\n";
//     return {};
//   }
// 
//   auto ttLoadOp = convertLayoutOp.getSrc().getDefiningOp<tt::LoadOp>();
//   if (!ttLoadOp) {
//     llvm::errs() << "failed to find load op\n";
//     return {};
//   }
// 
//   auto ttLoadOpPtr = ttLoadOp.getPtr();
//   Value forOpArg{};
// 
//   // need to look at init args of for op
//   if (auto blockArg = dyn_cast<BlockArgument>(ttLoadOpPtr)) {
//     auto parentOp = blockArg.getOwner()->getParentOp();
//     if (auto forOp = dyn_cast<mlir::scf::ForOp>(parentOp)) {
//       auto initArgs = forOp.getInitArgs();
//       auto argIndex = blockArg.getArgNumber() - forOp.getNumInductionVars();
//       if (argIndex < 0 || argIndex >= initArgs.size()) {
//         return {};
//       }
// 
//       forOpArg = initArgs[argIndex];
//     } else {
//       llvm::errs() << "failed to find parent for block argument\n";
//       return {};
//     }
//   } else {
//     llvm::errs() << "failed to find block argument\n";
//     return {};
//   }
// 
//   if (!forOpArg) {
//     llvm::errs() << "failed to find for op argument\n";
//     return {};
//   }
// 
//   // at this point, defining op of forOpArg should be addptr op
//   auto addPtrResult = forOpArg;
//   while (auto addPtrOp = addPtrResult.getDefiningOp<tt::AddPtrOp>()) {
//     addPtrResult = addPtrOp.getPtr();
//   }
// 
//   auto ttSplatOp = addPtrResult.getDefiningOp<tt::SplatOp>();
//   if (!ttSplatOp) {
//     llvm::errs() << "failed to find splat op\n";
//     return {};
//   }
// 
//   return ttSplatOp.getSrc();
// }
// 
// Value findStoreBasePtr(PatternRewriter &rewriter, Value val, tt::DotOp dotOp) {
//   auto dotResultUsers = val.getUsers();
// 
//   ttg::ConvertLayoutOp convertLayoutOp;
//   for (auto user : dotResultUsers) {
//     auto convLOp = dyn_cast<ttg::ConvertLayoutOp>(user);
//     if (convLOp) {
//       convertLayoutOp = convLOp;
//       break;
//     }
//   }
//   if (!convertLayoutOp) {
//     llvm::errs()
//         << "failed to find convert layout op that uses result of tt.dot\n";
//     return {};
//   }
// 
//   auto convertLayoutResult = convertLayoutOp.getResult();
//   return {};
// }
// 
// class DotToMetalMatmul : public OpRewritePattern<tt::DotOp> {
// public:
//   DotToMetalMatmul(MLIRContext *context, PatternBenefit benefit = 1)
//       : OpRewritePattern(context, benefit) {}
// 
//   LogicalResult matchAndRewrite(tt::DotOp dotOp,
//                                 PatternRewriter &rewriter) const override {
//     auto oldRetType = cast<RankedTensorType>(dotOp.getResult().getType());
//     auto elemType = oldRetType.getElementType();
//     if (!elemType.isF32()) {
//       return rewriter.notifyMatchFailure(dotOp, "expected f32 element type");
//     }
//     auto retShape = oldRetType.getShape();
//     int numWarps = ttg::lookupNumWarps(dotOp);
// 
//     // find base ptrs for A and B
//     Value aBasePtr = findBasePtr(rewriter, dotOp.getA(), dotOp);
//     Value bBasePtr = findBasePtr(rewriter, dotOp.getB(), dotOp);
//     if (!aBasePtr || !bBasePtr) {
//       llvm::errs() << "failed to find base ptrs\n";
//       return rewriter.notifyMatchFailure(
//           dotOp, "failed to find base ptrs for A and B");
//     }
// 
//     // TODO need to look forward to find the store op that uses dot op
//     // accumulated result
//     Value cBasePtr = findStoreBasePtr(rewriter, dotOp.getC(), dotOp);
// 
//     // TODO need to get the strides for A, B, and C
// 
//     return success();
//   }
// };
// } // namespace
// 
// #define GEN_PASS_DEF_TRITONMETALGPUACCELERATEMATMUL
// #include "TritonMetalGPUTransforms/Passes.h.inc"
// 
// struct TritonMetalGPUAccelerateMatmulPass
//     : impl::TritonMetalGPUAccelerateMatmulBase<
//           TritonMetalGPUAccelerateMatmulPass> {
//   using Base::Base;
// 
//   void runOnOperation() override {
//     MLIRContext *context = &getContext();
//     ModuleOp m = getOperation();
// 
//     RewritePatternSet patterns(context);
//     patterns.add<DotToMetalMatmul>(context);
//     if (applyPatternsGreedily(m, std::move(patterns)).failed())
//       signalPassFailure();
//   };
// };
// 
// } // namespace mlir
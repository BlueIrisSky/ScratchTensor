import { TensorNode, getAllLeafParameters, unregisterTensorById } from './tensor.js';
import { OpNode } from './types.js';
import { onesNdArray } from './ndarray-utils.js';

let graphTrackingEnabled = true;

/**
 * [设置计算图追踪: <true/false>]
 */
export function setGraphTracking(enabled: boolean): void {
  graphTrackingEnabled = enabled;
}

/**
 * Check if dynamic computation graph tracking is currently enabled.
 */
export function isGraphTracking(): boolean {
  return graphTrackingEnabled;
}

/**
 * [对 Tensor <target> 进行反向传播]
 * Implements Eager Graph Cleanup: after backward, non-leaf intermediate
 * tensors are detached from the graph and removed from the global registry
 * (see docs/autograd.md).
 */
export function backward(target: TensorNode): void {
  if (!target.creatorOp && !target.requiresGrad) {
    return;
  }

  // Build reverse topological order of OpNodes reachable from target
  const visitedOps = new Set<OpNode>();
  const topoOrder: OpNode[] = [];
  const visitedTensors = new Set<TensorNode>();

  function buildTopo(tensor: TensorNode) {
    if (visitedTensors.has(tensor)) return;
    visitedTensors.add(tensor);

    if (tensor.creatorOp && !visitedOps.has(tensor.creatorOp)) {
      visitedOps.add(tensor.creatorOp);
      for (const input of tensor.creatorOp.inputs) {
        buildTopo(input);
      }
      topoOrder.push(tensor.creatorOp);
    }
  }

  buildTopo(target);

  // Initialize target grad to ones if null (e.g. loss scalar 1.0)
  if (!target.grad) {
    target.grad = onesNdArray(target.shape);
  }

  // Reverse topological traversal: execute backwardFn
  for (let i = topoOrder.length - 1; i >= 0; i--) {
    const op = topoOrder[i];
    const gradOutput = op.output.grad;
    if (gradOutput) {
      op.backwardFn(gradOutput);
    }
  }

  // 4.1 & 4.2: Eager Graph Cleanup (Post-Backward Graph Release)
  // Release creatorOp and temporary gradients on non-leaf intermediate Tensors.
  // Also remove them from the global registry so their NDArray data can be GC'd.
  for (const tensor of visitedTensors) {
    if (!tensor.isLeaf && tensor.creatorOp !== null) {
      tensor.creatorOp = null;
      tensor.grad = null;
      if (tensor !== target) {
        unregisterTensorById(tensor.id);
      }
    }
  }
}

/**
 * [清空所有参数梯度] / [清空grad]
 */
export function clearAllGradients(): void {
  const leaves = getAllLeafParameters();
  for (const leaf of leaves) {
    leaf.clearGrad();
  }
}

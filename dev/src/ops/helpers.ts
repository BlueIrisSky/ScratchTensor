import { OpNode } from '../types.js';
import { TensorNode, nextId, registerTensor } from '../tensor.js';
import { isGraphTracking } from '../autograd.js';

export function requiresGradAny(nodes: TensorNode[]): boolean {
  for (let i = 0; i < nodes.length; i++) {
    if (nodes[i].requiresGrad) return true;
  }
  return false;
}

export function markRequiresGrad(op: OpNode, nodes: TensorNode[]): void {
  const grad = requiresGradAny(nodes) && isGraphTracking();
  op.output.requiresGrad = grad;
  op.output.isLeaf = false;
  if (grad) {
    op.output.creatorOp = op;
  }
  registerTensor(op.output);
}

export function initOpOutput(op: OpNode, outName: string, outData: any, gradInputs: TensorNode[] = op.inputs): TensorNode {
  const requiresGrad = requiresGradAny(gradInputs) && isGraphTracking();
  const output = new TensorNode(nextId(), outName, outData, requiresGrad);
  output.isLeaf = false;
  if (requiresGrad) {
    output.creatorOp = op;
  }
  registerTensor(output);
  return output;
}

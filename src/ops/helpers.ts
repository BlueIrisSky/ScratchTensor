import { OpNode } from '../types.js';
import { TensorNode, nextId, registerTensor, getTensorByName } from '../tensor.js';
import { isGraphTracking } from '../autograd.js';

export function requiresGradAny(nodes: TensorNode[]): boolean {
  for (let i = 0; i < nodes.length; i++) {
    if (nodes[i].requiresGrad) return true;
  }
  return false;
}

/** Reject accidental replacement of model parameters by operation results. */
export function assertOpOutputName(name: string): void {
  const existing = getTensorByName(name);
  if (existing?.isLeaf && existing.requiresGrad) {
    throw new Error(`Operation output "${name}" cannot overwrite a trainable leaf tensor; choose a different output name`);
  }
}

export function markRequiresGrad(op: OpNode, nodes: TensorNode[]): void {
  assertOpOutputName(op.output.name);
  const grad = requiresGradAny(nodes) && isGraphTracking();
  op.output.requiresGrad = grad;
  op.output.isLeaf = false;
  if (grad) {
    op.output.creatorOp = op;
  }
  registerTensor(op.output);
}

export function initOpOutput(op: OpNode, outName: string, outData: any, gradInputs: TensorNode[] = op.inputs): TensorNode {
  assertOpOutputName(outName);
  const requiresGrad = requiresGradAny(gradInputs) && isGraphTracking();
  const output = new TensorNode(nextId(), outName, outData, requiresGrad);
  output.isLeaf = false;
  if (requiresGrad) {
    output.creatorOp = op;
  }
  registerTensor(output);
  return output;
}

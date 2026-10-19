import { TensorNode } from '../tensor.js';
import { OpNode } from '../types.js';
import { isGraphTracking } from '../autograd.js';
import { initOpOutput } from './helpers.js';
import {
  scaledDotProductAttentionNdArray,
  scaledDotProductAttentionGradNdArray
} from '../ndarray-utils.js';

export class ScaledDotProductAttentionOp implements OpNode {
  opName = 'ScaledDotProductAttention';
  inputs: TensorNode[];
  output: TensorNode;
  private attn: any;
  private dropoutMask: Float32Array | null;
  private scale: number;
  private isTraining: boolean;
  private dropoutP: number;

  constructor(
    Q: TensorNode,
    K: TensorNode,
    V: TensorNode,
    outName: string,
    isCausal = false,
    dropoutP = 0,
    scale?: number,
    attnMask?: TensorNode
  ) {
    this.inputs = [Q, K, V];
    this.dropoutP = dropoutP;
    this.isTraining = isGraphTracking();
    const maskData = attnMask ? attnMask.data : undefined;
    const { out, attn, dropoutMask, scale: effScale } = scaledDotProductAttentionNdArray(
      Q.data, K.data, V.data, this.isTraining, dropoutP, isCausal, scale, maskData
    );
    this.attn = attn;
    this.dropoutMask = dropoutMask;
    this.scale = effScale;
    this.output = initOpOutput(this, outName, out);
  }

  backwardFn(gradOutput: any): void {
    const [Q, K, V] = this.inputs;
    const { dQ, dK, dV } = scaledDotProductAttentionGradNdArray(
      gradOutput, Q.data, K.data, V.data, this.attn, this.dropoutMask, this.scale, this.isTraining, this.dropoutP
    );
    if (Q.requiresGrad) Q.addGrad(dQ);
    if (K.requiresGrad) K.addGrad(dK);
    if (V.requiresGrad) V.addGrad(dV);
  }
}

export function scaledDotProductAttention(
  Q: TensorNode,
  K: TensorNode,
  V: TensorNode,
  outName: string,
  isCausal = false,
  dropoutP = 0,
  scale?: number,
  attnMask?: TensorNode
): TensorNode {
  return new ScaledDotProductAttentionOp(Q, K, V, outName, isCausal, dropoutP, scale, attnMask).output;
}

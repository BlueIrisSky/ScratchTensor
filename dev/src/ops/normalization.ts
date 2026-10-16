import { TensorNode } from '../tensor.js';
import { OpNode } from '../types.js';
import { isGraphTracking } from '../autograd.js';
import { initOpOutput } from './helpers.js';
import {
  layerNormNdArray,
  layerNormGradNdArray,
  dropoutNdArray,
  createNdArray
} from '../ndarray-utils.js';

export class LayerNormOp implements OpNode {
  opName = 'LayerNorm';
  inputs: TensorNode[];
  output: TensorNode;
  private n: number;
  private mean: any;
  private invStd: any;

  constructor(X: TensorNode, n: number, outName: string) {
    this.inputs = [X];
    this.n = n;
    const { out, mean, invStd } = layerNormNdArray(X.data, n);
    this.mean = mean;
    this.invStd = invStd;
    this.output = initOpOutput(this, outName, out);
  }

  backwardFn(gradOutput: any): void {
    const [X] = this.inputs;
    if (X.requiresGrad) {
      X.addGrad(layerNormGradNdArray(gradOutput, X.data, this.mean, this.invStd, this.n));
    }
  }
}

export class DropoutOp implements OpNode {
  opName = 'Dropout';
  inputs: TensorNode[];
  output: TensorNode;
  private mask: Float32Array;
  private scale: number;

  constructor(X: TensorNode, p: number, outName: string) {
    this.inputs = [X];
    const isTraining = isGraphTracking();
    const { out, mask } = dropoutNdArray(X.data, p, isTraining);
    this.mask = mask;
    this.scale = isTraining ? (p >= 1 ? 1.0 : 1.0 / (1.0 - p)) : 1.0;
    this.output = initOpOutput(this, outName, out);
  }

  backwardFn(gradOutput: any): void {
    const [X] = this.inputs;
    if (X.requiresGrad) {
      const gData = gradOutput.data;
      const mask = this.mask;
      const scale = this.scale;
      const size = gData.length;
      const buf = new Float32Array(size);
      for (let i = 0; i < size; i++) {
        buf[i] = gData[i] * mask[i] * scale;
      }
      X.addGrad(createNdArray(buf, gradOutput.shape.slice()));
    }
  }
}

export function layerNorm(X: TensorNode, n: number, outName: string): TensorNode {
  return new LayerNormOp(X, n, outName).output;
}

export function dropout(X: TensorNode, p: number, outName: string): TensorNode {
  return new DropoutOp(X, p, outName).output;
}

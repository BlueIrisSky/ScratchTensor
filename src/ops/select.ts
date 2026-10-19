import { TensorNode } from '../tensor.js';
import { OpNode } from '../types.js';
import { initOpOutput } from './helpers.js';
import {
  whereNdArray,
  computeBroadcastShape,
  broadcastNdArray,
  clampNdArray,
  sumToShapeNdArray,
  createNdArray
} from '../ndarray-utils.js';

export class WhereOp implements OpNode {
  opName = 'Where';
  inputs: TensorNode[];
  output: TensorNode;
  private broadcastShape: number[];

  constructor(cond: TensorNode, a: TensorNode, b: TensorNode, outName: string) {
    this.inputs = [cond, a, b];
    this.broadcastShape = computeBroadcastShape([cond.shape, a.shape, b.shape]);
    this.output = initOpOutput(this, outName, whereNdArray(cond.data, a.data, b.data), [a, b]);
  }

  backwardFn(gradOutput: any): void {
    const [cond, a, b] = this.inputs;
    const condBroadcast = broadcastNdArray(cond.data, this.broadcastShape);
    const cData = condBroadcast.data;
    const gData = gradOutput.data;
    const size = gData.length;
    if (a.requiresGrad) {
      const dABuf = new Float32Array(size);
      for (let i = 0; i < size; i++) {
        dABuf[i] = cData[i] ? gData[i] : 0;
      }
      a.addGrad(sumToShapeNdArray(createNdArray(dABuf, gradOutput.shape.slice()), a.shape));
    }
    if (b.requiresGrad) {
      const dBBuf = new Float32Array(size);
      for (let i = 0; i < size; i++) {
        dBBuf[i] = cData[i] ? 0 : gData[i];
      }
      b.addGrad(sumToShapeNdArray(createNdArray(dBBuf, gradOutput.shape.slice()), b.shape));
    }
  }
}

export class ClampOp implements OpNode {
  opName = 'Clamp';
  inputs: TensorNode[];
  output: TensorNode;
  private min: number;
  private max: number;

  constructor(X: TensorNode, min: number, max: number, outName: string) {
    this.inputs = [X];
    this.min = min;
    this.max = max;
    this.output = initOpOutput(this, outName, clampNdArray(X.data, min, max));
  }

  backwardFn(gradOutput: any): void {
    const [X] = this.inputs;
    if (X.requiresGrad) {
      const gData = gradOutput.data;
      const xData = X.data.data;
      const size = gData.length;
      const dXBuf = new Float32Array(size);
      const min = this.min;
      const max = this.max;
      for (let i = 0; i < size; i++) {
        const v = xData[i];
        dXBuf[i] = v >= min && v <= max ? gData[i] : 0;
      }
      X.addGrad(createNdArray(dXBuf, gradOutput.shape.slice()));
    }
  }
}

export function where(cond: TensorNode, a: TensorNode, b: TensorNode, outName: string): TensorNode {
  return new WhereOp(cond, a, b, outName).output;
}

export function clamp(X: TensorNode, min: number, max: number, outName: string): TensorNode {
  return new ClampOp(X, min, max, outName).output;
}

import { TensorNode } from '../tensor.js';
import { OpNode } from '../types.js';
import { initOpOutput } from './helpers.js';
import {
  sumAlongDimNdArray,
  meanNdArray,
  maxNdArray,
  minNdArray,
  unsqueezeNdArray,
  expandNdArray,
  broadcastNdArray,
  normalizeDim,
  getAxisStrides,
  createNdArray
} from '../ndarray-utils.js';

export class SumOp implements OpNode {
  opName = 'Sum';
  inputs: TensorNode[];
  output: TensorNode;
  private dim: number | null;
  private keepdim: boolean;

  constructor(X: TensorNode, dim: number | null, outName: string, keepdim = false) {
    this.inputs = [X];
    this.dim = dim;
    this.keepdim = keepdim;
    this.output = initOpOutput(this, outName, sumAlongDimNdArray(X.data, dim, keepdim));
  }

  backwardFn(gradOutput: any): void {
    const [X] = this.inputs;
    if (X.requiresGrad) {
      if (this.dim === null || this.dim === undefined) {
        X.addGrad(broadcastNdArray(gradOutput, X.shape));
      } else {
        const d = normalizeDim(this.dim, X.shape.length);
        const aligned = this.keepdim ? gradOutput : unsqueezeNdArray(gradOutput, d);
        X.addGrad(expandNdArray(aligned, X.shape));
      }
    }
  }
}

export class MeanOp implements OpNode {
  opName = 'Mean';
  inputs: TensorNode[];
  output: TensorNode;
  private dim: number | null;
  private keepdim: boolean;

  constructor(X: TensorNode, dim: number | null, outName: string, keepdim = false) {
    this.inputs = [X];
    this.dim = dim;
    this.keepdim = keepdim;
    this.output = initOpOutput(this, outName, meanNdArray(X.data, dim, keepdim));
  }

  backwardFn(gradOutput: any): void {
    const [X] = this.inputs;
    if (X.requiresGrad) {
      const gData = gradOutput.data;
      if (this.dim === null || this.dim === undefined) {
        const scale = 1.0 / X.data.data.length;
        const buf = new Float32Array(gData.length);
        for (let i = 0; i < buf.length; i++) buf[i] = gData[i] * scale;
        const scaled = createNdArray(buf, gradOutput.shape.slice());
        X.addGrad(broadcastNdArray(scaled, X.shape));
      } else {
        const d = normalizeDim(this.dim, X.shape.length);
        const scale = 1.0 / X.shape[d];
        const buf = new Float32Array(gData.length);
        for (let i = 0; i < buf.length; i++) buf[i] = gData[i] * scale;
        const scaled = createNdArray(buf, gradOutput.shape.slice());
        const aligned = this.keepdim ? scaled : unsqueezeNdArray(scaled, d);
        X.addGrad(expandNdArray(aligned, X.shape));
      }
    }
  }
}

function scatterExtremaGrad(
  xNd: any,
  gradOutput: any,
  dim: number | null,
  findMax: boolean
): any {
  const xData: Float32Array = xNd.data;
  const xOff: number = xNd.offset;
  const gData: Float32Array = gradOutput.data;
  const gOff: number = gradOutput.offset;
  const gradBuf = new Float32Array(xData.length);

  if (dim === null || dim === undefined) {
    let bestVal = findMax ? -Infinity : Infinity;
    let bestIdx = 0;
    for (let i = 0; i < xData.length; i++) {
      const v = xData[xOff + i];
      if (findMax ? v > bestVal : v < bestVal) {
        bestVal = v;
        bestIdx = i;
      }
    }
    gradBuf[bestIdx] = gData[gOff];
    return createNdArray(gradBuf, xNd.shape.slice());
  }

  const d = normalizeDim(dim, xNd.shape.length);
  const { outerSize, dimSize, innerSize } = getAxisStrides(xNd.shape, d);
  const outerStride = dimSize * innerSize;

  for (let outer = 0; outer < outerSize; outer++) {
    const outBase = outer * innerSize;
    const inOuterBase = xOff + outer * outerStride;
    for (let inner = 0; inner < innerSize; inner++) {
      let bestVal = findMax ? -Infinity : Infinity;
      let bestIdx = inOuterBase + inner;
      for (let k = 0; k < dimSize; k++) {
        const idx = inOuterBase + k * innerSize + inner;
        const v = xData[idx];
        if (findMax ? v > bestVal : v < bestVal) {
          bestVal = v;
          bestIdx = idx;
        }
      }
      gradBuf[bestIdx] = gData[gOff + outBase + inner];
    }
  }

  return createNdArray(gradBuf, xNd.shape.slice());
}

export class MaxOp implements OpNode {
  opName = 'Max';
  inputs: TensorNode[];
  output: TensorNode;
  private dim: number | null;

  constructor(X: TensorNode, dim: number | null, outName: string, keepdim = false) {
    this.inputs = [X];
    this.dim = dim;
    this.output = initOpOutput(this, outName, maxNdArray(X.data, dim, keepdim));
  }

  backwardFn(gradOutput: any): void {
    const [X] = this.inputs;
    if (X.requiresGrad) {
      X.addGrad(scatterExtremaGrad(X.data, gradOutput, this.dim, true));
    }
  }
}

export class MinOp implements OpNode {
  opName = 'Min';
  inputs: TensorNode[];
  output: TensorNode;
  private dim: number | null;

  constructor(X: TensorNode, dim: number | null, outName: string, keepdim = false) {
    this.inputs = [X];
    this.dim = dim;
    this.output = initOpOutput(this, outName, minNdArray(X.data, dim, keepdim));
  }

  backwardFn(gradOutput: any): void {
    const [X] = this.inputs;
    if (X.requiresGrad) {
      X.addGrad(scatterExtremaGrad(X.data, gradOutput, this.dim, false));
    }
  }
}

export function sum(X: TensorNode, dim: number | null, outName: string, keepdim = false): TensorNode {
  return new SumOp(X, dim, outName, keepdim).output;
}

export function mean(X: TensorNode, dim: number | null, outName: string, keepdim = false): TensorNode {
  return new MeanOp(X, dim, outName, keepdim).output;
}

export function max(X: TensorNode, dim: number | null, outName: string, keepdim = false): TensorNode {
  return new MaxOp(X, dim, outName, keepdim).output;
}

export function min(X: TensorNode, dim: number | null, outName: string, keepdim = false): TensorNode {
  return new MinOp(X, dim, outName, keepdim).output;
}

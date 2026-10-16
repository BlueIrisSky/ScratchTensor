import { TensorNode } from '../tensor.js';
import { OpNode } from '../types.js';
import { initOpOutput } from './helpers.js';
import {
  createNdArray,
  zerosNdArray,
  resolveReshapeShape,
  unsqueezeNdArray,
  squeezeNdArray,
  expandNdArray,
  sumToShapeNdArray,
  concatNdArray,
  splitNdArray,
  stackNdArray,
  sliceNdArray,
  transposeNdArray,
  normalizeDim,
  getAxisStrides
} from '../ndarray-utils.js';

export class ReshapeOp implements OpNode {
  opName = 'Reshape';
  inputs: TensorNode[];
  output: TensorNode;

  constructor(X: TensorNode, newShape: number[], outName: string) {
    this.inputs = [X];
    const resolvedShape = resolveReshapeShape(X.shape, newShape);
    const outData = createNdArray(X.data.data, resolvedShape);
    this.output = initOpOutput(this, outName, outData);
  }

  backwardFn(gradOutput: any): void {
    const [X] = this.inputs;
    if (X.requiresGrad) {
      X.addGrad(createNdArray(gradOutput.data, X.shape));
    }
  }
}

export class TransposeOp implements OpNode {
  opName = 'Transpose';
  inputs: TensorNode[];
  output: TensorNode;
  private dim1: number;
  private dim2: number;

  constructor(X: TensorNode, dim1: number, dim2: number, outName: string) {
    this.inputs = [X];
    this.dim1 = dim1;
    this.dim2 = dim2;
    this.output = initOpOutput(this, outName, transposeNdArray(X.data, dim1, dim2));
  }

  backwardFn(gradOutput: any): void {
    const [X] = this.inputs;
    if (X.requiresGrad) X.addGrad(transposeNdArray(gradOutput, this.dim1, this.dim2));
  }
}

export class UnsqueezeOp implements OpNode {
  opName = 'Unsqueeze';
  inputs: TensorNode[];
  output: TensorNode;
  private dim: number;

  constructor(X: TensorNode, dim: number, outName: string) {
    this.dim = dim;
    this.inputs = [X];
    this.output = initOpOutput(this, outName, unsqueezeNdArray(X.data, dim));
  }

  backwardFn(gradOutput: any): void {
    const [X] = this.inputs;
    if (X.requiresGrad) {
      X.addGrad(squeezeNdArray(gradOutput, this.dim));
    }
  }
}

export class SqueezeOp implements OpNode {
  opName = 'Squeeze';
  inputs: TensorNode[];
  output: TensorNode;
  private originalShape: number[];

  constructor(X: TensorNode, outName: string, dim?: number) {
    this.inputs = [X];
    this.originalShape = X.shape.slice();
    this.output = initOpOutput(this, outName, squeezeNdArray(X.data, dim));
  }

  backwardFn(gradOutput: any): void {
    const [X] = this.inputs;
    if (X.requiresGrad) {
      X.addGrad(createNdArray(gradOutput.data, this.originalShape.slice()));
    }
  }
}

export class ExpandOp implements OpNode {
  opName = 'Expand';
  inputs: TensorNode[];
  output: TensorNode;
  private originalShape: number[];

  constructor(X: TensorNode, targetShape: number[], outName: string) {
    this.inputs = [X];
    this.originalShape = X.shape.slice();
    this.output = initOpOutput(this, outName, expandNdArray(X.data, targetShape));
  }

  backwardFn(gradOutput: any): void {
    const [X] = this.inputs;
    if (X.requiresGrad) {
      X.addGrad(sumToShapeNdArray(gradOutput, this.originalShape));
    }
  }
}

export class ConcatOp implements OpNode {
  opName = 'Concat';
  inputs: TensorNode[];
  output: TensorNode;
  private dim: number;

  constructor(inputs: TensorNode[], dim: number, outName: string) {
    this.inputs = inputs;
    this.dim = dim;
    this.output = initOpOutput(this, outName, concatNdArray(inputs.map(n => n.data), dim));
  }

  backwardFn(gradOutput: any): void {
    const d = normalizeDim(this.dim, gradOutput.shape.length);
    let offset = 0;
    for (const input of this.inputs) {
      const len = input.shape[d];
      if (input.requiresGrad) {
        input.addGrad(sliceNdArray(gradOutput, d, offset, offset + len));
      }
      offset += len;
    }
  }
}

export class StackOp implements OpNode {
  opName = 'Stack';
  inputs: TensorNode[];
  output: TensorNode;
  private dim: number;

  constructor(inputs: TensorNode[], dim: number, outName: string) {
    this.inputs = inputs;
    this.dim = dim;
    this.output = initOpOutput(this, outName, stackNdArray(inputs.map(n => n.data), dim));
  }

  backwardFn(gradOutput: any): void {
    const slices = splitNdArray(gradOutput, this.dim, this.inputs.length);
    for (let i = 0; i < this.inputs.length; i++) {
      const input = this.inputs[i];
      if (input.requiresGrad) {
        input.addGrad(squeezeNdArray(slices[i], this.dim));
      }
    }
  }
}

export class SliceOp implements OpNode {
  opName = 'Slice';
  inputs: TensorNode[];
  output: TensorNode;
  private dim: number;
  private start: number;
  private end: number;
  private originalShape: number[];

  constructor(X: TensorNode, dim: number, start: number, end: number, outName: string) {
    this.inputs = [X];
    this.dim = normalizeDim(dim, X.shape.length);
    this.start = start;
    this.end = end;
    this.originalShape = X.shape.slice();
    this.output = initOpOutput(this, outName, sliceNdArray(X.data, this.dim, start, end));
  }

  backwardFn(gradOutput: any): void {
    const [X] = this.inputs;
    if (X.requiresGrad) {
      const d = this.dim;
      const start = this.start;
      const sliceLen = this.end - start;
      const dX = zerosNdArray(this.originalShape);
      const { outerSize, dimSize, innerSize } = getAxisStrides(this.originalShape, d);
      const dxOuterStride = dimSize * innerSize;
      const gradOuterStride = sliceLen * innerSize;
      const startOffset = start * innerSize;
      const dxData = dX.data;
      const gData: Float32Array = gradOutput.data;

      for (let outer = 0; outer < outerSize; outer++) {
        const gradStart = outer * gradOuterStride;
        const dstStart = outer * dxOuterStride + startOffset;
        dxData.set(gData.subarray(gradStart, gradStart + gradOuterStride), dstStart);
      }
      X.addGrad(dX);
    }
  }
}

export function split(X: TensorNode, dim: number, n: number, prefix: string): TensorNode[] {
  const d = normalizeDim(dim, X.shape.length);
  const dimSize = X.shape[d];
  if (dimSize % n !== 0) {
    throw new Error(`Cannot split dim ${d} of size ${dimSize} into ${n} equal parts`);
  }
  const chunk = dimSize / n;
  const results = new Array<TensorNode>(n);
  for (let i = 0; i < n; i++) {
    const start = i * chunk;
    results[i] = new SliceOp(X, d, start, start + chunk, `${prefix}_${i}`).output;
  }
  return results;
}

export function reshape(X: TensorNode, newShape: number[], outName: string): TensorNode {
  return new ReshapeOp(X, newShape, outName).output;
}

export function transpose(X: TensorNode, dim1: number, dim2: number, outName: string): TensorNode {
  return new TransposeOp(X, dim1, dim2, outName).output;
}

export function unsqueeze(X: TensorNode, dim: number, outName: string): TensorNode {
  return new UnsqueezeOp(X, dim, outName).output;
}

export function squeeze(X: TensorNode, outName: string, dim?: number): TensorNode {
  return new SqueezeOp(X, outName, dim).output;
}

export function expand(X: TensorNode, targetShape: number[], outName: string): TensorNode {
  return new ExpandOp(X, targetShape, outName).output;
}

export function concat(inputs: TensorNode[], dim: number, outName: string): TensorNode {
  return new ConcatOp(inputs, dim, outName).output;
}

export function slice(X: TensorNode, dim: number, start: number, end: number, outName: string): TensorNode {
  return new SliceOp(X, dim, start, end, outName).output;
}

export function stack(inputs: TensorNode[], dim: number, outName: string): TensorNode {
  return new StackOp(inputs, dim, outName).output;
}

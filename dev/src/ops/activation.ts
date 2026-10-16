import { TensorNode } from '../tensor.js';
import { OpNode } from '../types.js';
import { initOpOutput } from './helpers.js';
import {
  reluNdArray,
  reluGradNdArray,
  geluNdArray,
  geluGradNdArray,
  sigmoidNdArray,
  sigmoidGradNdArray,
  tanhNdArray,
  tanhGradNdArray,
  softmaxNdArray,
  softmaxGradNdArray
} from '../ndarray-utils.js';

export class ReLUOp implements OpNode {
  opName = 'ReLU';
  inputs: TensorNode[];
  output: TensorNode;

  constructor(X: TensorNode, outName: string) {
    this.inputs = [X];
    this.output = initOpOutput(this, outName, reluNdArray(X.data));
  }

  backwardFn(gradOutput: any): void {
    const [X] = this.inputs;
    if (X.requiresGrad) {
      X.addGrad(reluGradNdArray(gradOutput, X.data));
    }
  }
}

export class GeluOp implements OpNode {
  opName = 'GELU';
  inputs: TensorNode[];
  output: TensorNode;

  constructor(X: TensorNode, outName: string) {
    this.inputs = [X];
    this.output = initOpOutput(this, outName, geluNdArray(X.data));
  }

  backwardFn(gradOutput: any): void {
    const [X] = this.inputs;
    if (X.requiresGrad) X.addGrad(geluGradNdArray(gradOutput, X.data));
  }
}

export class SigmoidOp implements OpNode {
  opName = 'Sigmoid';
  inputs: TensorNode[];
  output: TensorNode;

  constructor(X: TensorNode, outName: string) {
    this.inputs = [X];
    this.output = initOpOutput(this, outName, sigmoidNdArray(X.data));
  }

  backwardFn(gradOutput: any): void {
    const [X] = this.inputs;
    if (X.requiresGrad) X.addGrad(sigmoidGradNdArray(gradOutput, this.output.data));
  }
}

export class TanhOp implements OpNode {
  opName = 'Tanh';
  inputs: TensorNode[];
  output: TensorNode;

  constructor(X: TensorNode, outName: string) {
    this.inputs = [X];
    this.output = initOpOutput(this, outName, tanhNdArray(X.data));
  }

  backwardFn(gradOutput: any): void {
    const [X] = this.inputs;
    if (X.requiresGrad) X.addGrad(tanhGradNdArray(gradOutput, this.output.data));
  }
}

export class SoftmaxOp implements OpNode {
  opName = 'Softmax';
  inputs: TensorNode[];
  output: TensorNode;
  private dim: number;

  constructor(X: TensorNode, dim: number, outName: string) {
    this.inputs = [X];
    this.dim = dim;
    this.output = initOpOutput(this, outName, softmaxNdArray(X.data, dim));
  }

  backwardFn(gradOutput: any): void {
    const [X] = this.inputs;
    if (X.requiresGrad) X.addGrad(softmaxGradNdArray(gradOutput, this.output.data, this.dim));
  }
}

export function relu(X: TensorNode, outName: string): TensorNode {
  return new ReLUOp(X, outName).output;
}

export function gelu(X: TensorNode, outName: string): TensorNode {
  return new GeluOp(X, outName).output;
}

export function sigmoid(X: TensorNode, outName: string): TensorNode {
  return new SigmoidOp(X, outName).output;
}

export function tanh(X: TensorNode, outName: string): TensorNode {
  return new TanhOp(X, outName).output;
}

export function softmax(X: TensorNode, dim: number, outName: string): TensorNode {
  return new SoftmaxOp(X, dim, outName).output;
}

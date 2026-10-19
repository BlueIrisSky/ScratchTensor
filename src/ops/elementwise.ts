import { TensorNode } from '../tensor.js';
import { OpNode } from '../types.js';
import { initOpOutput } from './helpers.js';
import {
  powNdArray,
  expNdArray,
  logNdArray,
  sqrtNdArray,
  absNdArray,
  createNdArray,
  mulNdArray
} from '../ndarray-utils.js';

export class PowOp implements OpNode {
  opName = 'Pow';
  inputs: TensorNode[];
  output: TensorNode;
  private p: number;

  constructor(X: TensorNode, p: number, outName: string) {
    this.inputs = [X];
    this.p = p;
    this.output = initOpOutput(this, outName, powNdArray(X.data, p));
  }

  backwardFn(gradOutput: any): void {
    const [X] = this.inputs;
    if (X.requiresGrad) {
      const gData = gradOutput.data;
      const xData = X.data.data;
      const size = gData.length;
      const buf = new Float32Array(size);
      const p = this.p;
      if (p === 2) {
        for (let i = 0; i < size; i++) {
          buf[i] = 2.0 * gData[i] * xData[i];
        }
      } else {
        const pm1 = p - 1;
        for (let i = 0; i < size; i++) {
          buf[i] = gData[i] * p * Math.pow(xData[i], pm1);
        }
      }
      X.addGrad(createNdArray(buf, gradOutput.shape.slice()));
    }
  }
}

export class ExpOp implements OpNode {
  opName = 'Exp';
  inputs: TensorNode[];
  output: TensorNode;

  constructor(X: TensorNode, outName: string) {
    this.inputs = [X];
    this.output = initOpOutput(this, outName, expNdArray(X.data));
  }

  backwardFn(gradOutput: any): void {
    const [X] = this.inputs;
    if (X.requiresGrad) X.addGrad(mulNdArray(gradOutput, this.output.data));
  }
}

export class LogOp implements OpNode {
  opName = 'Log';
  inputs: TensorNode[];
  output: TensorNode;

  constructor(X: TensorNode, outName: string) {
    this.inputs = [X];
    this.output = initOpOutput(this, outName, logNdArray(X.data));
  }

  backwardFn(gradOutput: any): void {
    const [X] = this.inputs;
    if (X.requiresGrad) {
      const gData = gradOutput.data;
      const xData = X.data.data;
      const size = gData.length;
      const buf = new Float32Array(size);
      for (let i = 0; i < size; i++) {
        buf[i] = gData[i] / xData[i];
      }
      X.addGrad(createNdArray(buf, gradOutput.shape.slice()));
    }
  }
}

export class SqrtOp implements OpNode {
  opName = 'Sqrt';
  inputs: TensorNode[];
  output: TensorNode;

  constructor(X: TensorNode, outName: string) {
    this.inputs = [X];
    this.output = initOpOutput(this, outName, sqrtNdArray(X.data));
  }

  backwardFn(gradOutput: any): void {
    const [X] = this.inputs;
    if (X.requiresGrad) {
      const gData = gradOutput.data;
      const outData = this.output.data.data;
      const size = gData.length;
      const buf = new Float32Array(size);
      for (let i = 0; i < size; i++) {
        buf[i] = gData[i] / (2.0 * outData[i]);
      }
      X.addGrad(createNdArray(buf, gradOutput.shape.slice()));
    }
  }
}

export class AbsOp implements OpNode {
  opName = 'Abs';
  inputs: TensorNode[];
  output: TensorNode;

  constructor(X: TensorNode, outName: string) {
    this.inputs = [X];
    this.output = initOpOutput(this, outName, absNdArray(X.data));
  }

  backwardFn(gradOutput: any): void {
    const [X] = this.inputs;
    if (X.requiresGrad) {
      const gData = gradOutput.data;
      const xData = X.data.data;
      const size = gData.length;
      const buf = new Float32Array(size);
      for (let i = 0; i < size; i++) {
        const v = xData[i];
        buf[i] = v > 0 ? gData[i] : v < 0 ? -gData[i] : 0;
      }
      X.addGrad(createNdArray(buf, gradOutput.shape.slice()));
    }
  }
}

export function pow(X: TensorNode, p: number, outName: string): TensorNode {
  return new PowOp(X, p, outName).output;
}

export function exp(X: TensorNode, outName: string): TensorNode {
  return new ExpOp(X, outName).output;
}

export function log(X: TensorNode, outName: string): TensorNode {
  return new LogOp(X, outName).output;
}

export function sqrt(X: TensorNode, outName: string): TensorNode {
  return new SqrtOp(X, outName).output;
}

export function abs(X: TensorNode, outName: string): TensorNode {
  return new AbsOp(X, outName).output;
}

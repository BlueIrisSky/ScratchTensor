import { TensorNode } from '../tensor.js';
import { OpNode } from '../types.js';
import { initOpOutput } from './helpers.js';
import {
  addNdArray,
  subNdArray,
  mulNdArray,
  divNdArray,
  matMulNdArray,
  transposeNdArray,
  sumToShapeNdArray,
  broadcastNdArray,
  createNdArray
} from '../ndarray-utils.js';

export class MatMulOp implements OpNode {
  opName = 'MatMul';
  inputs: TensorNode[];
  output: TensorNode;

  constructor(A: TensorNode, B: TensorNode, outName: string) {
    this.inputs = [A, B];
    this.output = initOpOutput(this, outName, matMulNdArray(A.data, B.data));
  }

  backwardFn(gradOutput: any): void {
    const [A, B] = this.inputs;
    const rankA = A.shape.length;
    const rankB = B.shape.length;

    if (rankA === 1 && rankB === 1) {
      const scale = gradOutput.data[gradOutput.offset];
      if (A.requiresGrad) {
        const bData = B.data.data;
        const bOff = B.data.offset;
        const buf = new Float32Array(A.data.data.length);
        for (let i = 0; i < buf.length; i++) {
          buf[i] = scale * bData[i + bOff];
        }
        A.addGrad(createNdArray(buf, A.shape));
      }
      if (B.requiresGrad) {
        const aData = A.data.data;
        const aOff = A.data.offset;
        const buf = new Float32Array(B.data.data.length);
        for (let i = 0; i < buf.length; i++) {
          buf[i] = scale * aData[i + aOff];
        }
        B.addGrad(createNdArray(buf, B.shape));
      }
      return;
    }

    // 2D x 1D: matrix-vector
    if (rankA === 2 && rankB === 1) {
      const M = A.shape[0];
      const K = A.shape[1];
      const gData = gradOutput.data;
      const gOff = gradOutput.offset;
      if (A.requiresGrad) {
        const bData = B.data.data;
        const bOff = B.data.offset;
        const buf = new Float32Array(M * K);
        for (let i = 0; i < M; i++) {
          const g = gData[i + gOff];
          const row = i * K;
          for (let k = 0; k < K; k++) {
            buf[row + k] = g * bData[k + bOff];
          }
        }
        A.addGrad(createNdArray(buf, A.shape));
      }
      if (B.requiresGrad) {
        const aData = A.data.data;
        const aOff = A.data.offset;
        const buf = new Float32Array(K);
        for (let i = 0; i < M; i++) {
          const g = gData[i + gOff];
          const aRow = aOff + i * K;
          for (let k = 0; k < K; k++) {
            buf[k] += aData[aRow + k] * g;
          }
        }
        B.addGrad(createNdArray(buf, B.shape));
      }
      return;
    }

    // 1D x 2D: vector-matrix
    if (rankA === 1 && rankB === 2) {
      const K = A.shape[0];
      const N = B.shape[1];
      const gData = gradOutput.data;
      const gOff = gradOutput.offset;
      if (A.requiresGrad) {
        const bData = B.data.data;
        const bOff = B.data.offset;
        const buf = new Float32Array(K);
        for (let k = 0; k < K; k++) {
          let sum = 0;
          const bRow = bOff + k * N;
          for (let j = 0; j < N; j++) {
            sum += bData[bRow + j] * gData[j + gOff];
          }
          buf[k] = sum;
        }
        A.addGrad(createNdArray(buf, A.shape));
      }
      if (B.requiresGrad) {
        const aData = A.data.data;
        const aOff = A.data.offset;
        const buf = new Float32Array(K * N);
        for (let k = 0; k < K; k++) {
          const a = aData[k + aOff];
          const row = k * N;
          for (let j = 0; j < N; j++) {
            buf[row + j] = a * gData[j + gOff];
          }
        }
        B.addGrad(createNdArray(buf, B.shape));
      }
      return;
    }

    // General batched case: both >= 2D with broadcasting.
    if (A.requiresGrad) {
      const Bt = transposeNdArray(B.data, rankB - 2, rankB - 1);
      A.addGrad(sumToShapeNdArray(matMulNdArray(gradOutput, Bt), A.shape));
    }
    if (B.requiresGrad) {
      const At = transposeNdArray(A.data, rankA - 2, rankA - 1);
      B.addGrad(sumToShapeNdArray(matMulNdArray(At, gradOutput), B.shape));
    }
  }
}

export class AddOp implements OpNode {
  opName = 'Add';
  inputs: TensorNode[];
  output: TensorNode;

  constructor(A: TensorNode, B: TensorNode, outName: string) {
    this.inputs = [A, B];
    this.output = initOpOutput(this, outName, addNdArray(A.data, B.data));
  }

  backwardFn(gradOutput: any): void {
    const [A, B] = this.inputs;
    if (A.requiresGrad) {
      A.addGrad(sumToShapeNdArray(gradOutput, A.shape));
    }
    if (B.requiresGrad) {
      B.addGrad(sumToShapeNdArray(gradOutput, B.shape));
    }
  }
}

export class SubOp implements OpNode {
  opName = 'Sub';
  inputs: TensorNode[];
  output: TensorNode;

  constructor(A: TensorNode, B: TensorNode, outName: string) {
    this.inputs = [A, B];
    this.output = initOpOutput(this, outName, subNdArray(A.data, B.data));
  }

  backwardFn(gradOutput: any): void {
    const [A, B] = this.inputs;
    if (A.requiresGrad) {
      A.addGrad(sumToShapeNdArray(gradOutput, A.shape));
    }
    if (B.requiresGrad) {
      const gData = gradOutput.data;
      const negBuf = new Float32Array(gData.length);
      for (let i = 0; i < negBuf.length; i++) {
        negBuf[i] = -gData[i];
      }
      const negGrad = createNdArray(negBuf, gradOutput.shape.slice());
      B.addGrad(sumToShapeNdArray(negGrad, B.shape));
    }
  }
}

export class MulOp implements OpNode {
  opName = 'Mul';
  inputs: TensorNode[];
  output: TensorNode;

  constructor(A: TensorNode, B: TensorNode, outName: string) {
    this.inputs = [A, B];
    this.output = initOpOutput(this, outName, mulNdArray(A.data, B.data));
  }

  backwardFn(gradOutput: any): void {
    const [A, B] = this.inputs;
    if (A.requiresGrad) {
      const bBroadcast = broadcastNdArray(B.data, gradOutput.shape);
      A.addGrad(sumToShapeNdArray(mulNdArray(gradOutput, bBroadcast), A.shape));
    }
    if (B.requiresGrad) {
      const aBroadcast = broadcastNdArray(A.data, gradOutput.shape);
      B.addGrad(sumToShapeNdArray(mulNdArray(gradOutput, aBroadcast), B.shape));
    }
  }
}

export class DivOp implements OpNode {
  opName = 'Div';
  inputs: TensorNode[];
  output: TensorNode;

  constructor(A: TensorNode, B: TensorNode, outName: string) {
    this.inputs = [A, B];
    this.output = initOpOutput(this, outName, divNdArray(A.data, B.data));
  }

  backwardFn(gradOutput: any): void {
    const [A, B] = this.inputs;
    const bBroadcast = (A.requiresGrad || B.requiresGrad)
      ? broadcastNdArray(B.data, gradOutput.shape)
      : null;
    if (A.requiresGrad && bBroadcast) {
      A.addGrad(sumToShapeNdArray(divNdArray(gradOutput, bBroadcast), A.shape));
    }
    if (B.requiresGrad && bBroadcast) {
      const aBroadcast = broadcastNdArray(A.data, gradOutput.shape);
      const gData = gradOutput.data;
      const aData = aBroadcast.data;
      const bData = bBroadcast.data;
      const size = gData.length;
      const negBuf = new Float32Array(size);
      for (let i = 0; i < size; i++) {
        const b = bData[i];
        negBuf[i] = (-gData[i] * aData[i]) / (b * b);
      }
      const negGrad = createNdArray(negBuf, gradOutput.shape.slice());
      B.addGrad(sumToShapeNdArray(negGrad, B.shape));
    }
  }
}

export function matmul(A: TensorNode, B: TensorNode, outName: string): TensorNode {
  return new MatMulOp(A, B, outName).output;
}

export function add(A: TensorNode, B: TensorNode, outName: string): TensorNode {
  return new AddOp(A, B, outName).output;
}

export function sub(A: TensorNode, B: TensorNode, outName: string): TensorNode {
  return new SubOp(A, B, outName).output;
}

export function mul(A: TensorNode, B: TensorNode, outName: string): TensorNode {
  return new MulOp(A, B, outName).output;
}

export function div(A: TensorNode, B: TensorNode, outName: string): TensorNode {
  return new DivOp(A, B, outName).output;
}

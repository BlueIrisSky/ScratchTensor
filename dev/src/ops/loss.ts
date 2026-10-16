import { TensorNode } from '../tensor.js';
import { OpNode } from '../types.js';
import { isGraphTracking } from '../autograd.js';
import { initOpOutput } from './helpers.js';
import {
  mseLossNdArray,
  crossEntropyLossNdArray,
  bceLossNdArray,
  l1LossNdArray,
  createNdArray
} from '../ndarray-utils.js';

function addScaledLossGrad(target: TensorNode, grad: any, gradOutput: any): void {
  if (!target.requiresGrad || !grad) return;
  const scale = gradOutput.data[0];
  if (scale === 1.0) {
    target.addGrad(grad);
  } else {
    const gData = grad.data;
    const scaledBuf = new Float32Array(gData.length);
    for (let i = 0; i < scaledBuf.length; i++) {
      scaledBuf[i] = gData[i] * scale;
    }
    target.addGrad(createNdArray(scaledBuf, grad.shape.slice()));
  }
}

export class MSELossOp implements OpNode {
  opName = 'MSELoss';
  inputs: TensorNode[];
  output: TensorNode;
  private _gradPred: any = null;

  constructor(pred: TensorNode, target: TensorNode, outName: string) {
    this.inputs = [pred, target];
    const computeGrad = pred.requiresGrad && isGraphTracking();
    const { lossValue, gradPred } = mseLossNdArray(pred.data, target.data, computeGrad);
    const lossNdArray = createNdArray(new Float32Array([lossValue]), []);
    this._gradPred = gradPred;
    this.output = initOpOutput(this, outName, lossNdArray, [pred]);
  }

  backwardFn(gradOutput: any): void {
    addScaledLossGrad(this.inputs[0], this._gradPred, gradOutput);
  }
}

export class CrossEntropyLossOp implements OpNode {
  opName = 'CrossEntropyLoss';
  inputs: TensorNode[];
  output: TensorNode;
  private _gradLogits: any = null;

  constructor(logits: TensorNode, labels: TensorNode, outName: string) {
    this.inputs = [logits, labels];
    const computeGrad = logits.requiresGrad && isGraphTracking();
    const { lossValue, gradLogits } = crossEntropyLossNdArray(logits.data, labels.data, computeGrad);
    const lossNdArray = createNdArray(new Float32Array([lossValue]), []);
    this._gradLogits = gradLogits;
    this.output = initOpOutput(this, outName, lossNdArray, [logits]);
  }

  backwardFn(gradOutput: any): void {
    addScaledLossGrad(this.inputs[0], this._gradLogits, gradOutput);
  }
}

export class BCELossOp implements OpNode {
  opName = 'BCELoss';
  inputs: TensorNode[];
  output: TensorNode;
  private _gradPred: any = null;

  constructor(pred: TensorNode, target: TensorNode, outName: string) {
    this.inputs = [pred, target];
    const computeGrad = pred.requiresGrad && isGraphTracking();
    const { lossValue, gradPred } = bceLossNdArray(pred.data, target.data, computeGrad);
    const lossNdArray = createNdArray(new Float32Array([lossValue]), []);
    this._gradPred = gradPred;
    this.output = initOpOutput(this, outName, lossNdArray, [pred]);
  }

  backwardFn(gradOutput: any): void {
    addScaledLossGrad(this.inputs[0], this._gradPred, gradOutput);
  }
}

export class L1LossOp implements OpNode {
  opName = 'L1Loss';
  inputs: TensorNode[];
  output: TensorNode;
  private _gradPred: any = null;

  constructor(pred: TensorNode, target: TensorNode, outName: string) {
    this.inputs = [pred, target];
    const computeGrad = pred.requiresGrad && isGraphTracking();
    const { lossValue, gradPred } = l1LossNdArray(pred.data, target.data, computeGrad);
    const lossNdArray = createNdArray(new Float32Array([lossValue]), []);
    this._gradPred = gradPred;
    this.output = initOpOutput(this, outName, lossNdArray, [pred]);
  }

  backwardFn(gradOutput: any): void {
    addScaledLossGrad(this.inputs[0], this._gradPred, gradOutput);
  }
}

export function mseLoss(pred: TensorNode, target: TensorNode, outName: string): TensorNode {
  return new MSELossOp(pred, target, outName).output;
}

export function crossEntropyLoss(logits: TensorNode, labels: TensorNode, outName: string): TensorNode {
  return new CrossEntropyLossOp(logits, labels, outName).output;
}

export function bceLoss(pred: TensorNode, target: TensorNode, outName: string): TensorNode {
  return new BCELossOp(pred, target, outName).output;
}

export function l1Loss(pred: TensorNode, target: TensorNode, outName: string): TensorNode {
  return new L1LossOp(pred, target, outName).output;
}

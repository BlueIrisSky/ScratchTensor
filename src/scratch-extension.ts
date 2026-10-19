import {
  createTensor,
  createZerosTensor,
  createOnesTensor,
  createRandomNormalTensor,
  getTensorByName,
  getTensorGrad,
  setGraphTracking,
  backward,
  clearAllGradients,
  matmul,
  add,
  sub,
  mul,
  div,
  relu,
  gelu,
  sigmoid,
  tanh,
  softmax,
  mseLoss,
  crossEntropyLoss,
  bceLoss,
  l1Loss,
  scaledDotProductAttention,
  reshape,
  transpose,
  pow,
  exp,
  log,
  sqrt,
  abs,
  sum,
  mean,
  max,
  min,
  unsqueeze,
  squeeze,
  expand,
  concat,
  split,
  slice,
  stack,
  where,
  clamp,
  dropout,
  layerNorm,
  exportStateDict,
  loadStateDict,
  sgdStep,
  adamStep,
  adamWStep,
  clipGradNorm,
  setLrScale,
  toJsArray,
  unregisterTensor,
  clearTensorRegistry
} from './index.js';
import { getExtensionInfo } from './extension/metadata.js';
import { installBlockErrorHandlers } from './extension/errors.js';

declare const Scratch: any;

export class ScratchTensorExtension {
  constructor() {
    installBlockErrorHandlers(this, this.getInfo().blocks.map(block => block.opcode));
  }

  getInfo() {
    return getExtensionInfo();
  }

  private requireTensor(name: any) {
    const tensor = getTensorByName(String(name));
    if (!tensor) throw new Error(`Tensor ${JSON.stringify(String(name))} does not exist`);
    return tensor;
  }

  private requireTensorList(json: string) {
    const names: unknown = JSON.parse(json);
    if (!Array.isArray(names) || names.length === 0 || names.some(name => typeof name !== 'string')) {
      throw new Error('NAMES must be a non-empty JSON array of tensor names');
    }
    return names.map(name => this.requireTensor(name));
  }

  private reductionDimension(value: unknown): number | null {
    if (value === '全部') return null;
    if (typeof value === 'number' && Number.isSafeInteger(value)) return value;
    if (typeof value === 'string' && /^[+-]?\d+$/u.test(value.trim())) {
      const dim = Number(value);
      if (Number.isSafeInteger(dim)) return dim;
    }
    throw new Error('DIM must be an integer axis or "全部"');
  }

  //Tensor Creation

  createTensorBlock(args: any) {
    const data = JSON.parse(args.DATA);
    createTensor(args.NAME, data, args.REQ_GRAD);
  }

  createZerosTensorBlock(args: any) {
    const shape = JSON.parse(args.SHAPE);
    createZerosTensor(args.NAME, shape, args.REQ_GRAD);
  }

  createOnesTensorBlock(args: any) {
    const shape = JSON.parse(args.SHAPE);
    createOnesTensor(args.NAME, shape, args.REQ_GRAD);
  }

  createRandomNormalTensorBlock(args: any) {
    const shape = JSON.parse(args.SHAPE);
    createRandomNormalTensor(args.NAME, shape, args.REQ_GRAD);
  }

  // ========== Tensor Query ==========

  getTensorPropertyBlock(args: any): string | number {
    const tensor = this.requireTensor(args.NAME);
    switch (args.PROPERTY) {
      case '数值':
        return JSON.stringify(tensor.toArray());
      case '标量值':
        return tensor.item();
      case 'Shape':
        return JSON.stringify(tensor.shape);
      case '梯度':
        const grad = getTensorGrad(args.NAME);
        return grad ? JSON.stringify(toJsArray(grad)) : 'null';
      default:
        throw new Error(`Unknown tensor property: ${String(args.PROPERTY)}`);
    }
  }

  // ========== Tensor Deletion ==========

  deleteTensorBlock(args: any) {
    unregisterTensor(String(args.NAME));
  }

  deleteAllTensorsBlock() {
    clearTensorRegistry();
  }

  // ========== Autograd ==========

  setGraphTrackingBlock(args: any) {
    setGraphTracking(Boolean(args.ENABLED));
  }

  backwardBlock(args: any) {
    const tensor = this.requireTensor(args.TENSOR);
    backward(tensor);
  }

  clearGradBlock() {
    clearAllGradients();
  }

  // ========== Operators ==========

  binaryOpBlock(args: any) {
    const A = this.requireTensor(args.A);
    const B = this.requireTensor(args.B);
    switch (args.OP) {
      case '+':
        add(A, B, args.OUT);
        break;
      case '-':
        sub(A, B, args.OUT);
        break;
      case '*':
        mul(A, B, args.OUT);
        break;
      case '/':
        div(A, B, args.OUT);
        break;
      case '@':
        matmul(A, B, args.OUT);
        break;
      default:
        throw new Error(`Unknown operation: ${String(args.OP)}`);
    }
  }

  // ========== Activation Functions ==========

  activationBlock(args: any) {
    const X = this.requireTensor(args.X);
    switch (args.ACT) {
      case 'ReLU':
        relu(X, args.OUT);
        break;
      case 'GELU':
        gelu(X, args.OUT);
        break;
      case 'Sigmoid':
        sigmoid(X, args.OUT);
        break;
      case 'Tanh':
        tanh(X, args.OUT);
        break;
      default:
        throw new Error(`Unknown activation: ${String(args.ACT)}`);
    }
  }

  softmaxBlock(args: any) {
    const X = this.requireTensor(args.X);
    softmax(X, Number(args.DIM), args.OUT);
  }

  scaledDotProductAttentionBlock(args: any) {
    const Q = this.requireTensor(args.Q);
    const K = this.requireTensor(args.K);
    const V = this.requireTensor(args.V);
    const isCausal = Boolean(args.CAUSAL);
    const dropoutP = Number(args.DROPOUT);
    const scaleVal = Number(args.SCALE);
    if ((typeof args.SCALE !== 'number' && typeof args.SCALE !== 'string') ||
        String(args.SCALE).trim() === '' || !Number.isFinite(scaleVal) || scaleVal < 0) {
      throw new Error('SCALE must be a finite non-negative number (0 selects automatic scaling)');
    }
    const scale = scaleVal === 0 ? undefined : scaleVal;
    scaledDotProductAttention(Q, K, V, args.OUT, isCausal, dropoutP, scale);
  }

  // ========== Loss Functions ==========

  mseLossBlock(args: any) {
    const pred = this.requireTensor(args.PRED);
    const target = this.requireTensor(args.TARGET);
    mseLoss(pred, target, args.OUT);
  }

  crossEntropyLossBlock(args: any) {
    const logits = this.requireTensor(args.LOGITS);
    const labels = this.requireTensor(args.LABELS);
    crossEntropyLoss(logits, labels, args.OUT);
  }

  bceLossBlock(args: any) {
    const pred = this.requireTensor(args.PRED);
    const target = this.requireTensor(args.TARGET);
    bceLoss(pred, target, args.OUT);
  }

  l1LossBlock(args: any) {
    const pred = this.requireTensor(args.PRED);
    const target = this.requireTensor(args.TARGET);
    l1Loss(pred, target, args.OUT);
  }

  // ========== Shape Operations ==========

  reshapeBlock(args: any) {
    const X = this.requireTensor(args.X);
    reshape(X, JSON.parse(args.SHAPE), args.OUT);
  }

  unsqueezeBlock(args: any) {
    const X = this.requireTensor(args.X);
    unsqueeze(X, Number(args.DIM), args.OUT);
  }

  squeezeBlock(args: any) {
    const X = this.requireTensor(args.X);
    squeeze(X, args.OUT);
  }

  expandBlock(args: any) {
    const X = this.requireTensor(args.X);
    expand(X, JSON.parse(args.SHAPE), args.OUT);
  }

  concatBlock(args: any) {
    const tensors = this.requireTensorList(args.NAMES);
    concat(tensors, Number(args.DIM), args.OUT);
  }

  splitBlock(args: any) {
    const X = this.requireTensor(args.X);
    split(X, Number(args.DIM), Number(args.N), args.PREFIX);
  }

  stackBlock(args: any) {
    const tensors = this.requireTensorList(args.NAMES);
    stack(tensors, Number(args.DIM), args.OUT);
  }

  sliceBlock(args: any) {
    const X = this.requireTensor(args.X);
    slice(X, Number(args.DIM), Number(args.START), Number(args.END), args.OUT);
  }

  whereBlock(args: any) {
    const cond = this.requireTensor(args.COND);
    const a = this.requireTensor(args.A);
    const b = this.requireTensor(args.B);
    where(cond, a, b, args.OUT);
  }

  clampBlock(args: any) {
    const X = this.requireTensor(args.X);
    clamp(X, Number(args.MIN), Number(args.MAX), args.OUT);
  }

  // ========== Element-wise Math ==========

  powBlock(args: any) {
    const X = this.requireTensor(args.X);
    pow(X, Number(args.P), args.OUT);
  }

  unaryMathBlock(args: any) {
    const X = this.requireTensor(args.X);
    switch (args.OP) {
      case 'Exp':
        exp(X, args.OUT);
        break;
      case 'Log':
        log(X, args.OUT);
        break;
      case 'Sqrt':
        sqrt(X, args.OUT);
        break;
      case 'Abs':
        abs(X, args.OUT);
        break;
      default:
        throw new Error(`Unknown unary operation: ${String(args.OP)}`);
    }
  }

  // ========== Reductions & Matrix Ops ==========

  reductionBlock(args: any) {
    const X = this.requireTensor(args.X);
    const dim = this.reductionDimension(args.DIM);
    const keepdim = Boolean(args.KEEPDIM);
    switch (args.OP) {
      case '求和(Sum)':
        sum(X, dim, args.OUT, keepdim);
        break;
      case '求平均值(Mean)':
        mean(X, dim, args.OUT, keepdim);
        break;
      case '求最大值(Max)':
        max(X, dim, args.OUT, keepdim);
        break;
      case '求最小值(Min)':
        min(X, dim, args.OUT, keepdim);
        break;
      default:
        throw new Error(`Unknown reduction: ${String(args.OP)}`);
    }
  }

  transposeBlock(args: any) {
    const X = this.requireTensor(args.X);
    transpose(X, Number(args.DIM1), Number(args.DIM2), args.OUT);
  }

  // ========== Regularization ==========

  layerNormBlock(args: any) {
    const X = this.requireTensor(args.X);
    layerNorm(X, Number(args.N), args.OUT);
  }

  dropoutBlock(args: any) {
    const X = this.requireTensor(args.X);
    dropout(X, Number(args.P), args.OUT);
  }

  // ========== Optimizers ==========

  sgdFullBlock(args: any) {
    sgdStep(Number(args.LR), Number(args.MOMENTUM), Number(args.WD));
  }

  adamBlock(args: any) {
    adamStep(Number(args.LR), Number(args.B1), Number(args.B2), Number(args.WD));
  }

  adamWBlock(args: any) {
    adamWStep(Number(args.LR), Number(args.B1), Number(args.B2), Number(args.WD));
  }

  clipGradNormBlock(args: any) {
    clipGradNorm(Number(args.MAX_NORM));
  }

  setLrScaleBlock(args: any) {
    setLrScale(Number(args.SCALE));
  }

  // ========== Serialization ==========

  exportStateDictBlock(): string {
    return exportStateDict();
  }

  loadStateDictBlock(args: any) {
    loadStateDict(args.JSON_STR);
  }
}

if (typeof Scratch !== 'undefined' && Scratch.extensions) {
  Scratch.extensions.register(new ScratchTensorExtension());
}

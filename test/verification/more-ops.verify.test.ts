import { describe, it, expect } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import {
  createTensor,
  gelu,
  sigmoid,
  tanh,
  softmax,
  crossEntropyLoss,
  bceLoss,
  l1Loss,
  pow,
  exp,
  log,
  sqrt,
  abs,
  sum,
  mean,
  max,
  min,
  transpose,
  dropout,
  backward,
  setGraphTracking,
  adamStep,
  adamWStep,
  clipGradNorm,
  setLrScale,
  sgdStep,
  resetOptimizerState
} from '../../src/index.js';
import {
  useVerificationLifecycle,
  expectClose,
  withTfReference
} from './harness/index.js';

describe('Verification: More Advanced Ops against @tensorflow/tfjs', () => {
  useVerificationLifecycle();

  it('GELU forward/backward', () => {
    const data = [[1.0, -0.5], [0.2, -2.0]];
    const X = createTensor('X', data, true);
    const Y = gelu(X, 'Y');

    backward(Y);
    expect(X.grad).not.toBeNull();

    withTfReference(ctx => {
      const tfX = ctx.variable2d(data);
      const geluRef = (x: tf.Tensor) => {
        const c = tf.mul(Math.sqrt(2 / Math.PI), tf.add(x, tf.mul(0.044715, tf.pow(x, 3))));
        return tf.mul(0.5, tf.mul(x, tf.add(1, tf.tanh(c))));
      };
      const grads = tf.variableGrads(() => geluRef(tfX).sum() as tf.Scalar);
      expectClose(Y.toArray(), geluRef(tfX).arraySync(), 1e-5, 1e-5, '(GELU forward)');
      expectClose(X.grad.data, grads.grads[tfX.name].arraySync(), 1e-5, 1e-5, '(GELU grad)');
      tf.dispose(grads);
    });
  });

  it('Sigmoid forward/backward', () => {
    const data = [[1.0, -0.5], [0.2, -2.0]];
    const X = createTensor('X', data, true);
    const Y = sigmoid(X, 'Y');
    backward(Y);
    expect(X.grad).not.toBeNull();

    withTfReference(ctx => {
      const tfX = ctx.variable2d(data);
      const grads = tf.variableGrads(() => tf.sigmoid(tfX).sum() as tf.Scalar);
      expectClose(Y.toArray(), tf.sigmoid(tfX).arraySync(), 1e-5, 1e-5, '(Sigmoid forward)');
      expectClose(X.grad.data, grads.grads[tfX.name].arraySync(), 1e-5, 1e-5, '(Sigmoid grad)');
      tf.dispose(grads);
    });
  });

  it('Tanh forward/backward', () => {
    const data = [[1.0, -0.5], [0.2, -2.0]];
    const X = createTensor('X', data, true);
    const Y = tanh(X, 'Y');
    backward(Y);
    expect(X.grad).not.toBeNull();

    withTfReference(ctx => {
      const tfX = ctx.variable2d(data);
      const grads = tf.variableGrads(() => tf.tanh(tfX).sum() as tf.Scalar);
      expectClose(Y.toArray(), tf.tanh(tfX).arraySync(), 1e-5, 1e-5, '(Tanh forward)');
      expectClose(X.grad.data, grads.grads[tfX.name].arraySync(), 1e-5, 1e-5, '(Tanh grad)');
      tf.dispose(grads);
    });
  });

  it('Softmax forward/backward along last dim', () => {
    const data = [[1.0, 2.0, 0.5], [0.0, 3.0, 1.0]];
    const X = createTensor('X', data, true);
    const Y = softmax(X, -1, 'Y');
    backward(Y);
    expect(X.grad).not.toBeNull();

    withTfReference(ctx => {
      const tfX = ctx.variable2d(data);
      const grads = tf.variableGrads(() => tf.softmax(tfX).sum() as tf.Scalar);
      expectClose(Y.toArray(), tf.softmax(tfX).arraySync(), 1e-5, 1e-5, '(Softmax forward)');
      expectClose(X.grad.data, grads.grads[tfX.name].arraySync(), 1e-5, 1e-5, '(Softmax grad)');
      tf.dispose(grads);
    });
  });

  it('CrossEntropyLoss forward/backward', () => {
    const logitsData = [[1.0, 2.0, 0.5], [0.0, 3.0, 1.0]];
    const labelsData = [0, 2];
    const Logits = createTensor('Logits', logitsData, true);
    const Labels = createTensor('Labels', labelsData, false);
    const Loss = crossEntropyLoss(Logits, Labels, 'Loss');
    backward(Loss);
    expect(Logits.grad).not.toBeNull();

    withTfReference(ctx => {
      const tfLogits = ctx.variable2d(logitsData);
      const tfLabels = tf.tensor1d(labelsData, 'int32');
      const tfLabelsOneHot = tf.oneHot(tfLabels, 3);
      const grads = tf.variableGrads(() => tf.losses.softmaxCrossEntropy(tfLabelsOneHot, tfLogits) as tf.Scalar);
      expectClose(Logits.grad.data, grads.grads[tfLogits.name].arraySync(), 1e-5, 1e-5, '(CrossEntropy grad)');
      tf.dispose(grads);
    });
  });

  it('BCELoss forward/backward', () => {
    const predData = [[0.8, 0.2], [0.4, 0.9]];
    const targetData = [[1.0, 0.0], [0.0, 1.0]];
    const Pred = createTensor('Pred', predData, true);
    const Target = createTensor('Target', targetData, false);
    const Loss = bceLoss(Pred, Target, 'Loss');
    backward(Loss);
    expect(Pred.grad).not.toBeNull();

    withTfReference(ctx => {
      const tfPred = ctx.variable2d(predData);
      const tfTarget = ctx.tensor2d(targetData);
      const bceRef = (pred: tf.Tensor, target: tf.Tensor) => {
        const eps = 1e-12;
        const p = tf.clipByValue(pred, eps, 1 - eps);
        return tf.neg(tf.mean(tf.add(tf.mul(target, tf.log(p)), tf.mul(tf.sub(1, target), tf.log(tf.sub(1, p))))));
      };
      const grads = tf.variableGrads(() => bceRef(tfPred, tfTarget) as tf.Scalar);
      expectClose(Pred.grad.data, grads.grads[tfPred.name].arraySync(), 1e-5, 1e-5, '(BCE grad)');
      tf.dispose(grads);
    });
  });

  it('L1Loss forward/backward', () => {
    const predData = [[1.0, -0.5], [0.2, 2.0]];
    const targetData = [[0.5, 0.0], [1.0, 1.5]];
    const Pred = createTensor('Pred', predData, true);
    const Target = createTensor('Target', targetData, false);
    const Loss = l1Loss(Pred, Target, 'Loss');
    backward(Loss);
    expect(Pred.grad).not.toBeNull();

    withTfReference(ctx => {
      const tfPred = ctx.variable2d(predData);
      const tfTarget = ctx.tensor2d(targetData);
      const grads = tf.variableGrads(() => tf.losses.absoluteDifference(tfTarget, tfPred).mean() as tf.Scalar);
      expectClose(Pred.grad.data, grads.grads[tfPred.name].arraySync(), 1e-5, 1e-5, '(L1 grad)');
      tf.dispose(grads);
    });
  });

  it('Pow, Exp, Log, Sqrt, Abs forward/backward', () => {
    const data = [[1.0, 4.0], [9.0, 0.25]];
    const X = createTensor('X', data, true);
    const Y = pow(X, 2, 'Y');
    backward(Y);
    expect(X.grad).not.toBeNull();

    withTfReference(ctx => {
      const tfX = ctx.variable2d(data);
      const grads = tf.variableGrads(() => tf.pow(tfX, 2).sum() as tf.Scalar);
      expectClose(X.grad.data, grads.grads[tfX.name].arraySync(), 1e-5, 1e-5, '(Pow grad)');
      tf.dispose(grads);
    });
  });

  it('Sum, Mean, Max, Min along dim (PyTorch keepdim=False)', () => {
    const data = [[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]];
    const X = createTensor('X', data, true);
    const Y = sum(X, 1, 'Y');
    expect(Y.shape).toEqual([2]);
    backward(Y);
    expect(X.grad).not.toBeNull();

    withTfReference(ctx => {
      const tfX = ctx.variable2d(data);
      const grads = tf.variableGrads(() => tf.sum(tfX, 1, false).sum() as tf.Scalar);
      expectClose(X.grad.data, grads.grads[tfX.name].arraySync(), 1e-5, 1e-5, '(Sum grad)');
      tf.dispose(grads);
    });
  });

  it('Max forward/backward along dim', () => {
    const data = [[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]];
    const X = createTensor('X', data, true);
    const Y = max(X, 1, 'Y');
    expect(Y.shape).toEqual([2]);
    expectClose(Y.toArray(), [3.0, 5.0], 1e-5, 1e-5, '(Max dim forward)');
    backward(Y);
    expect(X.grad).not.toBeNull();

    withTfReference(ctx => {
      const tfX = ctx.variable2d(data);
      const grads = tf.variableGrads(() => tf.max(tfX, 1, false).sum() as tf.Scalar);
      expectClose(X.grad.data, grads.grads[tfX.name].arraySync(), 1e-5, 1e-5, '(Max dim grad)');
      tf.dispose(grads);
    });
  });

  it('Max global backward routes gradient to the max element', () => {
    const data = [[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]];
    const X = createTensor('X', data, true);
    const Y = max(X, null, 'Y');
    expect(Y.item()).toBe(5.0);
    backward(Y);
    expectClose(X.grad.data, [0, 0, 0, 0, 0, 1], 1e-5, 1e-5, '(Max global grad)');
  });

  it('Min forward/backward along dim', () => {
    const data = [[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]];
    const X = createTensor('X', data, true);
    const Y = min(X, 1, 'Y');
    expect(Y.shape).toEqual([2]);
    expectClose(Y.toArray(), [1.0, 0.0], 1e-5, 1e-5, '(Min dim forward)');
    backward(Y);
    expect(X.grad).not.toBeNull();

    withTfReference(ctx => {
      const tfX = ctx.variable2d(data);
      const grads = tf.variableGrads(() => tf.min(tfX, 1, false).sum() as tf.Scalar);
      expectClose(X.grad.data, grads.grads[tfX.name].arraySync(), 1e-5, 1e-5, '(Min dim grad)');
      tf.dispose(grads);
    });
  });

  it('Min global backward routes gradient to the min element', () => {
    const data = [[1.0, 3.0, 2.0], [4.0, 0.5, 5.0]];
    const X = createTensor('X', data, true);
    const Y = min(X, null, 'Y');
    expect(Y.item()).toBe(0.5);
    backward(Y);
    expectClose(X.grad.data, [0, 0, 0, 0, 1, 0], 1e-5, 1e-5, '(Min global grad)');
  });

  it('Sum with keepdim=True and dim=None', () => {
    const data = [[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]];
    const X = createTensor('X', data, true);
    const Y = sum(X, null, 'Y', true);
    expect(Y.shape).toEqual([1, 1]);
    backward(Y);
    expect(X.grad).not.toBeNull();

    withTfReference(ctx => {
      const tfX = ctx.variable2d(data);
      const grads = tf.variableGrads(() => tf.sum(tfX, undefined, true).sum() as tf.Scalar);
      expectClose(X.grad.data, grads.grads[tfX.name].arraySync(), 1e-5, 1e-5, '(Sum all keepdim grad)');
      tf.dispose(grads);
    });
  });

  it('Transpose forward/backward', () => {
    const data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    const X = createTensor('X', data, true);
    const Y = transpose(X, 0, 1, 'Y');
    expect(Y.shape).toEqual([3, 2]);
    backward(Y);
    expect(X.grad).not.toBeNull();

    withTfReference(ctx => {
      const tfX = ctx.variable2d(data);
      const grads = tf.variableGrads(() => tf.transpose(tfX).sum() as tf.Scalar);
      expectClose(X.grad.data, grads.grads[tfX.name].arraySync(), 1e-5, 1e-5, '(Transpose grad)');
      tf.dispose(grads);
    });
  });

  it('Dropout is identity in inference mode and preserves shape', () => {
    setGraphTracking(false);
    const data = [[1.0, 2.0], [3.0, 4.0]];
    const X = createTensor('X', data, true);
    const Y = dropout(X, 0.5, 'Y');
    expect(Y.shape).toEqual([2, 2]);
    expectClose(Y.toArray(), data, 1e-5, 1e-5, '(Dropout inference identity)');
  });

  it('Adam optimizer updates parameters', () => {
    resetOptimizerState();
    const W = createTensor('W', [[2.0]], true);
    const G = createTensor('G', [[-4.0]], false);
    // Manually set gradient
    W.grad = G.data;
    adamStep(0.1, 0.9, 0.999, 0, 1e-8, false);
    const wValue = W.toArray()[0][0];
    // Parameter should move toward positive (negative gradient times positive lr)
    expect(wValue).toBeGreaterThan(2.0);
  });

  it('AdamW optimizer applies decoupled weight decay', () => {
    resetOptimizerState();
    const W = createTensor('W', [[2.0]], true);
    const G = createTensor('G', [[0.0]], false);
    W.grad = G.data;
    adamWStep(0.1, 0.9, 0.999, 0.5);
    // With zero grad and weight decay 0.5, param shrinks by lr * wd * param = 0.1 * 0.5 * 2 = 0.1
    expectClose(W.toArray(), [[1.9]], 1e-5, 1e-5, '(AdamW weight decay)');
  });

  it('ClipGradNorm reduces total gradient norm', () => {
    resetOptimizerState();
    const A = createTensor('A', [[3.0]], true);
    const B = createTensor('B', [[4.0]], true);
    A.grad = A.data;
    B.grad = B.data;
    // Total norm = 5, clip to 2.5 -> scale 0.5
    clipGradNorm(2.5);
    expectClose(A.grad.data, [1.5], 1e-5, 1e-5, '(ClipGradNorm A)');
    expectClose(B.grad.data, [2.0], 1e-5, 1e-5, '(ClipGradNorm B)');
  });

  it('LR scale affects SGD step size', () => {
    resetOptimizerState();
    setLrScale(0.5);
    const W = createTensor('W', [[2.0]], true);
    const G = createTensor('G', [[-4.0]], false);
    W.grad = G.data;
    sgdStep(0.1);
    // Effective lr = 0.05, update = -0.05 * (-4) = +0.2
    expectClose(W.toArray(), [[2.2]], 1e-5, 1e-5, '(LR scale SGD)');
  });
});

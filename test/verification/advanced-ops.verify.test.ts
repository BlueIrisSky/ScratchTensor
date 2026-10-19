import { describe, it, expect } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import {
  createTensor,
  unsqueeze,
  squeeze,
  expand,
  concat,
  stack,
  split,
  slice,
  where,
  clamp,
  backward
} from '../../src/index.js';
import {
  useVerificationLifecycle,
  expectClose,
  withTfReference
} from './harness/index.js';

describe('Verification: Advanced Tensor Ops against @tensorflow/tfjs', () => {
  useVerificationLifecycle();

  it('unsqueeze and squeeze forward/backward', () => {
    const data = [[1, 2], [3, 4]];
    const X = createTensor('X', data, true);
    const Y = unsqueeze(X, 0, 'Y');
    expect(Y.shape).toEqual([1, 2, 2]);

    backward(Y);
    expect(X.grad).not.toBeNull();

    withTfReference(ctx => {
      const tfX = ctx.variable2d(data);
      const grads = tf.variableGrads(() => {
        const tfY = tf.expandDims(tfX, 0);
        return tfY.sum() as tf.Scalar;
      });
      expectClose(X.grad.data, grads.grads[tfX.name].arraySync(), 1e-5, 1e-5, '(unsqueeze grad)');
      tf.dispose(grads);
    });
  });

  it('expand forward/backward', () => {
    const data = [[1, 2]]; // [1, 2]
    const X = createTensor('X', data, true);
    const Y = expand(X, [3, 2], 'Y');
    expect(Y.shape).toEqual([3, 2]);

    backward(Y);
    expect(X.grad).not.toBeNull();

    withTfReference(ctx => {
      const tfX = ctx.variable2d(data);
      const grads = tf.variableGrads(() => {
        const tfY = tf.broadcastTo(tfX, [3, 2]);
        return tfY.sum() as tf.Scalar;
      });
      expectClose(X.grad.data, grads.grads[tfX.name].arraySync(), 1e-5, 1e-5, '(expand grad)');
      tf.dispose(grads);
    });
  });

  it('concat forward/backward', () => {
    const aData = [[1, 2], [3, 4]];
    const bData = [[5, 6], [7, 8]];
    const A = createTensor('A', aData, true);
    const B = createTensor('B', bData, true);
    const Y = concat([A, B], 1, 'Y');
    expect(Y.shape).toEqual([2, 4]);

    backward(Y);
    expect(A.grad).not.toBeNull();
    expect(B.grad).not.toBeNull();

    withTfReference(ctx => {
      const tfA = ctx.variable2d(aData);
      const tfB = ctx.variable2d(bData);
      const grads = tf.variableGrads(() => tf.concat([tfA, tfB], 1).sum() as tf.Scalar);
      expectClose(A.grad.data, grads.grads[tfA.name].arraySync(), 1e-5, 1e-5, '(concat dA)');
      expectClose(B.grad.data, grads.grads[tfB.name].arraySync(), 1e-5, 1e-5, '(concat dB)');
      tf.dispose(grads);
    });
  });

  it('stack forward/backward', () => {
    const aData = [[1, 2], [3, 4]];
    const bData = [[5, 6], [7, 8]];
    const A = createTensor('A', aData, true);
    const B = createTensor('B', bData, true);
    const Y = stack([A, B], 0, 'Y');
    expect(Y.shape).toEqual([2, 2, 2]);

    backward(Y);
    expect(A.grad).not.toBeNull();
    expect(B.grad).not.toBeNull();

    withTfReference(ctx => {
      const tfA = ctx.variable2d(aData);
      const tfB = ctx.variable2d(bData);
      const grads = tf.variableGrads(() => tf.stack([tfA, tfB], 0).sum() as tf.Scalar);
      expectClose(A.grad.data, grads.grads[tfA.name].arraySync(), 1e-5, 1e-5, '(stack dA)');
      expectClose(B.grad.data, grads.grads[tfB.name].arraySync(), 1e-5, 1e-5, '(stack dB)');
      tf.dispose(grads);
    });
  });

  it('stack along negative dimension matches TensorFlow', () => {
    const aData = [[1, 2], [3, 4]];
    const bData = [[5, 6], [7, 8]];
    const A = createTensor('A', aData, true);
    const B = createTensor('B', bData, true);
    const Y = stack([A, B], -1, 'Y');
    expect(Y.shape).toEqual([2, 2, 2]);

    backward(Y);
    expect(A.grad).not.toBeNull();
    expect(B.grad).not.toBeNull();

    withTfReference(ctx => {
      const tfA = ctx.variable2d(aData);
      const tfB = ctx.variable2d(bData);
      const grads = tf.variableGrads(() => tf.stack([tfA, tfB], -1).sum() as tf.Scalar);
      expectClose(Y.toArray(), tf.stack([tfA, tfB], -1).arraySync(), 1e-5, 1e-5, '(stack -1 forward)');
      expectClose(A.grad.data, grads.grads[tfA.name].arraySync(), 1e-5, 1e-5, '(stack -1 dA)');
      expectClose(B.grad.data, grads.grads[tfB.name].arraySync(), 1e-5, 1e-5, '(stack -1 dB)');
      tf.dispose(grads);
    });
  });

  it('split along negative dimension', () => {
    const data = [[1, 2, 3, 4], [5, 6, 7, 8]];
    const X = createTensor('X', data, true);
    const parts = split(X, -1, 2, 'part');
    expect(parts.length).toBe(2);
    expect(parts[0].shape).toEqual([2, 2]);
    expect(parts[1].shape).toEqual([2, 2]);
    expect(parts[0].toArray()).toEqual([[1, 2], [5, 6]]);
    expect(parts[1].toArray()).toEqual([[3, 4], [7, 8]]);

    backward(parts[0]);
    expect(X.grad).not.toBeNull();
    withTfReference(ctx => {
      const tfX = ctx.variable2d(data);
      const grads = tf.variableGrads(() => tf.slice(tfX, [0, 0], [2, 2]).sum() as tf.Scalar);
      expectClose(X.grad.data, grads.grads[tfX.name].arraySync(), 1e-5, 1e-5, '(split -1 grad)');
      tf.dispose(grads);
    });
  });

  it('slice forward/backward', () => {
    const data = [[1, 2, 3], [4, 5, 6]];
    const X = createTensor('X', data, true);
    const Y = slice(X, 1, 1, 3, 'Y');
    expect(Y.shape).toEqual([2, 2]);

    backward(Y);
    expect(X.grad).not.toBeNull();

    withTfReference(ctx => {
      const tfX = ctx.variable2d(data);
      const grads = tf.variableGrads(() => {
        const tfY = tf.slice(tfX, [0, 1], [2, 2]);
        return tfY.sum() as tf.Scalar;
      });
      expectClose(X.grad.data, grads.grads[tfX.name].arraySync(), 1e-5, 1e-5, '(slice grad)');
      tf.dispose(grads);
    });
  });

  it('where forward/backward', () => {
    const condData = [[1, 0], [0, 1]];
    const aData = [[1, 2], [3, 4]];
    const bData = [[5, 6], [7, 8]];
    const Cond = createTensor('Cond', condData, false);
    const A = createTensor('A', aData, true);
    const B = createTensor('B', bData, true);
    const Y = where(Cond, A, B, 'Y');
    expect(Y.shape).toEqual([2, 2]);

    backward(Y);
    expect(A.grad).not.toBeNull();
    expect(B.grad).not.toBeNull();

    withTfReference(ctx => {
      const tfA = ctx.variable2d(aData);
      const tfB = ctx.variable2d(bData);
      const grads = tf.variableGrads(() => {
        const tfCond = tf.tensor2d(condData, [2, 2], 'bool');
        return tf.where(tfCond, tfA, tfB).sum() as tf.Scalar;
      });
      expectClose(A.grad.data, grads.grads[tfA.name].arraySync(), 1e-5, 1e-5, '(where dA)');
      expectClose(B.grad.data, grads.grads[tfB.name].arraySync(), 1e-5, 1e-5, '(where dB)');
      tf.dispose(grads);
    });
  });

  it('where forward/backward with broadcasting', () => {
    const condData = [[1, 0]]; // [1, 2]
    const aData = [1, 2]; // [2] broadcasts to [1, 2]
    const bData = [[5, 6], [7, 8]]; // [2, 2]
    const Cond = createTensor('Cond', condData, false);
    const A = createTensor('A', aData, true);
    const B = createTensor('B', bData, true);
    const Y = where(Cond, A, B, 'Y');
    expect(Y.shape).toEqual([2, 2]);
    expect(Y.toArray()).toEqual([[1, 6], [1, 8]]);

    backward(Y);
    expect(A.grad).not.toBeNull();
    expect(B.grad).not.toBeNull();

    withTfReference(ctx => {
      const tfA = ctx.variable1d(aData);
      const tfB = ctx.variable2d(bData);
      const grads = tf.variableGrads(() => {
        const tfCond = tf.tensor2d(condData, [1, 2], 'bool');
        return tf.where(tfCond, tfA, tfB).sum() as tf.Scalar;
      });
      expectClose(A.grad.data, grads.grads[tfA.name].arraySync(), 1e-5, 1e-5, '(where broadcast dA)');
      expectClose(B.grad.data, grads.grads[tfB.name].arraySync(), 1e-5, 1e-5, '(where broadcast dB)');
      tf.dispose(grads);
    });
  });

  it('clamp forward/backward', () => {
    const data = [[-2, 0.5], [3, -0.1]];
    const X = createTensor('X', data, true);
    const Y = clamp(X, -1, 1, 'Y');
    expect(Y.shape).toEqual([2, 2]);

    backward(Y);
    expect(X.grad).not.toBeNull();

    withTfReference(ctx => {
      const tfX = ctx.variable2d(data);
      const grads = tf.variableGrads(() => {
        const tfY = tf.clipByValue(tfX, -1, 1);
        return tfY.sum() as tf.Scalar;
      });
      expectClose(X.grad.data, grads.grads[tfX.name].arraySync(), 1e-5, 1e-5, '(clamp grad)');
      tf.dispose(grads);
    });
  });
});

import { describe, it, expect } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { createTensor, matmul, backward } from '../../src/index.js';
import {
  useVerificationLifecycle,
  expectClose,
  withTfReference,
  verifyEagerGraphCleanup
} from './harness/index.js';

describe('Verification: MatMulOp against @tensorflow/tfjs', () => {
  useVerificationLifecycle();

  it('forward [2x2] * [2x2] square matrices', () => {
    const aData = [[1.2, -0.5], [3.0, 4.1]];
    const bData = [[0.8, 2.0], [-1.5, 0.3]];

    const C = matmul(createTensor('A', aData), createTensor('B', bData), 'C');

    withTfReference(ctx => {
      const tfC = tf.matMul(ctx.tensor2d(aData), ctx.tensor2d(bData));
      expect(C.shape).toEqual(tfC.shape);
      expectClose(C.toArray(), tfC.arraySync(), 1e-5, 1e-5, '(MatMul Square Forward)');
    });
  });

  it('forward [3x4] * [4x2] rectangular matrices', () => {
    const aData = [[1, 2, -1, 0.5], [0, -3, 4, 2], [2.5, 1.1, -0.2, 3.3]];
    const bData = [[0.5, -1], [1.2, 0.4], [-2, 3], [0.1, 0.2]];

    const C = matmul(createTensor('A', aData), createTensor('B', bData), 'C');

    withTfReference(ctx => {
      const tfC = tf.matMul(ctx.tensor2d(aData), ctx.tensor2d(bData));
      expect(C.shape).toEqual(tfC.shape);
      expectClose(C.toArray(), tfC.arraySync(), 1e-5, 1e-5, '(MatMul Rect Forward)');
    });
  });

  it('backward dA and dB with both inputs requiring grad', () => {
    const aData = [[1.5, -2.0, 0.5], [0.8, 3.2, -1.1]];
    const bData = [[-1.0, 2.0], [0.5, -0.5], [1.2, 3.0]];

    const A = createTensor('A', aData, true);
    const B = createTensor('B', bData, true);
    const C = matmul(A, B, 'C');
    backward(C);

    expect(A.grad).not.toBeNull();
    expect(B.grad).not.toBeNull();

    withTfReference(ctx => {
      const tfA = ctx.variable2d(aData);
      const tfB = ctx.variable2d(bData);
      const grads = tf.variableGrads(() => tf.matMul(tfA, tfB).sum() as tf.Scalar);
      expectClose(A.grad.data, grads.grads[tfA.name].arraySync(), 1e-5, 1e-5, '(MatMul dA)');
      expectClose(B.grad.data, grads.grads[tfB.name].arraySync(), 1e-5, 1e-5, '(MatMul dB)');
      tf.dispose(grads);
    });

    verifyEagerGraphCleanup([C]);
  });

  it('backward only A requires grad (partial)', () => {
    const aData = [[2.0, 1.0], [-1.0, 3.0]];
    const bData = [[0.5, -1.5], [2.2, 0.7]];

    const A = createTensor('A', aData, true);
    const B = createTensor('B', bData, false);
    const C = matmul(A, B, 'C');
    backward(C);

    expect(A.grad).not.toBeNull();
    expect(B.grad).toBeNull();

    withTfReference(ctx => {
      const tfA = ctx.variable2d(aData);
      const tfB = ctx.tensor2d(bData);
      const grads = tf.variableGrads(() => tf.matMul(tfA, tfB).sum() as tf.Scalar);
      expectClose(A.grad.data, grads.grads[tfA.name].arraySync(), 1e-5, 1e-5, '(MatMul partial dA)');
      tf.dispose(grads);
    });

    verifyEagerGraphCleanup([C]);
  });

  it('1D x 1D dot product', () => {
    const aData = [1.0, 2.0, 3.0];
    const bData = [4.0, 5.0, 6.0];
    const A = createTensor('A', aData, true);
    const B = createTensor('B', bData, true);
    const C = matmul(A, B, 'C');
    expect(C.shape).toEqual([]);
    expect(C.toArray()).toEqual(32); // 1*4 + 2*5 + 3*6 = 32
    backward(C);
    expect(A.grad).not.toBeNull();
    expect(B.grad).not.toBeNull();
    expectClose(A.grad.data, [4, 5, 6], 1e-5, 1e-5, '(MatMul 1D dA)');
    expectClose(B.grad.data, [1, 2, 3], 1e-5, 1e-5, '(MatMul 1D dB)');
  });

  it('2D x 1D matrix-vector', () => {
    const aData = [[1.0, 2.0], [3.0, 4.0]];
    const bData = [5.0, 6.0];
    const A = createTensor('A', aData, true);
    const B = createTensor('B', bData, true);
    const C = matmul(A, B, 'C');
    expect(C.shape).toEqual([2]);
    backward(C);

    withTfReference(ctx => {
      const tfA = ctx.variable2d(aData);
      const tfB = ctx.variable1d(bData);
      const grads = tf.variableGrads(() => tf.squeeze(tf.matMul(tfA, tf.expandDims(tfB, 1)), [1]).sum() as tf.Scalar);
      expectClose(A.grad.data, grads.grads[tfA.name].arraySync(), 1e-5, 1e-5, '(MatMul 2D-1D dA)');
      expectClose(B.grad.data, grads.grads[tfB.name].arraySync(), 1e-5, 1e-5, '(MatMul 2D-1D dB)');
      tf.dispose(grads);
    });
  });

  it('1D x 2D vector-matrix', () => {
    const aData = [1.0, 2.0];
    const bData = [[3.0, 4.0], [5.0, 6.0]];
    const A = createTensor('A', aData, true);
    const B = createTensor('B', bData, true);
    const C = matmul(A, B, 'C');
    expect(C.shape).toEqual([2]);
    backward(C);

    withTfReference(ctx => {
      const tfA = ctx.variable1d(aData);
      const tfB = ctx.variable2d(bData);
      const grads = tf.variableGrads(() => tf.squeeze(tf.matMul(tf.expandDims(tfA, 0), tfB), [0]).sum() as tf.Scalar);
      expectClose(A.grad.data, grads.grads[tfA.name].arraySync(), 1e-5, 1e-5, '(MatMul 1D-2D dA)');
      expectClose(B.grad.data, grads.grads[tfB.name].arraySync(), 1e-5, 1e-5, '(MatMul 1D-2D dB)');
      tf.dispose(grads);
    });
  });

  it('broadcasts 2D x 3D like torch.matmul', () => {
    const aData = [[1.0, 2.0], [3.0, 4.0]];
    const bData = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]];

    const A = createTensor('A', aData, true);
    const B = createTensor('B', bData, true);
    const C = matmul(A, B, 'C');

    withTfReference(ctx => {
      const tfA = ctx.tensor2d(aData);
      const tfB = tf.tensor3d(bData, [2, 2, 2]);
      const tfC = tf.matMul(tfA, tfB);
      expect(C.shape).toEqual(tfC.shape);
      expectClose(C.toArray(), tfC.arraySync(), 1e-5, 1e-5, '(MatMul 2D-3D broadcast forward)');
    });

    // Verify broadcast backward numerically: gradient of the 2D operand must be
    // summed over the broadcast batch dims back to [2,2] (PyTorch semantics).
    backward(C);
    expect(A.grad).not.toBeNull();
    expect(B.grad).not.toBeNull();
    expect(A.grad.shape).toEqual([2, 2]);
    expect(B.grad.shape).toEqual([2, 2, 2]);

    withTfReference(ctx => {
      const tfA = ctx.variable2d(aData);
      const tfB = ctx.variable3d(bData);
      const grads = tf.variableGrads(() => {
        const a3d = tf.tile(tf.expandDims(tfA, 0), [2, 1, 1]);
        return tf.matMul(a3d, tfB).sum() as tf.Scalar;
      });
      expectClose(A.grad.data, grads.grads[tfA.name].arraySync(), 1e-5, 1e-5, '(MatMul 2D-3D broadcast dA)');
      expectClose(B.grad.data, grads.grads[tfB.name].arraySync(), 1e-5, 1e-5, '(MatMul 2D-3D broadcast dB)');
      tf.dispose(grads);
    });
  });

  it('broadcasts 4D x 4D with compatible batch dimensions', () => {
    const aData = [[[[1.0, 2.0], [3.0, 4.0]]], [[[5.0, 6.0], [7.0, 8.0]]]];
    const bData = [[[[1.0, 0.0], [0.0, 1.0]]]];

    const A = createTensor('A', aData, true);
    const B = createTensor('B', bData, true);
    const C = matmul(A, B, 'C');

    withTfReference(ctx => {
      const tfA = tf.tensor4d(aData, [2, 1, 2, 2]);
      const tfB = tf.tensor4d(bData, [1, 1, 2, 2]);
      const tfC = tf.matMul(tfA, tfB);
      expect(C.shape).toEqual(tfC.shape);
      expectClose(C.toArray(), tfC.arraySync(), 1e-5, 1e-5, '(MatMul 4D broadcast forward)');
      tfA.dispose();
      tfB.dispose();
      tfC.dispose();
    });

    // Backward: B was broadcast over batch dim 0, so dB must sum back to [1,1,2,2].
    backward(C);
    expect(A.grad.shape).toEqual([2, 1, 2, 2]);
    expect(B.grad.shape).toEqual([1, 1, 2, 2]);

    withTfReference(ctx => {
      const tfA = tf.tensor4d(aData, [2, 1, 2, 2]);
      const tfB = tf.variable(tf.tensor4d(bData, [1, 1, 2, 2]));
      const grads = tf.variableGrads(() => {
        const b4d = tf.tile(tfB, [2, 1, 1, 1]);
        return tf.matMul(tfA, b4d).sum() as tf.Scalar;
      });
      expectClose(B.grad.data, grads.grads[tfB.name].arraySync(), 1e-5, 1e-5, '(MatMul 4D broadcast dB)');
      tf.dispose(grads);
      tfA.dispose();
      tfB.dispose();
    });
  });
});

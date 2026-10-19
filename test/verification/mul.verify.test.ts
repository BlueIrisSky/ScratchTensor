import { describe, it, expect } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { createTensor, mul, backward } from '../../src/index.js';
import {
  useVerificationLifecycle,
  expectClose,
  withTfReference,
  verifyEagerGraphCleanup
} from './harness/index.js';

describe('Verification: MulOp against @tensorflow/tfjs', () => {
  useVerificationLifecycle();

  it('forward [2x3] * [2x3] element-wise', () => {
    const aData = [[2.0, -1.5, 3.0], [0.5, -4.0, 1.2]];
    const bData = [[-3.0, 2.0, 0.5], [4.0, 1.5, -2.0]];

    const C = mul(createTensor('A', aData), createTensor('B', bData), 'C');

    withTfReference(ctx => {
      const tfC = tf.mul(ctx.tensor2d(aData), ctx.tensor2d(bData));
      expect(C.shape).toEqual(tfC.shape);
      expectClose(C.toArray(), tfC.arraySync(), 1e-5, 1e-5, '(Mul Forward)');
    });
  });

  it('forward [3x3] * [3x1] column broadcast', () => {
    const aData = [[2.0, -1.5, 3.0], [0.5, -4.0, 1.2], [1.0, 2.0, -3.0]];
    const bData = [[2.0], [-1.0], [3.0]];

    const C = mul(createTensor('A', aData), createTensor('B', bData), 'C');

    withTfReference(ctx => {
      const tfC = tf.mul(ctx.tensor2d(aData), ctx.tensor2d(bData));
      expect(C.shape).toEqual(tfC.shape);
      expectClose(C.toArray(), tfC.arraySync(), 1e-5, 1e-5, '(Mul Column Broadcast Forward)');
    });
  });

  it('backward dA and dB element-wise derivative', () => {
    const aData = [[1.5, -2.0], [3.0, 0.4]];
    const bData = [[4.0, 1.1], [-2.5, 5.0]];

    const A = createTensor('A', aData, true);
    const B = createTensor('B', bData, true);
    const C = mul(A, B, 'C');
    backward(C);

    withTfReference(ctx => {
      const tfA = ctx.variable2d(aData);
      const tfB = ctx.variable2d(bData);
      const grads = tf.variableGrads(() => tf.mul(tfA, tfB).sum() as tf.Scalar);
      expectClose(A.grad.data, grads.grads[tfA.name].arraySync(), 1e-5, 1e-5, '(Mul dA)');
      expectClose(B.grad.data, grads.grads[tfB.name].arraySync(), 1e-5, 1e-5, '(Mul dB)');
      tf.dispose(grads);
    });

    verifyEagerGraphCleanup([C]);
  });

  it('backward dA and dB column broadcast [3x3] * [3x1]', () => {
    const aData = [[1.5, -2.0, 3.0], [3.0, 0.4, -1.0], [2.0, -3.0, 1.0]];
    const bData = [[4.0], [1.1], [-2.5]];

    const A = createTensor('A', aData, true);
    const B = createTensor('B', bData, true);
    const C = mul(A, B, 'C');
    backward(C);

    withTfReference(ctx => {
      const tfA = ctx.variable2d(aData);
      const tfB = ctx.variable2d(bData);
      const grads = tf.variableGrads(() => tf.mul(tfA, tfB).sum() as tf.Scalar);
      expectClose(A.grad.data, grads.grads[tfA.name].arraySync(), 1e-5, 1e-5, '(Mul column dA)');
      expectClose(B.grad.data, grads.grads[tfB.name].arraySync(), 1e-5, 1e-5, '(Mul column dB)');
      tf.dispose(grads);
    });

    verifyEagerGraphCleanup([C]);
  });
});

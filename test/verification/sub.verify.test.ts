import { describe, it, expect } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { createTensor, sub, backward } from '../../src/index.js';
import {
  useVerificationLifecycle,
  expectClose,
  withTfReference,
  verifyEagerGraphCleanup
} from './harness/index.js';

describe('Verification: SubOp against @tensorflow/tfjs', () => {
  useVerificationLifecycle();

  it('forward [2x3] - [2x3]', () => {
    const aData = [[5.5, -2.0, 1.1], [0.0, 3.4, -4.2]];
    const bData = [[1.2, 3.5, -0.9], [2.0, -1.1, 4.2]];

    const C = sub(createTensor('A', aData), createTensor('B', bData), 'C');

    withTfReference(ctx => {
      const tfC = tf.sub(ctx.tensor2d(aData), ctx.tensor2d(bData));
      expect(C.shape).toEqual(tfC.shape);
      expectClose(C.toArray(), tfC.arraySync(), 1e-5, 1e-5, '(Sub Forward)');
    });
  });

  it('forward [3x4] - [3x1] column broadcast', () => {
    const aData = [[5.5, -2.0, 1.1], [0.0, 3.4, -4.2], [1.0, -1.0, 2.0]];
    const bData = [[1.0], [2.0], [3.0]];

    const C = sub(createTensor('A', aData), createTensor('B', bData), 'C');

    withTfReference(ctx => {
      const tfC = tf.sub(ctx.tensor2d(aData), ctx.tensor2d(bData));
      expect(C.shape).toEqual(tfC.shape);
      expectClose(C.toArray(), tfC.arraySync(), 1e-5, 1e-5, '(Sub Column Broadcast Forward)');
    });
  });

  it('backward dA (+1) and dB (-1)', () => {
    const aData = [[10.0, -3.0], [2.5, 0.0]];
    const bData = [[4.0, 1.5], [-2.0, 3.0]];

    const A = createTensor('A', aData, true);
    const B = createTensor('B', bData, true);
    const C = sub(A, B, 'C');
    backward(C);

    withTfReference(ctx => {
      const tfA = ctx.variable2d(aData);
      const tfB = ctx.variable2d(bData);
      const grads = tf.variableGrads(() => tf.sub(tfA, tfB).sum() as tf.Scalar);
      expectClose(A.grad.data, grads.grads[tfA.name].arraySync(), 1e-5, 1e-5, '(Sub dA)');
      expectClose(B.grad.data, grads.grads[tfB.name].arraySync(), 1e-5, 1e-5, '(Sub dB)');
      tf.dispose(grads);
    });

    verifyEagerGraphCleanup([C]);
  });

  it('backward dA and dB column broadcast [3x3] - [3x1]', () => {
    const aData = [[5.5, -2.0, 1.1], [0.0, 3.4, -4.2], [1.0, -1.0, 2.0]];
    const bData = [[1.2], [-0.5], [3.0]];

    const A = createTensor('A', aData, true);
    const B = createTensor('B', bData, true);
    const C = sub(A, B, 'C');
    backward(C);

    withTfReference(ctx => {
      const tfA = ctx.variable2d(aData);
      const tfB = ctx.variable2d(bData);
      const grads = tf.variableGrads(() => tf.sub(tfA, tfB).sum() as tf.Scalar);
      expectClose(A.grad.data, grads.grads[tfA.name].arraySync(), 1e-5, 1e-5, '(Sub column dA)');
      expectClose(B.grad.data, grads.grads[tfB.name].arraySync(), 1e-5, 1e-5, '(Sub column dB)');
      tf.dispose(grads);
    });

    verifyEagerGraphCleanup([C]);
  });
});

import { describe, it, expect } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { createTensor, div, backward } from '../../src/index.js';
import {
  useVerificationLifecycle,
  expectClose,
  withTfReference,
  verifyEagerGraphCleanup
} from './harness/index.js';

describe('Verification: DivOp against @tensorflow/tfjs', () => {
  useVerificationLifecycle();

  it('forward [2x3] / [2x3] element-wise', () => {
    const aData = [[2.0, -3.0, 4.0], [5.0, 6.0, -7.0]];
    const bData = [[1.0, 1.5, 2.0], [-2.0, 3.0, 0.5]];

    const C = div(createTensor('A', aData), createTensor('B', bData), 'C');

    withTfReference(ctx => {
      const tfC = tf.div(ctx.tensor2d(aData), ctx.tensor2d(bData));
      expect(C.shape).toEqual(tfC.shape);
      expectClose(C.toArray(), tfC.arraySync(), 1e-5, 1e-5, '(Div Forward)');
    });
  });

  it('forward [3x3] / [3x1] column broadcast', () => {
    const aData = [[2.0, -3.0, 4.0], [5.0, 6.0, -7.0], [1.0, -1.0, 2.0]];
    const bData = [[2.0], [-1.0], [4.0]];

    const C = div(createTensor('A', aData), createTensor('B', bData), 'C');

    withTfReference(ctx => {
      const tfC = tf.div(ctx.tensor2d(aData), ctx.tensor2d(bData));
      expect(C.shape).toEqual(tfC.shape);
      expectClose(C.toArray(), tfC.arraySync(), 1e-5, 1e-5, '(Div Column Broadcast Forward)');
    });
  });

  it('backward dA and dB element-wise', () => {
    const aData = [[4.0, 9.0], [1.0, -2.0]];
    const bData = [[2.0, 3.0], [0.5, -1.0]];

    const A = createTensor('A', aData, true);
    const B = createTensor('B', bData, true);
    const C = div(A, B, 'C');
    backward(C);

    withTfReference(ctx => {
      const tfA = ctx.variable2d(aData);
      const tfB = ctx.variable2d(bData);
      const grads = tf.variableGrads(() => tf.div(tfA, tfB).sum() as tf.Scalar);
      expectClose(A.grad.data, grads.grads[tfA.name].arraySync(), 1e-5, 1e-5, '(Div dA)');
      expectClose(B.grad.data, grads.grads[tfB.name].arraySync(), 1e-5, 1e-5, '(Div dB)');
      tf.dispose(grads);
    });

    verifyEagerGraphCleanup([C]);
  });

  it('backward scalar divisor', () => {
    const aData = [[2.0, 4.0], [6.0, 8.0]];
    const bData = [[2.0]];

    const A = createTensor('A', aData, true);
    const B = createTensor('B', bData, true);
    const C = div(A, B, 'C');
    backward(C);

    withTfReference(ctx => {
      const tfA = ctx.variable2d(aData);
      const tfB = ctx.variable2d(bData);
      const grads = tf.variableGrads(() => tf.div(tfA, tfB).sum() as tf.Scalar);
      expectClose(A.grad.data, grads.grads[tfA.name].arraySync(), 1e-5, 1e-5, '(Div scalar dA)');
      expectClose(B.grad.data, grads.grads[tfB.name].arraySync(), 1e-5, 1e-5, '(Div scalar dB)');
      tf.dispose(grads);
    });

    verifyEagerGraphCleanup([C]);
  });

  it('backward dA and dB column broadcast [3x3] / [3x1]', () => {
    const aData = [[4.0, 9.0, 1.0], [1.0, -2.0, 3.0], [2.0, 5.0, -6.0]];
    const bData = [[2.0], [0.5], [-1.0]];

    const A = createTensor('A', aData, true);
    const B = createTensor('B', bData, true);
    const C = div(A, B, 'C');
    backward(C);

    withTfReference(ctx => {
      const tfA = ctx.variable2d(aData);
      const tfB = ctx.variable2d(bData);
      const grads = tf.variableGrads(() => tf.div(tfA, tfB).sum() as tf.Scalar);
      expectClose(A.grad.data, grads.grads[tfA.name].arraySync(), 1e-5, 1e-5, '(Div column dA)');
      expectClose(B.grad.data, grads.grads[tfB.name].arraySync(), 1e-5, 1e-5, '(Div column dB)');
      tf.dispose(grads);
    });

    verifyEagerGraphCleanup([C]);
  });
});

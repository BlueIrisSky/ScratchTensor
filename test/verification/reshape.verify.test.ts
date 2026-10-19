import { describe, it, expect } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { createTensor, reshape, backward } from '../../src/index.js';
import {
  useVerificationLifecycle,
  expectClose,
  withTfReference,
  verifyEagerGraphCleanup
} from './harness/index.js';

describe('Verification: ReshapeOp against @tensorflow/tfjs', () => {
  useVerificationLifecycle();

  it('forward [2x3] -> [3x2] and [6]', () => {
    const xData = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

    const Y1 = reshape(createTensor('X1', xData), [3, 2], 'Y1');
    const Y2 = reshape(createTensor('X2', xData), [6], 'Y2');

    withTfReference(ctx => {
      const tfX = ctx.tensor2d(xData);
      const tfY1 = tf.reshape(tfX, [3, 2]);
      const tfY2 = tf.reshape(tfX, [6]);

      expect(Y1.shape).toEqual(tfY1.shape);
      expectClose(Y1.toArray(), tfY1.arraySync(), 1e-5, 1e-5, '(Reshape 2D->2D)');

      expect(Y2.shape).toEqual(tfY2.shape);
      expectClose(Y2.toArray(), tfY2.arraySync(), 1e-5, 1e-5, '(Reshape 2D->1D)');
    });
  });

  it('backward dX shape restoration', () => {
    const xData = [[1.5, -2.5, 3.5], [-4.5, 5.5, -6.5]];

    const X = createTensor('X', xData, true);
    const Y = reshape(X, [6], 'Y');
    backward(Y);

    expect(X.grad.shape).toEqual([2, 3]);

    withTfReference(ctx => {
      const tfX = ctx.variable2d(xData);
      const grads = tf.variableGrads(() => tf.reshape(tfX, [6]).sum() as tf.Scalar);
      expectClose(X.grad.data, grads.grads[tfX.name].arraySync(), 1e-5, 1e-5, '(Reshape dX)');
      tf.dispose(grads);
    });

    verifyEagerGraphCleanup([Y]);
  });
});

import { describe, it, expect } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { createTensor, relu, backward } from '../../src/index.js';
import {
  useVerificationLifecycle,
  expectClose,
  withTfReference,
  verifyEagerGraphCleanup
} from './harness/index.js';

describe('Verification: ReLUOp against @tensorflow/tfjs', () => {
  useVerificationLifecycle();

  it('forward across positive, negative, and zero entries', () => {
    const xData = [[-3.5, 0.0, 2.1], [10.0, -0.01, -100.0]];

    const Y = relu(createTensor('X', xData), 'Y');

    withTfReference(ctx => {
      const tfY = tf.relu(ctx.tensor2d(xData));
      expect(Y.shape).toEqual(tfY.shape);
      expectClose(Y.toArray(), tfY.arraySync(), 1e-5, 1e-5, '(ReLU Forward)');
    });
  });

  it('backward dX masking non-positive gradients to zero', () => {
    const xData = [[-2.0, 5.0, 0.0], [3.3, -1.1, 0.001]];

    const X = createTensor('X', xData, true);
    const Y = relu(X, 'Y');
    backward(Y);

    withTfReference(ctx => {
      const tfX = ctx.variable2d(xData);
      const grads = tf.variableGrads(() => tf.relu(tfX).sum() as tf.Scalar);
      expectClose(X.grad.data, grads.grads[tfX.name].arraySync(), 1e-5, 1e-5, '(ReLU dX)');
      tf.dispose(grads);
    });

    verifyEagerGraphCleanup([Y]);
  });
});

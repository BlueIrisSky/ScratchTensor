import { describe, it, expect } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { createTensor, mseLoss, backward } from '../../src/index.js';
import {
  useVerificationLifecycle,
  expectClose,
  withTfReference,
  verifyEagerGraphCleanup
} from './harness/index.js';

describe('Verification: MSELossOp against @tensorflow/tfjs', () => {
  useVerificationLifecycle();

  it('forward scalar loss value', () => {
    const predData = [[1.5, 2.0], [-0.5, 3.1]];
    const targetData = [[1.0, 2.5], [0.0, 3.0]];

    const Loss = mseLoss(createTensor('Pred', predData), createTensor('Target', targetData), 'Loss');

    withTfReference(ctx => {
      const tfLoss = tf.losses.meanSquaredError(ctx.tensor2d(targetData), ctx.tensor2d(predData));
      expectClose(Loss.toArray(), tfLoss.arraySync(), 1e-5, 1e-5, '(MSELoss Forward)');
    });
  });

  it('backward gradient dPred', () => {
    const predData = [[2.0, -1.0, 0.5], [1.1, 0.0, -2.2]];
    const targetData = [[1.0, 0.0, 0.5], [2.0, -1.0, -2.0]];

    const Pred = createTensor('Pred', predData, true);
    const Target = createTensor('Target', targetData, false);
    const Loss = mseLoss(Pred, Target, 'Loss');
    backward(Loss);

    withTfReference(ctx => {
      const tfPred = ctx.variable2d(predData);
      const tfTarget = ctx.tensor2d(targetData);
      const grads = tf.variableGrads(() => tf.losses.meanSquaredError(tfTarget, tfPred as tf.Tensor2D) as tf.Scalar);
      expectClose(Pred.grad.data, grads.grads[tfPred.name].arraySync(), 1e-5, 1e-5, '(MSELoss dPred)');
      tf.dispose(grads);
    });

    verifyEagerGraphCleanup([Loss]);
  });
});

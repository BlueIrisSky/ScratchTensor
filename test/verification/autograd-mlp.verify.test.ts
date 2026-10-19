import { describe, it, expect } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { createTensor, matmul, add, relu, mseLoss, backward } from '../../src/index.js';
import {
  useVerificationLifecycle,
  expectClose,
  withTfReference,
  verifyEagerGraphCleanup
} from './harness/index.js';

describe('Verification: MLP End-to-End Backpropagation against @tensorflow/tfjs', () => {
  useVerificationLifecycle();

  it('full forward + backward (MatMul -> Add -> ReLU -> MSELoss) gradient alignment', () => {
    const xData = [[1.0, -0.5, 2.0], [0.0, 1.5, -1.0]];
    const wData = [[0.5, -0.2], [1.1, 0.3], [-0.8, 1.4]];
    const bData = [[0.1, -0.1]];
    const targetData = [[1.0, 0.0], [0.0, 1.0]];

    // ScratchTensor forward + backward
    const X = createTensor('X', xData, false);
    const W = createTensor('W', wData, true);
    const b = createTensor('b', bData, true);
    const Target = createTensor('Target', targetData, false);

    const Z = matmul(X, W, 'Z');
    const H = add(Z, b, 'H');
    const A = relu(H, 'A');
    const Loss = mseLoss(A, Target, 'Loss');
    backward(Loss);

    // TF.js reference
    withTfReference(ctx => {
      const tfX = ctx.tensor2d(xData);
      const tfW = ctx.variable2d(wData);
      const tfB = ctx.variable2d(bData);
      const tfTarget = ctx.tensor2d(targetData);

      const grads = tf.variableGrads(() => {
        const z = tf.matMul(tfX, tfW);
        const h = tf.add(z, tfB);
        const a = tf.relu(h);
        return tf.losses.meanSquaredError(tfTarget, a) as tf.Scalar;
      });

      expect(W.grad).not.toBeNull();
      expectClose(W.grad.data, grads.grads[tfW.name].arraySync(), 1e-5, 1e-5, '(MLP dW)');

      expect(b.grad).not.toBeNull();
      expectClose(b.grad.data, grads.grads[tfB.name].arraySync(), 1e-5, 1e-5, '(MLP db)');

      tf.dispose(grads);
    });

    verifyEagerGraphCleanup([Z, H, A, Loss]);
  });
});

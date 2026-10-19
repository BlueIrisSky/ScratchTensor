import { describe, it, expect } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { createTensor, layerNorm, backward } from '../../src/index.js';
import {
  useVerificationLifecycle,
  expectClose,
  withTfReference
} from './harness/index.js';

function layerNormRef(x: tf.Tensor, n: number, eps = 1e-5): tf.Tensor {
  const rank = x.shape.length;
  const axes = Array.from({ length: n }, (_, i) => rank - n + i);
  const mean = tf.mean(x, axes, true);
  const variance = tf.mean(tf.square(tf.sub(x, mean)), axes, true);
  return tf.div(tf.sub(x, mean), tf.sqrt(tf.add(variance, eps)));
}

describe('Verification: LayerNormOp against TensorFlow reference', () => {
  useVerificationLifecycle();

  it('forward normalizes last 1 dimension on 2D input', () => {
    const data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    const Y = layerNorm(createTensor('X', data), 1, 'Y');

    withTfReference(ctx => {
      const tfX = ctx.tensor2d(data);
      const tfY = layerNormRef(tfX, 1);
      expect(Y.shape).toEqual(tfY.shape);
      expectClose(Y.toArray(), tfY.arraySync(), 1e-5, 1e-5, '(LayerNorm forward 2D)');
    });
  });

  it('backward dX for last-1-dim normalization', () => {
    const data = [[1.0, 2.0, 0.5], [-1.0, 3.0, 2.0]];
    const X = createTensor('X', data, true);
    const Y = layerNorm(X, 1, 'Y');
    backward(Y);

    withTfReference(ctx => {
      const tfX = ctx.variable2d(data);
      const grads = tf.variableGrads(() => layerNormRef(tfX, 1).sum() as tf.Scalar);
      expectClose(X.grad.data, grads.grads[tfX.name].arraySync(), 1e-5, 1e-5, '(LayerNorm dX)');
      tf.dispose(grads);
    });
  });

  it('forward normalizes last 2 dimensions on 3D input', () => {
    const data = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]];
    const Y = layerNorm(createTensor('X', data), 2, 'Y');

    withTfReference(ctx => {
      const tfX = ctx.tensor3d(data);
      const tfY = layerNormRef(tfX, 2);
      expect(Y.shape).toEqual(tfY.shape);
      expectClose(Y.toArray(), tfY.arraySync(), 1e-5, 1e-5, '(LayerNorm forward 3D)');
    });
  });
});

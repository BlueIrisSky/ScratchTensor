import { describe, it, expect } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import {
  createTensor,
  scaledDotProductAttention,
  backward,
  setGraphTracking
} from '../../src/index.js';
import {
  useVerificationLifecycle,
  expectClose
} from './harness/index.js';

function referenceScaledDotProductAttention(
  q: tf.Tensor,
  k: tf.Tensor,
  v: tf.Tensor,
  isCausal = false,
  scale?: number
): tf.Tensor {
  const qShape = q.shape;
  const D = qShape[qShape.length - 1];
  const effScale = scale ?? 1.0 / Math.sqrt(D);
  const scores = tf.mul(tf.matMul(q, k, false, true), effScale);
  let masked = scores;
  if (isCausal) {
    const Lq = qShape[qShape.length - 2];
    const Lk = k.shape[k.shape.length - 2];
    const mask = tf.ones([Lq, Lk]).cumsum(1).sub(1).mul(-Infinity);
    masked = tf.add(scores, mask);
  }
  const attn = tf.softmax(masked, -1);
  return tf.matMul(attn, v);
}

describe('Verification: Scaled Dot Product Attention', () => {
  useVerificationLifecycle();

  it('forward matches TensorFlow reference (no mask, no dropout)', () => {
    const qData = [[[[1.0, 0.5], [0.2, 1.5]], [[0.3, 1.0], [1.2, 0.4]]]];
    const kData = [[[[0.8, 0.1], [0.5, 0.9]], [[0.2, 0.7], [0.6, 0.3]]]];
    const vData = [[[[1.0, 0.0], [0.0, 1.0]], [[0.5, 0.5], [1.0, 0.0]]]];

    const Q = createTensor('Q', qData, true);
    const K = createTensor('K', kData, true);
    const V = createTensor('V', vData, true);
    const Out = scaledDotProductAttention(Q, K, V, 'Out', false, 0);

    expect(Out.shape).toEqual([1, 2, 2, 2]);

    const tfQ = tf.tensor4d(qData, [1, 2, 2, 2]);
    const tfK = tf.tensor4d(kData, [1, 2, 2, 2]);
    const tfV = tf.tensor4d(vData, [1, 2, 2, 2]);
    const ref = referenceScaledDotProductAttention(tfQ, tfK, tfV);
    expectClose(Out.toArray(), ref.arraySync(), 1e-4, 1e-4, '(SDPA forward)');
  });

  it('causal mask forward matches TensorFlow reference', () => {
    const qData = [[[[1.0, 0.5], [0.2, 1.5], [0.3, 0.8]], [[0.3, 1.0], [1.2, 0.4], [0.1, 0.9]]]];
    const kData = [[[[0.8, 0.1], [0.5, 0.9], [0.2, 0.7]], [[0.2, 0.7], [0.6, 0.3], [0.9, 0.5]]]];
    const vData = [[[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], [[0.5, 0.5], [1.0, 0.0], [0.0, 1.0]]]];

    const Q = createTensor('Q', qData, true);
    const K = createTensor('K', kData, true);
    const V = createTensor('V', vData, true);
    const Out = scaledDotProductAttention(Q, K, V, 'Out', true, 0);

    expect(Out.shape).toEqual([1, 2, 3, 2]);

    const tfQ = tf.tensor4d(qData, [1, 2, 3, 2]);
    const tfK = tf.tensor4d(kData, [1, 2, 3, 2]);
    const tfV = tf.tensor4d(vData, [1, 2, 3, 2]);
    const ref = referenceScaledDotProductAttention(tfQ, tfK, tfV, true);
    expectClose(Out.toArray(), ref.arraySync(), 1e-4, 1e-4, '(SDPA causal forward)');
  });

  it('custom scale matches TensorFlow reference', () => {
    const qData = [[[[1.0, 0.5], [0.2, 1.5]], [[0.3, 1.0], [1.2, 0.4]]]];
    const kData = [[[[0.8, 0.1], [0.5, 0.9]], [[0.2, 0.7], [0.6, 0.3]]]];
    const vData = [[[[1.0, 0.0], [0.0, 1.0]], [[0.5, 0.5], [1.0, 0.0]]]];

    const Q = createTensor('Q', qData, true);
    const K = createTensor('K', kData, true);
    const V = createTensor('V', vData, true);
    const Out = scaledDotProductAttention(Q, K, V, 'Out', false, 0, 0.5);

    const tfQ = tf.tensor4d(qData, [1, 2, 2, 2]);
    const tfK = tf.tensor4d(kData, [1, 2, 2, 2]);
    const tfV = tf.tensor4d(vData, [1, 2, 2, 2]);
    const ref = referenceScaledDotProductAttention(tfQ, tfK, tfV, false, 0.5);
    expectClose(Out.toArray(), ref.arraySync(), 1e-4, 1e-4, '(SDPA scale forward)');
  });

  it('backward gradients align with TensorFlow reference', () => {
    const qData = [[[[1.0, 0.5], [0.2, 1.5]], [[0.3, 1.0], [1.2, 0.4]]]];
    const kData = [[[[0.8, 0.1], [0.5, 0.9]], [[0.2, 0.7], [0.6, 0.3]]]];
    const vData = [[[[1.0, 2.0], [0.0, 0.5]], [[0.5, 1.5], [1.0, 0.0]]]];

    const Q = createTensor('Q', qData, true);
    const K = createTensor('K', kData, true);
    const V = createTensor('V', vData, true);
    const Out = scaledDotProductAttention(Q, K, V, 'Out', false, 0);
    backward(Out);

    expect(Q.grad).not.toBeNull();
    expect(K.grad).not.toBeNull();
    expect(V.grad).not.toBeNull();

    const tfQ = tf.variable(tf.tensor4d(qData, [1, 2, 2, 2]));
    const tfK = tf.variable(tf.tensor4d(kData, [1, 2, 2, 2]));
    const tfV = tf.variable(tf.tensor4d(vData, [1, 2, 2, 2]));
    const grads = tf.variableGrads(() => {
      const out = referenceScaledDotProductAttention(tfQ, tfK, tfV);
      return out.sum() as tf.Scalar;
    });
    expectClose(Q.grad.data, grads.grads[tfQ.name].arraySync(), 1e-4, 1e-4, '(SDPA dQ)');
    expectClose(K.grad.data, grads.grads[tfK.name].arraySync(), 1e-4, 1e-4, '(SDPA dK)');
    expectClose(V.grad.data, grads.grads[tfV.name].arraySync(), 1e-4, 1e-4, '(SDPA dV)');
  });

  it('inference mode with dropout_p > 0 is identity', () => {
    setGraphTracking(false);
    const qData = [[[[1.0, 0.5], [0.2, 1.5]], [[0.3, 1.0], [1.2, 0.4]]]];
    const kData = [[[[0.8, 0.1], [0.5, 0.9]], [[0.2, 0.7], [0.6, 0.3]]]];
    const vData = [[[[1.0, 0.0], [0.0, 1.0]], [[0.5, 0.5], [1.0, 0.0]]]];

    const Q = createTensor('Q', qData, true);
    const K = createTensor('K', kData, true);
    const V = createTensor('V', vData, true);
    const Out = scaledDotProductAttention(Q, K, V, 'Out', false, 0.5);

    const tfQ = tf.tensor4d(qData, [1, 2, 2, 2]);
    const tfK = tf.tensor4d(kData, [1, 2, 2, 2]);
    const tfV = tf.tensor4d(vData, [1, 2, 2, 2]);
    const ref = referenceScaledDotProductAttention(tfQ, tfK, tfV);
    expectClose(Out.toArray(), ref.arraySync(), 1e-4, 1e-4, '(SDPA inference dropout identity)');
  });
});

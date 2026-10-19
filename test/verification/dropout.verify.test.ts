import { describe, it, expect, vi, afterEach } from 'vitest';
import {
  createTensor,
  dropout,
  backward,
  mul,
  getTensorByName
} from '../../src/index.js';
import { useVerificationLifecycle, expectClose, flattenArray } from './harness/index.js';

describe('Verification: Dropout training mode', () => {
  useVerificationLifecycle();

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('forward applies the inverted-dropout mask with 1/(1-p) scaling', () => {
    // Deterministic Bernoulli sequence: keep = Math.random() >= p
    const values = [0.6, 0.4, 0.7, 0.2];
    let idx = 0;
    vi.spyOn(Math, 'random').mockImplementation(() => (idx < values.length ? values[idx++] : 0.9));

    const X = createTensor('X', [[1, 2], [3, 4]], true);
    const Y = dropout(X, 0.5, 'Y');

    // mask = [1, 0, 1, 0], scale = 2
    expectClose(Y.toArray(), [[2, 0], [6, 0]], 0, 0);
    expect(Y.shape).toEqual([2, 2]);
    expect(Y.creatorOp).not.toBeNull();
    expect(getTensorByName('Y')).toBe(Y);
  });

  it('backward routes the gradient through the mask and scale', () => {
    const values = [0.6, 0.4, 0.7, 0.2];
    let idx = 0;
    vi.spyOn(Math, 'random').mockImplementation(() => (idx < values.length ? values[idx++] : 0.9));

    const X = createTensor('X', [[1, 2], [3, 4]], true);
    const W = createTensor('W', [[1, 2], [3, 4]], false);
    const Y = dropout(X, 0.5, 'Y');
    const Z = mul(Y, W, 'Z');
    backward(Z);

    // gradY = W = [[1,2],[3,4]]; gradX = gradY * mask * scale = [[2,0],[6,0]]
    expectClose(X.grad.data, [[2, 0], [6, 0]], 0, 0);

    // Eager cleanup: Y is non-leaf and released from the registry
    expect(Y.creatorOp).toBeNull();
    expect(getTensorByName('Y')).toBeUndefined();
  });

  it('p=0 dropout is an exact identity in training mode', () => {
    const X = createTensor('X', [[1.5, -2], [3, 4]], true);
    const Y = dropout(X, 0, 'Y');
    expectClose(Y.toArray(), X.toArray(), 0, 0);

    backward(Y);
    expectClose(X.grad.data, [1, 1, 1, 1], 0, 0);
  });

  it('p=1 dropout zeroes the output and the gradient', () => {
    const X = createTensor('X', [[1, 2], [3, 4]], true);
    const Y = dropout(X, 1, 'Y');
    expectClose(Y.toArray(), [[0, 0], [0, 0]], 0, 0);

    backward(Y);
    expectClose(X.grad.data, [0, 0, 0, 0], 0, 0);
  });

  it('drops approximately fraction p of elements over a large tensor', () => {
    const size = 4000;
    const data = new Array(size).fill(0).map((_, i) => (i % 7) + 1);
    const X = createTensor('X', data, false);
    const Y = dropout(X, 0.5, 'Y');

    const out = flattenArray(Y.toArray());
    const dropped = out.filter((v) => v === 0).length;
    expect(dropped).toBeGreaterThan(size * 0.4);
    expect(dropped).toBeLessThan(size * 0.6);

    // Surviving elements must be scaled by exactly 1/(1-p) = 2
    for (let i = 0; i < size; i++) {
      if (out[i] !== 0) {
        expect(out[i]).toBe(((i % 7) + 1) * 2);
      }
    }
  });
});

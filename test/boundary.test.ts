import { describe, it, expect, beforeEach } from 'vitest';
import {
  createTensor,
  createZerosTensor,
  getTensorByName,
  clearTensorRegistry,
  setGraphTracking,
  matmul,
  add,
  div,
  log,
  softmax,
  crossEntropyLoss,
  unsqueeze,
  squeeze,
  reshape,
  expand,
  concat,
  stack,
  split,
  slice,
  transpose,
  layerNorm,
  loadStateDict,
  backward,
  mseLoss
} from '../src/index.js';

describe('ScratchTensor Boundary Behaviors', () => {
  beforeEach(() => {
    clearTensorRegistry();
    setGraphTracking(true);
  });

  describe('Tensor creation & scalar helpers', () => {
    it('createTensor wraps a scalar number as a 0-D tensor', () => {
      const t = createTensor('s', 5.0);
      expect(t.shape).toEqual([]);
      expect(t.toArray()).toBe(5.0);
    });

    it('item() on scalar tensor returns its value', () => {
      const t = createTensor('s', 7.5);
      expect(t.item()).toBe(7.5);
    });

    it('item() throws on multi-element tensor', () => {
      const t = createTensor('m', [[1, 2], [3, 4]]);
      expect(() => t.item()).toThrow('item() can only be called on a tensor with one element');
    });

    it('createZerosTensor allows empty shape for scalar', () => {
      const t = createZerosTensor('z', []);
      expect(t.shape).toEqual([]);
      expect(t.toArray()).toBe(0);
    });
  });

  describe('MatMul broadcast and unsupported shapes', () => {
    it('broadcasts 2D x 3D like torch.matmul', () => {
      const A = createTensor('A', [[1, 2], [3, 4]]);
      const B = createTensor('B', [[[1, 2], [3, 4]]]);
      const Y = matmul(A, B, 'Y');
      expect(Y.shape).toEqual([1, 2, 2]);
      expect(Y.toArray()).toEqual([[[7, 10], [15, 22]]]);
    });

    it('throws on incompatible inner dimensions for 4D x 4D', () => {
      const A = createTensor('A', [[[[1, 2]]]]);
      const B = createTensor('B', [[[[3, 4]]]]);
      expect(() => matmul(A, B, 'Y')).toThrow('Incompatible matmul shapes');
    });
  });

  describe('Dimension / shape mismatches', () => {
    it('auto-derives a single -1 dimension in reshape', () => {
      const X = createTensor('X', [[1, 2, 3], [4, 5, 6]]);
      const flat = reshape(X, [-1], 'Flat');
      expect(flat.shape).toEqual([6]);
      expect(flat.toArray()).toEqual([1, 2, 3, 4, 5, 6]);

      const mat = reshape(X, [3, -1], 'Mat');
      expect(mat.shape).toEqual([3, 2]);
      expect(mat.toArray()).toEqual([[1, 2], [3, 4], [5, 6]]);
    });

    it('throws on reshape with mismatched element count', () => {
      const X = createTensor('X', [[1, 2], [3, 4]]);
      expect(() => reshape(X, [3], 'Y')).toThrow('size mismatch');
    });

    it('throws on reshape with more than one -1 dimension', () => {
      const X = createTensor('X', [[1, 2], [3, 4]]);
      expect(() => reshape(X, [-1, -1], 'Y')).toThrow('at most one -1');
    });

    it('throws on unsqueeze with out-of-range dim', () => {
      const X = createTensor('X', [[1, 2], [3, 4]]);
      expect(() => unsqueeze(X, 5, 'Y')).toThrow('out of range');
      expect(() => unsqueeze(X, -5, 'Y')).toThrow('out of range');
    });

    it('throws on squeeze of a dimension with size != 1', () => {
      const X = createTensor('X', [[1, 2], [3, 4]]);
      expect(() => squeeze(X, 'Y', 0)).toThrow('not 1');
    });

    it('throws when expand cannot broadcast', () => {
      const X = createTensor('X', [1, 2, 3]);
      expect(() => expand(X, [2, 3, 4], 'Y')).toThrow('Cannot broadcast');
    });

    it('throws on concat with mismatched ranks', () => {
      const A = createTensor('A', [[1, 2]]);
      const B = createTensor('B', [1, 2]);
      expect(() => concat([A, B], 0, 'Y')).toThrow('same rank');
    });

    it('throws on concat with mismatched non-concat dimensions', () => {
      const A = createTensor('A', [[1, 2], [3, 4]]);
      const B = createTensor('B', [[1, 2, 3]]);
      expect(() => concat([A, B], 0, 'Y')).toThrow('shape mismatch');
    });

    it('throws on slice with invalid range', () => {
      const X = createTensor('X', [[1, 2, 3], [4, 5, 6]]);
      expect(() => slice(X, 1, 2, 5, 'Y')).toThrow('Invalid slice range');
      expect(() => slice(X, 1, -1, 2, 'Y')).toThrow('Invalid slice range');
    });

    it('throws on split when dimension is not evenly divisible', () => {
      const X = createTensor('X', [[1, 2, 3], [4, 5, 6]]);
      expect(() => split(X, 1, 2, 'p')).toThrow('Cannot split');
    });

    it('throws on stack with empty input list', () => {
      expect(() => stack([], 0, 'Y')).toThrow('requires at least one ndarray');
    });

    it('throws on transpose with out-of-range dims', () => {
      const X = createTensor('X', [[1, 2], [3, 4]]);
      expect(() => transpose(X, 0, 5, 'Y')).toThrow('out of range');
    });
  });

  describe('Loss and activation shape checks', () => {
    it('crossEntropyLoss throws on 1D logits', () => {
      const Logits = createTensor('Logits', [1, 2, 3], true);
      const Labels = createTensor('Labels', [0], false);
      expect(() => crossEntropyLoss(Logits, Labels, 'Loss')).toThrow('expects 2D logits');
    });

    it('softmax throws on out-of-range dim', () => {
      const X = createTensor('X', [[1, 2], [3, 4]], true);
      expect(() => softmax(X, 5, 'Y')).toThrow('out of range');
    });
  });

  describe('LayerNorm constraints', () => {
    it('layerNorm throws when n exceeds tensor rank', () => {
      const X = createTensor('X', [[1, 2], [3, 4]], true);
      expect(() => layerNorm(X, 3, 'Y')).toThrow('Cannot normalize last');
    });
  });

  describe('Serialization format errors', () => {
    it('loadStateDict throws on missing tensors array', () => {
      expect(() => loadStateDict('{"framework":"ScratchTensor"}')).toThrow('Invalid StateDict format');
    });

    it('loadStateDict throws on invalid JSON', () => {
      expect(() => loadStateDict('not-json')).toThrow();
    });
  });

  describe('Numeric edge cases', () => {
    it('division by zero produces Infinity', () => {
      const A = createTensor('A', [[1, 2]]);
      const B = createTensor('B', [[0, 0.5]]);
      const Y = div(A, B, 'Y');
      expect(Y.toArray()).toEqual([[Infinity, 4]]);
    });

    it('log of zero produces -Infinity', () => {
      const X = createTensor('X', [[1, 0]]);
      const Y = log(X, 'Y');
      expect((Y.toArray() as number[][])[0][1]).toBe(-Infinity);
    });

    it('add with mismatched non-broadcastable shapes throws', () => {
      const A = createTensor('A', [[1, 2, 3]]);
      const B = createTensor('B', [[1, 2]]);
      expect(() => add(A, B, 'Y')).toThrow('Cannot broadcast');
    });

    it('backward on a non-scalar without explicit grad is allowed but uses ones', () => {
      const X = createTensor('X', [[1, 2]], true);
      const Y = add(X, X, 'Y');
      backward(Y);
      expect(Y.grad).toBeNull(); // eager cleanup
      expect(X.grad).not.toBeNull();
    });
  });
});

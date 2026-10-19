import { describe, it, expect, beforeEach } from 'vitest';
import {
  createTensor,
  createZerosTensor,
  createOnesTensor,
  createRandomNormalTensor,
  getTensorByName,
  getAllLeafParameters,
  clearTensorRegistry,
  matmul,
  add,
  sub,
  mul,
  div,
  mseLoss,
  backward,
  clearAllGradients,
  exportStateDict,
  loadStateDict,
  setGraphTracking
} from '../src/index.js';

describe('ScratchTensor Core (Self-Contained NDArray Backend)', () => {
  beforeEach(() => {
    clearTensorRegistry();
    setGraphTracking(true);
  });

  it('creates tensors backed by self-contained NDArray and registers them by name/id', () => {
    const t1 = createTensor('A', [[1, 2], [3, 4]], true);
    expect(t1.name).toBe('A');
    expect(t1.shape).toEqual([2, 2]);
    expect(t1.toArray()).toEqual([[1, 2], [3, 4]]);
    expect(t1.requiresGrad).toBe(true);
    expect(getTensorByName('A')).toBe(t1);

    const zeros = createZerosTensor('Z', [3, 4]);
    expect(zeros.shape).toEqual([3, 4]);
    expect(zeros.toArray()[0]).toEqual([0, 0, 0, 0]);
  });

  it('performs forward computation (MatMul + Add) correctly', () => {
    // X: [1, 2] = [[1, 2]]
    const X = createTensor('X', [[1, 2]]);
    // W: [2, 1] = [[2], [3]]
    const W = createTensor('W', [[2], [3]], true);
    // b: [1] = [0.5]
    const b = createTensor('b', [0.5], true);

    const h = matmul(X, W, 'H'); // 1*2 + 2*3 = 8
    expect(h.shape).toEqual([1, 1]);
    expect(h.toArray()).toEqual([[8]]);

    const y = add(h, b, 'Y');
    expect(y.toArray()).toEqual([[8.5]]);
  });

  it('computes backward gradients and performs eager graph cleanup', () => {
    // X = [[1, 2]], W = [[2], [3]], b = [0.5]
    const X = createTensor('X', [[1, 2]], false);
    const W = createTensor('W', [[2], [3]], true);
    const b = createTensor('b', [0.5], true);
    const target = createTensor('Target', [[10.5]], false);

    const h = matmul(X, W, 'H');
    const pred = add(h, b, 'Pred');
    const loss = mseLoss(pred, target, 'Loss');

    // Before backward: intermediate nodes H, Pred, Loss should have creatorOp defined
    expect(h.creatorOp).not.toBeNull();
    expect(pred.creatorOp).not.toBeNull();
    expect(loss.creatorOp).not.toBeNull();

    backward(loss);

    // W.grad and b.grad should be populated
    expect(W.grad).not.toBeNull();
    expect(b.grad).not.toBeNull();

    // Eager Graph Cleanup check: intermediate non-leaf nodes should have creatorOp reset to null
    expect(h.creatorOp).toBeNull();
    expect(pred.creatorOp).toBeNull();
    expect(loss.creatorOp).toBeNull();

    // Intermediate non-target tensors should be removed from registry,
    // while user-created leaf tensors (including requiresGrad=false inputs/targets) must remain registered
    expect(getTensorByName('H')).toBeUndefined();
    expect(getTensorByName('Pred')).toBeUndefined();
    expect(getTensorByName('Loss')).toBe(loss);
    expect(getTensorByName('X')).toBe(X);
    expect(getTensorByName('Target')).toBe(target);
    expect(getTensorByName('W')).toBe(W);
    expect(getTensorByName('b')).toBe(b);
  });

  it('exports and loads model parameters using Name-First StateDict JSON', () => {
    createTensor('Weight_W', [[1.5], [-2.5]], true);
    createTensor('Bias_b', [0.1], true);

    const jsonStr = exportStateDict();
    const parsed = JSON.parse(jsonStr);
    expect(parsed.framework).toBe('ScratchTensor');
    expect(parsed.tensors.length).toBe(2);

    // Modify tensor data in memory
    const w = getTensorByName('Weight_W')!;
    w.data.set(0, 0, 999);

    // Restore from jsonStr
    loadStateDict(jsonStr);
    expect(getTensorByName('Weight_W')!.toArray()).toEqual([[1.5], [-2.5]]);
  });

  it('throws when loading a checkpoint with mismatched shape', () => {
    createTensor('Weight_W', [[1.5], [-2.5]], true);

    const badStateDict = JSON.stringify({
      framework: 'ScratchTensor',
      tensors: [
        {
          name: 'Weight_W',
          id: 101,
          shape: [2, 2],
          requires_grad: true,
          data: [[1, 0], [0, 1]]
        }
      ]
    });

    expect(() => loadStateDict(badStateDict)).toThrow('Shape mismatch');
  });

  it('broadcasts size-1 dimensions between operands of different ranks', () => {
    // [1,2] + [2] should broadcast [2] to [1,2]
    const a = createTensor('A', [[1, 3]], true);
    const b = createTensor('B', [10, 20], false);
    const c = add(a, b, 'C');
    expect(c.shape).toEqual([1, 2]);
    expect(c.toArray()).toEqual([[11, 23]]);

    // [2] + [1,2] should squeeze [1,2] to [2]
    const d = createTensor('D', [1, 2], true);
    const e = createTensor('E', [[10, 20]], false);
    const f = add(d, e, 'F');
    expect(f.shape).toEqual([2]);
    expect(f.toArray()).toEqual([11, 22]);

    // [2,2] + [2,1] broadcasts [2,1] to [2,2]
    const g = createTensor('G', [[10, 20], [30, 40]], true);
    const h = createTensor('H', [[1], [2]], false);
    const i = add(g, h, 'I');
    expect(i.shape).toEqual([2, 2]);
    expect(i.toArray()).toEqual([[11, 21], [32, 42]]);

    // scalar [1] + [2,2]
    const j = createTensor('J', [5], false);
    const k = createTensor('K', [[1, 2], [3, 4]], true);
    const l = mul(j, k, 'L');
    expect(l.shape).toEqual([2, 2]);
    expect(l.toArray()).toEqual([[5, 10], [15, 20]]);
  });

  it('preserves shape [1,1] instead of flattening to [1]', () => {
    const a = createTensor('A', [[8]], true);
    const b = createTensor('B', [0.5], true);
    const y = add(a, b, 'Y');
    expect(y.shape).toEqual([1, 1]);
    expect(y.toArray()).toEqual([[8.5]]);
  });

  it('createZerosTensor fills the entire tensor with zeros', () => {
    const z = createZerosTensor('Z', [2, 3], true);
    expect(z.shape).toEqual([2, 3]);
    expect(z.toArray()).toEqual([[0, 0, 0], [0, 0, 0]]);
    expect(z.requiresGrad).toBe(true);
    expect(getTensorByName('Z')).toBe(z);
  });

  it('createOnesTensor fills the entire tensor with ones', () => {
    const o = createOnesTensor('O', [3, 1]);
    expect(o.shape).toEqual([3, 1]);
    expect(o.toArray()).toEqual([[1], [1], [1]]);
    expect(o.requiresGrad).toBe(false);
    expect(getAllLeafParameters()).toEqual([]);
  });

  it('createRandomNormalTensor samples ~N(0,1) values', () => {
    const r = createRandomNormalTensor('R', [100, 100], true);
    expect(r.shape).toEqual([100, 100]);

    const flat = (r.toArray() as number[][]).reduce((acc, row) => acc.concat(row), [] as number[]);
    expect(flat.length).toBe(10000);
    for (const v of flat) {
      expect(Number.isFinite(v)).toBe(true);
    }

    const mean = flat.reduce((a, b) => a + b, 0) / flat.length;
    const std = Math.sqrt(flat.reduce((a, b) => a + (b - mean) ** 2, 0) / flat.length);
    expect(Math.abs(mean)).toBeLessThan(0.05);
    expect(std).toBeGreaterThan(0.9);
    expect(std).toBeLessThan(1.1);

    expect(getAllLeafParameters().length).toBe(1);
    expect(getAllLeafParameters()[0]).toBe(r);
  });

  it('re-registering the same name removes the stale tensor from the id registry', () => {
    createTensor('W', [1], true);
    createTensor('W', [2], true);

    expect(getTensorByName('W')!.toArray()).toEqual([2]);

    const leaves = getAllLeafParameters();
    expect(leaves.length).toBe(1);
    expect(leaves[0].toArray()).toEqual([2]);

    const parsed = JSON.parse(exportStateDict());
    expect(parsed.tensors.length).toBe(1);
    expect(parsed.tensors[0].name).toBe('W');
  });
});

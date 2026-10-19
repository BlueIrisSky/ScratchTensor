import { beforeEach, describe, expect, it } from 'vitest';
import {
  add,
  adamStep,
  adamWStep,
  backward,
  clearAllGradients,
  clearTensorRegistry,
  clipGradNorm,
  createTensor,
  getAllLeafParameters,
  getTensorByName,
  mul,
  setGraphTracking,
  setLrScale,
  sgdStep,
  split
} from '../src/index.js';
import { getLrScale } from '../src/optim.js';

describe('Parameter and optimizer safety', () => {
  beforeEach(() => {
    clearTensorRegistry();
    setGraphTracking(true);
  });

  it('clips both gradients from an addition exactly once', () => {
    const a = createTensor('A', [1], true);
    const b = createTensor('B', [2], true);
    backward(add(a, b, 'sum'));

    clipGradNorm(1);
    const aGrad = a.grad.data[0];
    const bGrad = b.grad.data[0];
    expect(aGrad).toBeCloseTo(1 / Math.sqrt(2), 6);
    expect(bGrad).toBeCloseTo(1 / Math.sqrt(2), 6);
    expect(Math.hypot(aGrad, bGrad)).toBeCloseTo(1, 6);

    sgdStep(0.1, 0);
    expect(a.item()).toBeCloseTo(1 - 0.1 / Math.sqrt(2), 6);
    expect(b.item()).toBeCloseTo(2 - 0.1 / Math.sqrt(2), 6);
  });

  it('keeps accumulated gradients independent between parameters', () => {
    const a = createTensor('A', [1], true);
    const b = createTensor('B', [2], true);
    backward(add(a, b, 'sum'));
    backward(mul(a, a, 'square'));
    expect(a.grad.data[0]).toBe(3);
    expect(b.grad.data[0]).toBe(1);

    clipGradNorm(1);
    expect(a.grad.data[0]).toBeCloseTo(3 / Math.sqrt(10), 6);
    expect(b.grad.data[0]).toBeCloseTo(1 / Math.sqrt(10), 6);
  });

  it.each([true, false])('rejects replacing a trainable input when graph tracking is %s', (tracking) => {
    const w = createTensor('W', [2], true);
    setGraphTracking(tracking);
    expect(() => mul(w, w, 'W')).toThrow(/output "W".*trainable leaf/);
    expect(getTensorByName('W')).toBe(w);
    expect(getAllLeafParameters()).toEqual([w]);
    expect(w.item()).toBe(2);
    expect(w.grad).toBeNull();
  });

  it('rejects replacing a trainable parameter that is not an operation input', () => {
    const w = createTensor('W', [2], true);
    const x = createTensor('X', [3]);
    expect(() => add(x, x, 'W')).toThrow(/trainable leaf/);
    expect(getTensorByName('W')).toBe(w);
  });

  it('allows temporary output names to be reused within a differentiable computation', () => {
    const w = createTensor('W', [2], true);
    const first = mul(w, w, 'temporary');
    const second = add(first, w, 'temporary');
    expect(getTensorByName('temporary')).toBe(second);
    expect(second.item()).toBe(6);

    backward(second);
    expect(w.grad.data[0]).toBe(5);
    expect(getTensorByName('temporary')).toBe(second);
  });

  it('allows replacing an ordinary data tensor and explicitly recreating a parameter', () => {
    const x = createTensor('X', [3]);
    expect(add(x, x, 'X').item()).toBe(6);

    createTensor('W', [2], true);
    const replacement = createTensor('W', [7], true);
    expect(getTensorByName('W')).toBe(replacement);
    expect(getAllLeafParameters()).toEqual([replacement]);
  });

  it('checks all split destinations before replacing any of them', () => {
    const x = createTensor('X', [1, 2, 3, 4]);
    const first = createTensor('part_0', [9, 9]);
    const parameter = createTensor('part_1', [8, 8], true);
    expect(() => split(x, 0, 2, 'part')).toThrow(/output "part_1".*trainable leaf/);
    expect(getTensorByName('part_0')).toBe(first);
    expect(getTensorByName('part_1')).toBe(parameter);
    expect(first.toArray()).toEqual([9, 9]);
  });

  it.each([
    ['SGD learning rate', () => sgdStep(NaN)],
    ['SGD negative learning rate', () => sgdStep(-0.1)],
    ['SGD momentum', () => sgdStep(0.1, Infinity)],
    ['SGD negative momentum', () => sgdStep(0.1, -0.1)],
    ['SGD weight decay', () => sgdStep(0.1, 0.9, NaN)],
    ['SGD negative weight decay', () => sgdStep(0.1, 0.9, -0.1)],
    ['Adam learning rate', () => adamStep(Infinity)],
    ['Adam Beta1 equal to one', () => adamStep(0.1, 1)],
    ['Adam Beta1 below zero', () => adamStep(0.1, -0.1)],
    ['Adam Beta1 non-finite', () => adamStep(0.1, NaN)],
    ['Adam Beta2 equal to one', () => adamStep(0.1, 0.9, 1)],
    ['Adam Beta2 below zero', () => adamStep(0.1, 0.9, -0.1)],
    ['Adam Beta2 non-finite', () => adamStep(0.1, 0.9, Infinity)],
    ['Adam weight decay', () => adamStep(0.1, 0.9, 0.999, -1)],
    ['Adam epsilon equal to zero', () => adamStep(0.1, 0.9, 0.999, 0, 0)],
    ['Adam non-finite epsilon', () => adamStep(0.1, 0.9, 0.999, 0, Infinity)],
    ['AdamW Beta1', () => adamWStep(0.1, 1)],
    ['AdamW weight decay', () => adamWStep(0.1, 0.9, 0.999, Infinity)]
  ])('rejects invalid %s before changing any weights or gradients', (_label, step) => {
    const a = createTensor('A', [1], true);
    const b = createTensor('B', [2], true);
    backward(add(a, b, 'sum'));

    expect(step).toThrow();
    expect(a.toArray()).toEqual([1]);
    expect(b.toArray()).toEqual([2]);
    expect(a.grad.data[0]).toBe(1);
    expect(b.grad.data[0]).toBe(1);
  });

  it('does not poison Adam moments or advance its step count after invalid Beta1', () => {
    const w = createTensor('W', [1], true);
    backward(mul(w, w, 'square'));
    adamStep(0.1);
    expect(() => adamStep(0.1, 1)).toThrow(/Beta1/);
    clearAllGradients();
    backward(add(w, createTensor('C', [0]), 'sum'));
    adamStep(0.1);
    const afterRejectedStep = w.item();

    clearTensorRegistry();
    const reference = createTensor('reference', [1], true);
    backward(mul(reference, reference, 'square'));
    adamStep(0.1);
    clearAllGradients();
    backward(add(reference, createTensor('C', [0]), 'sum'));
    adamStep(0.1);
    expect(afterRejectedStep).toBe(reference.item());
  });

  it('rejects an overflowing scaled learning rate before updating parameters', () => {
    const w = createTensor('W', [1], true);
    backward(add(w, w, 'sum'));
    setLrScale(Number.MAX_VALUE);
    expect(() => sgdStep(2)).toThrow(/Scaled learning rate/);
    expect(() => adamStep(2)).toThrow(/Scaled learning rate/);
    expect(w.item()).toBe(1);
  });

  it.each([NaN, Infinity, -Infinity, -1])('rejects invalid learning rate scale %s without changing the previous scale', (scale) => {
    setLrScale(0.5);
    expect(() => setLrScale(scale)).toThrow(/Learning rate scale/);
    expect(getLrScale()).toBe(0.5);
  });

  it.each([NaN, Infinity, -Infinity])('rejects invalid clip norm %s without changing gradients', (maxNorm) => {
    const w = createTensor('W', [2], true);
    backward(mul(w, w, 'square'));
    expect(() => clipGradNorm(maxNorm)).toThrow(/Maximum gradient norm/);
    expect(w.grad.data[0]).toBe(4);
  });

  it('preserves non-positive clip no-ops and allows zero learning rate and scale', () => {
    const w = createTensor('W', [2], true);
    backward(mul(w, w, 'square'));
    clipGradNorm(0);
    clipGradNorm(-1);
    expect(w.grad.data[0]).toBe(4);
    sgdStep(0, 0);
    adamStep(0, 0, 0);
    setLrScale(0);
    sgdStep(1, 0);
    adamWStep(1, 0, 0);
    expect(w.item()).toBe(2);
  });
});

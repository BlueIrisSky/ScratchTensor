import { beforeEach, describe, expect, it } from 'vitest';
import {
  TensorNode, clearTensorRegistry, createTensor, getTensorByName,
  softmax, sum, mean, max, min, transpose, concat, split, slice, unsqueeze, stack
} from '../src/index.js';
import { ScratchTensorExtension } from '../src/scratch-extension.js';

const invalidAxes = [0.5, -0.5, NaN, Infinity, -Infinity, Number.MAX_SAFE_INTEGER + 1];
const axisCalls: [string, (x: TensorNode, dim: number) => unknown][] = [
  ['softmax', (x, dim) => softmax(x, dim, 'bad')],
  ['sum', (x, dim) => sum(x, dim, 'bad')],
  ['mean', (x, dim) => mean(x, dim, 'bad')],
  ['max', (x, dim) => max(x, dim, 'bad')],
  ['min', (x, dim) => min(x, dim, 'bad')],
  ['transpose first axis', (x, dim) => transpose(x, dim, 0, 'bad')],
  ['transpose second axis', (x, dim) => transpose(x, 0, dim, 'bad')],
  ['concat', (x, dim) => concat([x], dim, 'bad')],
  ['split', (x, dim) => split(x, dim, 2, 'bad')],
  ['slice axis', (x, dim) => slice(x, dim, 0, 1, 'bad')],
  ['unsqueeze', (x, dim) => unsqueeze(x, dim, 'bad')],
  ['stack', (x, dim) => stack([x], dim, 'bad')]
];

describe('axis and slice index validation', () => {
  beforeEach(() => clearTensorRegistry());

  it.each(axisCalls)('%s rejects fractional, nonfinite and unsafe axes before registering output', (_, operation) => {
    const x = createTensor('X', [[1, 2], [3, 4]], true);
    const previous = createTensor('bad', [99]);
    const previousSplit = createTensor('bad_0', [98]);
    for (const dim of invalidAxes) {
      expect(() => operation(x, dim)).toThrow(/finite safe integer/);
      expect(getTensorByName('bad')).toBe(previous);
      expect(previous.toArray()).toEqual([99]);
      expect(getTensorByName('bad_0')).toBe(previousSplit);
      expect(getTensorByName('bad_1')).toBeUndefined();
    }
    expect(x.toArray()).toEqual([[1, 2], [3, 4]]);
    expect(x.grad).toBeNull();
  });

  it('rejects invalid slice starts and ends without truncating or replacing output', () => {
    const x = createTensor('X', [[1, 2], [3, 4]]);
    const previous = createTensor('bad', [99]);
    for (const value of invalidAxes) {
      expect(() => slice(x, 0, value, 2, 'bad')).toThrow(/finite safe integers/);
      expect(() => slice(x, 0, 0, value, 'bad')).toThrow(/finite safe integers/);
      expect(getTensorByName('bad')).toBe(previous);
    }
    expect(() => slice(x, 0, 1, 1, 'bad')).toThrow(/Invalid slice range/);
  });

  it('keeps negative axes, insertion endpoints and integer slice ranges working', () => {
    const x = createTensor('X', [[1, 2], [3, 4]]);
    expect(unsqueeze(x, -3, 'front').shape).toEqual([1, 2, 2]);
    expect(unsqueeze(x, -1, 'back').shape).toEqual([2, 2, 1]);
    expect(stack([x], -3, 'stack_front').shape).toEqual([1, 2, 2]);
    expect(stack([x], 2, 'stack_back').shape).toEqual([2, 2, 1]);
    expect(transpose(x, -1, -2, 'transposed').toArray()).toEqual([[1, 3], [2, 4]]);
    expect(slice(x, -1, 0, 1, 'first_column').toArray()).toEqual([[1], [3]]);
    expect(sum(x, -1, 'row_sums').toArray()).toEqual([3, 7]);
    for (const row of softmax(x, -1, 'probabilities').toArray()) {
      expect(row[0] + row[1]).toBeCloseTo(1);
    }
  });

  it('reports malformed numeric block inputs instead of publishing NaN or truncated results', () => {
    createTensor('X', [[1, 2], [3, 4]]);
    const extension = new ScratchTensorExtension();
    expect(() => extension.softmaxBlock({ X: 'X', DIM: 'NaN', OUT: 'bad' }))
      .toThrow(/ScratchTensor\.softmaxBlock:.*finite safe integer/);
    expect(() => extension.unsqueezeBlock({ X: 'X', DIM: '0.5', OUT: 'bad' }))
      .toThrow(/ScratchTensor\.unsqueezeBlock:.*finite safe integer/);
    expect(() => extension.stackBlock({ NAMES: '["X"]', DIM: '0.5', OUT: 'bad' }))
      .toThrow(/ScratchTensor\.stackBlock:.*finite safe integer/);
    expect(() => extension.sliceBlock({ X: 'X', DIM: 0, START: '0.5', END: '1.5', OUT: 'bad' }))
      .toThrow(/ScratchTensor\.sliceBlock:.*finite safe integers/);
    expect(getTensorByName('bad')).toBeUndefined();
  });
});

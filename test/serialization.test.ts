import { beforeEach, describe, expect, it } from 'vitest';
import {
  clearTensorRegistry,
  createTensor,
  createZerosTensor,
  getAllLeafParameters,
  getTensorById,
  getTensorByName,
  resetIdCounter
} from '../src/tensor.js';
import { fromJsArray } from '../src/ndarray-utils.js';
import { getLrScale, setLrScale, sgdStep } from '../src/optim.js';
import { exportStateDict, loadStateDict } from '../src/serialization.js';

function record(overrides: Record<string, unknown> = {}) {
  return { name: 'W', id: 900, shape: [2], requires_grad: true, data: [3, 4], ...overrides };
}

function checkpoint(tensors: unknown[], metadata: Record<string, unknown> = {}) {
  return JSON.stringify({ framework: 'ScratchTensor', tensors, ...metadata });
}

describe('StateDict serialization', () => {
  beforeEach(() => {
    clearTensorRegistry();
    resetIdCounter();
  });

  it('round-trips scalar, nested, and zero-sized leaf parameters with fresh IDs', () => {
    const originals = [
      createTensor('scalar', 2.5, true),
      createTensor('matrix', [[1, 2], [3, 4]], true),
      createZerosTensor('empty', [0, 3], true),
      createZerosTensor('nestedEmpty', [2, 0, 3], true)
    ];
    createTensor('input', [10], false);
    const intermediate = createTensor('intermediate', [20], true);
    intermediate.isLeaf = false;
    const json = exportStateDict();
    const parsed = JSON.parse(json);
    expect(parsed.framework).toBe('ScratchTensor');
    expect(parsed.tensors.map((item: { name: string }) => item.name)).toEqual(originals.map(t => t.name));

    clearTensorRegistry();
    loadStateDict(json);
    for (const original of originals) {
      const restored = getTensorByName(original.name)!;
      expect(restored).not.toBe(original);
      expect(restored.id).not.toBe(original.id);
      expect(getTensorById(restored.id)).toBe(restored);
      expect(restored.shape).toEqual(original.shape);
      expect(restored.toArray()).toEqual(original.toArray());
      expect(restored.requiresGrad).toBe(true);
      expect(restored.grad).toBeNull();
    }
    expect(getTensorByName('input')).toBeUndefined();
    expect(getTensorByName('intermediate')).toBeUndefined();
  });

  it('updates an existing object by name and preserves its ID and requiresGrad flag', () => {
    const parameter = createTensor('W', [1, 2], false);
    const id = parameter.id;
    parameter.grad = fromJsArray([7, 8]).ndarray;
    const untouched = createTensor('untouched', [6], true);
    const untouchedGrad = untouched.grad = fromJsArray([9]).ndarray;

    loadStateDict(checkpoint([record()]));

    expect(getTensorByName('W')).toBe(parameter);
    expect(getTensorById(id)).toBe(parameter);
    expect(parameter.id).toBe(id);
    expect(parameter.requiresGrad).toBe(false);
    expect(parameter.toArray()).toEqual([3, 4]);
    expect(parameter.grad).toBeNull();
    expect(getTensorById(900)).toBeUndefined();
    expect(untouched.toArray()).toEqual([6]);
    expect(untouched.grad).toBe(untouchedGrad);
  });

  it('keeps existing momentum state and learning-rate scale when loading weights', () => {
    const parameter = createTensor('W', [1], true);
    parameter.grad = fromJsArray([2]).ndarray;
    sgdStep(0.1, 0.5);
    setLrScale(0.5);

    loadStateDict(checkpoint([record({ shape: [1], data: [10], requires_grad: false })]));

    expect(parameter.requiresGrad).toBe(true);
    expect(getLrScale()).toBe(0.5);
    expect(parameter.grad).toBeNull();
    parameter.grad = fromJsArray([0]).ndarray;
    sgdStep(0.1, 0.5);
    expect(parameter.toArray()[0]).toBeCloseTo(9.95, 5);
  });

  it('allocates a fresh ID even when checkpoint metadata names another live tensor ID', () => {
    const other = createTensor('other', [7], true);
    loadStateDict(checkpoint([record({ id: other.id, requires_grad: false })]));

    const loaded = getTensorByName('W')!;
    expect(loaded.id).not.toBe(other.id);
    expect(loaded.requiresGrad).toBe(false);
    expect(loaded.toArray()).toEqual([3, 4]);
    expect(getTensorById(other.id)).toBe(other);
    expect(getTensorByName('other')).toBe(other);
    expect(other.toArray()).toEqual([7]);
  });

  it('accepts omitted framework and ID metadata and ignores unknown fields', () => {
    loadStateDict(JSON.stringify({
      comment: 'handwritten checkpoint',
      tensors: [record({ id: undefined, name: ' W ', extra: { ignored: true } })]
    }));
    expect(getTensorByName(' W ')!.toArray()).toEqual([3, 4]);
    expect(getTensorByName('W')).toBeUndefined();
  });

  it('allows an empty tensors array without changing registered parameters', () => {
    const parameter = createTensor('W', [1], true);
    loadStateDict(checkpoint([]));
    expect(getAllLeafParameters()).toEqual([parameter]);
    expect(parameter.toArray()).toEqual([1]);
  });

  it.each([
    { shape: [2], data: [1] },
    { shape: [2], data: [[1], [2]] },
    { shape: [], data: [1] },
    { shape: [1], data: 1 },
    { shape: [2, 2], data: [[1, 2], [3]] },
    { shape: [2, 2], data: [[1, 2], [3, 4, 5]] },
    { shape: [2, 1], data: [[1], 2] },
    { shape: [0], data: [1] }
  ])('rejects data that disagrees with its declared shape: %j', ({ shape, data }) => {
    expect(() => loadStateDict(checkpoint([record({ shape, data })]))).toThrow(/Shape mismatch.*"W".*data/);
    expect(getTensorByName('W')).toBeUndefined();
  });

  it('does not accept malformed data merely because the declared shape matches an existing tensor', () => {
    const parameter = createTensor('W', [1, 2], true);
    const data = parameter.data;
    const grad = parameter.grad = fromJsArray([5, 6]).ndarray;
    expect(() => loadStateDict(checkpoint([record({ data: [99] })]))).toThrow('Shape mismatch');
    expect(parameter.data).toBe(data);
    expect(parameter.grad).toBe(grad);
    expect(parameter.shape).toEqual([2]);
  });

  it.each([
    record({ name: 'bad', data: [3] }),
    record({ name: 'bad', data: [3, null] }),
    record({ name: 'bad', requires_grad: 'true' }),
    record({ name: 'bad', shape: [1], data: [3] }),
    record({ name: 'W' }),
    record({ name: 'new' }),
    null
  ])('does not update, clear gradients, create tensors, or consume IDs before a later failure: %j', bad => {
    const parameter = createTensor('W', [1, 2], true);
    const data = parameter.data;
    const grad = parameter.grad = fromJsArray([5, 6]).ndarray;
    const existingBad = createTensor('bad', [7, 8], true);
    const before = getAllLeafParameters();

    expect(() => loadStateDict(checkpoint([
      record(),
      record({ name: 'new' }),
      bad
    ]))).toThrow();

    expect(parameter.data).toBe(data);
    expect(parameter.grad).toBe(grad);
    expect(parameter.toArray()).toEqual([1, 2]);
    expect(existingBad.toArray()).toEqual([7, 8]);
    expect(getTensorByName('new')).toBeUndefined();
    expect(getAllLeafParameters()).toEqual(before);
    expect(createTensor('after', 0).id).toBe(existingBad.id + 1);
  });

  it.each([null, true, false, '3', {}, []])('refuses nonnumeric scalar data without coercion: %j', data => {
    expect(() => loadStateDict(checkpoint([record({ shape: [], data })]))).toThrow(/"W".*data/);
    expect(getTensorByName('W')).toBeUndefined();
  });

  it.each([null, true, '3', {}])('reports the exact location of a nonnumeric nested value: %j', value => {
    expect(() => loadStateDict(checkpoint([
      record({ shape: [1, 2], data: [[1, value]] })
    ]))).toThrow(/"W".*data\[0\]\[1\].*finite number/);
  });

  it.each([NaN, Infinity, -Infinity])('refuses to export %s rather than serializing it as null', value => {
    createTensor('W', [[1, value]], true);
    expect(() => exportStateDict()).toThrow(/"W".*data\[0\]\[1\].*finite number/);
  });

  it.each(['1e400', '-1e400'])('refuses nonfinite numbers parsed from valid JSON: %s', value => {
    const json = checkpoint([record({ data: [1, 'placeholder'] })]).replace('"placeholder"', value);
    expect(() => loadStateDict(json)).toThrow(/"W".*data\[1\].*finite number/);
    expect(getTensorByName('W')).toBeUndefined();
  });

  it('rejects finite JSON numbers that overflow float32 storage', () => {
    expect(() => loadStateDict(checkpoint([record({ data: [1, 1e39] })]))).toThrow(/"W".*data\[1\].*float32/);
    expect(getTensorByName('W')).toBeUndefined();
  });

  it.each([
    { name: undefined }, { name: '' }, { name: null }, { name: 42 },
    { id: null }, { id: '101' }, { id: 1.5 }, { id: Number.MAX_SAFE_INTEGER + 1 },
    { shape: undefined }, { shape: null }, { shape: '2' },
    { shape: [-1] }, { shape: [1.5] }, { shape: ['2'] }, { shape: [null] },
    { shape: [Number.MAX_SAFE_INTEGER + 1] },
    { requires_grad: undefined }, { requires_grad: null }, { requires_grad: 1 },
    { requires_grad: 'true' }, { data: undefined }
  ])('rejects malformed record fields: %j', overrides => {
    expect(() => loadStateDict(checkpoint([record(overrides)]))).toThrow(/Invalid StateDict|Shape mismatch/);
    expect(getAllLeafParameters()).toEqual([]);
  });

  it.each([null, [], 'tensor', 42])('rejects a non-object tensor record: %j', item => {
    expect(() => loadStateDict(checkpoint([item]))).toThrow(/tensors\[0\].*tensor object/);
  });

  it.each([null, [], 42, 'text', {}, { tensors: {} }])('rejects malformed top-level documents: %j', input => {
    expect(() => loadStateDict(JSON.stringify(input))).toThrow('Invalid StateDict format');
  });

  it.each([null, 42, 'OtherFramework'])('rejects an explicitly unsupported framework: %j', framework => {
    expect(() => loadStateDict(checkpoint([record()], { framework }))).toThrow('Invalid StateDict framework');
    expect(getTensorByName('W')).toBeUndefined();
  });

  it('rejects non-text input and invalid JSON', () => {
    expect(() => loadStateDict(null as unknown as string)).toThrow('expected JSON text');
    expect(() => loadStateDict('{')).toThrow();
  });
});

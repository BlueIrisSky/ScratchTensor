import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { ScratchTensorExtension } from '../src/scratch-extension.js';
import {
  getTensorByName,
  getAllLeafParameters,
  clearTensorRegistry,
  resetIdCounter,
  setGraphTracking,
  resetOptimizerState,
  setLrScale
} from '../src/index.js';
import { expectClose } from './verification/harness/index.js';

const ext = new ScratchTensorExtension();

describe('ScratchTensorExtension block layer', () => {
  beforeEach(() => {
    clearTensorRegistry();
    resetIdCounter();
    setGraphTracking(true);
    resetOptimizerState();
    setLrScale(1);
  });

  afterEach(() => {
    clearTensorRegistry();
  });

  it('getInfo exposes a well-formed Scratch 3 extension descriptor', () => {
    const info = ext.getInfo();
    expect(info.id).toBe('ScratchTensor');
    expect(info.name).toBe('ScratchTensor');
    expect(info.blocks.length).toBe(41);

    const opcodes = info.blocks.map((b: any) => b.opcode);
    expect(new Set(opcodes).size).toBe(opcodes.length);
    for (const expected of [
      'createTensorBlock',
      'binaryOpBlock',
      'activationBlock',
      'softmaxBlock',
      'scaledDotProductAttentionBlock',
      'backwardBlock',
      'reductionBlock',
      'layerNormBlock',
      'dropoutBlock',
      'sgdFullBlock',
      'adamWBlock',
      'exportStateDictBlock'
    ]) {
      expect(opcodes).toContain(expected);
    }

    for (const block of info.blocks) {
      expect(['command', 'reporter']).toContain(block.blockType);
      expect(typeof block.text).toBe('string');
    }
    const reporters = info.blocks.filter((b: any) => b.blockType === 'reporter').map((b: any) => b.opcode);
    expect(reporters).toEqual(['getTensorPropertyBlock', 'exportStateDictBlock']);

    expect(info.menus.opMenu.items).toEqual(['+', '-', '*', '/', '@']);
    expect(info.menus.activationMenu.items).toEqual(['ReLU', 'GELU', 'Sigmoid', 'Tanh']);
    expect(info.menus.unaryMathMenu.items).toEqual(['Exp', 'Log', 'Sqrt', 'Abs']);
    expect(info.menus.tensorPropertyMenu.items).toEqual(['数值', '标量值', 'Shape', '梯度']);
    expect(info.menus.reductionMenu.items).toEqual(['求和(Sum)', '求平均值(Mean)', '求最大值(Max)', '求最小值(Min)']);
  });

  it('createTensorBlock parses JSON data and registers the tensor', () => {
    ext.createTensorBlock({ NAME: 'A', DATA: '[[1, 2], [3, 4]]', REQ_GRAD: true });
    const t = getTensorByName('A')!;
    expect(t.shape).toEqual([2, 2]);
    expect(t.toArray()).toEqual([[1, 2], [3, 4]]);
    expect(t.requiresGrad).toBe(true);
  });

  it('createTensorBlock throws on invalid JSON data', () => {
    expect(() => ext.createTensorBlock({ NAME: 'B', DATA: 'oops', REQ_GRAD: false })).toThrow();
    expect(getTensorByName('B')).toBeUndefined();
  });

  it('createZerosTensorBlock / createOnesTensorBlock parse shape strings', () => {
    ext.createZerosTensorBlock({ SHAPE: '[2, 3]', NAME: 'Z', REQ_GRAD: false });
    expect(getTensorByName('Z')!.toArray()).toEqual([[0, 0, 0], [0, 0, 0]]);

    ext.createOnesTensorBlock({ SHAPE: '[1, 2]', NAME: 'O', REQ_GRAD: true });
    expect(getTensorByName('O')!.toArray()).toEqual([[1, 1]]);
    expect(getTensorByName('O')!.requiresGrad).toBe(true);
  });

  it('createRandomNormalTensorBlock produces the requested shape with finite values', () => {
    ext.createRandomNormalTensorBlock({ SHAPE: '[3, 4]', NAME: 'R', REQ_GRAD: false });
    const r = getTensorByName('R')!;
    expect(r.shape).toEqual([3, 4]);
    const flat = (r.toArray() as number[][]).reduce((acc, row) => acc.concat(row), [] as number[]);
    expect(flat.length).toBe(12);
    for (const v of flat) {
      expect(Number.isFinite(v)).toBe(true);
    }
  });

  it('getTensorPropertyBlock reports 数值 / Shape / 梯度 / 标量值', () => {
    ext.createTensorBlock({ NAME: 'A', DATA: '[[1, 2], [3, 4]]', REQ_GRAD: false });
    expect(ext.getTensorPropertyBlock({ NAME: 'A', PROPERTY: '数值' })).toBe('[[1,2],[3,4]]');
    expect(ext.getTensorPropertyBlock({ NAME: 'A', PROPERTY: 'Shape' })).toBe('[2,2]');
    expect(ext.getTensorPropertyBlock({ NAME: 'A', PROPERTY: '梯度' })).toBe('null');
    expect(() => ext.getTensorPropertyBlock({ NAME: 'missing', PROPERTY: '数值' }))
      .toThrow(/ScratchTensor\.getTensorPropertyBlock:.*missing.*does not exist/);
    expect(() => ext.getTensorPropertyBlock({ NAME: 'A', PROPERTY: '未知' }))
      .toThrow(/ScratchTensor\.getTensorPropertyBlock:.*Unknown tensor property/);
  });

  it('getTensorPropertyBlock 标量值 returns a number and throws for multiple elements', () => {
    ext.createTensorBlock({ NAME: 'S', DATA: '7.5', REQ_GRAD: false });
    expect(ext.getTensorPropertyBlock({ NAME: 'S', PROPERTY: '标量值' })).toBe(7.5);

    ext.createTensorBlock({ NAME: 'M', DATA: '[[1, 2]]', REQ_GRAD: false });
    expect(() => ext.getTensorPropertyBlock({ NAME: 'M', PROPERTY: '标量值' }))
      .toThrow(/ScratchTensor\.getTensorPropertyBlock:.*item\(\)/);
  });

  it('getTensorPropertyBlock 梯度 returns serialized grad after backward', () => {
    ext.createTensorBlock({ NAME: 'W', DATA: '[[2]]', REQ_GRAD: true });
    ext.createTensorBlock({ NAME: 'X', DATA: '[[3]]', REQ_GRAD: false });
    ext.binaryOpBlock({ A: 'W', B: 'X', OP: '*', OUT: 'Y' });
    ext.backwardBlock({ TENSOR: 'Y' });
    expect(ext.getTensorPropertyBlock({ NAME: 'W', PROPERTY: '梯度' })).toBe('[[3]]');
  });

  it('deleteTensorBlock and deleteAllTensorsBlock remove tensors from the registry', () => {
    ext.createTensorBlock({ NAME: 'A', DATA: '[1]', REQ_GRAD: false });
    ext.createTensorBlock({ NAME: 'B', DATA: '[2]', REQ_GRAD: true });
    ext.deleteTensorBlock({ NAME: 'A' });
    expect(getTensorByName('A')).toBeUndefined();
    expect(getTensorByName('B')).toBeDefined();
    ext.deleteAllTensorsBlock();
    expect(getTensorByName('B')).toBeUndefined();
    expect(getAllLeafParameters()).toEqual([]);
  });

  it('setGraphTrackingBlock toggles graph construction', () => {
    ext.createTensorBlock({ NAME: 'A', DATA: '[[1, 2]]', REQ_GRAD: true });
    ext.setGraphTrackingBlock({ ENABLED: false });
    ext.binaryOpBlock({ A: 'A', B: 'A', OP: '+', OUT: 'C' });
    expect(getTensorByName('C')!.creatorOp).toBeNull();
    ext.setGraphTrackingBlock({ ENABLED: true });
    ext.binaryOpBlock({ A: 'A', B: 'A', OP: '+', OUT: 'D' });
    expect(getTensorByName('D')!.creatorOp).not.toBeNull();
  });

  it('binaryOpBlock dispatches + - * / @ and rejects invalid input', () => {
    ext.createTensorBlock({ NAME: 'A', DATA: '[[1, 2], [3, 4]]', REQ_GRAD: false });
    ext.createTensorBlock({ NAME: 'B', DATA: '[[10, 20], [30, 40]]', REQ_GRAD: false });

    ext.binaryOpBlock({ A: 'A', B: 'B', OP: '+', OUT: 'Add' });
    expect(getTensorByName('Add')!.toArray()).toEqual([[11, 22], [33, 44]]);

    ext.binaryOpBlock({ A: 'A', B: 'B', OP: '-', OUT: 'Sub' });
    expect(getTensorByName('Sub')!.toArray()).toEqual([[-9, -18], [-27, -36]]);

    ext.binaryOpBlock({ A: 'A', B: 'B', OP: '*', OUT: 'Mul' });
    expect(getTensorByName('Mul')!.toArray()).toEqual([[10, 40], [90, 160]]);

    ext.binaryOpBlock({ A: 'B', B: 'A', OP: '/', OUT: 'Div' });
    expect(getTensorByName('Div')!.toArray()).toEqual([[10, 10], [10, 10]]);

    ext.binaryOpBlock({ A: 'A', B: 'B', OP: '@', OUT: 'Mat' });
    expect(getTensorByName('Mat')!.toArray()).toEqual([[70, 100], [150, 220]]);

    expect(() => ext.binaryOpBlock({ A: 'A', B: 'B', OP: '%', OUT: 'Noop' })).toThrow(/ScratchTensor\.binaryOpBlock:.*Unknown operation/);
    expect(getTensorByName('Noop')).toBeUndefined();
    expect(() => ext.binaryOpBlock({ A: 'missing', B: 'B', OP: '+', OUT: 'Noop2' })).toThrow(/ScratchTensor\.binaryOpBlock:.*missing/);
    expect(getTensorByName('Noop2')).toBeUndefined();
  });

  it('activationBlock dispatches ReLU / GELU / Sigmoid / Tanh', () => {
    ext.createTensorBlock({ NAME: 'X', DATA: '[[-1, 0], [1, 2]]', REQ_GRAD: false });

    ext.activationBlock({ X: 'X', ACT: 'ReLU', OUT: 'R' });
    expect(getTensorByName('R')!.toArray()).toEqual([[0, 0], [1, 2]]);

    ext.activationBlock({ X: 'X', ACT: 'Sigmoid', OUT: 'S' });
    expectClose(
      getTensorByName('S')!.toArray(),
      [[1 / (1 + Math.E), 0.5], [1 / (1 + 1 / Math.E), 1 / (1 + 1 / (Math.E * Math.E))]],
      1e-4,
      1e-4
    );

    ext.activationBlock({ X: 'X', ACT: 'Tanh', OUT: 'T' });
    expectClose(
      getTensorByName('T')!.toArray(),
      [[-0.76159416, 0], [0.76159416, 0.96402758]],
      1e-4,
      1e-4
    );

    ext.activationBlock({ X: 'X', ACT: 'GELU', OUT: 'G' });
    const g = getTensorByName('G')!.toArray();
    expect(g[0][1]).toBe(0);
    expect(g[0][0]).toBeLessThan(0);
    expectClose(g[1][1], 1.9547, 1e-3, 1e-3);

    expect(() => ext.activationBlock({ X: 'missing', ACT: 'ReLU', OUT: 'Noop' })).toThrow(/ScratchTensor\.activationBlock:.*missing/);
    expect(getTensorByName('Noop')).toBeUndefined();
    expect(() => ext.activationBlock({ X: 'X', ACT: 'Unknown', OUT: 'Noop2' })).toThrow(/ScratchTensor\.activationBlock:.*Unknown activation/);
    expect(getTensorByName('Noop2')).toBeUndefined();
  });

  it('softmaxBlock applies softmax along the given dim', () => {
    ext.createTensorBlock({ NAME: 'X', DATA: '[[1, 2]]', REQ_GRAD: false });
    ext.softmaxBlock({ X: 'X', DIM: -1, OUT: 'S' });
    const e1 = Math.exp(1);
    const e2 = Math.exp(2);
    expectClose(getTensorByName('S')!.toArray(), [[e1 / (e1 + e2), e2 / (e1 + e2)]], 1e-4, 1e-4);
  });

  it('loss blocks compute scalar losses', () => {
    ext.createTensorBlock({ NAME: 'Pred', DATA: '[[1], [2]]', REQ_GRAD: true });
    ext.createTensorBlock({ NAME: 'Target', DATA: '[[3], [4]]', REQ_GRAD: false });

    ext.mseLossBlock({ PRED: 'Pred', TARGET: 'Target', OUT: 'L1' });
    expectClose(getTensorByName('L1')!.item(), 4, 1e-4, 1e-4);

    ext.l1LossBlock({ PRED: 'Pred', TARGET: 'Target', OUT: 'L2' });
    expectClose(getTensorByName('L2')!.item(), 2, 1e-4, 1e-4);

    ext.createTensorBlock({ NAME: 'P2', DATA: '[[0.5]]', REQ_GRAD: false });
    ext.createTensorBlock({ NAME: 'T2', DATA: '[[1]]', REQ_GRAD: false });
    ext.bceLossBlock({ PRED: 'P2', TARGET: 'T2', OUT: 'L3' });
    expectClose(getTensorByName('L3')!.item(), -Math.log(0.5), 1e-4, 1e-4);

    ext.createTensorBlock({ NAME: 'Logits', DATA: '[[1, 2], [2, 1]]', REQ_GRAD: false });
    ext.createTensorBlock({ NAME: 'Labels', DATA: '[0, 1]', REQ_GRAD: false });
    ext.crossEntropyLossBlock({ LOGITS: 'Logits', LABELS: 'Labels', OUT: 'L4' });
    const expectedCE = -Math.log(Math.exp(1) / (Math.exp(1) + Math.exp(2)));
    expectClose(getTensorByName('L4')!.item(), expectedCE, 1e-4, 1e-4);

    expect(() => ext.mseLossBlock({ PRED: 'missing', TARGET: 'Target', OUT: 'Noop' })).toThrow(/ScratchTensor\.mseLossBlock:.*missing/);
    expect(getTensorByName('Noop')).toBeUndefined();
  });

  it('shape blocks: reshape / unsqueeze / squeeze / expand', () => {
    ext.createTensorBlock({ NAME: 'A', DATA: '[[1, 2], [3, 4]]', REQ_GRAD: false });
    ext.reshapeBlock({ X: 'A', SHAPE: '[4]', OUT: 'B' });
    expect(getTensorByName('B')!.toArray()).toEqual([1, 2, 3, 4]);

    ext.unsqueezeBlock({ X: 'A', DIM: 0, OUT: 'U' });
    expect(getTensorByName('U')!.shape).toEqual([1, 2, 2]);

    ext.createTensorBlock({ NAME: 'C', DATA: '[[1, 2]]', REQ_GRAD: false });
    ext.squeezeBlock({ X: 'C', OUT: 'S' });
    expect(getTensorByName('S')!.shape).toEqual([2]);

    ext.expandBlock({ X: 'C', SHAPE: '[3, 2]', OUT: 'E' });
    expect(getTensorByName('E')!.toArray()).toEqual([[1, 2], [1, 2], [1, 2]]);

    expect(() => ext.reshapeBlock({ X: 'A', SHAPE: 'bad-json', OUT: 'Noop' })).toThrow(/ScratchTensor\.reshapeBlock:.*/);
    expect(getTensorByName('Noop')).toBeUndefined();
  });

  it('concatBlock and stackBlock join lists of named tensors', () => {
    ext.createTensorBlock({ NAME: 'A', DATA: '[[1, 2]]', REQ_GRAD: false });
    ext.createTensorBlock({ NAME: 'B', DATA: '[[3, 4]]', REQ_GRAD: false });

    ext.concatBlock({ NAMES: '["A", "B"]', DIM: 0, OUT: 'C' });
    expect(getTensorByName('C')!.toArray()).toEqual([[1, 2], [3, 4]]);

    ext.stackBlock({ NAMES: '["A", "B"]', DIM: 0, OUT: 'S' });
    expect(getTensorByName('S')!.shape).toEqual([2, 1, 2]);

    expect(() => ext.concatBlock({ NAMES: 'not-json', DIM: 0, OUT: 'Noop' })).toThrow(/ScratchTensor\.concatBlock:.*/);
    expect(getTensorByName('Noop')).toBeUndefined();
    ext.stackBlock({ NAMES: '["A"]', DIM: 0, OUT: 'SingleStack' });
    expect(getTensorByName('SingleStack')!.toArray()).toEqual([[[1, 2]]]);
    ext.concatBlock({ NAMES: '["A"]', DIM: 0, OUT: 'SingleConcat' });
    expect(getTensorByName('SingleConcat')!.toArray()).toEqual([[1, 2]]);
  });

  it('splitBlock registers parts under prefix names', () => {
    ext.createTensorBlock({ NAME: 'A', DATA: '[[1, 2], [3, 4]]', REQ_GRAD: false });
    ext.splitBlock({ X: 'A', DIM: 0, N: 2, PREFIX: 'part' });
    expect(getTensorByName('part_0')!.toArray()).toEqual([[1, 2]]);
    expect(getTensorByName('part_1')!.toArray()).toEqual([[3, 4]]);
  });

  it('sliceBlock / whereBlock / clampBlock', () => {
    ext.createTensorBlock({ NAME: 'A', DATA: '[[1, 2, 3], [4, 5, 6]]', REQ_GRAD: false });
    ext.sliceBlock({ X: 'A', DIM: 1, START: 0, END: 2, OUT: 'S' });
    expect(getTensorByName('S')!.toArray()).toEqual([[1, 2], [4, 5]]);

    ext.createTensorBlock({ NAME: 'Cond', DATA: '[[1, 0], [0, 1]]', REQ_GRAD: false });
    ext.createTensorBlock({ NAME: 'Xt', DATA: '[[1, 2], [3, 4]]', REQ_GRAD: false });
    ext.createTensorBlock({ NAME: 'Yt', DATA: '[[10, 20], [30, 40]]', REQ_GRAD: false });
    ext.whereBlock({ COND: 'Cond', A: 'Xt', B: 'Yt', OUT: 'W' });
    expect(getTensorByName('W')!.toArray()).toEqual([[1, 20], [30, 4]]);

    ext.createTensorBlock({ NAME: 'X2', DATA: '[[-5, 0], [3, 9]]', REQ_GRAD: false });
    ext.clampBlock({ X: 'X2', MIN: -1, MAX: 1, OUT: 'Cl' });
    expect(getTensorByName('Cl')!.toArray()).toEqual([[-1, 0], [1, 1]]);
  });

  it('powBlock and unaryMathBlock dispatch element-wise math', () => {
    ext.createTensorBlock({ NAME: 'A', DATA: '[[1, 2], [3, 4]]', REQ_GRAD: false });
    ext.powBlock({ X: 'A', P: 2, OUT: 'P' });
    expect(getTensorByName('P')!.toArray()).toEqual([[1, 4], [9, 16]]);

    ext.createTensorBlock({ NAME: 'E', DATA: '[0, 1]', REQ_GRAD: false });
    ext.unaryMathBlock({ X: 'E', OP: 'Exp', OUT: 'Ex' });
    expectClose(getTensorByName('Ex')!.toArray(), [1, Math.E], 1e-4, 1e-4);
    ext.unaryMathBlock({ X: 'E', OP: 'Log', OUT: 'Ln' });
    expect(getTensorByName('Ln')!.toArray()).toEqual([-Infinity, 0]);

    ext.createTensorBlock({ NAME: 'Q', DATA: '[0, 4, 9]', REQ_GRAD: false });
    ext.unaryMathBlock({ X: 'Q', OP: 'Sqrt', OUT: 'Sq' });
    expect(getTensorByName('Sq')!.toArray()).toEqual([0, 2, 3]);

    ext.createTensorBlock({ NAME: 'N', DATA: '[[-1, 2]]', REQ_GRAD: false });
    ext.unaryMathBlock({ X: 'N', OP: 'Abs', OUT: 'Ab' });
    expect(getTensorByName('Ab')!.toArray()).toEqual([[1, 2]]);

    expect(() => ext.unaryMathBlock({ X: 'N', OP: 'Unknown', OUT: 'Noop' })).toThrow(/ScratchTensor\.unaryMathBlock:.*Unknown unary operation/);
    expect(getTensorByName('Noop')).toBeUndefined();
  });

  it('reductionBlock dispatches sum/mean/max/min with dim and keepdim', () => {
    ext.createTensorBlock({ NAME: 'A', DATA: '[[1, 2], [3, 4]]', REQ_GRAD: false });

    ext.reductionBlock({ X: 'A', DIM: 0, OP: '求和(Sum)', KEEPDIM: false, OUT: 'Sum0' });
    expect(getTensorByName('Sum0')!.toArray()).toEqual([4, 6]);

    ext.reductionBlock({ X: 'A', DIM: 0, OP: '求和(Sum)', KEEPDIM: true, OUT: 'SumK' });
    expect(getTensorByName('SumK')!.toArray()).toEqual([[4, 6]]);

    ext.reductionBlock({ X: 'A', DIM: 1, OP: '求平均值(Mean)', KEEPDIM: false, OUT: 'Mean1' });
    expect(getTensorByName('Mean1')!.toArray()).toEqual([1.5, 3.5]);

    ext.reductionBlock({ X: 'A', DIM: 0, OP: '求最大值(Max)', KEEPDIM: true, OUT: 'Max0' });
    expect(getTensorByName('Max0')!.toArray()).toEqual([[3, 4]]);

    ext.reductionBlock({ X: 'A', DIM: -1, OP: '求最小值(Min)', KEEPDIM: false, OUT: 'Min1' });
    expect(getTensorByName('Min1')!.toArray()).toEqual([1, 3]);

    ext.reductionBlock({ X: 'A', DIM: '全部', OP: '求和(Sum)', KEEPDIM: false, OUT: 'SumAll' });
    expect(getTensorByName('SumAll')!.item()).toBe(10);
  });

  it('transposeBlock swaps the two dims', () => {
    ext.createTensorBlock({ NAME: 'A', DATA: '[[1, 2], [3, 4]]', REQ_GRAD: false });
    ext.transposeBlock({ X: 'A', DIM1: 0, DIM2: 1, OUT: 'T' });
    expect(getTensorByName('T')!.toArray()).toEqual([[1, 3], [2, 4]]);
  });

  it('layerNormBlock normalizes the last N dims', () => {
    ext.createTensorBlock({ NAME: 'A', DATA: '[[1, 2]]', REQ_GRAD: false });
    ext.layerNormBlock({ X: 'A', N: 1, OUT: 'N' });
    expectClose(getTensorByName('N')!.toArray(), [[-0.99998, 0.99998]], 1e-3, 1e-3);
  });

  it('dropoutBlock registers an output (identity when p=0)', () => {
    ext.createTensorBlock({ NAME: 'A', DATA: '[[1, 2], [3, 4]]', REQ_GRAD: true });
    ext.dropoutBlock({ X: 'A', P: 0, OUT: 'D' });
    expect(getTensorByName('D')!.toArray()).toEqual([[1, 2], [3, 4]]);
    expect(getTensorByName('D')!.creatorOp).not.toBeNull();
  });

  it('optimizer blocks update leaf parameters, clip gradients and scale lr', () => {
    ext.createTensorBlock({ NAME: 'W', DATA: '[[2]]', REQ_GRAD: true });
    ext.createTensorBlock({ NAME: 'G', DATA: '[[-4]]', REQ_GRAD: false });
    const w = getTensorByName('W')!;
    w.grad = getTensorByName('G')!.data;

    ext.sgdFullBlock({ LR: 0.1, MOMENTUM: 0, WD: 0 });
    expectClose(w.toArray(), [[2.4]], 1e-5, 1e-5);

    w.grad = getTensorByName('G')!.data;
    ext.clipGradNormBlock({ MAX_NORM: 1 });
    expectClose(w.grad.data, [-1], 1e-5, 1e-5);

    ext.setLrScaleBlock({ SCALE: 0.5 });
    ext.sgdFullBlock({ LR: 0.1, MOMENTUM: 0, WD: 0 });
    expectClose(w.toArray(), [[2.45]], 1e-5, 1e-5);
  });

  it('adamBlock and adamWBlock update leaf parameters', () => {
    ext.createTensorBlock({ NAME: 'W1', DATA: '[[1]]', REQ_GRAD: true });
    const w1 = getTensorByName('W1')!;
    w1.grad = w1.data;
    ext.adamBlock({ LR: 0.1, B1: 0.9, B2: 0.999, WD: 0 });
    expect(w1.toArray()[0][0]).toBeLessThan(1);

    ext.createTensorBlock({ NAME: 'W2', DATA: '[[1]]', REQ_GRAD: true });
    const w2 = getTensorByName('W2')!;
    w2.grad = w2.data;
    ext.adamWBlock({ LR: 0.1, B1: 0.9, B2: 0.999, WD: 0 });
    expect(w2.toArray()[0][0]).toBeLessThan(1);
  });

  it('exportStateDictBlock / loadStateDictBlock round-trip weights', () => {
    ext.createTensorBlock({ NAME: 'W', DATA: '[[1.5], [-2.5]]', REQ_GRAD: true });
    const json = ext.exportStateDictBlock();
    const parsed = JSON.parse(json);
    expect(parsed.framework).toBe('ScratchTensor');
    expect(parsed.tensors.length).toBe(1);

    getTensorByName('W')!.data.set(0, 0, 999);
    ext.loadStateDictBlock({ JSON_STR: json });
    expect(getTensorByName('W')!.toArray()).toEqual([[1.5], [-2.5]]);
  });

  it('backwardBlock and clearGradBlock drive the autograd lifecycle', () => {
    ext.createTensorBlock({ NAME: 'W', DATA: '[[2]]', REQ_GRAD: true });
    ext.createTensorBlock({ NAME: 'X', DATA: '[[3]]', REQ_GRAD: false });
    ext.binaryOpBlock({ A: 'W', B: 'X', OP: '*', OUT: 'Y' });
    ext.backwardBlock({ TENSOR: 'Y' });
    expect(getTensorByName('W')!.grad).not.toBeNull();
    expect(getTensorByName('X')).toBeDefined();
    ext.clearGradBlock();
    expect(getTensorByName('W')!.grad).toBeNull();
    expect(() => ext.backwardBlock({ TENSOR: 'missing' }))
      .toThrow(/ScratchTensor\.backwardBlock:.*missing.*does not exist/);
  });

  it('preserves requiresGrad=false input and target tensors across multiple training iterations', () => {
    ext.createTensorBlock({ NAME: 'x', DATA: '[[1, 2]]', REQ_GRAD: false });
    ext.createTensorBlock({ NAME: 'y', DATA: '[[1]]', REQ_GRAD: false });
    ext.createTensorBlock({ NAME: 'w', DATA: '[[0.5], [0.5]]', REQ_GRAD: true });

    const losses: number[] = [];
    for (let step = 0; step < 3; step++) {
      ext.binaryOpBlock({ A: 'x', B: 'w', OP: '@', OUT: 'pred' });
      ext.mseLossBlock({ PRED: 'pred', TARGET: 'y', OUT: 'loss' });
      ext.backwardBlock({ TENSOR: 'loss' });
      ext.adamBlock({ LR: 0.1, B1: 0.9, B2: 0.999, WD: 0 });
      losses.push(Number(ext.getTensorPropertyBlock({ NAME: 'loss', PROPERTY: '标量值' })));
      ext.clearGradBlock();
    }

    expect(losses[1]).toBeLessThan(losses[0]);
    expect(losses[2]).toBeLessThan(losses[1]);
  });

  it('scaledDotProductAttentionBlock computes attention with default and explicit scale', () => {
    ext.createTensorBlock({ NAME: 'Q', DATA: '[[[[1, 0], [0, 1]]]]', REQ_GRAD: false });
    ext.createTensorBlock({ NAME: 'K', DATA: '[[[[1, 0], [0, 1]]]]', REQ_GRAD: false });
    ext.createTensorBlock({ NAME: 'V', DATA: '[[[[1, 0], [0, 1]]]]', REQ_GRAD: false });

    // SCALE=0 → default 1/sqrt(D); causal mask keeps lower-triangular attention
    ext.scaledDotProductAttentionBlock({ Q: 'Q', K: 'K', V: 'V', CAUSAL: true, DROPOUT: 0, SCALE: 0, OUT: 'Out' });
    const r = Math.exp(1 / Math.sqrt(2));
    expectClose(getTensorByName('Out')!.toArray(), [[[[1, 0], [1 / (1 + r), r / (1 + r)]]]], 1e-4, 1e-4);

    // SCALE=1 → no sqrt(D) scaling; full attention over 2x2 identity Q,K
    ext.scaledDotProductAttentionBlock({ Q: 'Q', K: 'K', V: 'V', CAUSAL: false, DROPOUT: 0, SCALE: 1, OUT: 'Out2' });
    const s = Math.exp(1) / (1 + Math.exp(1));
    expectClose(getTensorByName('Out2')!.toArray(), [[[[s, 1 - s], [1 - s, s]]]], 1e-4, 1e-4);

    expect(() => ext.scaledDotProductAttentionBlock({ Q: 'missing', K: 'K', V: 'V', CAUSAL: false, DROPOUT: 0, SCALE: 0, OUT: 'Noop' })).toThrow(/ScratchTensor\.scaledDotProductAttentionBlock:.*missing/);
    expect(getTensorByName('Noop')).toBeUndefined();
  });

  it('exposes explicit reduction, scale and slice input contracts without changing slot IDs', () => {
    const info = ext.getInfo();
    const reduction = info.blocks.find(block => block.opcode === 'reductionBlock')!;
    expect(reduction.arguments!.DIM).toEqual({ type: 'string', defaultValue: '-1' });
    const attention = info.blocks.find(block => block.opcode === 'scaledDotProductAttentionBlock')!;
    expect(attention.text).toContain('缩放（0=自动）: [SCALE]');
    const slicing = info.blocks.find(block => block.opcode === 'sliceBlock')!;
    expect(slicing.text).toContain('从索引 [START] 到索引 [END]（不含）');
    const load = info.blocks.find(block => block.opcode === 'loadStateDictBlock')!;
    expect(() => ext.loadStateDictBlock({ JSON_STR: load.arguments!.JSON_STR!.defaultValue })).not.toThrow();
  });

  it.each(['数值', '标量值', 'Shape', '梯度'])('rejects missing tensor property %s, distinct from an absent gradient', property => {
    ext.createTensorBlock({ NAME: 'A', DATA: '[1]', REQ_GRAD: true });
    expect(ext.getTensorPropertyBlock({ NAME: 'A', PROPERTY: '梯度' })).toBe('null');
    expect(() => ext.getTensorPropertyBlock({ NAME: 'missing', PROPERTY: property }))
      .toThrow(/ScratchTensor\.getTensorPropertyBlock:.*missing.*does not exist/);
  });

  it.each([
    ['binaryOpBlock', { A: 'A', B: 'missing', OP: '+' }],
    ['activationBlock', { X: 'missing', ACT: 'ReLU' }],
    ['softmaxBlock', { X: 'missing', DIM: -1 }],
    ['scaledDotProductAttentionBlock', { Q: 'A', K: 'A', V: 'missing', SCALE: 0, DROPOUT: 0 }],
    ['mseLossBlock', { PRED: 'A', TARGET: 'missing' }],
    ['crossEntropyLossBlock', { LOGITS: 'A', LABELS: 'missing' }],
    ['bceLossBlock', { PRED: 'A', TARGET: 'missing' }],
    ['l1LossBlock', { PRED: 'A', TARGET: 'missing' }],
    ['reshapeBlock', { X: 'missing', SHAPE: '[4]' }],
    ['unsqueezeBlock', { X: 'missing', DIM: 0 }],
    ['squeezeBlock', { X: 'missing' }],
    ['expandBlock', { X: 'missing', SHAPE: '[2,2]' }],
    ['splitBlock', { X: 'missing', DIM: 0, N: 2, PREFIX: 'part' }],
    ['sliceBlock', { X: 'missing', DIM: 0, START: 0, END: 1 }],
    ['whereBlock', { COND: 'A', A: 'A', B: 'missing' }],
    ['clampBlock', { X: 'missing', MIN: 0, MAX: 1 }],
    ['powBlock', { X: 'missing', P: 2 }],
    ['unaryMathBlock', { X: 'missing', OP: 'Abs' }],
    ['reductionBlock', { X: 'missing', DIM: -1, OP: '求和(Sum)' }],
    ['transposeBlock', { X: 'missing', DIM1: 0, DIM2: 1 }],
    ['layerNormBlock', { X: 'missing', N: 1 }],
    ['dropoutBlock', { X: 'missing', P: 0 }],
  ])('%s rejects a missing required input instead of reusing a stale output', (opcode, args) => {
    ext.createTensorBlock({ NAME: 'A', DATA: '[[1,2],[3,4]]', REQ_GRAD: false });
    ext.createTensorBlock({ NAME: 'previous', DATA: '[99]', REQ_GRAD: false });
    const previous = getTensorByName('previous');
    expect(() => (ext as any)[opcode as string]({ ...args as object, OUT: 'previous' }))
      .toThrow(`ScratchTensor.${opcode}: Tensor "missing" does not exist`);
    expect(getTensorByName('previous')).toBe(previous);
    expect(getTensorByName('previous')!.toArray()).toEqual([99]);
  });

  it.each(['reshapeBlock', 'expandBlock'])('%s propagates malformed JSON and incompatible shapes', opcode => {
    ext.createTensorBlock({ NAME: 'A', DATA: '[[1,2],[3,4]]', REQ_GRAD: false });
    for (const shape of ['invalid-json', '[3]']) {
      expect(() => (ext as any)[opcode]({ X: 'A', SHAPE: shape, OUT: 'bad' }))
        .toThrow(`ScratchTensor.${opcode}:`);
      expect(getTensorByName('bad')).toBeUndefined();
    }
  });

  it.each(['concatBlock', 'stackBlock'])('%s requires every named tensor and a non-empty array', opcode => {
    ext.createTensorBlock({ NAME: 'A', DATA: '[[1,2]]', REQ_GRAD: false });
    ext.createTensorBlock({ NAME: 'B', DATA: '[[3,4]]', REQ_GRAD: false });
    expect(() => (ext as any)[opcode]({ NAMES: '["A","missing","B"]', DIM: 0, OUT: 'bad' }))
      .toThrow(`ScratchTensor.${opcode}: Tensor "missing" does not exist`);
    for (const names of ['invalid-json', '[]', '{}', 'null', '"A"', '["A",1]']) {
      expect(() => (ext as any)[opcode]({ NAMES: names, DIM: 0, OUT: 'bad' }))
        .toThrow(`ScratchTensor.${opcode}:`);
    }
    ext.createTensorBlock({ NAME: 'C', DATA: '[[5]]', REQ_GRAD: false });
    expect(() => (ext as any)[opcode]({ NAMES: '["A","C"]', DIM: 0, OUT: 'bad' }))
      .toThrow(`ScratchTensor.${opcode}:`);
    expect(getTensorByName('bad')).toBeUndefined();
  });

  it.each(['NaN', 'last', '', ' ', ' 全部 ', '1.5', '1e0', '1.0', NaN, Infinity, 1.5, null, undefined, true, []])(
    'rejects ambiguous or noninteger reduction DIM %j', dim => {
      ext.createTensorBlock({ NAME: 'A', DATA: '[[1,2],[3,4]]', REQ_GRAD: false });
      expect(() => ext.reductionBlock({ X: 'A', DIM: dim, OP: '求和(Sum)', KEEPDIM: false, OUT: 'bad' }))
        .toThrow(/ScratchTensor\.reductionBlock: DIM must be an integer axis or "全部"/);
      expect(getTensorByName('bad')).toBeUndefined();
    }
  );

  it('distinguishes the last reduction axis from all axes and accepts integer text', () => {
    ext.createTensorBlock({ NAME: 'A', DATA: '[[1,2],[3,4]]', REQ_GRAD: false });
    ext.reductionBlock({ X: 'A', DIM: '-1', OP: '求和(Sum)', KEEPDIM: false, OUT: 'last' });
    ext.reductionBlock({ X: 'A', DIM: '全部', OP: '求和(Sum)', KEEPDIM: false, OUT: 'all' });
    expect(getTensorByName('last')!.toArray()).toEqual([3, 7]);
    expect(getTensorByName('all')!.item()).toBe(10);
  });

  it.each([-1, NaN, Infinity, -Infinity, 'auto', '', ' ', null, undefined, false])(
    'rejects invalid attention SCALE %j instead of choosing automatic scaling', scale => {
      ext.createTensorBlock({ NAME: 'Q', DATA: '[[[[1]]]]', REQ_GRAD: false });
      expect(() => ext.scaledDotProductAttentionBlock({ Q: 'Q', K: 'Q', V: 'Q',
        CAUSAL: false, DROPOUT: 0, SCALE: scale, OUT: 'bad' }))
        .toThrow(/ScratchTensor\.scaledDotProductAttentionBlock: SCALE/);
      expect(getTensorByName('bad')).toBeUndefined();
    }
  );

  it.each([
    ['reshapeBlock', { X: 'missing', SHAPE: '[1]', OUT: 'result' }],
    ['getTensorPropertyBlock', { NAME: 'missing', PROPERTY: '标量值' }],
  ])('%s shows its error and retires the current VM thread before throwing', (opcode, args) => {
    const target = {};
    const thread = { stack: ['greenFlag', 'procedure', 'failingBlock'], status: 0 };
    const events: unknown[][] = [];
    const emit = vi.fn((...event: unknown[]) => events.push(event));
    const retireThread = vi.fn((activeThread: typeof thread) => {
      activeThread.stack = [];
      activeThread.status = 4;
    });
    const stopThisScript = vi.fn();
    const util = { target, thread, runtime: { emit }, sequencer: { retireThread }, stopThisScript };
    expect(() => (ext as any)[opcode as string](args, util))
      .toThrow(`ScratchTensor.${opcode}: Tensor "missing" does not exist`);
    expect(events).toEqual([['SAY', target, 'say', `ScratchTensor.${opcode}: Tensor "missing" does not exist`]]);
    expect(retireThread).toHaveBeenCalledExactlyOnceWith(thread);
    expect(stopThisScript).not.toHaveBeenCalled();
    expect(thread.stack).toEqual([]);
    expect(thread.status).toBe(4);
  });

  it('falls back to stopThisScript when thread retirement is unavailable', () => {
    const stopThisScript = vi.fn();
    expect(() => (ext as any).backwardBlock({ TENSOR: 'missing' }, { stopThisScript }))
      .toThrow(/ScratchTensor\.backwardBlock:.*missing/);
    expect(stopThisScript).toHaveBeenCalledOnce();
  });

  it('keeps deletion of a missing name idempotent', () => {
    expect(() => ext.deleteTensorBlock({ NAME: 'missing' })).not.toThrow();
  });
});

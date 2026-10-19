import { describe, it, expect } from 'vitest';
import {
  createTensor,
  matmul,
  add,
  mseLoss,
  backward,
  clearAllGradients,
  sgdStep,
  resetOptimizerState,
  resetIdCounter
} from '../../src/index.js';
import {
  useVerificationLifecycle,
  expectClose
} from './harness/index.js';

describe('Verification: SGD Optimizer', () => {
  useVerificationLifecycle();

  it('simple SGD step updates leaf parameters in correct direction', () => {
    // X = [[1, 2]], W = [[2], [3]], target = [[10]]
    const X = createTensor('X', [[1, 2]], false);
    const W = createTensor('W', [[2], [3]], true);
    const target = createTensor('Target', [[10]], false);

    const pred = matmul(X, W, 'Pred');  // 1*2 + 2*3 = 8
    const loss = mseLoss(pred, target, 'Loss');
    backward(loss);

    // W.grad should be populated
    expect(W.grad).not.toBeNull();
    const wBefore = W.toArray();

    // Single SGD step with lr=0.01
    sgdStep(0.01);

    const wAfter = W.toArray();
    // Weights should have moved (not equal to before)
    expect(wAfter).not.toEqual(wBefore);
  });

  it('training loop reduces loss over iterations', () => {
    resetOptimizerState();
    const X = createTensor('X', [[1, 0.5]], false);
    const W = createTensor('W', [[0.1], [0.1]], true);
    const b = createTensor('b', [0], true);
    const target = createTensor('Target', [[5.0]], false);

    let firstLoss = 0;
    let lastLoss = 0;

    for (let i = 0; i < 50; i++) {
      const h = matmul(X, W, 'H');
      const pred = add(h, b, 'Pred');
      const loss = mseLoss(pred, target, 'Loss');

      const lossVal = loss.toArray();
      if (i === 0) firstLoss = lossVal;
      if (i === 49) lastLoss = lossVal;

      backward(loss);
      sgdStep(0.05);
      clearAllGradients();
    }

    // Loss should decrease significantly over 50 iterations
    expect(lastLoss).toBeLessThan(firstLoss * 0.01);
  });

  it('SGD with momentum converges faster than vanilla SGD', () => {
    resetOptimizerState();
    const X = createTensor('X_m', [[1, 2, 3]], false);
    const W = createTensor('W_m', [[0.01], [0.01], [0.01]], true);
    const target = createTensor('T_m', [[10.0]], false);

    let firstLoss = 0;
    for (let i = 0; i < 50; i++) {
      const pred = matmul(X, W, 'P');
      const loss = mseLoss(pred, target, 'L');
      if (i === 0) firstLoss = loss.toArray();
      backward(loss);
      sgdStep(0.01, 0.9);
      clearAllGradients();
    }

    const finalPred = matmul(X, W, 'P_final');
    const finalLoss = mseLoss(finalPred, target, 'L_final');
    const momentumLoss = finalLoss.toArray();

    // Momentum SGD should converge loss significantly from initial (~100) in 50 steps
    expect(momentumLoss).toBeLessThan(firstLoss * 0.05);
  });

  it('resetIdCounter resets SGD momentum state', () => {
    const W = createTensor('W', [[2.0]], true);
    const G = createTensor('G', [[-4.0]], false);

    W.grad = G.data;
    sgdStep(0.1, 0.9);
    const afterFirstStep = (W.toArray() as number[][])[0][0];

    resetIdCounter();
    W.clearGrad();
    W.grad = G.data;
    sgdStep(0.1, 0.9);
    const afterResetStep = (W.toArray() as number[][])[0][0];

    // With velocity reset to zero, the second step should just subtract lr*grad.
    expect(afterResetStep).toBeCloseTo(afterFirstStep + 0.4, 1e-10);
  });
});

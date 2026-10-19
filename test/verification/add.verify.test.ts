import { describe, it, expect } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { createTensor, add, backward } from '../../src/index.js';
import {
  useVerificationLifecycle,
  expectClose,
  withTfReference,
  verifyEagerGraphCleanup
} from './harness/index.js';

describe('Verification: AddOp against @tensorflow/tfjs', () => {
  useVerificationLifecycle();

  it('forward [2x3] + [2x3] same-shape', () => {
    const aData = [[1.5, -2.1, 0.0], [3.2, 4.4, -1.8]];
    const bData = [[-0.5, 2.1, 10.0], [1.1, -2.2, 0.8]];

    const C = add(createTensor('A', aData), createTensor('B', bData), 'C');

    withTfReference(ctx => {
      const tfC = tf.add(ctx.tensor2d(aData), ctx.tensor2d(bData));
      expect(C.shape).toEqual(tfC.shape);
      expectClose(C.toArray(), tfC.arraySync(), 1e-5, 1e-5, '(Add Same-Shape Forward)');
    });
  });

  it('forward [3x4] + [3x1] column broadcast', () => {
    const aData = [[1, 2, 3, 4], [-1, -2, -3, -4], [0.5, 1.5, -0.5, -1.5]];
    const bData = [[10.0], [-5.0], [2.0]];

    const C = add(createTensor('A', aData), createTensor('B', bData), 'C');

    withTfReference(ctx => {
      const tfC = tf.add(ctx.tensor2d(aData), ctx.tensor2d(bData));
      expect(C.shape).toEqual(tfC.shape);
      expectClose(C.toArray(), tfC.arraySync(), 1e-5, 1e-5, '(Add Column Broadcast Forward)');
    });
  });

  it('backward 2D + 1D broadcast [3x4] + [1x4]', () => {
    const aData = [[1, 2, 3, 4], [-1, -2, -3, -4], [0.5, 1.5, -0.5, -1.5]];
    const bData = [[10.0, -5.0, 0.1, 2.2]];

    const A = createTensor('A', aData, true);
    const B = createTensor('B', bData, true);
    const C = add(A, B, 'C');
    backward(C);

    withTfReference(ctx => {
      const tfA = ctx.variable2d(aData);
      const tfB = ctx.variable2d(bData);
      const grads = tf.variableGrads(() => tf.add(tfA, tfB).sum() as tf.Scalar);

      expect(C.shape).toEqual([3, 4]);
      expectClose(A.grad.data, grads.grads[tfA.name].arraySync(), 1e-5, 1e-5, '(Add dA)');
      expectClose(B.grad.data, grads.grads[tfB.name].arraySync(), 1e-5, 1e-5, '(Add dB Broadcast)');
      tf.dispose(grads);
    });

    verifyEagerGraphCleanup([C]);
  });

  it('backward only B requires grad', () => {
    const aData = [[1.1, 2.2], [3.3, 4.4]];
    const bData = [[-1.1, -2.2], [-3.3, -4.4]];

    const A = createTensor('A', aData, false);
    const B = createTensor('B', bData, true);
    const C = add(A, B, 'C');
    backward(C);

    expect(A.grad).toBeNull();
    expect(B.grad).not.toBeNull();

    withTfReference(ctx => {
      const tfA = ctx.tensor2d(aData);
      const tfB = ctx.variable2d(bData);
      const grads = tf.variableGrads(() => tf.add(tfA, tfB).sum() as tf.Scalar);
      expectClose(B.grad.data, grads.grads[tfB.name].arraySync(), 1e-5, 1e-5, '(Add dB only)');
      tf.dispose(grads);
    });

    verifyEagerGraphCleanup([C]);
  });

  it('backward 2D + 1D column broadcast [3x4] + [3x1]', () => {
    const aData = [[1, 2, 3, 4], [-1, -2, -3, -4], [0.5, 1.5, -0.5, -1.5]];
    const bData = [[10.0], [-5.0], [2.0]];

    const A = createTensor('A', aData, true);
    const B = createTensor('B', bData, true);
    const C = add(A, B, 'C');
    backward(C);

    withTfReference(ctx => {
      const tfA = ctx.variable2d(aData);
      const tfB = ctx.variable2d(bData);
      const grads = tf.variableGrads(() => tf.add(tfA, tfB).sum() as tf.Scalar);

      expect(C.shape).toEqual([3, 4]);
      expectClose(A.grad.data, grads.grads[tfA.name].arraySync(), 1e-5, 1e-5, '(Add column dA)');
      expectClose(B.grad.data, grads.grads[tfB.name].arraySync(), 1e-5, 1e-5, '(Add column dB Broadcast)');
      tf.dispose(grads);
    });

    verifyEagerGraphCleanup([C]);
  });
});

import * as tf from '@tensorflow/tfjs';

/**
 * Context object provided to TF.js reference computation callbacks.
 * All tensors created through this context are automatically tracked
 * and disposed when the `withTfReference` scope exits.
 */
export interface TfContext {
  /** Create and track a tf.tensor2d */
  tensor2d(data: number[][]): tf.Tensor2D;
  /** Create and track a tf.variable from 2D data */
  variable2d(data: number[][]): tf.Variable;
  /** Create and track a tf.tensor1d */
  tensor1d(data: number[]): tf.Tensor1D;
  /** Create and track a tf.variable from 1D data */
  variable1d(data: number[]): tf.Variable;
  /** Create and track a tf.tensor3d */
  tensor3d(data: number[][][]): tf.Tensor3D;
  /** Create and track a tf.variable from 3D data */
  variable3d(data: number[][][]): tf.Variable;
}

/**
 * Execute a TF.js reference computation with automatic resource management.
 * All tensors created through the `ctx` object are automatically disposed
 * when the callback completes, preventing TF.js memory leaks in tests.
 *
 * @example
 * ```typescript
 * const [tfResult, grads] = withTfReference(ctx => {
 *   const a = ctx.variable2d(aData);
 *   const b = ctx.variable2d(bData);
 *   const lossFn = () => tf.add(a, b).sum() as tf.Scalar;
 *   return [tf.add(a, b), tf.variableGrads(lossFn)];
 * });
 * ```
 */
export function withTfReference<T>(fn: (ctx: TfContext) => T): T {
  const tracked: tf.Tensor[] = [];

  const ctx: TfContext = {
    tensor2d(data: number[][]) {
      const t = tf.tensor2d(data);
      tracked.push(t);
      return t;
    },
    variable2d(data: number[][]) {
      const v = tf.variable(tf.tensor2d(data));
      tracked.push(v);
      return v;
    },
    tensor1d(data: number[]) {
      const t = tf.tensor1d(data);
      tracked.push(t);
      return t;
    },
    variable1d(data: number[]) {
      const v = tf.variable(tf.tensor1d(data));
      tracked.push(v);
      return v;
    },
    tensor3d(data: number[][][]) {
      const t = tf.tensor3d(data);
      tracked.push(t);
      return t;
    },
    variable3d(data: number[][][]) {
      const v = tf.variable(tf.tensor3d(data));
      tracked.push(v);
      return v;
    }
  };

  try {
    return fn(ctx);
  } finally {
    for (const t of tracked) {
      try { t.dispose(); } catch { /* already disposed */ }
    }
  }
}

/**
 * Compute TF.js gradients for variable tensors with automatic resource cleanup.
 *
 * @param lossFn - Function that computes a scalar loss from TF.js variables
 * @returns The grads result from tf.variableGrads, caller must dispose the result
 */
export function computeTfGrads(lossFn: () => tf.Scalar): { value: tf.Scalar; grads: tf.NamedTensorMap } {
  return tf.variableGrads(lossFn);
}

/**
 * Dispose a tf.variableGrads result safely.
 */
export function disposeTfGrads(result: { value: tf.Scalar; grads: tf.NamedTensorMap }): void {
  tf.dispose(result);
}

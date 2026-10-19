import { getAllLeafParameters, registerResetIdHook, registerClearRegistryHook } from './tensor.js';

registerResetIdHook(resetOptimizerState);
registerClearRegistryHook(resetOptimizerState);

/**
 * Internal momentum velocity buffers keyed by tensor ID.
 * Automatically created on first optimizer step for each leaf parameter.
 */
const momentumBuffers = new Map<number, Float32Array>();

/**
 * Adam first and second moment buffers keyed by tensor ID.
 */
const firstMoment = new Map<number, Float32Array>();
const secondMoment = new Map<number, Float32Array>();
const adamStepCount = new Map<number, number>();

/**
 * Global learning rate scale factor for scheduling.
 */
let lrScale = 1.0;

function requireFinite(name: string, value: number): void {
  if (!Number.isFinite(value)) {
    throw new Error(`${name} must be finite, got ${value}`);
  }
}

function requireNonNegative(name: string, value: number): void {
  requireFinite(name, value);
  if (value < 0) {
    throw new Error(`${name} must be non-negative, got ${value}`);
  }
}

function requireBeta(name: string, value: number): void {
  requireFinite(name, value);
  if (value < 0 || value >= 1) {
    throw new Error(`${name} must be in [0, 1), got ${value}`);
  }
}

function scaledLearningRate(lr: number): number {
  requireNonNegative('Learning rate', lr);
  const effectiveLr = lr * lrScale;
  requireFinite('Scaled learning rate', effectiveLr);
  return effectiveLr;
}

/**
 * Set the global learning rate scale factor.
 */
export function setLrScale(scale: number): void {
  requireNonNegative('Learning rate scale', scale);
  lrScale = scale;
}

/**
 * Get the current global learning rate scale factor.
 */
export function getLrScale(): number {
  return lrScale;
}

/**
 * [执行 SGD 优化器 学习率: <lr> 动量(Momentum): <momentum> 权重衰减(L2): <weight_decay>]
 *
 * Performs a single step of Stochastic Gradient Descent with optional
 * momentum and L2 weight decay on all leaf parameters (requiresGrad && isLeaf).
 *
 * Update rule (with momentum):
 *   v_t = momentum * v_{t-1} + grad + weight_decay * param
 *   param = param - lr * v_t
 *
 * Without momentum (momentum = 0):
 *   param = param - lr * (grad + weight_decay * param)
 */
export function sgdStep(lr: number, momentum = 0.9, weightDecay = 0): void {
  const effectiveLr = scaledLearningRate(lr);
  requireNonNegative('Momentum', momentum);
  requireNonNegative('Weight decay', weightDecay);
  const leaves = getAllLeafParameters();

  for (const param of leaves) {
    if (!param.grad) continue;

    const paramData: Float32Array = param.data.data;
    const gradData: Float32Array = param.grad.data;
    const size = paramData.length;

    if (momentum > 0) {
      // Initialize momentum buffer if not exists
      if (!momentumBuffers.has(param.id)) {
        momentumBuffers.set(param.id, new Float32Array(size));
      }
      const velocity = momentumBuffers.get(param.id)!;

      for (let i = 0; i < size; i++) {
        // v = momentum * v + grad + weight_decay * param
        velocity[i] = momentum * velocity[i] + gradData[i];
        if (weightDecay > 0) {
          velocity[i] += weightDecay * paramData[i];
        }
        // param = param - lr * v
        paramData[i] -= effectiveLr * velocity[i];
      }
    } else {
      // Simple SGD without momentum
      for (let i = 0; i < size; i++) {
        let grad = gradData[i];
        if (weightDecay > 0) {
          grad += weightDecay * paramData[i];
        }
        paramData[i] -= effectiveLr * grad;
      }
    }
  }
}

/**
 * [执行 Adam 优化器 学习率: <lr> Beta1: <beta1> Beta2: <beta2> 权重衰减: <weight_decay>]
 *
 * Performs a single Adam step. If decoupled is false, weight decay is applied
 * to the gradient (classic Adam). If decoupled is true, weight decay is applied
 * directly to the parameter (AdamW).
 */
export function adamStep(
  lr: number,
  beta1 = 0.9,
  beta2 = 0.999,
  weightDecay = 0,
  eps = 1e-8,
  decoupled = false
): void {
  const effectiveLr = scaledLearningRate(lr);
  requireBeta('Beta1', beta1);
  requireBeta('Beta2', beta2);
  requireNonNegative('Weight decay', weightDecay);
  requireFinite('Epsilon', eps);
  if (eps <= 0) {
    throw new Error(`Epsilon must be positive, got ${eps}`);
  }
  const leaves = getAllLeafParameters();

  for (const param of leaves) {
    if (!param.grad) continue;

    const paramData: Float32Array = param.data.data;
    const gradData: Float32Array = param.grad.data;
    const size = paramData.length;

    if (!firstMoment.has(param.id)) {
      firstMoment.set(param.id, new Float32Array(size));
      secondMoment.set(param.id, new Float32Array(size));
      adamStepCount.set(param.id, 0);
    }
    const m = firstMoment.get(param.id)!;
    const v = secondMoment.get(param.id)!;
    let step = adamStepCount.get(param.id)! + 1;
    adamStepCount.set(param.id, step);

    const biasCorr1 = 1 - Math.pow(beta1, step);
    const biasCorr2 = 1 - Math.pow(beta2, step);

    for (let i = 0; i < size; i++) {
      let g = gradData[i];
      if (decoupled) {
        paramData[i] -= effectiveLr * weightDecay * paramData[i];
      } else if (weightDecay > 0) {
        g += weightDecay * paramData[i];
      }
      m[i] = beta1 * m[i] + (1 - beta1) * g;
      v[i] = beta2 * v[i] + (1 - beta2) * g * g;
      const mHat = m[i] / biasCorr1;
      const vHat = v[i] / biasCorr2;
      paramData[i] -= effectiveLr * mHat / (Math.sqrt(vHat) + eps);
    }
  }
}

/**
 * [执行 AdamW 优化器 学习率: <lr> Beta1: <beta1> Beta2: <beta2> 权重衰减: <weight_decay>]
 */
export function adamWStep(lr: number, beta1 = 0.9, beta2 = 0.999, weightDecay = 0.01, eps = 1e-8): void {
  adamStep(lr, beta1, beta2, weightDecay, eps, true);
}

/**
 * [对所有参数执行梯度裁剪(Clip Grad Norm) 最大阈值: <max_norm>]
 *
 * Clips the global L2 norm of all leaf gradients to maxNorm.
 */
export function clipGradNorm(maxNorm: number): void {
  requireFinite('Maximum gradient norm', maxNorm);
  if (maxNorm <= 0) return;
  const leaves = getAllLeafParameters();
  let totalNormSq = 0;
  for (const param of leaves) {
    if (!param.grad) continue;
    const gradData = param.grad.data;
    for (let i = 0; i < gradData.length; i++) {
      totalNormSq += gradData[i] * gradData[i];
    }
  }
  const totalNorm = Math.sqrt(totalNormSq);
  if (totalNorm > maxNorm) {
    const scale = maxNorm / totalNorm;
    for (const param of leaves) {
      if (!param.grad) continue;
      const gradData = param.grad.data;
      for (let i = 0; i < gradData.length; i++) {
        gradData[i] *= scale;
      }
    }
  }
}

/**
 * Reset all optimizer internal state (momentum buffers, Adam moments, LR scale).
 * Useful for testing and when restarting training.
 */
export function resetOptimizerState(): void {
  momentumBuffers.clear();
  firstMoment.clear();
  secondMoment.clear();
  adamStepCount.clear();
  lrScale = 1.0;
}

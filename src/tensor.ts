import { OpNode } from './types.js';
import {
  fromJsArray,
  zerosNdArray,
  onesNdArray,
  randomNormalNdArray,
  toJsArray,
  addNdArray,
  cloneNdArray
} from './ndarray-utils.js';

let currentId = 100;

const resetIdHooks: (() => void)[] = [];

export function registerResetIdHook(fn: () => void): void {
  resetIdHooks.push(fn);
}

export function nextId(): number {
  return ++currentId;
}

export function resetIdCounter(start = 100): void {
  currentId = start;
  for (const hook of resetIdHooks) {
    hook();
  }
}

export class TensorNode {
  public id: number;                  // Global unique tensor identifier
  public name: string;                // Global registered name
  public data: any;                   // Underlying self-contained NDArray instance
  public grad: any | null = null;     // Accumulated gradient NDArray
  public requiresGrad: boolean;       // Whether this tensor requires gradient
  public creatorOp: OpNode | null;    // Creator operation in autograd dynamic graph
  public isLeaf: boolean;             // Whether this is a leaf tensor (not created by an operator)

  constructor(id: number, name: string, data: any, requiresGrad: boolean) {
    this.id = id;
    this.name = name;
    this.data = data;
    this.requiresGrad = requiresGrad;
    this.creatorOp = null;
    this.isLeaf = true;
  }

  get shape(): number[] {
    return Array.from(this.data.shape);
  }

  /**
   * Convert underlying ndarray to JavaScript nested array or scalar.
   */
  toArray(): any {
    return toJsArray(this.data);
  }

  /**
   * Return the scalar value of a single-element tensor.
   * Mirrors PyTorch's tensor.item().
   */
  item(): number {
    if (this.data.data.length !== 1) {
      throw new Error(`item() can only be called on a tensor with one element, got shape ${JSON.stringify(this.shape)}`);
    }
    return this.data.data[0];
  }

  /**
   * Add gradient to this tensor's grad buffer.
   */
  addGrad(gradNdArray: any): void {
    if (!this.requiresGrad) return;
    if (this.grad === null) {
      // Gradients are mutable during clipping; each tensor must own its buffer.
      this.grad = cloneNdArray(gradNdArray);
    } else {
      this.grad = addNdArray(this.grad, gradNdArray);
    }
  }

  /**
   * Clear accumulated gradient.
   */
  clearGrad(): void {
    this.grad = null;
  }
}

// Global Registry for scratch tensor binding
const tensorByName = new Map<string, TensorNode>();
const tensorById = new Map<number, TensorNode>();

/**
 * Register a tensor in the global registry.
 * If the name is already taken by another tensor, the stale ID entry is
 * removed so the replaced tensor no longer receives optimizer updates.
 */
export function registerTensor(tensor: TensorNode): void {
  const existing = tensorByName.get(tensor.name);
  if (existing && existing.id !== tensor.id) {
    tensorById.delete(existing.id);
  }
  tensorByName.set(tensor.name, tensor);
  tensorById.set(tensor.id, tensor);
}

/**
 * Get tensor by global registered name.
 */
export function getTensorByName(name: string): TensorNode | undefined {
  return tensorByName.get(name);
}

/**
 * Get tensor by integer ID.
 */
export function getTensorById(id: number): TensorNode | undefined {
  return tensorById.get(id);
}

/**
 * Get all registered tensors that are leaf parameters (requiresGrad = true && isLeaf = true).
 */
export function getAllLeafParameters(): TensorNode[] {
  const leaves: TensorNode[] = [];
  for (const tensor of tensorById.values()) {
    if (tensor.requiresGrad && tensor.isLeaf) {
      leaves.push(tensor);
    }
  }
  return leaves;
}

/**
 * Get gradient of a registered tensor by name.
 * Returns null if tensor does not exist or has no gradient.
 */
export function getTensorGrad(name: string): any | null {
  const tensor = tensorByName.get(name);
  if (!tensor || !tensor.grad) return null;
  return tensor.grad;
}

const clearRegistryHooks: (() => void)[] = [];

/**
 * Register a hook to be called when the global tensor registry is cleared.
 */
export function registerClearRegistryHook(fn: () => void): void {
  clearRegistryHooks.push(fn);
}

/**
 * Remove a tensor from the global registry by registered name.
 */
export function unregisterTensor(name: string): boolean {
  const tensor = tensorByName.get(name);
  if (!tensor) return false;
  tensorByName.delete(name);
  tensorById.delete(tensor.id);
  return true;
}

/**
 * Remove a tensor from the global registry by its integer ID.
 */
export function unregisterTensorById(id: number): boolean {
  const tensor = tensorById.get(id);
  if (!tensor) return false;
  tensorById.delete(id);
  if (tensorByName.get(tensor.name) === tensor) {
    tensorByName.delete(tensor.name);
  }
  return true;
}

/**
 * Clear global tensor registry.
 */
export function clearTensorRegistry(): void {
  tensorByName.clear();
  tensorById.clear();
  for (const hook of clearRegistryHooks) {
    hook();
  }
}

/**
 * [创建 Tensor 命名为 "name" 并赋值 <多维数组> 需梯度: <requiresGrad>]
 */
export function createTensor(name: string, dataOrArray: number | any[], requiresGrad = false): TensorNode {
  const { ndarray } = fromJsArray(dataOrArray);
  const id = nextId();
  const tensor = new TensorNode(id, name, ndarray, requiresGrad);
  registerTensor(tensor);
  return tensor;
}

/**
 * [创建形状为 <Shape数组> 的全0 Tensor 命名为 "name" 需梯度: <requiresGrad>]
 */
export function createZerosTensor(name: string, shape: number[], requiresGrad = false): TensorNode {
  const ndarray = zerosNdArray(shape);
  const id = nextId();
  const tensor = new TensorNode(id, name, ndarray, requiresGrad);
  registerTensor(tensor);
  return tensor;
}

/**
 * [创建形状为 <Shape数组> 的全1 Tensor 命名为 "name" 需梯度: <requiresGrad>]
 */
export function createOnesTensor(name: string, shape: number[], requiresGrad = false): TensorNode {
  const ndarray = onesNdArray(shape);
  const id = nextId();
  const tensor = new TensorNode(id, name, ndarray, requiresGrad);
  registerTensor(tensor);
  return tensor;
}

/**
 * [创建形状为 <Shape数组> 的随机正态分布 Tensor 命名为 "name" 需梯度: <requiresGrad>]
 */
export function createRandomNormalTensor(name: string, shape: number[], requiresGrad = false): TensorNode {
  const ndarray = randomNormalNdArray(shape);
  const id = nextId();
  const tensor = new TensorNode(id, name, ndarray, requiresGrad);
  registerTensor(tensor);
  return tensor;
}

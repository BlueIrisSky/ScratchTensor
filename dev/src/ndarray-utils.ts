/**
 * Self-contained N-Dimensional Array (NDArray) linear algebra component.
 */
export class NDArray {
  public dtype: string;
  public data: Float32Array;
  public shape: number[];
  public strides: number[];
  public offset: number;
  public order: string;

  constructor(
    dtype: string,
    data: Float32Array,
    shape: number[],
    strides: number[],
    offset = 0,
    order = 'row-major'
  ) {
    this.dtype = dtype;
    this.data = data;
    this.shape = shape;
    this.strides = strides;
    this.offset = offset;
    this.order = order;
  }

  /**
   * Get element at multi-dimensional indices.
   */
  get(...indices: number[]): number {
    let idx = this.offset;
    for (let i = 0; i < indices.length; i++) {
      idx += indices[i] * this.strides[i];
    }
    return this.data[idx];
  }

  /**
   * Set element at multi-dimensional indices to given value.
   * Last argument is value, preceding arguments are indices.
   */
  set(...args: number[]): void {
    const val = args[args.length - 1];
    let idx = this.offset;
    for (let i = 0; i < args.length - 1; i++) {
      idx += args[i] * this.strides[i];
    }
    this.data[idx] = val;
  }
}

/**
 * Compute row-major strides for a given shape.
 */
export function computeRowMajorStrides(shape: number[]): number[] {
  const rank = shape.length;
  if (rank === 0) return [1];
  const strides = new Array<number>(rank);
  let stride = 1;
  for (let i = rank - 1; i >= 0; i--) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return strides;
}

/**
 * Get total number of elements in a shape.
 */
export function getShapeSize(shape: number[]): number {
  let size = 1;
  for (let i = 0; i < shape.length; i++) {
    size *= shape[i];
  }
  return size;
}

/**
 * Normalize a possibly negative dimension index into [0, ndim).
 */
export function normalizeDim(dim: number, ndim: number): number {
  const d = dim < 0 ? dim + ndim : dim;
  if (d < 0 || d >= ndim) {
    throw new Error(`Dimension ${dim} out of range for ${ndim}-D tensor`);
  }
  return d;
}

/**
 * Decompose shape around axis `d` into (outerSize, dimSize, innerSize).
 */
export function getAxisStrides(shape: number[], d: number): { outerSize: number; dimSize: number; innerSize: number } {
  let outerSize = 1;
  for (let i = 0; i < d; i++) outerSize *= shape[i];
  const dimSize = shape[d];
  let innerSize = 1;
  for (let i = d + 1; i < shape.length; i++) innerSize *= shape[i];
  return { outerSize, dimSize, innerSize };
}

/**
 * Resolve a reshape target shape: validate total element count and
 * auto-derive at most one `-1` dimension from the original shape.
 */
export function resolveReshapeShape(originalShape: number[], newShape: number[]): number[] {
  let inferredIdx = -1;
  let knownSize = 1;
  for (let i = 0; i < newShape.length; i++) {
    const s = newShape[i];
    if (!Number.isInteger(s) || s < -1) {
      throw new Error(`Invalid reshape shape: ${JSON.stringify(newShape)}`);
    }
    if (s === -1) {
      if (inferredIdx !== -1) {
        throw new Error('Reshape shape can contain at most one -1 dimension');
      }
      inferredIdx = i;
    } else {
      knownSize *= s;
    }
  }
  const origSize = getShapeSize(originalShape);
  if (inferredIdx !== -1) {
    if (knownSize === 0 || origSize % knownSize !== 0) {
      throw new Error(`Cannot infer -1 dimension: ${origSize} elements are not divisible by ${knownSize}`);
    }
    const resolved = newShape.slice();
    resolved[inferredIdx] = origSize / knownSize;
    return resolved;
  }
  if (knownSize !== origSize) {
    throw new Error(`Reshape size mismatch: ${origSize} elements cannot be reshaped to ${JSON.stringify(newShape)}`);
  }
  return newShape;
}

/**
 * Create an NDArray from a Float32Array buffer and shape.
 */
export function createNdArray(buffer: Float32Array, shape: number[]): NDArray {
  return new NDArray('float32', buffer, shape, computeRowMajorStrides(shape), 0, 'row-major');
}

/**
 * Create an all-zeros ndarray with the given shape.
 */
export function zerosNdArray(shape: number[]): NDArray {
  return createNdArray(new Float32Array(getShapeSize(shape)), shape);
}

/**
 * Create an all-ones ndarray with the given shape.
 */
export function onesNdArray(shape: number[]): NDArray {
  const buffer = new Float32Array(getShapeSize(shape));
  buffer.fill(1.0);
  return createNdArray(buffer, shape);
}

/**
 * Create a random normal distribution ndarray (Box-Muller transform).
 */
export function randomNormalNdArray(shape: number[], mean = 0, std = 1): NDArray {
  const size = getShapeSize(shape);
  const buffer = new Float32Array(size);
  const twoPi = 2.0 * Math.PI;
  for (let i = 0; i < size; i += 2) {
    const u1 = Math.max(1e-15, Math.random());
    const u2 = twoPi * Math.random();
    const r = Math.sqrt(-2.0 * Math.log(u1)) * std;
    buffer[i] = mean + r * Math.cos(u2);
    if (i + 1 < size) {
      buffer[i + 1] = mean + r * Math.sin(u2);
    }
  }
  return createNdArray(buffer, shape);
}

/**
 * Convert a nested JS Array or scalar into an NDArray and inferred shape.
 */
export function fromJsArray(input: number | any[]): { ndarray: NDArray; shape: number[] } {
  if (typeof input === 'number') {
    const buffer = new Float32Array([input]);
    return { ndarray: createNdArray(buffer, []), shape: [] };
  }

  const shape: number[] = [];
  let curr: any = input;
  while (Array.isArray(curr)) {
    shape.push(curr.length);
    curr = curr[0];
  }

  const buffer = new Float32Array(getShapeSize(shape));
  let idx = 0;
  function flatten(arr: any[]) {
    for (let i = 0; i < arr.length; i++) {
      const item = arr[i];
      if (Array.isArray(item)) {
        flatten(item);
      } else {
        buffer[idx++] = Number(item);
      }
    }
  }
  flatten(input);

  return { ndarray: createNdArray(buffer, shape), shape };
}

/**
 * Convert an NDArray back to a JS nested Array (or scalar).
 */
export function toJsArray(nd: any): any {
  const shape: number[] = nd.shape;
  if (shape.length === 0) {
    return nd.data[nd.offset];
  }
  return toNestedArray(nd.data, shape, nd.offset, nd.strides, 0);
}

function toNestedArray(
  data: Float32Array,
  shape: number[],
  baseOffset: number,
  strides: number[],
  dim: number
): any {
  const size = shape[dim];
  const stride = strides[dim];
  const result = new Array(size);
  if (dim === shape.length - 1) {
    for (let i = 0; i < size; i++) {
      result[i] = data[baseOffset + i * stride];
    }
  } else {
    for (let i = 0; i < size; i++) {
      result[i] = toNestedArray(data, shape, baseOffset + i * stride, strides, dim + 1);
    }
  }
  return result;
}

/**
 * Clone an ndarray.
 */
export function cloneNdArray(nd: any): NDArray {
  return createNdArray(new Float32Array(nd.data), nd.shape.slice());
}

export function arraysEqual(a: number[], b: number[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

function isSingleElement(nd: any): boolean {
  return nd.data.length === 1 && (nd.shape.length === 0 || (nd.shape.length === 1 && nd.shape[0] === 1));
}

/**
 * Element-wise addition of two ndarrays.
 */
export function addNdArray(a: any, b: any): NDArray {
  const aData = a.data;
  const bData = b.data;
  const sizeA = aData.length;
  const sizeB = bData.length;

  if (sizeA === sizeB && arraysEqual(a.shape, b.shape)) {
    const outBuf = new Float32Array(sizeA);
    for (let i = 0; i < sizeA; i++) outBuf[i] = aData[i] + bData[i];
    return createNdArray(outBuf, a.shape.slice());
  }
  if (isSingleElement(a)) {
    const outBuf = new Float32Array(sizeB);
    const val = aData[0];
    for (let i = 0; i < sizeB; i++) outBuf[i] = val + bData[i];
    return createNdArray(outBuf, b.shape.slice());
  }
  if (isSingleElement(b)) {
    const outBuf = new Float32Array(sizeA);
    const val = bData[0];
    for (let i = 0; i < sizeA; i++) outBuf[i] = aData[i] + val;
    return createNdArray(outBuf, a.shape.slice());
  }
  if (sizeB > 0 && isSuffixBroadcast(b.shape, a.shape)) {
    const outBuf = new Float32Array(sizeA);
    const outer = sizeA / sizeB;
    for (let o = 0; o < outer; o++) {
      const base = o * sizeB;
      for (let j = 0; j < sizeB; j++) outBuf[base + j] = aData[base + j] + bData[j];
    }
    return createNdArray(outBuf, a.shape.slice());
  }

  const bBroadcast = broadcastNdArray(b, a.shape);
  return addNdArray(a, bBroadcast);
}

/**
 * Element-wise subtraction of two ndarrays.
 */
export function subNdArray(a: any, b: any): NDArray {
  const aData = a.data;
  const bData = b.data;
  const sizeA = aData.length;
  const sizeB = bData.length;

  if (sizeA === sizeB && arraysEqual(a.shape, b.shape)) {
    const outBuf = new Float32Array(sizeA);
    for (let i = 0; i < sizeA; i++) outBuf[i] = aData[i] - bData[i];
    return createNdArray(outBuf, a.shape.slice());
  }
  if (isSingleElement(a)) {
    const outBuf = new Float32Array(sizeB);
    const val = aData[0];
    for (let i = 0; i < sizeB; i++) outBuf[i] = val - bData[i];
    return createNdArray(outBuf, b.shape.slice());
  }
  if (isSingleElement(b)) {
    const outBuf = new Float32Array(sizeA);
    const val = bData[0];
    for (let i = 0; i < sizeA; i++) outBuf[i] = aData[i] - val;
    return createNdArray(outBuf, a.shape.slice());
  }
  if (sizeB > 0 && isSuffixBroadcast(b.shape, a.shape)) {
    const outBuf = new Float32Array(sizeA);
    const outer = sizeA / sizeB;
    for (let o = 0; o < outer; o++) {
      const base = o * sizeB;
      for (let j = 0; j < sizeB; j++) outBuf[base + j] = aData[base + j] - bData[j];
    }
    return createNdArray(outBuf, a.shape.slice());
  }

  const bBroadcast = broadcastNdArray(b, a.shape);
  return subNdArray(a, bBroadcast);
}

/**
 * Element-wise multiplication of two ndarrays.
 */
export function mulNdArray(a: any, b: any): NDArray {
  const aData = a.data;
  const bData = b.data;
  const sizeA = aData.length;
  const sizeB = bData.length;

  if (sizeA === sizeB && arraysEqual(a.shape, b.shape)) {
    const outBuf = new Float32Array(sizeA);
    for (let i = 0; i < sizeA; i++) outBuf[i] = aData[i] * bData[i];
    return createNdArray(outBuf, a.shape.slice());
  }
  if (isSingleElement(a)) {
    const outBuf = new Float32Array(sizeB);
    const val = aData[0];
    for (let i = 0; i < sizeB; i++) outBuf[i] = val * bData[i];
    return createNdArray(outBuf, b.shape.slice());
  }
  if (isSingleElement(b)) {
    const outBuf = new Float32Array(sizeA);
    const val = bData[0];
    for (let i = 0; i < sizeA; i++) outBuf[i] = aData[i] * val;
    return createNdArray(outBuf, a.shape.slice());
  }
  if (sizeB > 0 && isSuffixBroadcast(b.shape, a.shape)) {
    const outBuf = new Float32Array(sizeA);
    const outer = sizeA / sizeB;
    for (let o = 0; o < outer; o++) {
      const base = o * sizeB;
      for (let j = 0; j < sizeB; j++) outBuf[base + j] = aData[base + j] * bData[j];
    }
    return createNdArray(outBuf, a.shape.slice());
  }

  const bBroadcast = broadcastNdArray(b, a.shape);
  return mulNdArray(a, bBroadcast);
}

/**
 * Element-wise division of two ndarrays.
 */
export function divNdArray(a: any, b: any): NDArray {
  const aData = a.data;
  const bData = b.data;
  const sizeA = aData.length;
  const sizeB = bData.length;

  if (sizeA === sizeB && arraysEqual(a.shape, b.shape)) {
    const outBuf = new Float32Array(sizeA);
    for (let i = 0; i < sizeA; i++) outBuf[i] = aData[i] / bData[i];
    return createNdArray(outBuf, a.shape.slice());
  }
  if (isSingleElement(a)) {
    const outBuf = new Float32Array(sizeB);
    const val = aData[0];
    for (let i = 0; i < sizeB; i++) outBuf[i] = val / bData[i];
    return createNdArray(outBuf, b.shape.slice());
  }
  if (isSingleElement(b)) {
    const outBuf = new Float32Array(sizeA);
    const val = bData[0];
    for (let i = 0; i < sizeA; i++) outBuf[i] = aData[i] / val;
    return createNdArray(outBuf, a.shape.slice());
  }
  if (sizeB > 0 && isSuffixBroadcast(b.shape, a.shape)) {
    const outBuf = new Float32Array(sizeA);
    const outer = sizeA / sizeB;
    for (let o = 0; o < outer; o++) {
      const base = o * sizeB;
      for (let j = 0; j < sizeB; j++) outBuf[base + j] = aData[base + j] / bData[j];
    }
    return createNdArray(outBuf, a.shape.slice());
  }

  const bBroadcast = broadcastNdArray(b, a.shape);
  return divNdArray(a, bBroadcast);
}

/**
 * Matrix multiplication supporting PyTorch torch.matmul semantics.
 * Supports: 1D x 1D, 2D x 1D, 1D x 2D, 2D x 2D, and arbitrary batched
 * broadcasting for tensors of rank >= 2 (e.g. [B,M,K] x [K,N]).
 */
export function matMulNdArray(a: any, b: any): NDArray {
  const rankA = a.shape.length;
  const rankB = b.shape.length;
  const aData = a.data;
  const bData = b.data;
  const aOff = a.offset;
  const bOff = b.offset;

  // 1D x 1D: dot product scalar
  if (rankA === 1 && rankB === 1) {
    const K = a.shape[0];
    let sum = 0;
    for (let k = 0; k < K; k++) {
      sum += aData[aOff + k] * bData[bOff + k];
    }
    return createNdArray(new Float32Array([sum]), []);
  }

  // 2D x 1D: matrix-vector -> 1D [M]
  if (rankA === 2 && rankB === 1) {
    const M = a.shape[0];
    const K = a.shape[1];
    const outBuf = new Float32Array(M);
    for (let i = 0; i < M; i++) {
      let sum = 0;
      const aRow = aOff + i * K;
      for (let k = 0; k < K; k++) {
        sum += aData[aRow + k] * bData[bOff + k];
      }
      outBuf[i] = sum;
    }
    return createNdArray(outBuf, [M]);
  }

  // 1D x 2D: vector-matrix -> 1D [N]
  if (rankA === 1 && rankB === 2) {
    const K = a.shape[0];
    const N = b.shape[1];
    const outBuf = new Float32Array(N);
    for (let k = 0; k < K; k++) {
      const ak = aData[aOff + k];
      const bRow = bOff + k * N;
      for (let j = 0; j < N; j++) {
        outBuf[j] += ak * bData[bRow + j];
      }
    }
    return createNdArray(outBuf, [N]);
  }

  // Fast path 2D x 2D: matrix-matrix -> 2D [M, N]
  if (rankA === 2 && rankB === 2) {
    const M = a.shape[0];
    const K = a.shape[1];
    const N = b.shape[1];
    if (K !== b.shape[0]) {
      throw new Error(`Incompatible matmul shapes: ${JSON.stringify(a.shape)} x ${JSON.stringify(b.shape)}`);
    }
    const outBuf = new Float32Array(M * N);
    for (let i = 0; i < M; i++) {
      const outRow = i * N;
      const aRow = aOff + i * K;
      for (let k = 0; k < K; k++) {
        const aik = aData[aRow + k];
        const bRow = bOff + k * N;
        for (let j = 0; j < N; j++) {
          outBuf[outRow + j] += aik * bData[bRow + j];
        }
      }
    }
    return createNdArray(outBuf, [M, N]);
  }

  // General batched case: both tensors rank >= 2 with broadcasting.
  return matMulBroadcastNdArray(a, b);
}

function broadcastBatchShapes(shapeA: number[], shapeB: number[]): number[] {
  const maxLen = Math.max(shapeA.length, shapeB.length);
  const result = new Array<number>(maxLen);
  for (let i = 0; i < maxLen; i++) {
    const idxA = shapeA.length - 1 - i;
    const idxB = shapeB.length - 1 - i;
    const dimA = idxA >= 0 ? shapeA[idxA] : 1;
    const dimB = idxB >= 0 ? shapeB[idxB] : 1;
    if (dimA !== dimB && dimA !== 1 && dimB !== 1) {
      throw new Error(`Cannot broadcast matmul batch shapes ${JSON.stringify(shapeA)} and ${JSON.stringify(shapeB)}`);
    }
    result[maxLen - 1 - i] = Math.max(dimA, dimB);
  }
  return result;
}

function matMulBroadcastNdArray(a: any, b: any): NDArray {
  const aShape: number[] = a.shape;
  const bShape: number[] = b.shape;
  const rankA = aShape.length;
  const rankB = bShape.length;
  if (rankA < 2 || rankB < 2) {
    throw new Error(`matMulBroadcastNdArray requires inputs of rank >= 2, got ${rankA} and ${rankB}`);
  }
  const M = aShape[rankA - 2];
  const K = aShape[rankA - 1];
  const N = bShape[rankB - 1];
  const Kb = bShape[rankB - 2];
  if (K !== Kb) {
    throw new Error(`Incompatible matmul shapes: ${JSON.stringify(aShape)} x ${JSON.stringify(bShape)}`);
  }
  const batchShapeA = aShape.slice(0, -2);
  const batchShapeB = bShape.slice(0, -2);
  const batchShape = broadcastBatchShapes(batchShapeA, batchShapeB);
  const batchRank = batchShape.length;
  const batchSize = getShapeSize(batchShape);
  const outShape = [...batchShape, M, N];
  const outBuf = new Float32Array(batchSize * M * N);

  const aData = a.data;
  const bData = b.data;
  const aStrides = a.strides;
  const bStrides = b.strides;

  // Precompute broadcast batch strides aligned with batchShape
  const aBatchStrides = new Array<number>(batchRank).fill(0);
  const bBatchStrides = new Array<number>(batchRank).fill(0);
  const offsetA = batchRank - batchShapeA.length;
  const offsetB = batchRank - batchShapeB.length;
  for (let i = 0; i < batchShapeA.length; i++) {
    aBatchStrides[offsetA + i] = batchShapeA[i] === 1 ? 0 : aStrides[i];
  }
  for (let i = 0; i < batchShapeB.length; i++) {
    bBatchStrides[offsetB + i] = batchShapeB[i] === 1 ? 0 : bStrides[i];
  }

  const aStrideM = aStrides[rankA - 2];
  const aStrideK = aStrides[rankA - 1];
  const bStrideK = bStrides[rankB - 2];
  const bStrideN = bStrides[rankB - 1];
  const mn = M * N;

  for (let batch = 0; batch < batchSize; batch++) {
    let tmp = batch;
    let aBatchOffset = a.offset;
    let bBatchOffset = b.offset;
    for (let d = batchRank - 1; d >= 0; d--) {
      const coord = tmp % batchShape[d];
      tmp = (tmp - coord) / batchShape[d];
      aBatchOffset += coord * aBatchStrides[d];
      bBatchOffset += coord * bBatchStrides[d];
    }

    const outBatchBase = batch * mn;
    for (let i = 0; i < M; i++) {
      const outRow = outBatchBase + i * N;
      const aRow = aBatchOffset + i * aStrideM;
      for (let k = 0; k < K; k++) {
        const aik = aData[aRow + k * aStrideK];
        const bRow = bBatchOffset + k * bStrideK;
        if (bStrideN === 1) {
          for (let j = 0; j < N; j++) {
            outBuf[outRow + j] += aik * bData[bRow + j];
          }
        } else {
          for (let j = 0; j < N; j++) {
            outBuf[outRow + j] += aik * bData[bRow + j * bStrideN];
          }
        }
      }
    }
  }
  return createNdArray(outBuf, outShape);
}

/**
 * ReLU activation forward: max(0, x).
 */
export function reluNdArray(a: any): NDArray {
  const aData = a.data;
  const size = aData.length;
  const outBuf = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    const v = aData[i];
    outBuf[i] = v > 0 ? v : 0;
  }
  return createNdArray(outBuf, a.shape.slice());
}

/**
 * ReLU activation backward.
 */
export function reluGradNdArray(gradOut: any, forwardInput: any): NDArray {
  const gData = gradOut.data;
  const xData = forwardInput.data;
  const size = gData.length;
  const outBuf = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    outBuf[i] = xData[i] > 0 ? gData[i] : 0;
  }
  return createNdArray(outBuf, gradOut.shape.slice());
}

/**
 * Mean Squared Error loss between pred and target.
 */
export function mseLossNdArray(pred: any, target: any, computeGrad = true): { lossValue: number; gradPred: NDArray | null } {
  const pData = pred.data;
  const tData = target.data;
  const size = pData.length;
  let sumSq = 0;
  if (!computeGrad) {
    for (let i = 0; i < size; i++) {
      const diff = pData[i] - tData[i];
      sumSq += diff * diff;
    }
    return { lossValue: sumSq / size, gradPred: null };
  }
  const gradBuf = new Float32Array(size);
  const invScale = 2.0 / size;
  for (let i = 0; i < size; i++) {
    const diff = pData[i] - tData[i];
    sumSq += diff * diff;
    gradBuf[i] = diff * invScale;
  }
  return {
    lossValue: sumSq / size,
    gradPred: createNdArray(gradBuf, pred.shape.slice())
  };
}

// ============================================================================
// Advanced Tensor Structure Operations
// ============================================================================

function isCompatibleBroadcast(srcShape: number[], targetShape: number[]): boolean {
  if (srcShape.length > targetShape.length) return false;
  const offset = targetShape.length - srcShape.length;
  for (let i = 0; i < srcShape.length; i++) {
    const s = srcShape[i];
    const t = targetShape[offset + i];
    if (s !== 1 && s !== t) return false;
  }
  return true;
}

/**
 * Returns true if srcShape broadcasts into targetShape purely by repeating
 * the contiguous flat buffer of srcShape along leading dimensions.
 */
function isSuffixBroadcast(srcShape: number[], targetShape: number[]): boolean {
  if (srcShape.length > targetShape.length) return false;
  const offset = targetShape.length - srcShape.length;
  let seenNonOne = false;
  for (let i = 0; i < srcShape.length; i++) {
    const s = srcShape[i];
    const t = targetShape[offset + i];
    if (s === t) {
      if (s !== 1) seenNonOne = true;
    } else if (s === 1) {
      if (seenNonOne) return false;
    } else {
      return false;
    }
  }
  return true;
}

/**
 * Insert a size-1 dimension at the specified axis (supports negative indexing and append).
 * PyTorch-compatible: dim in [-ndim-1, ndim].
 */
export function unsqueezeNdArray(nd: any, dim: number): NDArray {
  const ndim = nd.shape.length;
  const d = dim < 0 ? dim + ndim + 1 : dim;
  if (d < 0 || d > ndim) {
    throw new Error(`Dimension ${dim} out of range for ${ndim}-D tensor`);
  }
  const newShape = [...nd.shape.slice(0, d), 1, ...nd.shape.slice(d)];
  return createNdArray(nd.data, newShape);
}

/**
 * Remove all size-1 dimensions, or a specific dimension if dim is provided.
 */
export function squeezeNdArray(nd: any, dim?: number): NDArray {
  if (dim !== undefined) {
    const d = normalizeDim(dim, nd.shape.length);
    if (nd.shape[d] !== 1) {
      throw new Error(`Squeeze target dim ${d} has size ${nd.shape[d]}, not 1`);
    }
    const newShape = [...nd.shape.slice(0, d), ...nd.shape.slice(d + 1)];
    return createNdArray(nd.data, newShape);
  }
  const newShape = nd.shape.filter((s: number) => s !== 1);
  return createNdArray(nd.data, newShape);
}

/**
 * Broadcast an ndarray to a target shape (flat stride-based copy, no per-element array allocation).
 */
export function expandNdArray(nd: any, targetShape: number[]): NDArray {
  if (!isCompatibleBroadcast(nd.shape, targetShape)) {
    throw new Error(`Cannot broadcast shape ${JSON.stringify(nd.shape)} to ${JSON.stringify(targetShape)}`);
  }
  const rank = targetShape.length;
  const outSize = getShapeSize(targetShape);
  const outBuf = new Float32Array(outSize);
  if (outSize === 0) return createNdArray(outBuf, targetShape.slice());

  const srcData = nd.data;
  const srcOff = nd.offset;
  if (srcData.length === 1) {
    outBuf.fill(srcData[srcOff]);
    return createNdArray(outBuf, targetShape.slice());
  }
  if (srcOff === 0 && isSuffixBroadcast(nd.shape, targetShape)) {
    const srcLen = srcData.length;
    const outer = outSize / srcLen;
    for (let o = 0; o < outer; o++) {
      outBuf.set(srcData, o * srcLen);
    }
    return createNdArray(outBuf, targetShape.slice());
  }

  const srcShape: number[] = nd.shape;
  const srcStrides = computeRowMajorStrides(srcShape);
  const outStrides = computeRowMajorStrides(targetShape);
  const paddedSrcStrides = new Array<number>(rank).fill(0);
  const offset = rank - srcShape.length;
  for (let i = 0; i < srcShape.length; i++) {
    paddedSrcStrides[offset + i] = srcShape[i] === 1 ? 0 : srcStrides[i];
  }

  const lastDim = targetShape[rank - 1];
  const lastSrcStride = paddedSrcStrides[rank - 1];

  function copyRecursive(dim: number, srcBase: number, outBase: number): void {
    if (dim === rank - 1) {
      if (lastSrcStride === 0) {
        outBuf.fill(srcData[srcBase], outBase, outBase + lastDim);
      } else {
        for (let i = 0; i < lastDim; i++) {
          outBuf[outBase + i] = srcData[srcBase + i * lastSrcStride];
        }
      }
      return;
    }
    const len = targetShape[dim];
    const sStride = paddedSrcStrides[dim];
    const oStride = outStrides[dim];
    for (let i = 0; i < len; i++) {
      copyRecursive(dim + 1, srcBase + i * sStride, outBase + i * oStride);
    }
  }

  copyRecursive(0, srcOff, 0);
  return createNdArray(outBuf, targetShape.slice());
}

function squeezeToRank(shape: number[], targetRank: number): number[] | null {
  if (shape.length < targetRank) return null;
  if (shape.length === targetRank) return shape;
  const extra = shape.length - targetRank;

  let result = shape.slice();
  let removed = 0;
  for (let i = 0; i < result.length && removed < extra; i++) {
    if (result[i] === 1) {
      result.splice(i, 1);
      removed++;
      i--;
    }
  }
  if (removed === extra) return result;

  result = shape.slice();
  removed = 0;
  for (let i = result.length - 1; i >= 0 && removed < extra; i--) {
    if (result[i] === 1) {
      result.splice(i, 1);
      removed++;
    }
  }
  if (removed === extra) return result;

  return null;
}

export function broadcastNdArray(nd: any, targetShape: number[]): NDArray {
  if (arraysEqual(nd.shape, targetShape)) return nd;
  if (!isCompatibleBroadcast(nd.shape, targetShape)) {
    if (nd.shape.length > targetShape.length) {
      const squeezedShape = squeezeToRank(nd.shape, targetShape.length);
      if (squeezedShape && isCompatibleBroadcast(squeezedShape, targetShape)) {
        const squeezed = createNdArray(nd.data, squeezedShape);
        return expandNdArray(squeezed, targetShape);
      }
    }
    throw new Error(`Cannot broadcast shape ${JSON.stringify(nd.shape)} to ${JSON.stringify(targetShape)}`);
  }
  return expandNdArray(nd, targetShape);
}

/**
 * Sum an ndarray along the specified dimension using flat stride loops.
 */
export function sumAlongDimNdArray(nd: any, dim?: number | null, keepdim = false): NDArray {
  const data = nd.data;
  const offset = nd.offset;
  if (dim === null || dim === undefined) {
    let total = 0;
    for (let i = 0; i < data.length; i++) {
      total += data[offset + i];
    }
    const outShape = keepdim ? new Array<number>(nd.shape.length).fill(1) : [];
    return createNdArray(new Float32Array([total]), outShape);
  }
  const d = normalizeDim(dim, nd.shape.length);
  const { outerSize, dimSize, innerSize } = getAxisStrides(nd.shape, d);
  const outBuf = new Float32Array(outerSize * innerSize);
  const outerStride = dimSize * innerSize;

  for (let outer = 0; outer < outerSize; outer++) {
    const outBase = outer * innerSize;
    const inOuterBase = offset + outer * outerStride;
    for (let k = 0; k < dimSize; k++) {
      const rowBase = inOuterBase + k * innerSize;
      for (let inner = 0; inner < innerSize; inner++) {
        outBuf[outBase + inner] += data[rowBase + inner];
      }
    }
  }

  if (keepdim) {
    const keptShape = nd.shape.slice();
    keptShape[d] = 1;
    return createNdArray(outBuf, keptShape);
  }
  if (nd.shape.length === 1) {
    return createNdArray(outBuf, []);
  }
  return createNdArray(outBuf, [...nd.shape.slice(0, d), ...nd.shape.slice(d + 1)]);
}

/**
 * Concatenate a list of ndarrays along the given dimension using contiguous block copies.
 */
export function concatNdArray(nds: any[], dim: number): NDArray {
  if (nds.length === 0) throw new Error('concatNdArray requires at least one ndarray');
  const refShape: number[] = nds[0].shape;
  const ndim = refShape.length;
  const d = normalizeDim(dim, ndim);
  let totalDimSize = refShape[d];

  for (let i = 1; i < nds.length; i++) {
    const shape = nds[i].shape;
    if (shape.length !== ndim) throw new Error('All ndarrays must have the same rank for concat');
    for (let j = 0; j < ndim; j++) {
      if (j !== d && shape[j] !== refShape[j]) {
        throw new Error(`Concat shape mismatch at dim ${j}: ${shape[j]} vs ${refShape[j]}`);
      }
    }
    totalDimSize += shape[d];
  }

  const outShape = refShape.slice();
  outShape[d] = totalDimSize;
  const { outerSize, innerSize } = getAxisStrides(refShape, d);
  const outOuterStride = totalDimSize * innerSize;
  const outBuf = new Float32Array(outerSize * outOuterStride);

  let dimOffset = 0;
  for (let t = 0; t < nds.length; t++) {
    const nd = nds[t];
    const tData: Float32Array = nd.data;
    const tOff: number = nd.offset;
    const blockLen = nd.shape[d] * innerSize;
    const outSliceOffset = dimOffset * innerSize;
    for (let outer = 0; outer < outerSize; outer++) {
      const srcStart = tOff + outer * blockLen;
      const dstStart = outer * outOuterStride + outSliceOffset;
      outBuf.set(tData.subarray(srcStart, srcStart + blockLen), dstStart);
    }
    dimOffset += nd.shape[d];
  }

  return createNdArray(outBuf, outShape);
}

/**
 * Slice an ndarray along a dimension [start, end) using contiguous block copies.
 */
export function sliceNdArray(nd: any, dim: number, start: number, end: number): NDArray {
  const d = normalizeDim(dim, nd.shape.length);
  if (start < 0 || end > nd.shape[d] || start >= end) {
    throw new Error(`Invalid slice range [${start}, ${end}) on dim ${d} of size ${nd.shape[d]}`);
  }
  const newShape = nd.shape.slice();
  const sliceLen = end - start;
  newShape[d] = sliceLen;

  const { outerSize, dimSize, innerSize } = getAxisStrides(nd.shape, d);
  const inOuterStride = dimSize * innerSize;
  const outOuterStride = sliceLen * innerSize;
  const startOffset = start * innerSize;
  const outBuf = new Float32Array(outerSize * outOuterStride);
  const srcData: Float32Array = nd.data;
  const srcOff: number = nd.offset;

  for (let outer = 0; outer < outerSize; outer++) {
    const srcStart = srcOff + outer * inOuterStride + startOffset;
    outBuf.set(srcData.subarray(srcStart, srcStart + outOuterStride), outer * outOuterStride);
  }
  return createNdArray(outBuf, newShape);
}

/**
 * Split an ndarray evenly into n parts along a dimension.
 */
export function splitNdArray(nd: any, dim: number, n: number): NDArray[] {
  const d = normalizeDim(dim, nd.shape.length);
  const dimSize = nd.shape[d];
  if (dimSize % n !== 0) {
    throw new Error(`Cannot split dim ${d} of size ${dimSize} into ${n} equal parts`);
  }
  const chunk = dimSize / n;
  const results = new Array<NDArray>(n);
  for (let i = 0; i < n; i++) {
    const start = i * chunk;
    results[i] = sliceNdArray(nd, d, start, start + chunk);
  }
  return results;
}

/**
 * Stack a list of ndarrays along a new dimension.
 */
export function stackNdArray(nds: any[], dim: number): NDArray {
  if (nds.length === 0) throw new Error('stackNdArray requires at least one ndarray');
  const ndim = nds[0].shape.length;
  const d = dim < 0 ? dim + ndim + 1 : dim;
  if (d < 0 || d > ndim) {
    throw new Error(`Dimension ${dim} out of range for ${ndim}-D tensor`);
  }
  const unsqueezed = nds.map(nd => {
    const newShape = [...nd.shape.slice(0, d), 1, ...nd.shape.slice(d)];
    return createNdArray(nd.data, newShape);
  });
  return concatNdArray(unsqueezed, d);
}

export function computeBroadcastShape(shapes: number[][]): number[] {
  let maxRank = 0;
  for (let i = 0; i < shapes.length; i++) {
    if (shapes[i].length > maxRank) maxRank = shapes[i].length;
  }
  const result = new Array<number>(maxRank).fill(1);
  for (let sIdx = 0; sIdx < shapes.length; sIdx++) {
    const shape = shapes[sIdx];
    const offset = maxRank - shape.length;
    for (let i = 0; i < shape.length; i++) {
      const s = shape[i];
      const t = result[offset + i];
      if (s === 1) continue;
      if (t === 1) {
        result[offset + i] = s;
      } else if (t !== s) {
        throw new Error(`Cannot broadcast shapes ${JSON.stringify(shapes)}`);
      }
    }
  }
  return result;
}

/**
 * Sum a broadcasted gradient back to a target shape, matching PyTorch's sum_to_size.
 */
export function sumToShapeNdArray(grad: any, targetShape: number[]): NDArray {
  if (arraysEqual(grad.shape, targetShape)) return grad;

  // Fast path: reducing to a single-element tensor (e.g. scalar or [1] bias)
  const targetSize = getShapeSize(targetShape);
  if (targetSize === 1) {
    let total = 0;
    const gData = grad.data;
    for (let i = 0; i < gData.length; i++) total += gData[i];
    return createNdArray(new Float32Array([total]), targetShape.slice());
  }

  // Fast path: suffix reduction (e.g. [B, L, D] -> [D] or [1, D]) in a single flat pass
  if (isSuffixBroadcast(targetShape, grad.shape)) {
    const outBuf = new Float32Array(targetSize);
    const gData = grad.data;
    const outer = gData.length / targetSize;
    for (let o = 0; o < outer; o++) {
      const base = o * targetSize;
      for (let j = 0; j < targetSize; j++) {
        outBuf[j] += gData[base + j];
      }
    }
    return createNdArray(outBuf, targetShape.slice());
  }

  const offset = grad.shape.length - targetShape.length;
  let result = grad;
  for (let i = 0; i < targetShape.length; i++) {
    const expDim = grad.shape[offset + i];
    const targetDim = targetShape[i];
    if (targetDim === 1 && expDim !== 1) {
      result = sumAlongDimNdArray(result, offset + i, true);
    }
  }
  for (let i = 0; i < offset; i++) {
    result = sumAlongDimNdArray(result, 0, true);
  }
  if (targetShape.length < result.shape.length) {
    return createNdArray(result.data, targetShape.slice());
  }
  return result;
}

/**
 * Element-wise masked selection: cond ? a : b, with broadcasting.
 */
export function whereNdArray(cond: any, a: any, b: any): NDArray {
  const shape = computeBroadcastShape([cond.shape, a.shape, b.shape]);
  const condB = broadcastNdArray(cond, shape);
  const aB = broadcastNdArray(a, shape);
  const bB = broadcastNdArray(b, shape);
  const size = aB.data.length;
  const outBuf = new Float32Array(size);
  const cData = condB.data;
  const aData = aB.data;
  const bData = bB.data;
  for (let i = 0; i < size; i++) {
    outBuf[i] = cData[i] ? aData[i] : bData[i];
  }
  return createNdArray(outBuf, shape);
}

/**
 * Element-wise clamp to [min, max].
 */
export function clampNdArray(nd: any, min: number, max: number): NDArray {
  const data = nd.data;
  const size = data.length;
  const outBuf = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    const v = data[i];
    outBuf[i] = v < min ? min : v > max ? max : v;
  }
  return createNdArray(outBuf, nd.shape.slice());
}

// ============================================================================
// Element-wise Math Functions
// ============================================================================

export function powNdArray(nd: any, p: number): NDArray {
  const data = nd.data;
  const size = data.length;
  const outBuf = new Float32Array(size);
  if (p === 2) {
    for (let i = 0; i < size; i++) {
      const v = data[i];
      outBuf[i] = v * v;
    }
  } else {
    for (let i = 0; i < size; i++) {
      outBuf[i] = Math.pow(data[i], p);
    }
  }
  return createNdArray(outBuf, nd.shape.slice());
}

export function expNdArray(nd: any): NDArray {
  const data = nd.data;
  const size = data.length;
  const outBuf = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    outBuf[i] = Math.exp(data[i]);
  }
  return createNdArray(outBuf, nd.shape.slice());
}

export function logNdArray(nd: any): NDArray {
  const data = nd.data;
  const size = data.length;
  const outBuf = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    outBuf[i] = Math.log(data[i]);
  }
  return createNdArray(outBuf, nd.shape.slice());
}

export function sqrtNdArray(nd: any): NDArray {
  const data = nd.data;
  const size = data.length;
  const outBuf = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    outBuf[i] = Math.sqrt(data[i]);
  }
  return createNdArray(outBuf, nd.shape.slice());
}

export function absNdArray(nd: any): NDArray {
  const data = nd.data;
  const size = data.length;
  const outBuf = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    outBuf[i] = Math.abs(data[i]);
  }
  return createNdArray(outBuf, nd.shape.slice());
}

// ============================================================================
// Activation Functions
// ============================================================================

const SQRT_2_OVER_PI = Math.sqrt(2.0 / Math.PI);

export function geluNdArray(nd: any): NDArray {
  const data = nd.data;
  const size = data.length;
  const outBuf = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    const x = data[i];
    const c = SQRT_2_OVER_PI * (x + 0.044715 * x * x * x);
    outBuf[i] = 0.5 * x * (1.0 + Math.tanh(c));
  }
  return createNdArray(outBuf, nd.shape.slice());
}

export function geluGradNdArray(gradOut: any, x: any): NDArray {
  const gData = gradOut.data;
  const xData = x.data;
  const size = gData.length;
  const outBuf = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    const xv = xData[i];
    const xv2 = xv * xv;
    const c = SQRT_2_OVER_PI * (xv + 0.044715 * xv2 * xv);
    const tanhC = Math.tanh(c);
    const sech2 = 1.0 - tanhC * tanhC;
    const dCdx = SQRT_2_OVER_PI * (1.0 + 0.134145 * xv2);
    const dgelu = 0.5 * (1.0 + tanhC) + 0.5 * xv * sech2 * dCdx;
    outBuf[i] = gData[i] * dgelu;
  }
  return createNdArray(outBuf, gradOut.shape.slice());
}

export function sigmoidNdArray(nd: any): NDArray {
  const data = nd.data;
  const size = data.length;
  const outBuf = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    outBuf[i] = 1.0 / (1.0 + Math.exp(-data[i]));
  }
  return createNdArray(outBuf, nd.shape.slice());
}

export function sigmoidGradNdArray(gradOut: any, out: any): NDArray {
  const gData = gradOut.data;
  const oData = out.data;
  const size = gData.length;
  const outBuf = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    const s = oData[i];
    outBuf[i] = gData[i] * s * (1.0 - s);
  }
  return createNdArray(outBuf, gradOut.shape.slice());
}

export function tanhNdArray(nd: any): NDArray {
  const data = nd.data;
  const size = data.length;
  const outBuf = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    outBuf[i] = Math.tanh(data[i]);
  }
  return createNdArray(outBuf, nd.shape.slice());
}

export function tanhGradNdArray(gradOut: any, out: any): NDArray {
  const gData = gradOut.data;
  const oData = out.data;
  const size = gData.length;
  const outBuf = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    const t = oData[i];
    outBuf[i] = gData[i] * (1.0 - t * t);
  }
  return createNdArray(outBuf, gradOut.shape.slice());
}

export function softmaxNdArray(nd: any, dim: number): NDArray {
  const d = normalizeDim(dim, nd.shape.length);
  const outShape: number[] = nd.shape.slice();
  const { outerSize, dimSize, innerSize } = getAxisStrides(outShape, d);
  const outBuf = new Float32Array(outerSize * dimSize * innerSize);
  const data: Float32Array = nd.data;
  const off: number = nd.offset;
  const outerStride = dimSize * innerSize;

  if (innerSize === 1) {
    // Contiguous last-dimension fast path
    for (let outer = 0; outer < outerSize; outer++) {
      const base = outer * dimSize;
      const srcBase = off + base;
      let maxVal = -Infinity;
      for (let k = 0; k < dimSize; k++) {
        const v = data[srcBase + k];
        if (v > maxVal) maxVal = v;
      }
      let sumExp = 0;
      for (let k = 0; k < dimSize; k++) {
        const e = Math.exp(data[srcBase + k] - maxVal);
        outBuf[base + k] = e;
        sumExp += e;
      }
      const invSum = 1.0 / sumExp;
      for (let k = 0; k < dimSize; k++) {
        outBuf[base + k] *= invSum;
      }
    }
  } else {
    for (let outer = 0; outer < outerSize; outer++) {
      const outerBase = outer * outerStride;
      for (let inner = 0; inner < innerSize; inner++) {
        const base = outerBase + inner;
        let maxVal = -Infinity;
        for (let k = 0; k < dimSize; k++) {
          const v = data[off + base + k * innerSize];
          if (v > maxVal) maxVal = v;
        }
        let sumExp = 0;
        for (let k = 0; k < dimSize; k++) {
          const idx = base + k * innerSize;
          const e = Math.exp(data[off + idx] - maxVal);
          outBuf[idx] = e;
          sumExp += e;
        }
        const invSum = 1.0 / sumExp;
        for (let k = 0; k < dimSize; k++) {
          outBuf[base + k * innerSize] *= invSum;
        }
      }
    }
  }
  return createNdArray(outBuf, outShape);
}

export function softmaxGradNdArray(gradOutput: any, out: any, dim: number): NDArray {
  const d = normalizeDim(dim, out.shape.length);
  const outShape: number[] = out.shape.slice();
  const { outerSize, dimSize, innerSize } = getAxisStrides(outShape, d);
  const outBuf = new Float32Array(outerSize * dimSize * innerSize);
  const oData: Float32Array = out.data;
  const gData: Float32Array = gradOutput.data;
  const oOff: number = out.offset;
  const gOff: number = gradOutput.offset;
  const outerStride = dimSize * innerSize;

  if (innerSize === 1) {
    for (let outer = 0; outer < outerSize; outer++) {
      const base = outer * dimSize;
      let dot = 0;
      for (let k = 0; k < dimSize; k++) {
        dot += oData[oOff + base + k] * gData[gOff + base + k];
      }
      for (let k = 0; k < dimSize; k++) {
        outBuf[base + k] = oData[oOff + base + k] * (gData[gOff + base + k] - dot);
      }
    }
  } else {
    for (let outer = 0; outer < outerSize; outer++) {
      const outerBase = outer * outerStride;
      for (let inner = 0; inner < innerSize; inner++) {
        const base = outerBase + inner;
        let dot = 0;
        for (let k = 0; k < dimSize; k++) {
          const idx = base + k * innerSize;
          dot += oData[oOff + idx] * gData[gOff + idx];
        }
        for (let k = 0; k < dimSize; k++) {
          const idx = base + k * innerSize;
          outBuf[idx] = oData[oOff + idx] * (gData[gOff + idx] - dot);
        }
      }
    }
  }
  return createNdArray(outBuf, outShape);
}

// ============================================================================
// Loss Functions
// ============================================================================

export function crossEntropyLossNdArray(
  logits: any,
  labels: any,
  computeGrad = true
): { lossValue: number; gradLogits: NDArray | null } {
  const shape: number[] = logits.shape;
  if (shape.length !== 2) {
    throw new Error('CrossEntropyLoss expects 2D logits [N, C]');
  }
  const N = shape[0];
  const C = shape[1];
  const logitsFlat = logits.data;
  const labelsFlat = labels.data;
  const eps = 1e-12;
  const invN = 1.0 / N;

  const gradBuf = computeGrad ? new Float32Array(N * C) : null;
  let loss = 0;
  for (let i = 0; i < N; i++) {
    const rowBase = i * C;
    let maxVal = -Infinity;
    for (let j = 0; j < C; j++) {
      const v = logitsFlat[rowBase + j];
      if (v > maxVal) maxVal = v;
    }
    let sumExp = 0;
    for (let j = 0; j < C; j++) {
      sumExp += Math.exp(logitsFlat[rowBase + j] - maxVal);
    }
    const label = Math.round(labelsFlat[i]);
    const probLabel = Math.exp(logitsFlat[rowBase + label] - maxVal) / sumExp;
    loss -= Math.log(Math.max(probLabel, eps));

    if (gradBuf) {
      const scale = invN / sumExp;
      for (let j = 0; j < C; j++) {
        gradBuf[rowBase + j] = Math.exp(logitsFlat[rowBase + j] - maxVal) * scale;
      }
      gradBuf[rowBase + label] -= invN;
    }
  }
  loss *= invN;

  return {
    lossValue: loss,
    gradLogits: gradBuf ? createNdArray(gradBuf, shape.slice()) : null
  };
}

export function bceLossNdArray(
  pred: any,
  target: any,
  computeGrad = true
): { lossValue: number; gradPred: NDArray | null } {
  const pData = pred.data;
  const tData = target.data;
  const size = pData.length;
  const eps = 1e-12;
  let loss = 0;
  const gradBuf = computeGrad ? new Float32Array(size) : null;
  const invSize = 1.0 / size;
  for (let i = 0; i < size; i++) {
    const rawP = pData[i];
    const p = rawP < eps ? eps : rawP > 1 - eps ? 1 - eps : rawP;
    const t = tData[i];
    loss -= t * Math.log(p) + (1 - t) * Math.log(1 - p);
    if (gradBuf) {
      gradBuf[i] = ((p - t) / (p * (1 - p))) * invSize;
    }
  }
  return {
    lossValue: loss * invSize,
    gradPred: gradBuf ? createNdArray(gradBuf, pred.shape.slice()) : null
  };
}

export function l1LossNdArray(
  pred: any,
  target: any,
  computeGrad = true
): { lossValue: number; gradPred: NDArray | null } {
  const pData = pred.data;
  const tData = target.data;
  const size = pData.length;
  let loss = 0;
  const gradBuf = computeGrad ? new Float32Array(size) : null;
  const invSize = 1.0 / size;
  for (let i = 0; i < size; i++) {
    const diff = pData[i] - tData[i];
    loss += Math.abs(diff);
    if (gradBuf) {
      gradBuf[i] = (diff > 0 ? invSize : diff < 0 ? -invSize : 0);
    }
  }
  return {
    lossValue: loss * invSize,
    gradPred: gradBuf ? createNdArray(gradBuf, pred.shape.slice()) : null
  };
}

// ============================================================================
// Normalization Operations
// ============================================================================

/**
 * LayerNorm forward: normalize over the last n dimensions in a single flat pass per row.
 */
export function layerNormNdArray(nd: any, n: number): { out: NDArray; mean: NDArray; invStd: NDArray } {
  const shape: number[] = nd.shape;
  const rank = shape.length;
  if (n <= 0 || n > rank) {
    throw new Error(`Cannot normalize last ${n} dims of a ${rank}-D tensor`);
  }
  let leadingSize = 1;
  for (let i = 0; i < rank - n; i++) leadingSize *= shape[i];
  let normalizedSize = 1;
  for (let i = rank - n; i < rank; i++) normalizedSize *= shape[i];

  const xData: Float32Array = nd.data;
  const outBuf = new Float32Array(leadingSize * normalizedSize);
  const meanBuf = new Float32Array(leadingSize);
  const invStdBuf = new Float32Array(leadingSize);
  const invNorm = 1.0 / normalizedSize;

  for (let row = 0; row < leadingSize; row++) {
    const base = row * normalizedSize;
    let sum = 0;
    for (let j = 0; j < normalizedSize; j++) {
      sum += xData[base + j];
    }
    const m = sum * invNorm;
    meanBuf[row] = m;

    let sumSq = 0;
    for (let j = 0; j < normalizedSize; j++) {
      const diff = xData[base + j] - m;
      sumSq += diff * diff;
    }
    const invS = 1.0 / Math.sqrt(sumSq * invNorm + 1e-5);
    invStdBuf[row] = invS;

    for (let j = 0; j < normalizedSize; j++) {
      outBuf[base + j] = (xData[base + j] - m) * invS;
    }
  }

  const meanShape = shape.map((s: number, i: number) => (i < rank - n ? s : 1));
  return {
    out: createNdArray(outBuf, shape.slice()),
    mean: createNdArray(meanBuf, meanShape),
    invStd: createNdArray(invStdBuf, meanShape)
  };
}

/**
 * LayerNorm backward: single flat pass per row without intermediate broadcast allocations.
 */
export function layerNormGradNdArray(gradOutput: any, x: any, mean: any, invStd: any, n: number): NDArray {
  const shape: number[] = x.shape;
  const rank = shape.length;
  let leadingSize = 1;
  for (let i = 0; i < rank - n; i++) leadingSize *= shape[i];
  let normalizedSize = 1;
  for (let i = rank - n; i < rank; i++) normalizedSize *= shape[i];

  const xData: Float32Array = x.data;
  const gData: Float32Array = gradOutput.data;
  const meanData: Float32Array = mean.data;
  const invStdData: Float32Array = invStd.data;
  const dxBuf = new Float32Array(leadingSize * normalizedSize);
  const invNorm = 1.0 / normalizedSize;

  for (let row = 0; row < leadingSize; row++) {
    const base = row * normalizedSize;
    const m = meanData[row];
    const invS = invStdData[row];

    let sumGrad = 0;
    let sumGradOut = 0;
    for (let j = 0; j < normalizedSize; j++) {
      const dy = gData[base + j];
      const yHat = (xData[base + j] - m) * invS;
      sumGrad += dy;
      sumGradOut += dy * yHat;
    }
    const meanGrad = sumGrad * invNorm;
    const meanGradOut = sumGradOut * invNorm;

    for (let j = 0; j < normalizedSize; j++) {
      const dy = gData[base + j];
      const yHat = (xData[base + j] - m) * invS;
      dxBuf[base + j] = invS * (dy - meanGrad - yHat * meanGradOut);
    }
  }

  return createNdArray(dxBuf, shape.slice());
}

// ============================================================================
// Reduction Operations
// ============================================================================

export function maxNdArray(nd: any, dim?: number | null, keepdim = false): NDArray {
  const data = nd.data;
  const off = nd.offset;
  if (dim === null || dim === undefined) {
    let maxVal = -Infinity;
    for (let i = 0; i < data.length; i++) {
      const v = data[off + i];
      if (v > maxVal) maxVal = v;
    }
    const outShape = keepdim ? new Array<number>(nd.shape.length).fill(1) : [];
    return createNdArray(new Float32Array([maxVal]), outShape);
  }
  const d = normalizeDim(dim, nd.shape.length);
  const { outerSize, dimSize, innerSize } = getAxisStrides(nd.shape, d);
  const outBuf = new Float32Array(outerSize * innerSize).fill(-Infinity);
  const outerStride = dimSize * innerSize;

  for (let outer = 0; outer < outerSize; outer++) {
    const outBase = outer * innerSize;
    const inOuterBase = off + outer * outerStride;
    for (let k = 0; k < dimSize; k++) {
      const rowBase = inOuterBase + k * innerSize;
      for (let inner = 0; inner < innerSize; inner++) {
        const v = data[rowBase + inner];
        if (v > outBuf[outBase + inner]) outBuf[outBase + inner] = v;
      }
    }
  }

  if (keepdim) {
    const keptShape = nd.shape.slice();
    keptShape[d] = 1;
    return createNdArray(outBuf, keptShape);
  }
  if (nd.shape.length === 1) return createNdArray(outBuf, []);
  return createNdArray(outBuf, [...nd.shape.slice(0, d), ...nd.shape.slice(d + 1)]);
}

export function minNdArray(nd: any, dim?: number | null, keepdim = false): NDArray {
  const data = nd.data;
  const off = nd.offset;
  if (dim === null || dim === undefined) {
    let minVal = Infinity;
    for (let i = 0; i < data.length; i++) {
      const v = data[off + i];
      if (v < minVal) minVal = v;
    }
    const outShape = keepdim ? new Array<number>(nd.shape.length).fill(1) : [];
    return createNdArray(new Float32Array([minVal]), outShape);
  }
  const d = normalizeDim(dim, nd.shape.length);
  const { outerSize, dimSize, innerSize } = getAxisStrides(nd.shape, d);
  const outBuf = new Float32Array(outerSize * innerSize).fill(Infinity);
  const outerStride = dimSize * innerSize;

  for (let outer = 0; outer < outerSize; outer++) {
    const outBase = outer * innerSize;
    const inOuterBase = off + outer * outerStride;
    for (let k = 0; k < dimSize; k++) {
      const rowBase = inOuterBase + k * innerSize;
      for (let inner = 0; inner < innerSize; inner++) {
        const v = data[rowBase + inner];
        if (v < outBuf[outBase + inner]) outBuf[outBase + inner] = v;
      }
    }
  }

  if (keepdim) {
    const keptShape = nd.shape.slice();
    keptShape[d] = 1;
    return createNdArray(outBuf, keptShape);
  }
  if (nd.shape.length === 1) return createNdArray(outBuf, []);
  return createNdArray(outBuf, [...nd.shape.slice(0, d), ...nd.shape.slice(d + 1)]);
}

export function meanNdArray(nd: any, dim?: number | null, keepdim = false): NDArray {
  const data = nd.data;
  const off = nd.offset;
  if (dim === null || dim === undefined) {
    let total = 0;
    for (let i = 0; i < data.length; i++) {
      total += data[off + i];
    }
    const outShape = keepdim ? new Array<number>(nd.shape.length).fill(1) : [];
    return createNdArray(new Float32Array([total / data.length]), outShape);
  }
  const d = normalizeDim(dim, nd.shape.length);
  const result = sumAlongDimNdArray(nd, d, keepdim);
  const invSize = 1.0 / nd.shape[d];
  const rData = result.data;
  for (let i = 0; i < rData.length; i++) {
    rData[i] *= invSize;
  }
  return result;
}

// ============================================================================
// Transpose and BMM
// ============================================================================

export function transposeNdArray(nd: any, dim1: number, dim2: number): NDArray {
  const rank = nd.shape.length;
  const d1 = normalizeDim(dim1, rank);
  const d2 = normalizeDim(dim2, rank);
  if (d1 === d2) return cloneNdArray(nd);

  const outShape: number[] = nd.shape.slice();
  outShape[d1] = nd.shape[d2];
  outShape[d2] = nd.shape[d1];
  const outSize = getShapeSize(outShape);
  const outBuf = new Float32Array(outSize);
  const srcData: Float32Array = nd.data;
  const srcOff: number = nd.offset;

  // Fast path for 2D transpose
  if (rank === 2) {
    const M = nd.shape[0];
    const N = nd.shape[1];
    for (let i = 0; i < M; i++) {
      const rowBase = srcOff + i * N;
      for (let j = 0; j < N; j++) {
        outBuf[j * M + i] = srcData[rowBase + j];
      }
    }
    return createNdArray(outBuf, outShape);
  }

  const permutedSrcStrides = nd.strides.slice();
  permutedSrcStrides[d1] = nd.strides[d2];
  permutedSrcStrides[d2] = nd.strides[d1];
  const outStrides = computeRowMajorStrides(outShape);
  const lastDim = outShape[rank - 1];
  const lastSrcStride = permutedSrcStrides[rank - 1];

  function copyTranspose(dim: number, srcBase: number, outBase: number): void {
    if (dim === rank - 1) {
      for (let i = 0; i < lastDim; i++) {
        outBuf[outBase + i] = srcData[srcBase + i * lastSrcStride];
      }
      return;
    }
    const len = outShape[dim];
    const sStride = permutedSrcStrides[dim];
    const oStride = outStrides[dim];
    for (let i = 0; i < len; i++) {
      copyTranspose(dim + 1, srcBase + i * sStride, outBase + i * oStride);
    }
  }

  copyTranspose(0, srcOff, 0);
  return createNdArray(outBuf, outShape);
}

export function bmmNdArray(a: any, b: any): NDArray {
  const shapeA = a.shape;
  const shapeB = b.shape;
  if (shapeA.length !== 3 || shapeB.length !== 3) {
    throw new Error('BMM expects 3D inputs [B, M, K] and [B, K, N]');
  }
  const B = shapeA[0];
  const M = shapeA[1];
  const K = shapeA[2];
  const N = shapeB[2];
  if (shapeB[0] !== B || shapeB[1] !== K) {
    throw new Error(`BMM shape mismatch: ${JSON.stringify(shapeA)} vs ${JSON.stringify(shapeB)}`);
  }
  const outBuf = new Float32Array(B * M * N);
  const aData = a.data;
  const bData = b.data;
  const aOff = a.offset;
  const bOff = b.offset;
  const mk = M * K;
  const kn = K * N;
  const mn = M * N;

  for (let batch = 0; batch < B; batch++) {
    const aBatch = aOff + batch * mk;
    const bBatch = bOff + batch * kn;
    const outBatch = batch * mn;
    for (let i = 0; i < M; i++) {
      const aRow = aBatch + i * K;
      const outRow = outBatch + i * N;
      for (let k = 0; k < K; k++) {
        const aik = aData[aRow + k];
        const bRow = bBatch + k * N;
        for (let j = 0; j < N; j++) {
          outBuf[outRow + j] += aik * bData[bRow + j];
        }
      }
    }
  }
  return createNdArray(outBuf, [B, M, N]);
}

// ============================================================================
// Dropout
// ============================================================================

export function dropoutNdArray(nd: any, p: number, isTraining: boolean): { out: NDArray; mask: Float32Array } {
  const size = nd.data.length;
  const mask = new Float32Array(size).fill(1);
  if (!isTraining || p <= 0) {
    return { out: cloneNdArray(nd), mask };
  }
  if (p >= 1) {
    mask.fill(0);
    return { out: createNdArray(new Float32Array(size), nd.shape.slice()), mask };
  }
  const outBuf = new Float32Array(size);
  const scale = 1.0 / (1.0 - p);
  const data = nd.data;
  for (let i = 0; i < size; i++) {
    const keep = Math.random() >= p ? 1 : 0;
    mask[i] = keep;
    outBuf[i] = data[i] * keep * scale;
  }
  return { out: createNdArray(outBuf, nd.shape.slice()), mask };
}

// ============================================================================
// Scaled Dot Product Attention
// ============================================================================

function transposeLastTwoNdArray(nd: any): NDArray {
  const rank = nd.shape.length;
  if (rank < 2) throw new Error('transposeLastTwoNdArray requires at least 2D tensor');
  return transposeNdArray(nd, rank - 2, rank - 1);
}

function applyCausalMaskInPlace(scores: NDArray): void {
  const shape = scores.shape;
  const rank = shape.length;
  const Lq = shape[rank - 2];
  const Lk = shape[rank - 1];
  const outerSize = scores.data.length / (Lq * Lk);
  const buf = scores.data;
  for (let outer = 0; outer < outerSize; outer++) {
    const base = outer * Lq * Lk;
    for (let i = 0; i < Lq; i++) {
      const rowBase = base + i * Lk;
      for (let j = i + 1; j < Lk; j++) {
        buf[rowBase + j] = -Infinity;
      }
    }
  }
}

function flattenBatchHeadNdArray(nd: any): NDArray {
  const shape: number[] = nd.shape;
  if (shape.length < 4) throw new Error('flattenBatchHeadNdArray requires 4D tensor');
  return createNdArray(nd.data, [shape[0] * shape[1], ...shape.slice(2)]);
}

function unflattenBatchHeadNdArray(nd: any, B: number, H: number): NDArray {
  const shape: number[] = nd.shape;
  if (shape.length < 3 || shape[0] !== B * H) {
    throw new Error('unflattenBatchHeadNdArray: first dim must equal B*H');
  }
  return createNdArray(nd.data, [B, H, ...shape.slice(1)]);
}

export function scaledDotProductAttentionNdArray(
  q: any,
  k: any,
  v: any,
  isTraining: boolean,
  dropoutP: number,
  isCausal: boolean,
  scale?: number,
  attnMask?: any
): { out: NDArray; scores: NDArray; attn: NDArray; dropoutMask: Float32Array | null; scale: number } {
  const qShape: number[] = q.shape;
  const kShape: number[] = k.shape;
  const vShape: number[] = v.shape;
  if (qShape.length !== 4 || kShape.length !== 4 || vShape.length !== 4) {
    throw new Error('Scaled dot product attention expects 4D inputs [B, H, L, D]');
  }
  const B = qShape[0];
  const H = qShape[1];
  const Lq = qShape[2];
  const Dq = qShape[3];
  const Lk = kShape[2];
  const Dk = kShape[3];
  if (B !== kShape[0] || B !== vShape[0] || H !== kShape[1] || H !== vShape[1]) {
    throw new Error('Scaled dot product attention batch/head mismatch');
  }
  if (Lk !== vShape[2]) {
    throw new Error('Scaled dot product attention key/value sequence length mismatch');
  }
  if (Dk !== Dq) {
    throw new Error('Scaled dot product attention query/key head dimension mismatch');
  }

  const effScale = scale ?? 1.0 / Math.sqrt(Dq);

  const q3d = flattenBatchHeadNdArray(q);
  const k3d = flattenBatchHeadNdArray(k);
  const v3d = flattenBatchHeadNdArray(v);

  const kt3d = transposeLastTwoNdArray(k3d);
  let scores3d = bmmNdArray(q3d, kt3d);
  const sData = scores3d.data;
  for (let i = 0; i < sData.length; i++) sData[i] *= effScale;

  if (attnMask) {
    let mask3d = attnMask;
    if (mask3d.shape.length === 4) {
      mask3d = flattenBatchHeadNdArray(mask3d);
    }
    mask3d = broadcastNdArray(mask3d, [B * H, Lq, Lk]);
    scores3d = addNdArray(scores3d, mask3d);
  }

  if (isCausal) {
    applyCausalMaskInPlace(scores3d);
  }

  const attn3d = softmaxNdArray(scores3d, -1);

  let dropoutMask: Float32Array | null = null;
  let attn3dDropped = attn3d;
  if (isTraining && dropoutP > 0) {
    const { out, mask } = dropoutNdArray(attn3d, dropoutP, true);
    attn3dDropped = out;
    dropoutMask = mask;
  }

  const output3d = bmmNdArray(attn3dDropped, v3d);

  return {
    out: unflattenBatchHeadNdArray(output3d, B, H),
    scores: unflattenBatchHeadNdArray(scores3d, B, H),
    attn: unflattenBatchHeadNdArray(attn3dDropped, B, H),
    dropoutMask,
    scale: effScale
  };
}

export function scaledDotProductAttentionGradNdArray(
  gradOutput: any,
  q: any,
  k: any,
  v: any,
  attn: any,
  dropoutMask: Float32Array | null,
  scale: number,
  isTraining: boolean,
  dropoutP: number
): { dQ: NDArray; dK: NDArray; dV: NDArray } {
  const B = q.shape[0];
  const H = q.shape[1];

  const dO3d = flattenBatchHeadNdArray(gradOutput);
  const v3d = flattenBatchHeadNdArray(v);
  const k3d = flattenBatchHeadNdArray(k);
  const q3d = flattenBatchHeadNdArray(q);
  const attn3d = flattenBatchHeadNdArray(attn);

  const dV3d = bmmNdArray(transposeLastTwoNdArray(attn3d), dO3d);

  const dAttn3d = bmmNdArray(dO3d, transposeLastTwoNdArray(v3d));
  if (isTraining && dropoutP > 0 && dropoutMask) {
    const dropoutScale = 1.0 / (1.0 - dropoutP);
    const dAttnData = dAttn3d.data;
    for (let i = 0; i < dAttnData.length; i++) {
      dAttnData[i] *= dropoutMask[i] * dropoutScale;
    }
  }

  const dScores3d = softmaxGradNdArray(dAttn3d, attn3d, -1);
  const dScoresData = dScores3d.data;
  for (let i = 0; i < dScoresData.length; i++) {
    dScoresData[i] *= scale;
  }

  const dQ3d = bmmNdArray(dScores3d, k3d);
  const dK3d = bmmNdArray(transposeLastTwoNdArray(dScores3d), q3d);

  return {
    dQ: unflattenBatchHeadNdArray(dQ3d, B, H),
    dK: unflattenBatchHeadNdArray(dK3d, B, H),
    dV: unflattenBatchHeadNdArray(dV3d, B, H)
  };
}

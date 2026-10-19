import {
  TensorNode,
  getAllLeafParameters,
  getTensorByName,
  nextId,
  registerTensor
} from './tensor.js';
import { createNdArray, arraysEqual, NDArray } from './ndarray-utils.js';

export interface SerializedTensor {
  name: string;
  id: number;
  shape: number[];
  requires_grad: boolean;
  data: any;
}

export interface StateDict {
  framework: string;
  tensors: SerializedTensor[];
}

interface PreparedTensor {
  name: string;
  requiresGrad: boolean;
  data: NDArray;
  existing: TensorNode | undefined;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/** Validate against the declared shape, including dimensions after an empty axis. */
function validateData(data: unknown, shape: number[], location: string): number[] {
  const values: number[] = [];
  function visit(value: unknown, depth: number, path: string): void {
    if (depth < shape.length) {
      if (!Array.isArray(value) || value.length !== shape[depth]) {
        throw new Error(`Shape mismatch at ${path}: expected an array of length ${shape[depth]}`);
      }
      for (let i = 0; i < value.length; i++) {
        visit(value[i], depth + 1, `${path}[${i}]`);
      }
      return;
    }
    if (Array.isArray(value)) {
      throw new Error(`Shape mismatch at ${path}: expected a numeric scalar`);
    }
    if (typeof value !== 'number' || !Number.isFinite(value)) {
      throw new Error(`Invalid StateDict value at ${path}: expected a finite number`);
    }
    if (!Number.isFinite(Math.fround(value))) {
      throw new Error(`Invalid StateDict value at ${path}: number is outside the finite float32 range`);
    }
    values.push(value);
  }
  visit(data, 0, location);
  return values;
}

function validateRecord(item: unknown, index: number): {
  name: string;
  shape: number[];
  requiresGrad: boolean;
  values: number[];
} {
  const entry = `tensors[${index}]`;
  if (!isRecord(item)) {
    throw new Error(`Invalid StateDict format at ${entry}: expected a tensor object`);
  }
  if (typeof item.name !== 'string' || item.name.length === 0) {
    throw new Error(`Invalid StateDict format at ${entry}.name: expected a nonempty string`);
  }
  const location = `tensor ${JSON.stringify(item.name)} at ${entry}`;
  // IDs are optional metadata; runtime identity is always resolved by name.
  if ('id' in item && !Number.isSafeInteger(item.id)) {
    throw new Error(`Invalid StateDict format for ${location}.id: expected a safe integer`);
  }
  if (typeof item.requires_grad !== 'boolean') {
    throw new Error(`Invalid StateDict format for ${location}.requires_grad: expected a boolean`);
  }
  if (!Array.isArray(item.shape) || item.shape.some(dim => !Number.isSafeInteger(dim) || dim < 0)) {
    throw new Error(`Invalid StateDict format for ${location}.shape: expected nonnegative safe integer dimensions`);
  }
  const shape: number[] = item.shape;
  const values = validateData(item.data, shape, `${location}.data`);
  return { name: item.name, shape, requiresGrad: item.requires_grad, values };
}

/**
 * [获取当前模型权重字典 JSON 字符串]
 * Exports all leaf model parameters to StateDict JSON string.
 * Rejects non-finite values instead of allowing JSON to replace them with null.
 */
export function exportStateDict(): string {
  const leaves = getAllLeafParameters();
  const serializedList: SerializedTensor[] = leaves.map((tensor, index) => {
    const item = {
      name: tensor.name,
      id: tensor.id,
      shape: tensor.shape,
      requires_grad: tensor.requiresGrad,
      data: tensor.toArray()
    };
    validateRecord(item, index);
    return item;
  });

  const stateDict: StateDict = {
    framework: 'ScratchTensor',
    tensors: serializedList
  };

  return JSON.stringify(stateDict, null, 2);
}

/**
 * [从 StateDict 文本 <JSON文本字符串> 导入并恢复模型权重]
 * Validates and prepares the entire checkpoint before applying any changes.
 * Existing objects keep their ID, requiresGrad flag, and optimizer state.
 * Unknown fields are ignored; framework and ID metadata may be omitted.
 */
export function loadStateDict(jsonStr: string): void {
  if (typeof jsonStr !== 'string') {
    throw new Error('Invalid StateDict format: expected JSON text');
  }
  const stateDict: unknown = JSON.parse(jsonStr);
  if (!isRecord(stateDict) || !Array.isArray(stateDict.tensors)) {
    throw new Error('Invalid StateDict format: missing tensors array');
  }
  if ('framework' in stateDict && stateDict.framework !== 'ScratchTensor') {
    throw new Error('Invalid StateDict framework: expected "ScratchTensor"');
  }

  const names = new Set<string>();
  const prepared: PreparedTensor[] = stateDict.tensors.map((item, index) => {
    const record = validateRecord(item, index);
    if (names.has(record.name)) {
      throw new Error(`Invalid StateDict format at tensors[${index}].name: duplicate tensor name ${JSON.stringify(record.name)}`);
    }
    names.add(record.name);
    const existing = getTensorByName(record.name);
    if (existing && !arraysEqual(existing.shape, record.shape)) {
      throw new Error(
        `Shape mismatch for tensor ${JSON.stringify(record.name)}: existing ${JSON.stringify(existing.shape)} vs checkpoint ${JSON.stringify(record.shape)}`
      );
    }
    return {
      name: record.name,
      requiresGrad: record.requiresGrad,
      data: createNdArray(new Float32Array(record.values), record.shape),
      existing
    };
  });

  for (const item of prepared) {
    if (item.existing) {
      item.existing.data = item.data;
      item.existing.clearGrad();
    } else {
      registerTensor(new TensorNode(nextId(), item.name, item.data, item.requiresGrad));
    }
  }
}

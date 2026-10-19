import { expect } from 'vitest';

/**
 * Flatten nested array, TypedArray, or scalar into a 1D number array.
 */
export function flattenArray(data: any): number[] {
  if (data === null || data === undefined) return [];
  if (typeof data === 'number') return [data];
  if (Array.isArray(data)) {
    const result: number[] = [];
    for (const item of data) {
      result.push(...flattenArray(item));
    }
    return result;
  }
  if (data instanceof Float64Array || data instanceof Float32Array || data.length !== undefined) {
    return Array.from(data);
  }
  return [];
}

/**
 * Standard numerical precision comparison between actual and expected arrays/values.
 * Uses both absolute tolerance (atol) and relative tolerance (rtol):
 *   |actual - expected| <= atol + rtol * |expected|
 *
 * @param actual - The actual computed values (nested array, TypedArray, or scalar)
 * @param expected - The expected reference values
 * @param atol - Absolute tolerance (default: 1e-5)
 * @param rtol - Relative tolerance (default: 1e-5)
 * @param label - Context label for error messages
 */
export function expectClose(
  actual: any,
  expected: any,
  atol = 1e-5,
  rtol = 1e-5,
  label = ''
): void {
  const actualFlat = flattenArray(actual);
  const expectedFlat = flattenArray(expected);

  expect(actualFlat.length, `Length mismatch ${label}`).toBe(expectedFlat.length);

  for (let i = 0; i < actualFlat.length; i++) {
    const a = actualFlat[i];
    const e = expectedFlat[i];
    const diff = Math.abs(a - e);
    const tol = atol + rtol * Math.abs(e);
    if (diff > tol) {
      throw new Error(
        `Numerical mismatch at index ${i} ${label}: actual=${a}, expected=${e}, diff=${diff}, allowed_tol=${tol}`
      );
    }
  }
}

/**
 * Verify that two shapes are identical.
 */
export function expectShapeEqual(actual: number[], expected: number[], label = ''): void {
  expect(actual, `Shape mismatch ${label}`).toEqual(expected);
}

import { beforeEach, afterEach } from 'vitest';
import { clearTensorRegistry, resetIdCounter, setGraphTracking } from '../../../src/index.js';

/**
 * Standard verification test lifecycle.
 * Call once at the top of each `describe()` block to automatically register
 * beforeEach/afterEach hooks that:
 *   1. Clear the global tensor registry
 *   2. Reset ID counter to ensure deterministic IDs
 *   3. Enable graph tracking (default training mode)
 */
export function useVerificationLifecycle(): void {
  beforeEach(() => {
    clearTensorRegistry();
    resetIdCounter();
    setGraphTracking(true);
  });

  afterEach(() => {
    clearTensorRegistry();
  });
}

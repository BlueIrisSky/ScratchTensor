/**
 * ScratchTensor Unified Verification Test Harness
 *
 * Provides a complete toolkit for writing standardized verification tests
 * that compare ScratchTensor operations against TensorFlow.js reference
 * implementations. Import everything from this single entry point.
 *
 * @example
 * ```typescript
 * import {
 *   useVerificationLifecycle,
 *   expectClose,
 *   withTfReference,
 *   verifyEagerGraphCleanup
 * } from './harness/index.js';
 * ```
 */

// Test lifecycle management
export { useVerificationLifecycle } from './test-lifecycle.js';

// Numerical precision assertions
export { flattenArray, expectClose, expectShapeEqual } from './numerical-assert.js';

// TF.js reference computation with auto-cleanup
export { withTfReference, computeTfGrads, disposeTfGrads } from './tf-reference.js';
export type { TfContext } from './tf-reference.js';

// Eager Graph Cleanup verification
export { verifyEagerGraphCleanup } from './graph-cleanup.js';

import { expect } from 'vitest';
import { TensorNode } from '../../../src/index.js';

/**
 * Verify that after backward() execution, all intermediate (non-leaf) nodes
 * have their `creatorOp` cleared to null, confirming Eager Graph Cleanup
 * as specified in docs/architecture.md Section 4.
 *
 * @param intermediateNodes - Array of TensorNodes expected to be cleaned up
 */
export function verifyEagerGraphCleanup(intermediateNodes: TensorNode[]): void {
  for (const node of intermediateNodes) {
    if (!node.isLeaf) {
      expect(
        node.creatorOp,
        `Eager Graph Cleanup failed: Node "${node.name}" (id=${node.id}) should have creatorOp reset to null`
      ).toBeNull();
    }
  }
}

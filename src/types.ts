/**
 * OpNode Interface representing an operation in the computational graph.
 */
export interface OpNode {
  opName: string;                     // Operator name
  inputs: any[];                      // Array of TensorNode inputs
  output: any;                        // TensorNode output
  backwardFn(gradOutput: any): void;  // Chain rule backward propagation callback
}

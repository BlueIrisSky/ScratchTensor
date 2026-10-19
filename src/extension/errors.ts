interface ScratchBlockUtility {
  target?: unknown;
  thread?: unknown;
  runtime?: { emit?: (event: string, ...args: unknown[]) => unknown };
  sequencer?: { retireThread?: (thread: unknown) => void };
  stopThisScript?: () => void;
}

type BlockMethod = (args: any, util?: ScratchBlockUtility) => unknown;

/** Add consistent diagnostics and stop the failing Scratch thread for every block. */
export function installBlockErrorHandlers(extension: object, opcodes: readonly string[]): void {
  for (const opcode of opcodes) {
    const method = (extension as Record<string, BlockMethod>)[opcode];
    Object.defineProperty(extension, opcode, {
      configurable: true,
      writable: true,
      value: (args: any, util?: ScratchBlockUtility) => {
        try {
          return method.call(extension, args, util);
        } catch (cause) {
          const reason = cause instanceof Error ? cause.message : String(cause);
          const error = new Error(`ScratchTensor.${opcode}: ${reason}`, { cause });
          try {
            if (util?.target) util.runtime?.emit?.('SAY', util.target, 'say', error.message);
          } catch {
            // A renderer failure must not hide the original tensor error.
          } finally {
            // stopThisScript only returns from a custom block. Retire the whole
            // thread so its caller cannot continue using a previous output.
            if (util?.thread && util.sequencer?.retireThread) {
              util.sequencer.retireThread(util.thread);
            } else {
              util?.stopThisScript?.();
            }
          }
          throw error;
        }
      }
    });
  }
}

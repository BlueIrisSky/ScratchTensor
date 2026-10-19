import { describe, it, expect, vi, afterEach } from 'vitest';

describe('ScratchTensorExtension auto-registration', () => {
  afterEach(() => {
    delete (globalThis as any).Scratch;
    vi.resetModules();
  });

  it('registers itself when the global Scratch runtime is present', async () => {
    const register = vi.fn();
    (globalThis as any).Scratch = { extensions: { register } };

    await import('../src/scratch-extension.js');

    expect(register).toHaveBeenCalledTimes(1);
    const instance = register.mock.calls[0][0];
    expect(typeof instance.getInfo).toBe('function');
    expect(typeof instance.createTensorBlock).toBe('function');
    expect(typeof instance.binaryOpBlock).toBe('function');
    expect(instance.getInfo().id).toBe('ScratchTensor');
    expect(() => instance.backwardBlock({ TENSOR: 'missing' }))
      .toThrow(/ScratchTensor\.backwardBlock:.*missing.*does not exist/);
  });

  it('does not throw when the global Scratch runtime is absent', async () => {
    await expect(import('../src/scratch-extension.js')).resolves.toBeDefined();
  });
});

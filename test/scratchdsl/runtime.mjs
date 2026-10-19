import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFile, writeFile} from 'node:fs/promises';
import {createRequire} from 'node:module';
import path from 'node:path';
import vmModule from 'node:vm';

const require = createRequire(import.meta.url);
const embeddedPrefix = 'data:application/javascript;charset=utf-8;base64,';
const sleep = milliseconds => new Promise(resolve => setTimeout(resolve, milliseconds));
const plain = value => JSON.parse(JSON.stringify(value));
const isRetiredThread = thread => !!thread &&
  thread.status === thread.constructor.STATUS_DONE && thread.stack.length === 0;
let previousRun = Promise.resolve();

function evaluateExtension(source, register, filename, output = console) {
  // This is a host for the checked-in, trusted bundle, not a security boundary
  // for arbitrary JavaScript. No browser, Node imports, or network APIs are given
  // to the extension; its operators and registry execute without replacements.
  const context = vmModule.createContext({
    Scratch: {extensions: {unsandboxed: true, register}},
    console: output,
  }, {codeGeneration: {strings: false, wasm: false}});
  new vmModule.Script(source, {filename}).runInContext(context, {timeout: 2000});
  return context;
}

/** Read the public catalogue by actually evaluating the current bundled JS. */
export async function readMetadata(pathOrSource) {
  const source = Buffer.isBuffer(pathOrSource) ? pathOrSource.toString('utf8')
    : typeof pathOrSource === 'string' && /[\r\n]|Scratch\.extensions/u.test(pathOrSource)
      ? pathOrSource : await readFile(pathOrSource, 'utf8');
  const registered = [];
  evaluateExtension(source, extension => registered.push(extension), 'extension-metadata.js');
  assert.equal(registered.length, 1, 'the bundle must register exactly one extension');
  const metadata = plain(registered[0].getInfo());
  assert.equal(typeof metadata.id, 'string');
  assert(Array.isArray(metadata.blocks), 'getInfo() must return a block catalogue');
  return metadata;
}

function targetData(target) {
  const variables = {};
  const lists = {};
  for (const variable of Object.values(target.variables)) {
    if (variable.type === 'list') lists[variable.name] = structuredClone(variable.value);
    else if (variable.type === '') variables[variable.name] = structuredClone(variable.value);
  }
  return {
    name: target.getName(), isStage: target.isStage, variables, lists,
    ...(!target.isStage ? {x: target.x, y: target.y, visible: target.visible} : {}),
  };
}

function opcodeCounts(target) {
  const counts = {};
  for (const block of Object.values(target.blocks || {})) {
    if (block && !Array.isArray(block)) counts[block.opcode] = (counts[block.opcode] || 0) + 1;
  }
  return counts;
}

function assertSerialization(before, after) {
  assert.equal(after.targets.length, before.targets.length, 'VM target count changed');
  for (const original of before.targets) {
    const target = after.targets.find(item => item.name === original.name && item.isStage === original.isStage);
    assert(target, `VM lost target ${original.name}`);
    assert.deepEqual(opcodeCounts(target), opcodeCounts(original), `VM changed ${original.name} opcodes`);
    for (const kind of ['costumes', 'sounds']) {
      assert.deepEqual((target[kind] || []).map(asset => asset.md5ext),
        (original[kind] || []).map(asset => asset.md5ext), `VM changed ${kind} asset references`);
    }
  }
  assert.deepEqual(after.extensionURLs, before.extensionURLs, 'VM changed embedded extension URLs');
}

async function bounded(promise, timeoutMs, message) {
  let timer;
  try {
    return await Promise.race([promise, new Promise((_, reject) => {
      timer = setTimeout(() => reject(new Error(message)), timeoutMs);
    })]);
  } finally {
    clearTimeout(timer);
  }
}

async function executeProject(compiled, {
  timeoutMs = 10000, allowExtensionErrors = false, continueAfterExtensionErrors = false,
} = {}) {
  const vmPath = process.env.SCRATCH_VM_PATH;
  const vmRequire = vmPath ? createRequire(path.join(path.resolve(vmPath), 'package.json')) : require;
  const VirtualMachine = vmPath ? vmRequire('./src/index.js') : require('scratch-vm');
  // Resolve JSZip beside the VM, including when the VM is a separate pinned checkout.
  const resolveRequire = vmPath ? vmRequire : createRequire(require.resolve('scratch-vm'));
  const JSZip = resolveRequire('@turbowarp/jszip');
  const bytes = compiled.sb3 || await readFile(compiled.sb3Path);
  const zip = await JSZip.loadAsync(bytes);
  assert(zip.file('project.json'), 'the compiler output has no project.json');
  const project = JSON.parse(await zip.file('project.json').async('string'));
  const urls = new Set(Object.values(project.extensionURLs || {}));
  assert(urls.size > 0, 'this runtime test must embed an extension');
  for (const url of urls) assert(url.startsWith(embeddedPrefix), 'only embedded data-URL extensions may load');
  const assets = [];
  for (const target of project.targets) {
    for (const kind of ['costumes', 'sounds']) {
      for (const asset of target[kind] || []) {
        assert(zip.file(asset.md5ext), `missing packaged asset ${asset.md5ext}`);
        const data = await zip.file(asset.md5ext).async('nodebuffer');
        assert.equal(createHash('md5').update(data).digest('hex'), asset.assetId, 'asset hash mismatch');
        assets.push({target: target.name, kind, name: asset.name, md5ext: asset.md5ext});
      }
    }
  }

  const extensionCalls = {};
  const extensionErrors = [];
  const thrownExtensionErrors = new Set();
  const extensionErrorThreads = new Map();
  const handledExtensionThreads = new Set();
  const executionErrors = [];
  const extensionLogs = [];
  const registered = new Set();
  const spoken = [];
  const runtime = new VirtualMachine();
  runtime.setCompilerOptions({enabled: false});
  runtime.setTurboMode(true);
  runtime.securityManager.getSandboxMode = () => 'unsandboxed';
  runtime.securityManager.canLoadExtensionFromProject = url => urls.has(url);
  runtime.securityManager.canFetch = () => false;
  runtime.runtime.on('SAY', (_target, _type, message) => spoken.push(String(message)));
  const saved = new Map(['document', 'Scratch', 'ScratchExtensions'].map(name => [name,
    Object.getOwnPropertyDescriptor(globalThis, name)]));
  const began = Date.now();
  let outcome;
  let failure;
  globalThis.document = {
    createElement(tag) {
      assert.equal(tag, 'script', `unexpected browser operation ${tag}`);
      return {};
    },
    body: {appendChild(script) {
      assert(urls.has(script.src), 'VM attempted to load a script outside the project');
      const register = globalThis.Scratch.extensions.register;
      let context;
      // Registration runs synchronously; method dispatch begins after evaluation.
      context = evaluateExtension(Buffer.from(script.src.slice(embeddedPrefix.length), 'base64').toString('utf8'), extension => {
        const metadata = extension.getInfo();
        assert(!registered.has(metadata.id), `extension ${metadata.id} registered twice`);
        registered.add(metadata.id);
        const methods = new Set(metadata.blocks.filter(block => block && typeof block === 'object')
          .map(block => block.func || block.opcode).filter(Boolean));
        const invoke = new vmModule.Script('__invocation.method.apply(__invocation.self, __invocation.args)');
        for (const method of methods) {
          const original = extension[method];
          if (typeof original !== 'function') continue;
          extension[method] = function (...args) {
            const key = `${metadata.id}.${method}`;
            extensionCalls[key] = (extensionCalls[key] || 0) + 1;
            const record = error => {
              thrownExtensionErrors.add(error);
              const thread = args[1]?.thread;
              extensionErrorThreads.set(error, thread);
              extensionErrors.push({method: key, message: String(error?.message || error), args: plain(args[0] || {}),
                threadRetired: isRetiredThread(thread)});
              throw error;
            };
            context.__invocation = {method: original, self: this, args};
            try {
              const value = invoke.runInContext(context, {timeout: 2000});
              return value && typeof value.then === 'function' ? value.catch(record) : value;
            } catch (error) {
              return record(error);
            } finally {
              delete context.__invocation;
            }
          };
        }
        register(extension);
      }, 'project-embedded-extension.js', Object.fromEntries(['log', 'info', 'warn', 'error'].map(level => [level,
        (...args) => extensionLogs.push({level, message: args.map(String).join(' ')})])));
      return script;
    }},
  };
  try {
    await bounded(runtime.loadProject(bytes), timeoutMs, 'VM project loading timed out');
    for (const id of Object.keys(project.extensionURLs)) {
      assert(registered.has(id), `official VM loader did not register ${id}`);
      assert(runtime.extensionManager.isExtensionLoaded(id), `VM did not finish registering ${id}`);
    }
    assertSerialization(project, JSON.parse(runtime.toJSON()));
    runtime.greenFlag();
    const deadline = Date.now() + timeoutMs;
    let aborted = false;
    let executionFailure;
    let firstStep = true;
    // Use the VM's real scheduler without its browser interval. A synchronous
    // primitive failure can then be observed here instead of escaping as an
    // uncaught timer exception and retrying the same failed block indefinitely.
    while ((firstStep || runtime.runtime.threads.length) && Date.now() < deadline) {
      firstStep = false;
      try {
        runtime.runtime._step();
      } catch (error) {
        executionFailure = error;
        executionErrors.push({message: String(error?.message || error),
          fromExtension: thrownExtensionErrors.has(error)});
        if (continueAfterExtensionErrors && thrownExtensionErrors.has(error)) {
          // The extension must have retired the complete failing thread itself.
          // Never retry an active block or suppress an unrelated VM exception.
          const thread = extensionErrorThreads.get(error);
          assert(isRetiredThread(thread),
            'extension failure did not retire its complete thread');
          assert(!handledExtensionThreads.has(thread), 'VM retried a retired extension thread');
          handledExtensionThreads.add(thread);
          continue;
        }
        aborted = true;
        runtime.stopAll();
        break;
      }
      // Promise-returning primitives and timed Scratch blocks still get real
      // event-loop turns. No tensor operation or scheduling rule is replaced.
      await sleep(1);
      if (extensionErrors.length && !continueAfterExtensionErrors) {
        aborted = true;
        runtime.stopAll();
        break;
      }
    }
    const targets = runtime.runtime.targets.filter(target => target.isOriginal).map(targetData);
    const stage = targets.find(target => target.isStage);
    assert(stage, 'VM has no stage');
    outcome = {
      variables: stage.variables, lists: stage.lists, targets, extensionCalls, extensionErrors, executionErrors, extensionLogs, spoken,
      registeredExtensions: [...registered], project, assets, elapsedMs: Date.now() - began,
      aborted, greenFlagFinished: !aborted && runtime.runtime.threads.length === 0,
    };
    // An expected extension failure never permits unrelated scheduler errors.
    if (executionErrors.length && !thrownExtensionErrors.has(executionFailure)) throw executionFailure;
    if (!allowExtensionErrors && !continueAfterExtensionErrors) {
      assert.deepEqual(extensionErrors, [], 'extension block threw during execution');
    }
    if (continueAfterExtensionErrors) {
      assert(extensionErrors.every(error => error.threadRetired), 'an extension failure left an active thread');
    }
    assert(outcome.greenFlagFinished || (allowExtensionErrors && aborted && extensionErrors.length > 0),
      'green-flag scripts did not finish before the deadline');
    assert(Object.values(extensionCalls).some(count => count > 0), 'VM executed no extension blocks');
    assertSerialization(project, JSON.parse(runtime.toJSON()));
    return outcome;
  } catch (error) {
    failure = error;
    if (outcome) outcome.error = String(error?.stack || error);
    throw error;
  } finally {
    try {
      runtime.stopAll();
    } finally {
      runtime.quit();
      runtime.runtime.removeAllListeners();
      for (const [name, descriptor] of saved) {
        if (descriptor) Object.defineProperty(globalThis, name, descriptor);
        else delete globalThis[name];
      }
      await writeFile(path.join(compiled.directory, 'runtime.json'), `${JSON.stringify(outcome || {
        error: String(failure?.stack || failure), extensionCalls, extensionErrors, executionErrors, extensionLogs, spoken,
        registeredExtensions: [...registered], elapsedMs: Date.now() - began,
      }, (_key, value) => typeof value === 'number' && !Number.isFinite(value)
        ? {$number: String(value)} : value, 2)}\n`);
    }
  }
}

/** Real VM execution is serialized because its extension loader uses global APIs. */
export async function runProject(compiled, options) {
  const run = previousRun.then(() => executeProject(compiled, options));
  previousRun = run.catch(() => {});
  return run;
}

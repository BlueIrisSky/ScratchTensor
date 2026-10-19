import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { compileProject, loadFixture, extensionPath } from './compiler.mjs';
import { runProject, readMetadata } from './runtime.mjs';

const namespace = 'scratch-tensor-extension';
const q = JSON.stringify;
const extension = alias => `${q(extensionPath)}${alias ? ` as ${q(alias)}` : ''}`;
const blocks = compiled => compiled.project.targets.flatMap(target => Object.values(target.blocks));
function jsonVariable(result, name) {
  assert.equal(typeof result.variables[name], 'string', `${name} must be JSON text`);
  return JSON.parse(result.variables[name]);
}

function near(actual, expected, tolerance = 1e-6) {
  if (Array.isArray(expected)) {
    assert.ok(Array.isArray(actual), 'expected an array');
    assert.equal(actual.length, expected.length);
    actual.forEach((item, index) => near(item, expected[index], tolerance));
  } else {
    assert.equal(typeof actual, 'number');
    assert.ok(Number.isFinite(actual), `expected finite ${expected}, received ${actual}`);
    assert.ok(Math.abs(actual - expected) <= tolerance,
      `expected ${expected} ± ${tolerance}, received ${actual}`);
  }
}

async function rejectsCompilation(name, source, expected, options = {}) {
  await assert.rejects(
    compileProject({ name: `boundary-reject-${name}`, source, ...options }),
    error => {
      assert.equal(typeof error.compilerResult?.code, 'number', 'the compiler must have run');
      assert.notEqual(error.compilerResult.code, 0);
      assert.equal(error.compilerResult.timedOut, false);
      assert.match(String(error), expected);
      return true;
    },
    `${name} must be rejected by the real compiler`,
  );
}

test('bare default, Unicode, hyphen and dotted aliases retain the extension runtime identity', async () => {
  const original = await readFile(extensionPath);
  for (const [index, alias] of [undefined, '张量', '张量-tools.v2'].entries()) {
    const compiled = await compileProject({
      name: `boundary-namespace-${index}`,
      source: (await loadFixture('boundary-namespace')).replaceAll(namespace + '.', (alias ?? namespace) + '.'),
      extensionEntries: [extension(alias)],
    });
    assert.deepEqual(compiled.project.extensions, ['ScratchTensor']);
    assert.equal(blocks(compiled).filter(block => block.opcode === 'ScratchTensor_createTensorBlock').length, 1);
    const embedded = compiled.project.extensionURLs.ScratchTensor;
    assert.match(embedded, /^data:application\/javascript;charset=utf-8;base64,/);
    assert.deepEqual(Buffer.from(embedded.slice(embedded.indexOf(',') + 1), 'base64'), original);
    if (alias) assert.ok(blocks(compiled).every(block => !block.opcode.startsWith(`${alias}_`)));
  }
});

test('explicit opcodes preserve slot order and static menus, including star and slash', async () => {
  const compiled = await compileProject({
    name: 'boundary-explicit-menus',
    source: await loadFixture('boundary-explicit-menus'),
  });
  const binary = blocks(compiled).filter(block => block.opcode === 'ScratchTensor_binaryOpBlock');
  assert.deepEqual(binary.map(block => block.fields.OP[0]).sort(), ['*', '/']);
  assert.ok(binary.every(block => !Object.hasOwn(block.inputs, 'OP')),
    'acceptReporters:false menus belong in fields, not reporter shadows');
  const reduction = blocks(compiled).filter(block => block.opcode === 'ScratchTensor_reductionBlock');
  assert.deepEqual(reduction.map(block => block.fields.OP[0]), ['求和(Sum)', '求和(Sum)']);
  const result = await runProject(compiled);
  assert.deepEqual(jsonVariable(result, 'product'), [[2, 8], [18, 32]]);
  assert.deepEqual(jsonVariable(result, 'quotient'), [[2, 2], [2, 2]]);
  assert.deepEqual(jsonVariable(result, 'sum'), [6, 14]);
  assert.deepEqual(jsonVariable(result, 'quoted_sum'), [6, 14]);

  // ScratchTensor's static menus have equal labels and values. This real builtin
  // checks translated label/value mapping while the same extension is loaded.
  const translated = await compileProject({
    name: 'boundary-translated-menu',
    source: await loadFixture('boundary-translated-menu'),
  });
  assert.deepEqual(blocks(translated).filter(block => block.opcode === 'motion_goto_menu')
    .map(block => block.fields.TO[0]), ['_mouse_', '_mouse_']);
});

test('imports bind public procedures globally and private procedures locally with Boolean inputs', async () => {
  const compiled = await compileProject({
    name: 'boundary-import-procedures',
    source: await loadFixture('boundary-import-procedures'),
    files: {
      'src/common.sbdsl': await loadFixture('common'),
    },
  });
  const result = await runProject(compiled);
  assert.deepEqual(jsonVariable(result, 'global_result'), [8]);
  assert.deepEqual(jsonVariable(result, 'private_result'), [128]);
  assert.equal(Number(result.variables.factor), 3);
  assert.deepEqual(result.lists.calls, ['global']);
  const worker = result.targets.find(target => target.name === 'worker');
  assert.equal(Number(worker.variables.factor), 7);
  assert.deepEqual(worker.lists.calls, ['private']);
});

test('duplicate references to one extension deduplicate, while conflicting aliases fail', async () => {
  const compiled = await compileProject({
    name: 'boundary-duplicate-source',
    source: await loadFixture('boundary-duplicate-source'),
    extensionEntries: [extension('tensor'), extension('tensor')],
  });
  assert.deepEqual(compiled.project.extensions, ['ScratchTensor']);
  await rejectsCompilation('conflicting-aliases', await loadFixture('boundary-duplicate-source'), /命名空间|别名|不同.*名|多.*名/, {
    extensionEntries: [extension('tensor'), extension('other')],
  });
  await rejectsCompilation('old-namespace', await loadFixture('compile-cleanup'), /未知.*命名空间/, {
    extensionEntries: [extension('tensor')],
  });
});

test('unloaded namespaces, unknown opcodes and invalid slots fail before SB3 generation', async t => {
  const cases = [
    ['unloaded', 'cleanup', /未知.*命名空间/, { extensionEntries: [] }],
    ['opcode', 'opcode', /opcode|积木/],
    ['arity', 'arity', /参数|槽位/],
    ['boolean', 'boolean', /参数|类型|布尔/],
    ['shape', 'shape', /形状|类型|报告|积木/],
    ['menu', 'menu', /参数|菜单|选项/],
    ['dynamic-menu', 'dynamic-menu', /参数|菜单|类型/],
  ];
  for (const [name, fixture, expected, options] of cases) {
    await t.test(name, async () => rejectsCompilation(name, await loadFixture(`compile-${fixture}`), expected, options));
  }
});

test('extension configuration rejects every URL scheme without fetching it', async t => {
  for (const url of ['https://invalid.example/extension.js', 'http://invalid.example/extension.js',
    'file:///extension.js', 'data:application/javascript,void(0)']) {
    const scheme = url.slice(0, url.indexOf(':'));
    await t.test(scheme, async () => rejectsCompilation(`url-${scheme}`, await loadFixture('compile-wait'), /URL|本地|远程/, {
      extensionEntries: [`${q(url)} as "tensor"`],
    }));
  }
});

test('broadcast-and-wait orders cross-target batched matmul and elementwise broadcasting', async () => {
  const compiled = await compileProject({
    name: 'boundary-broadcast-batch',
    source: await loadFixture('boundary-broadcast-batch'),
  });
  const result = await runProject(compiled);
  assert.equal(Number(result.variables.ready), 1);
  assert.deepEqual(jsonVariable(result, 'product'), [[[2, 6], [6, 12]], [[10, 18], [14, 24]]]);
  assert.deepEqual(jsonVariable(result, 'biased'), [[[12, 26], [16, 32]], [[20, 38], [24, 44]]]);
});

test('large-logit softmax and causal attention stay finite and obey masked averaging', async () => {
  const compiled = await compileProject({
    name: 'boundary-softmax-attention',
    source: await loadFixture('boundary-softmax-attention'),
  });
  const result = await runProject(compiled);
  // exp(1000) would overflow. The ratio depends only on the logit difference 1.
  const p = 1 / (1 + Math.E);
  near(jsonVariable(result, 'probabilities'), [[p, 1 - p], [p, 1 - p]]);
  // Zero Q/K imply equal scores. Causal row 0 sees V[0], row 1 their mean.
  near(jsonVariable(result, 'causal'), [[[[2], [4]]]]);
  near(jsonVariable(result, 'unmasked'), [[[[4], [4]]]]);
});

test('StateDict exports only leaf parameters and restores data without replacing existing identity', async () => {
  const compiled = await compileProject({
    name: 'boundary-state-dict',
    source: await loadFixture('boundary-state-dict'),
  });
  const result = await runProject(compiled);
  const checkpoint = jsonVariable(result, 'checkpoint');
  const modified = jsonVariable(result, 'modified');
  const restored = jsonVariable(result, 'restored');
  assert.deepEqual(checkpoint.tensors.map(tensor => tensor.name), ['P']);
  assert.deepEqual(checkpoint.tensors[0].data, [1, 2]);
  assert.deepEqual(modified.tensors[0].data, [-8, -6]);
  assert.deepEqual(restored.tensors.map(tensor => tensor.name).sort(), ['P', 'newParam']);
  const parameter = restored.tensors.find(tensor => tensor.name === 'P');
  assert.equal(parameter.id, checkpoint.tensors[0].id);
  assert.deepEqual(parameter.data, [1, 2]);
  assert.equal(parameter.requires_grad, true);
  const created = restored.tensors.find(tensor => tensor.name === 'newParam');
  assert.notEqual(created.id, 900002, 'checkpoint IDs must not become runtime identities');
  assert.deepEqual(created.shape, [1]);
  assert.deepEqual(created.data, [5]);
  assert.equal(result.variables.gradient, 'null');
  assert.deepEqual(jsonVariable(result, 'frozen'), [3, 4]);
});

test('incompatible shapes produce an observed runtime failure, never a silent successful test', async () => {
  const compiled = await compileProject({
    name: 'boundary-invalid-runtime-shape',
    source: await loadFixture('boundary-invalid-runtime-shape'),
  });
  await assert.rejects(runProject(compiled), /shape|broadcast|维度|形状|扩展|extension/i);
  const result = await runProject(compiled, { allowExtensionErrors: true });
  assert.equal(result.extensionErrors.length, 1);
  assert.equal(result.extensionErrors[0].method, 'ScratchTensor.binaryOpBlock');
  assert.match(result.extensionErrors[0].message, /shape|broadcast|维度|形状/i);
  assert.equal(result.aborted, true);
  assert.equal(result.greenFlagFinished, false, 'a terminated failure is not a completed project');
});

test('non-finite scalar reports stay distinct from JSON null and a missing tensor', async () => {
  const compiled = await compileProject({
    name: 'boundary-nonfinite',
    source: await loadFixture('boundary-nonfinite'),
  });
  const result = await runProject(compiled, {allowExtensionErrors: true});
  assert.equal(result.variables.infinity_number, Infinity);
  assert.ok(Number.isNaN(result.variables.nan_number), 'NaN must not become a null or zero success');
  assert.equal(result.variables.infinity_json, 'null');
  assert.equal(result.variables.nan_json, 'null');
  assert.equal(result.variables.missing, '');
  assert.equal(result.extensionErrors.length, 1);
  assert.equal(result.extensionErrors[0].method, 'ScratchTensor.getTensorPropertyBlock');
  assert.match(result.extensionErrors[0].message, /absent.*does not exist/);
});


test('plain comparisons keep Boolean boundaries separate from menu delimiters', async () => {
  const compiled = await compileProject({
    name: 'boundary-plain-comparisons',
    source: await loadFixture('boundary-plain-comparisons'),
  });
  const result = await runProject(compiled);
  assert.equal(result.variables.passed, 6);
});

test('menus require single quotes and comparisons reject the starred spelling', async t => {
  for (const name of ['bare-menu', 'double-quoted-menu', 'unquoted-empty-menu', 'starred-less', 'starred-greater']) {
    await t.test(name, async () => rejectsCompilation(name, await loadFixture(`compile-${name}`),
      name.startsWith('starred-') ? /mismatched bracket/ : /menu values must use/i));
  }
});

test('reduction accepts all axes, rejects invalid strings, and slice excludes its end index', async () => {
  const metadata = await readMetadata(extensionPath);
  const reduction = metadata.blocks.find(block => block.opcode === 'reductionBlock');
  assert.deepEqual(reduction.arguments.DIM, {type: 'string', defaultValue: '-1'});
  const source = await loadFixture('boundary-reduction-slice');
  const result = await runProject(await compileProject({name: 'boundary-reduction-slice', source}),
    {continueAfterExtensionErrors: true});
  assert.equal(jsonVariable(result, 'all'), 10);
  assert.deepEqual(jsonVariable(result, 'rows'), [3, 7]);
  assert.deepEqual(jsonVariable(result, 'selected'), [[1], [3]]);
  assert.equal(result.extensionErrors.length, 4);
  const dimensions = result.extensionErrors.filter(error => error.method === 'ScratchTensor.reductionBlock');
  assert.equal(dimensions.length, 3);
  assert.deepEqual(dimensions.map(error => error.args.DIM).sort(), ['', '0.5', 'NaN']);
  assert(dimensions.every(error => /integer axis|全部/.test(error.message)));
  const attention = result.extensionErrors.find(error => error.method === 'ScratchTensor.scaledDotProductAttentionBlock');
  assert(attention);
  assert.match(attention.message, /SCALE|scale/);
  for (const variable of ['after_nan', 'after_empty', 'after_fraction', 'after_scale']) {
    assert.equal(Number(result.variables[variable]), 0);
  }
  assert.equal(Number(result.variables.finished), 1);
});

test('a late invalid StateDict record leaves existing values, gradients and registry unchanged in the VM', async () => {
  const source = await loadFixture('boundary-state-dict-atomic');
  const result = await runProject(await compileProject({name: 'boundary-state-dict-atomic', source}),
    {continueAfterExtensionErrors: true});
  assert.equal(result.extensionErrors.length, 1);
  assert.equal(result.extensionErrors[0].method, 'ScratchTensor.loadStateDictBlock');
  assert.match(result.extensionErrors[0].message, /bad.*data\[0\].*finite number/);
  assert.deepEqual(jsonVariable(result, 'after'), jsonVariable(result, 'before'));
  assert.deepEqual(jsonVariable(result, 'gradient_before'), [1, 1]);
  assert.deepEqual(jsonVariable(result, 'gradient_after'), [1, 1]);
  assert.equal(Number(result.variables.after_import), 0);
});

import assert from 'node:assert/strict';
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import test from 'node:test';
import { artifactRoot, compileProject, extensionPath, wrapScript } from './compiler.mjs';
import { loadDocumentationExamples } from './documents.mjs';
import { readMetadata } from './runtime.mjs';

test('all documented ScratchDSL examples compile and cover every public extension block', { timeout: 180_000 }, async t => {
  const report = { documents: [], examples: [], failures: [], coveredOpcodes: [], missingOpcodes: [] };
  const covered = new Set();
  await mkdir(artifactRoot, { recursive: true });
  try {
    const { documents, examples } = await loadDocumentationExamples();
    report.documents = documents.map(document => document.path);
    const metadata = await readMetadata(extensionPath);
    const expected = metadata.blocks
      .filter(block => typeof block === 'object' && block.opcode && !block.hideFromPalette)
      .map(block => `${metadata.id}_${block.opcode}`);
    assert.ok(expected.length > 0, 'The built extension must expose public blocks');
    for (const example of examples) {
      await t.test(`${example.document}:${example.line} (${example.kind})`, async () => {
        const item = { document: example.document, line: example.line, kind: example.kind, heading: example.heading };
        try {
          const options = { name: `docs-${example.id}`, source: wrapScript(example.source) };
          if (example.kind === 'sbdsl-build') {
            options.configSource = example.source;
            options.source = examples.find(value => value.document === 'docs/demo.md' && value.fullProgram && value.heading === 'Linear 层与完整训练').source;
          } else if (example.kind === 'extension-entry') {
            const alias = /\s+as\s+("[^"\n]+")$/u.exec(example.source);
            assert.ok(alias, 'Expected a documented extension alias');
            options.extensionEntries = [example.source];
            options.source = wrapScript(`${JSON.parse(alias[1])}.[删除所有 Tensor]`);
          }
          const result = await compileProject(options);
          item.artifacts = path.relative(artifactRoot, result.directory).split(path.sep).join('/');
          item.opcodes = [...new Set(result.project.targets.flatMap(target => Object.values(target.blocks))
            .filter(block => typeof block === 'object' && block.opcode?.startsWith(`${metadata.id}_`) && !block.opcode.startsWith(`${metadata.id}_menu_`))
            .map(block => block.opcode))].sort();
          for (const opcode of item.opcodes) covered.add(opcode);
          report.examples.push(item);
        } catch (error) {
          item.error = error.stack ?? String(error);
          if (error.directory) item.artifacts = path.relative(artifactRoot, error.directory).split(path.sep).join('/');
          report.failures.push(item);
          throw error;
        }
      });
    }
    report.coveredOpcodes = [...covered].sort();
    report.missingOpcodes = expected.filter(opcode => !covered.has(opcode)).sort();
    assert.deepEqual(report.missingOpcodes, [], 'Public extension blocks missing from compiled documentation');
    assert.equal(report.failures.length, 0, 'One or more documentation examples failed to compile');
  } catch (error) {
    report.error = error.stack ?? String(error);
    throw error;
  } finally {
    await writeFile(path.join(artifactRoot, 'documentation.json'), `${JSON.stringify(report, null, 2)}\n`);
  }
});

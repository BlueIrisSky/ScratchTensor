import assert from 'node:assert/strict';
import { readFile, readdir } from 'node:fs/promises';
import path from 'node:path';
import { repoRoot } from './compiler.mjs';

export const requiredDocuments = new Map([
  ['README.md', 3],
  ['docs/autograd.md', 3],
  ['docs/demo.md', 4],
  ['docs/fun.md', 12],
  ['docs/optim.md', 7],
  ['docs/tensor.md', 12]
]);

const extensionCall = /^(?:"(?:\\.|[^"\\])+"|[\p{L}\p{N}_-]+(?:\.[\p{L}\p{N}_-]+)*)(?:::"[^"]+")?\.[[(<]/u;

export function extractDocumentationExamples(markdown, document) {
  const text = markdown.replace(/\r\n/gu, '\n');
  const examples = [];
  const lines = text.split('\n');
  let heading = '';
  let index = 0;
  const add = (kind, source, line) => examples.push({
    document, line, kind, source, heading,
    id: `${document.replace(/[^a-zA-Z0-9]/gu, '-')}-${++index}`,
    fullProgram: kind === 'sbdsl' && /^\s*@(?:sprite|stage)\b/mu.test(source)
  });
  for (let cursor = 0; cursor < lines.length; cursor++) {
    const line = lines[cursor];
    const title = /^(#{1,6})\s+(.+)$/u.exec(line);
    if (title) heading = title[2];
    const fence = /^\s*(`{3,}|~{3,})([^\s]*)\s*$/u.exec(line);
    if (fence) {
      const opening = cursor;
      const source = [];
      const terminator = new RegExp(`^\\s*${fence[1][0]}{${fence[1].length},}\\s*$`, 'u');
      while (++cursor < lines.length && !terminator.test(lines[cursor])) source.push(lines[cursor]);
      assert.ok(cursor < lines.length, `${document}:${opening + 1}: unclosed Markdown code fence`);
      const language = fence[2];
      const code = `${source.join('\n')}\n`;
      if (language === 'sbdsl' || language === 'sbdsl-build') add(language, code, opening + 1);
      else {
        assert.ok(!/^scratch/u.test(language), `${document}:${opening + 1}: use the sbdsl code fence`);
        assert.ok(!/^\s*(?:scratch-tensor-extension\.\[|@sprite\b)/mu.test(code), `${document}:${opening + 1}: ScratchDSL example has an untested code-fence language (${language})`);
        if (language === 'json') JSON.parse(code);
      }
      continue;
    }
    for (const inline of line.matchAll(/`([^`\n]+)`/gu)) {
      if (extensionCall.test(inline[1])) add('inline', inline[1], cursor + 1);
      else if (/^"[^"\n]+\.js"\s+as\s+"[^"\n]+"$/u.test(inline[1])) add('extension-entry', inline[1], cursor + 1);
    }
  }
  return examples;
}

async function markdownFiles(directory) {
  const files = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const fullPath = path.join(directory, entry.name);
    if (entry.isDirectory()) files.push(...await markdownFiles(fullPath));
    else if (entry.isFile() && entry.name.endsWith('.md')) files.push(fullPath);
  }
  return files.sort();
}

export async function loadDocumentationExamples() {
  const paths = [path.join(repoRoot, 'README.md'), ...await markdownFiles(path.join(repoRoot, 'docs'))];
  const documents = [];
  for (const fullPath of paths) {
    const document = path.relative(repoRoot, fullPath).split(path.sep).join('/');
    const text = await readFile(fullPath, 'utf8');
    documents.push({ path: document, text, examples: extractDocumentationExamples(text, document) });
  }
  for (const [document, minimum] of requiredDocuments) {
    const found = documents.find(item => item.path === document);
    assert.ok(found, `Required documentation page disappeared: ${document}`);
    assert.ok(found.examples.length >= minimum, `${document}: expected at least ${minimum} executable examples, found ${found.examples.length}`);
  }
  const examples = documents.flatMap(document => document.examples);
  for (const heading of ['Linear 层与完整训练', '最小乘法与梯度验证']) {
    assert.equal(examples.filter(example => example.document === 'docs/demo.md' && example.heading === heading && example.fullProgram).length, 1,
      `docs/demo.md: expected exactly one complete program under ${heading}`);
  }
  return { documents, examples };
}

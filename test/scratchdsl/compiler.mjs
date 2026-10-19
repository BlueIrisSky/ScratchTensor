import { spawn } from 'node:child_process';
import { copyFile, mkdir, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import JSZip from '@turbowarp/jszip';

export const repoRoot = fileURLToPath(new URL('../../', import.meta.url));
export const extensionPath = path.join(repoRoot, 'dist/scratch-tensor-extension.js');
export const artifactRoot = path.join(repoRoot, 'dist/scratchdsl-tests');

export async function loadFixture(name) {
  if (!/^[a-zA-Z0-9_-]+$/u.test(name)) throw new Error(`Invalid fixture name: ${name}`);
  return readFile(new URL(`fixtures/${name}.sbdsl`, import.meta.url), 'utf8');
}

const timeoutMs = 60_000;
const icon = '<svg xmlns="http://www.w3.org/2000/svg" width="96" height="96" viewBox="0 0 96 96"><rect width="96" height="96" rx="16" fill="#ffab19"/></svg>\n';

export async function runCompiler(args, { cwd = repoRoot } = {}) {
  const binary = process.env.SCRATCHDSL_BIN || 'sbdsl';
  const started = Date.now();
  return new Promise((resolve, reject) => {
    const child = spawn(binary, args, { cwd, windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'] });
    const stdout = [];
    const stderr = [];
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      child.kill('SIGKILL');
    }, timeoutMs);
    child.stdout.on('data', chunk => stdout.push(chunk));
    child.stderr.on('data', chunk => stderr.push(chunk));
    child.once('error', error => {
      clearTimeout(timer);
      reject(new Error(`Cannot start ScratchDSL (${binary}). Build it and set SCRATCHDSL_BIN to its executable.`, { cause: error }));
    });
    child.once('close', (code, signal) => {
      clearTimeout(timer);
      const result = {
        binary, args, cwd, code, signal, timedOut,
        stdout: Buffer.concat(stdout).toString('utf8'),
        stderr: Buffer.concat(stderr).toString('utf8'),
        durationMs: Date.now() - started
      };
      if (timedOut) {
        const error = new Error(`ScratchDSL exceeded ${timeoutMs} ms: ${args.join(' ')}`);
        error.compilerResult = result;
        reject(error);
      } else {
        resolve(result);
      }
    });
  });
}

export function wrapScript(body, { declarations = [] } = {}) {
  if (/^\s*@(?:sprite|stage)\b/mu.test(body)) return body;
  const lifted = [...declarations];
  const statements = [];
  for (const line of body.trim().split(/\r?\n/u)) {
    if (/^@(?:var|list)\s/u.test(line)) lifted.push(line);
    else statements.push(line);
  }
  let script = statements.join('\n');
  if (/^(?:"[^"\n]+"|[\p{L}\p{N}_.-]+)(?:::"[^"]+")?\.[(<]/u.test(script.trim())) {
    lifted.push('@var __documentation_result = ""');
    script = `[将 $__documentation_result 设为 ${script}]`;
  }
  return `${lifted.join('\n')}\n@sprite "ScratchDSL 验证" {\n    @init {}\n    当绿旗被点击 {\n${script}\n    }\n}\n`;
}

async function writeWithin(directory, relative, contents) {
  const destination = path.resolve(directory, relative);
  if (path.relative(directory, destination).startsWith('..') || path.isAbsolute(relative)) {
    throw new Error(`Fixture path must stay inside its project: ${relative}`);
  }
  await mkdir(path.dirname(destination), { recursive: true });
  await writeFile(destination, contents);
}

async function findSb3s(directory) {
  const files = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const fullPath = path.join(directory, entry.name);
    if (entry.isDirectory()) files.push(...await findSb3s(fullPath));
    else if (entry.isFile() && entry.name.endsWith('.sb3')) files.push(fullPath);
  }
  return files;
}

// Each case gets a small repository-shaped sandbox so the documented
// ../dist extension path is exercised without rewriting its configuration.
export async function compileProject({ name, source, files = {}, extensionEntries, configSource }) {
  if (!/^[a-zA-Z0-9_-]+$/u.test(name)) throw new Error(`Invalid fixture name: ${name}`);
  const sandbox = path.join(artifactRoot, 'projects', name);
  const directory = path.join(sandbox, 'demo');
  await rm(sandbox, { recursive: true, force: true });
  await mkdir(path.join(sandbox, 'dist'), { recursive: true });
  await copyFile(extensionPath, path.join(sandbox, 'dist/scratch-tensor-extension.js'));
  await writeWithin(directory, 'src/main.sbdsl', source);
  await writeWithin(directory, 'assets/sbdsl.svg', icon);
  for (const [relative, contents] of Object.entries(files)) await writeWithin(directory, relative, contents);
  const entries = extensionEntries ?? [JSON.stringify('../dist/scratch-tensor-extension.js')];
  const config = configSource ?? `entry: "./src/main.sbdsl"\noutput: "./dist/project.sb3"\nextensions: [${entries.join('\n')}]\n`;
  const configPath = path.join(directory, 'sbdsl.buildsb');
  await writeFile(configPath, config);
  let result;
  try {
    result = await runCompiler(['build', configPath], { cwd: directory });
  } catch (error) {
    result = error.compilerResult ?? { error: error.message, cause: String(error.cause ?? '') };
    await writeFile(path.join(directory, 'compiler.json'), `${JSON.stringify(result, null, 2)}\n`);
    await writeFile(path.join(directory, 'compiler.log'), `${result.stdout ?? ''}${result.stderr ?? ''}\n${error.stack ?? error}\n`);
    error.directory = directory;
    throw error;
  }
  await writeFile(path.join(directory, 'compiler.json'), `${JSON.stringify(result, null, 2)}\n`);
  await writeFile(path.join(directory, 'compiler.log'), `${result.stdout}${result.stderr}`);
  if (result.code !== 0) {
    const error = new Error(`ScratchDSL build failed (${name}); artifacts: ${directory}\n${result.stderr || result.stdout}`);
    error.compilerResult = result;
    error.directory = directory;
    throw error;
  }
  const outputs = await findSb3s(sandbox);
  if (outputs.length !== 1) throw new Error(`Expected one generated SB3 in ${sandbox}, found ${outputs.length}`);
  const sb3Path = outputs[0];
  const sb3 = await readFile(sb3Path);
  const archive = await JSZip.loadAsync(sb3);
  const projectFile = archive.file('project.json');
  if (!projectFile) throw new Error(`Compiled SB3 has no project.json: ${sb3Path}`);
  const project = JSON.parse(await projectFile.async('string'));
  return { directory, sb3Path, sb3, project };
}

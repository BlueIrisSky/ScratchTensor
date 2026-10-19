import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { readFile, readdir, mkdir, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const directory = path.join(root, 'dist', 'release');
const payloadNames = ['scratch-tensor-extension.js', 'LICENSE'];
const names = [...payloadNames, 'release-manifest.json', 'SHA256SUMS.txt'];
const hash = data => createHash('sha256').update(data).digest('hex');
const readJson = async file => JSON.parse(await readFile(file, 'utf8'));

async function sourceMetadata() {
  const pkg = await readJson(path.join(root, 'package.json'));
  const lock = await readJson(path.join(root, 'package-lock.json'));
  assert.match(pkg.version, /^\d+\.\d+\.\d+$/, 'package version must be a release version');
  assert.equal(pkg.license, 'MIT', 'package license must be MIT');
  assert.equal(lock.version, pkg.version, 'lockfile version differs from package.json');
  assert.equal(lock.packages[''].version, pkg.version, 'lockfile root version differs');
  assert.equal(lock.packages[''].license, 'MIT', 'lockfile root license must be MIT');
  const gitCommit = execFileSync('git', ['rev-parse', 'HEAD'], { cwd: root, encoding: 'utf8' }).trim();
  assert.match(gitCommit, /^[0-9a-f]{40}$/, 'source commit must be a full Git SHA');
  if (process.env.GITHUB_SHA) assert.equal(gitCommit, process.env.GITHUB_SHA, 'checkout commit differs from workflow');
  if (process.env.GITHUB_REF_TYPE === 'tag') {
    assert.equal(process.env.GITHUB_REF_NAME, `v${pkg.version}`, 'release tag differs from package version');
  }
  const license = await readFile(path.join(root, 'LICENSE'));
  assert.match(license.toString('utf8'), /^MIT License\r?\n/, 'LICENSE must contain the MIT license');
  return { version: pkg.version, gitCommit, license };
}

async function prepare(metadata) {
  const extension = await readFile(path.join(root, 'dist', payloadNames[0]));
  assert.ok(extension.length > 0, 'build the extension before packaging');
  // This generated directory contains only the files promoted into a release.
  await rm(directory, { recursive: true, force: true });
  await mkdir(directory, { recursive: true });
  const payloads = [extension, metadata.license];
  const assets = [];
  for (const [index, name] of payloadNames.entries()) {
    const data = payloads[index];
    await writeFile(path.join(directory, name), data);
    assets.push({ name, bytes: data.length, sha256: hash(data) });
  }
  const manifest = {
    project: 'ScratchTensor', version: metadata.version, gitCommit: metadata.gitCommit,
    license: 'MIT', assets
  };
  await writeFile(path.join(directory, 'release-manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
  const checksums = [];
  for (const name of names.filter(name => name !== 'SHA256SUMS.txt')) {
    checksums.push(`${hash(await readFile(path.join(directory, name)))}  ${name}`);
  }
  await writeFile(path.join(directory, 'SHA256SUMS.txt'), checksums.join('\n') + '\n');
}

async function verify(metadata, uploadedFile) {
  assert.deepEqual((await readdir(directory)).sort(), [...names].sort(), 'release must contain exactly four assets');
  const files = new Map();
  for (const name of names) {
    const data = await readFile(path.join(directory, name));
    assert.ok(data.length > 0, `empty release asset: ${name}`);
    files.set(name, data);
  }
  assert.ok(files.get('LICENSE').equals(metadata.license), 'release LICENSE differs from source');
  const manifest = JSON.parse(files.get('release-manifest.json').toString('utf8'));
  assert.equal(manifest.project, 'ScratchTensor');
  assert.equal(manifest.version, metadata.version, 'release version differs from source');
  assert.equal(manifest.gitCommit, metadata.gitCommit, 'release commit differs from source');
  assert.equal(manifest.license, 'MIT');
  assert.deepEqual(manifest.assets.map(asset => asset.name).sort(), [...payloadNames].sort(), 'manifest payload set differs');
  for (const asset of manifest.assets) {
    const data = files.get(asset.name);
    assert.equal(asset.bytes, data.length, `manifest size differs: ${asset.name}`);
    assert.equal(asset.sha256, hash(data), `manifest hash differs: ${asset.name}`);
  }
  const expected = names.filter(name => name !== 'SHA256SUMS.txt')
    .map(name => `${hash(files.get(name))}  ${name}`).sort();
  const checksums = files.get('SHA256SUMS.txt').toString('utf8').trimEnd().split('\n').sort();
  assert.deepEqual(checksums, expected, 'checksum list differs from the release');
  if (uploadedFile) {
    const uploaded = (await readJson(uploadedFile)).assets;
    assert.deepEqual(uploaded.map(asset => asset.name).sort(), [...names].sort(), 'uploaded asset set differs');
    for (const asset of uploaded) {
      const data = files.get(asset.name);
      assert.equal(asset.size, data.length, `uploaded size differs: ${asset.name}`);
      if (asset.digest) assert.equal(asset.digest, `sha256:${hash(data)}`, `uploaded hash differs: ${asset.name}`);
    }
  }
  console.log(`Verified four release assets for ScratchTensor ${metadata.version} (${metadata.gitCommit})`);
}

const [command, option, uploadedFile, ...extra] = process.argv.slice(2);
assert.ok((command === 'prepare' && option === undefined) ||
  (command === 'verify' && (option === undefined || (option === '--uploaded' && uploadedFile && extra.length === 0))),
  'Usage: node scripts/release.mjs prepare | verify [--uploaded file.json]');
const metadata = await sourceMetadata();
if (command === 'prepare') await prepare(metadata);
await verify(metadata, uploadedFile);

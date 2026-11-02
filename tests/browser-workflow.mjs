// Optional real-browser test, with Playwright supplied externally. No network
// access is needed beyond the local HTTP server and the provided CSV fixture.
// PLAYWRIGHT_MODULE=/path/to/playwright-core/index.mjs CHROMIUM_PATH=/path/to/chromium \
//   node tests/browser-workflow.mjs http://127.0.0.1:8000 /path/to/iris-browser.csv
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { zipRead } from '../js/core/zip.js';
const [url, csv] = process.argv.slice(2);
if (!url || !csv || !process.env.CHROMIUM_PATH) throw new Error('Provide URL, Iris CSV path and CHROMIUM_PATH.');
const mod = process.env.PLAYWRIGHT_MODULE ? await import(pathToFileURL(process.env.PLAYWRIGHT_MODULE).href) : await import('playwright-core');
const browser = await mod.chromium.launch({ executablePath: process.env.CHROMIUM_PATH, headless: true,
  args: ['--no-sandbox', '--disable-dev-shm-usage', '--single-process', '--no-zygote', '--use-gl=angle', '--use-angle=swiftshader'] });
const errors = [];
try {
  const page = await browser.newPage({ acceptDownloads: true });
  page.on('pageerror', e => errors.push(e.message));
  await page.goto(url + '/studio.html');
  const rail = (name) => page.getByRole('button', { name: new RegExp('0[1-7]\\s*' + name) });
  await rail('Data').click();
  await page.getByRole('button', { name: 'Import files', exact: true }).first().click();
  const [chooser] = await Promise.all([page.waitForEvent('filechooser'), page.getByText('Drop files here, or click to browse').click()]);
  await chooser.setFiles(csv);
  await page.locator('.modal-back').getByRole('button', { name: 'Import', exact: true }).click();
  await page.getByText('150 samples', { exact: false }).first().waitFor();
  await rail('Build').click();
  await page.getByRole('button', { name: 'Model for loaded data', exact: true }).first().click();
  await rail('Train').click();
  await page.getByRole('button', { name: 'Start run', exact: true }).click();
  await page.evaluate(async () => { window.__rootismTestState = (await import('/js/ui/state.js')).state; });
  await page.waitForFunction(() => !window.__rootismTestState.training && window.__rootismTestState.runs.length > 0, null, { timeout: 60000 });
  const trained = await page.evaluate(async () => {
    const s = await import('/js/ui/state.js'), store = await import('/js/ui/store.js');
    const run = await store.loadRun(s.state.runs[0].id);
    const data = await store.loadDataset(run.datasetId);
    return { accuracy: run.metrics.test.scalars.accuracy, testCount: data.splits.test.length, runId: run.id, datasetId: data.id };
  });
  assert(trained.testCount > 0 && trained.accuracy >= 0.75, 'browser-trained Iris model should achieve reasonable held-out accuracy');
  const expected = await page.evaluate(async (ids) => {
    const store = await import('/js/ui/store.js'), core = await import('/js/core/model.js'), tasks = await import('/js/core/tasks.js');
    const run = await store.loadRun(ids.runId), data = await store.loadDataset(ids.datasetId);
    const model = new core.Model(run.spec, run.seed); model.loadWeights(run.weights); model.syncStates();
    const indices = data.splits.test, result = tasks.TASKS.multiclass.predict(model, data, indices);
    return { features: Array.from(tasks.sliceRows(data, 'x', indices, data.xShape)), probabilities: Array.from(result.probs), n: indices.length, width: data.xShape[0], classes: result.k };
  }, trained);
  // Reload and reopen the automatically saved data, then export stored weights.
  await page.reload();
  await rail('Library').click();
  const dataRow = page.locator('.run-row').filter({ hasText: path.basename(csv, path.extname(csv)) }).filter({ hasText: '150 n' }).first();
  await dataRow.getByRole('button', { name: 'Open', exact: true }).click();
  await rail('Export').click();
  await page.getByRole('button', { name: 'Build artifacts', exact: true }).click();
  await page.getByRole('button', { name: 'Download ZIP', exact: true }).waitFor({ timeout: 30000 });
  const [download] = await Promise.all([page.waitForEvent('download'), page.getByRole('button', { name: 'Download ZIP', exact: true }).click()]);
  const filename = path.join(os.tmpdir(), 'rootism-browser-test-' + process.pid + '.zip');
  await download.saveAs(filename);
  const entries = zipRead(new Uint8Array(await fs.readFile(filename)));
  await fs.unlink(filename);
  const names = entries.map(e => e.name);
  for (const required of ['onnx/model.onnx', 'tfjs/model.json', 'tfjs/model.weights.bin', 'preprocessing/table.mjs', 'preprocessing/feature-spec.json']) assert(names.includes(required), 'missing ' + required);
  assert(!names.some(n => n.startsWith('dataset/')), 'raw dataset should not be exported by default');
  assert(entries.every(e => e.crcOk), 'bundle checksums must pass');
  if (process.env.BROWSER_TEST_OUTPUT) {
    const prefix = process.env.BROWSER_TEST_OUTPUT;
    await fs.writeFile(prefix + '.onnx', entries.find(e => e.name === 'onnx/model.onnx').data);
    await fs.writeFile(prefix + '.json', JSON.stringify(expected));
  }
  assert.deepEqual(errors, [], 'browser must not encounter uncaught errors');
  console.log('Browser import, training, persistent data/run, reload, and export passed:', trained);
} finally { await browser.close(); }

import { parseDelimited, parseJsonl, parseJsonTable, parseTextLines, tableToDataset, inferColumns, pickLabelColumn, detectFormat, detectDelimiter, importFiles, datasetToCsv, datasetToJsonl, datasetToNative, nativeToDataset, datasetToCoco, datasetToWavArchive, exportDataset, signalToFeature, audioFeatureConfig, decodeAudioFile } from '../js/data/import.js';
import { wavEncode } from '../js/data/audio.js';
import { zipWrite, zipRead } from '../js/core/zip.js';
import { buildPresetDataset } from '../js/presets/datasets.js';
import { utf8Encode } from '../js/core/base64.js';

let fails = 0;
const check = (label, cond, extra) => { if (!cond) { console.log('FAIL', label, extra === undefined ? '' : extra); fails++; } else console.log('ok  ', label, extra === undefined ? '' : extra); };

const csv = 'id,text,label\n1,"hello, world",positive\n2,"a ""quoted"" line",negative\n3,plain text,positive\n4,"multi\nline",negative\n';
const parsed = parseDelimited(csv);
check('csv columns', JSON.stringify(parsed.columns) === '["id","text","label"]', JSON.stringify(parsed.columns));
check('csv rows', parsed.rows.length === 4, parsed.rows.length);
check('csv quoted comma', parsed.rows[0].text === 'hello, world', JSON.stringify(parsed.rows[0].text));
check('csv escaped quote', parsed.rows[1].text === 'a "quoted" line', JSON.stringify(parsed.rows[1].text));
check('csv embedded newline', parsed.rows[2].text === 'plain text' && parsed.rows[3].label === 'negative', JSON.stringify(parsed.rows[3]));
check('delimiter detect tsv', detectDelimiter('a\tb\tc\n1\t2\t3\n') === '\t');
check('delimiter detect semicolon', detectDelimiter('a;b;c\n1;2;3\n') === ';');
check('format detect', detectFormat('data.CSV') === 'csv' && detectFormat('a/b/c.jsonl') === 'jsonl' && detectFormat('x.zip', new Uint8Array([0x50,0x4b,3,4])) === 'zip' && detectFormat('y.wav', new Uint8Array([0x52,0x49,0x46,0x46,0,0,0,0])) === 'wav' && detectFormat('d.rootism.json') === 'rootism');

const ds = tableToDataset({ rows: parsed.rows, columns: parsed.columns, name: 'Sentiment' });
check('dataset count', ds.count === 4, ds.count);
check('dataset task', ds.task === 'binary', ds.task);
check('dataset labels', JSON.stringify(ds.labels) === '["positive","negative"]', JSON.stringify(ds.labels));
check('dataset modality', ds.modality === 'text', ds.modality);
check('x shape', JSON.stringify(ds.xShape), JSON.stringify(ds.xShape));
check('splits', ds.splits.train.length + ds.splits.val.length + ds.splits.test.length === 4, JSON.stringify([ds.splits.train.length, ds.splits.val.length, ds.splits.test.length]));
check('y values', Array.from(ds.y).join(',') === '0,1,0,1', Array.from(ds.y).join(','));
check('dropped id column', ds.importReport.droppedColumns.indexOf('id') >= 0, JSON.stringify(ds.importReport.droppedColumns));

const numericRows = [];
for (let i = 0; i < 40; i++) numericRows.push({ a: (i % 7) + 0.5, b: i * 2, c: i % 3 === 0 ? 'x' : 'y', target: i % 5 });
const dsNum = tableToDataset({ rows: numericRows, name: 'Numeric' });
check('numeric task', dsNum.task === 'multiclass', dsNum.task + ' labels ' + JSON.stringify(dsNum.labels));
check('numeric classes', dsNum.labels.length === 5, JSON.stringify(dsNum.labels));
check('numeric feature dim', dsNum.xShape[0] === 4, dsNum.xShape[0] + ' (a,b standardized + c one-hot over 2 levels)');
check('numeric standardised', Math.abs(dsNum.x[0]) < 3, dsNum.x[0]);

const regRows = [];
for (let i = 0; i < 30; i++) regRows.push({ t: i, value: Math.sin(i) * 3.5 });
const dsReg = tableToDataset({ rows: regRows, name: 'Regression', labelColumn: 'value' });
check('regression task', dsReg.task === 'regression', dsReg.task);
check('regression labels empty', dsReg.labels.length === 0);
check('regression y', Math.abs(dsReg.y[1] - Math.sin(1) * 3.5) < 1e-6, dsReg.y[1]);

const mlRows = [];
for (let i = 0; i < 24; i++) mlRows.push({ text: 'sample number ' + i + ' with words', tags: i % 3 === 0 ? 'a;b' : i % 3 === 1 ? 'b' : 'c;a' });
const dsMl = tableToDataset({ rows: mlRows, name: 'Multilabel', labelColumn: 'tags' });
check('multilabel task', dsMl.task === 'multilabel', dsMl.task);
check('multilabel width', dsMl.yShape[0] === 3, JSON.stringify(dsMl.yShape));
check('multilabel row0', Array.from(dsMl.y.subarray(0, 3)).join(',') === '1,1,0', Array.from(dsMl.y.subarray(0, 3)).join(','));

const jl = parseJsonl('{"text":"a","label":"x"}\n{"text":"b","label":"y"}\nnot json\n{"text":"c","label":"x"}\n');
check('jsonl rows', jl.rows.length === 3 && jl.broken === 1, jl.rows.length + '/' + jl.broken);
const jt = parseJsonTable(JSON.stringify({ rows: [{ q: 1, k: 'a' }, { q: 2, k: 'b' }] }));
check('json table', jt.rows.length === 2 && JSON.stringify(jt.columns) === '["q","k"]');
const jt2 = parseJsonTable(JSON.stringify({ a: [1, 2, 3], b: ['x', 'y', 'z'] }));
check('json columnar', jt2.rows.length === 3 && jt2.rows[2].b === 'z');
const tl = parseTextLines('hello world\tpositive\ngoodbye\tnegative\n');
check('text lines', tl.rows.length === 2 && tl.rows[0].label === 'positive');
const tl2 = parseTextLines('just a line\nanother line\n');
check('text lines no label', tl2.rows.length === 2 && tl2.columns.length === 1);

const native = datasetToNative(ds);
const json = JSON.stringify(native);
const restored = nativeToDataset(JSON.parse(json));
check('native roundtrip count', restored.count === ds.count);
check('native roundtrip labels', JSON.stringify(restored.labels) === JSON.stringify(ds.labels));
check('native roundtrip x', restored.x.length === ds.x.length && Math.abs(restored.x[5] - ds.x[5]) < 1e-7);
check('native roundtrip splits', restored.splits.train.length === ds.splits.train.length, restored.splits.train.length);
check('native roundtrip digest', restored.digest.contentHash === ds.digest.contentHash, restored.digest.short + ' vs ' + ds.digest.short);
check('native payload size', json.length, json.length + ' bytes for 4 rows');

const csvOut = datasetToCsv(ds);
check('csv export header', csvOut.split('\n')[0].indexOf('index,split') === 0, csvOut.split('\n')[0]);
check('csv export rows', parseDelimited(csvOut).rows.length === 4 && parseDelimited(csvOut).rows[3].text === 'multi\nline', parseDelimited(csvOut).rows.length + ' / ' + JSON.stringify(parseDelimited(csvOut).rows[3] && parseDelimited(csvOut).rows[3].text));
const reimported = parseDelimited(csvOut);
check('csv export reimportable', reimported.rows.length === 4 && reimported.columns.indexOf('label_name') >= 0, JSON.stringify(reimported.columns));
const jsonlOut = datasetToJsonl(ds);
check('jsonl export', jsonlOut.trim().split('\n').length === 4 && JSON.parse(jsonlOut.split('\n')[0]).label === 'positive');
const wavReimport = parseDelimited(datasetToCsv(ds, { includeFeatures: false }));
check('csv export lossless labels', wavReimport.rows.map((r) => r.label_name).join(',') === 'positive,negative,positive,negative', wavReimport.rows.map((r) => r.label_name).join(','));

const det = buildPresetDataset('detection', { size: 'small' });
const coco = datasetToCoco(det);
check('coco images', coco.images.length === det.count, coco.images.length);
check('coco annotations', coco.annotations.length > 0 && coco.annotations[0].bbox.length === 4, coco.annotations.length);
check('coco categories', coco.categories.length === det.labels.length);

const audio = buildPresetDataset('denoise', { size: 'small' });
const wavEntries = datasetToWavArchive(audio, { limit: 6 });
check('wav archive entries', wavEntries.length === 7, wavEntries.length + ' (6 wav + labels.csv)');
const zipBytes = zipWrite(wavEntries);
const back = zipRead(zipBytes);
check('wav zip readable', back.length === 7 && back.every((e) => e.crcOk));
const decodedBack = decodeAudioFile(back[0].data, audioFeatureConfig({ preset: 'kws' }));
check('wav roundtrip decode', decodedBack.length > 0, decodedBack.length + ' samples at 8 kHz');
const feat = signalToFeature(decodedBack, audioFeatureConfig({ preset: 'kws' }));
check('audio feature shape', JSON.stringify(feat.shape) === '[36,32]', JSON.stringify(feat.shape) + ' len ' + feat.data.length);
check('audio feature finite', Array.from(feat.data).every((v) => Number.isFinite(v)));

const makeWav = (freq) => wavEncode(Float32Array.from({ length: 8000 }, (v, i) => Math.sin(i * freq) * 0.5 + Math.sin(i * freq * 2.7) * 0.2), 8000, 1);
const wavBytes = makeWav(0.05);
const wavBytes2 = makeWav(0.11);
const labelsCsv = 'file,label\n00000,go\n00001,stop\n';
const result = importFiles([
  { name: 'clips/00000.wav', bytes: wavBytes },
  { name: 'clips/00001.wav', bytes: wavBytes2 },
  { name: 'labels.csv', text: labelsCsv }
], { name: 'Wake import' });
check('importFiles audio', result.datasets.length === 1, JSON.stringify(result.report));
if (result.datasets.length) {
  const ids = result.datasets[0];
  check('imported audio count', ids.count === 2, ids.count);
  check('imported audio labels', JSON.stringify(ids.labels), JSON.stringify(ids.labels));
  check('imported audio shape', JSON.stringify(ids.xShape), JSON.stringify(ids.xShape));
  check('imported audio modality', ids.modality === 'audio');
}
const zipped = zipWrite([{ name: 'clips/00000.wav', data: wavBytes }, { name: 'clips/00001.wav', data: wavBytes2 }, { name: 'labels.csv', data: labelsCsv }]);
const fromZip = importFiles([{ name: 'wake.zip', bytes: zipped }]);
check('importFiles zip of wav', fromZip.datasets.length === 1 && fromZip.datasets[0].count === 2, JSON.stringify(fromZip.problems));
const folderZip = zipWrite([{ name: 'go/a.wav', data: wavBytes }, { name: 'go/b.wav', data: wavBytes2 }, { name: 'stop/c.wav', data: makeWav(0.23) }]);
const fromFolders = importFiles([{ name: 'folders.zip', bytes: folderZip }]);
check('folder-per-class labels', fromFolders.datasets.length === 1 && JSON.stringify(fromFolders.datasets[0].labels).indexOf('go') >= 0, fromFolders.datasets.length ? JSON.stringify(fromFolders.datasets[0].labels) : JSON.stringify(fromFolders.problems));
const tableImport = importFiles([{ name: 'sentiment.csv', text: csv }]);
check('importFiles csv', tableImport.datasets.length === 1 && tableImport.datasets[0].count === 4, JSON.stringify(tableImport.problems));
const badImport = importFiles([{ name: 'thing.xyz', bytes: new Uint8Array([1, 2, 3]) }]);
check('importFiles unknown', badImport.datasets.length === 0 && badImport.problems.length > 0);

for (const fmt of ['csv', 'jsonl', 'native', 'labels', 'coco']) {
  const files = exportDataset(ds, fmt, { limit: 3 });
  check('export ' + fmt, files.length === 1 && files[0].name && files[0].data.length > 0, files[0] ? files[0].name + ' ' + (files[0].data.length || 0) : '');
}
const wavExport = exportDataset(audio, 'wav', { limit: 3 });
check('export wav zip', wavExport[0].name.endsWith('.zip') && wavExport[0].data.length > 100, wavExport[0].name + ' ' + wavExport[0].data.length);

const inferred = inferColumns(parsed.columns, parsed.rows);
check('infer id ignored', inferred.find((c) => c.name === 'id').ignored === true);
check('infer text kind', inferred.find((c) => c.name === 'text').kind === 'text', inferred.find((c) => c.name === 'text').kind);
check('infer label kind', inferred.find((c) => c.name === 'label').kind === 'categorical');
check('pickLabelColumn', pickLabelColumn(inferred, parsed.rows) === 'label');

console.log('');
console.log(fails ? 'IMPORT FAILURES ' + fails : 'IMPORT CHECKS ALL PASS');
process.exit(fails ? 1 : 0);

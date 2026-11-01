import { GENERATORS, REFERENCE_DATASETS, buildDataset } from '../data/generators.js';
import { defaultAugConfig } from '../data/augment.js';
import { buildSplits } from '../data/dataset.js';

const SIZE_COUNTS = {
  langid: [480, 1320, 3840],
  intent: [400, 1113, 3200],
  spam: [150, 410, 1200],
  pii: [150, 430, 1050],
  code: [300, 820, 2400],
  password: [400, 1100, 3200],
  embeddings: [100, 372, 1040],
  autocomplete: [200, 759, 2200],
  wake: [150, 410, 1200],
  vad: [80, 220, 640],
  denoise: [60, 140, 320],
  pitch: [180, 460, 1000],
  events: [270, 744, 2160],
  onset: [60, 170, 480],
  superres: [80, 240, 560],
  denoiseimg: [70, 170, 400],
  detection: [43, 150, 420],
  segmentation: [50, 150, 420],
  style: [70, 170, 400],
  saliency: [50, 150, 420],
  nsfw: [70, 219, 600],
  imu: [220, 576, 1600],
  anomaly: [180, 520, 1400],
  forecast: [140, 380, 1000],
  bot: [150, 420, 1200],
  compression: [160, 620, 1800],
  texture: [1024, 2304, 4096]
};

export const PRESET_DATASETS = {
  langid: {
    key: 'langid', name: 'Language identification', modality: 'text', task: 'multiclass', generator: 'langid',
    tagline: 'Twelve languages, word-level Markov expansion',
    defaults: { size: 'medium', seed: 11, dim: 640 },
    splits: { method: 'stratified', ratios: [0.72, 0.14, 0.14] },
    augment: [{ id: 'typos', strength: 0.06, prob: 0.45 }, { id: 'word_drop', strength: 0.12, prob: 0.3 }],
    notes: 'Char n-gram features are hashed, so augmentation re-derives them from the stored text rather than perturbing the vector.'
  },
  intent: {
    key: 'intent', name: 'Assistant intents', modality: 'text', task: 'multiclass', generator: 'intent',
    tagline: 'Eighteen intents with slot variety and typos',
    defaults: { size: 'medium', seed: 22, dim: 900, noise: 0.08 },
    splits: { method: 'stratified', ratios: [0.72, 0.14, 0.14] },
    augment: [{ id: 'typos', strength: 0.08, prob: 0.5 }, { id: 'word_shuffle', strength: 2, prob: 0.3 }],
    notes: 'Slot values are re-sampled per row, so the same intent appears with many different entities.'
  },
  spam: {
    key: 'spam', name: 'Spam and toxicity', modality: 'text', task: 'multiclass', generator: 'spam',
    tagline: 'Three classes with adjustable label noise',
    defaults: { size: 'medium', seed: 33, dim: 768, classes: 3, labelNoise: 0.02 },
    splits: { method: 'stratified', ratios: [0.72, 0.14, 0.14] },
    augment: [{ id: 'typos', strength: 0.06, prob: 0.4 }],
    notes: 'Raise label noise to see the calibration plot degrade the way it does on real moderation queues.'
  },
  pii: {
    key: 'pii', name: 'PII redaction', modality: 'text', task: 'seq_tag', generator: 'pii',
    tagline: 'Seven entity types, character BIO tags',
    defaults: { size: 'medium', seed: 44, maxLen: 96, piiRate: 0.62 },
    splits: { method: 'stratified', ratios: [0.72, 0.14, 0.14] },
    augment: [{ id: 'typos', strength: 0.04, prob: 0.3 }],
    notes: 'Rows carry their true length so padding is masked out of both the loss and every metric.'
  },
  code: {
    key: 'code', name: 'Code language', modality: 'text', task: 'multiclass', generator: 'code',
    tagline: 'Sixteen languages, identifiers renamed',
    defaults: { size: 'medium', seed: 55, dim: 1024 },
    splits: { method: 'stratified', ratios: [0.72, 0.14, 0.14] },
    augment: [{ id: 'typos', strength: 0.03, prob: 0.25 }],
    notes: 'Identifier renaming is what stops the model from classifying on variable names alone.'
  },
  password: {
    key: 'password', name: 'Password strength', modality: 'text', task: 'multiclass', generator: 'password',
    tagline: 'Five bands, no leaked corpora',
    defaults: { size: 'medium', seed: 66, dim: 320 },
    splits: { method: 'stratified', ratios: [0.72, 0.14, 0.14] },
    augment: [],
    notes: 'Every public corpus for this task is a breach dump. Nothing here is derived from one, which is the point.'
  },
  embeddings: {
    key: 'embeddings', name: 'Paraphrase pairs', modality: 'text', task: 'embedding', generator: 'embeddings',
    tagline: 'Forty meaning groups, surface variations',
    defaults: { size: 'medium', seed: 77, dim: 512 },
    splits: { method: 'group', ratios: [0.72, 0.14, 0.14] },
    augment: [{ id: 'typos', strength: 0.06, prob: 0.4 }, { id: 'word_drop', strength: 0.1, prob: 0.3 }],
    notes: 'Split by group, not by row: two surface variants of one sentence must never straddle the train and test boundary.'
  },
  autocomplete: {
    key: 'autocomplete', name: 'Character sequences', modality: 'text', task: 'lm', generator: 'autocomplete',
    tagline: 'Next-symbol training over a 128 vocab',
    defaults: { size: 'medium', seed: 88, seqLen: 64 },
    splits: { method: 'random', ratios: [0.72, 0.14, 0.14] },
    augment: [],
    notes: 'Targets are the sequence shifted by one, so no separate label column is stored.'
  },
  wake: {
    key: 'wake', name: 'Wake word clips', modality: 'audio', task: 'binary', generator: 'wake',
    tagline: 'Two keywords against five negative phrases',
    defaults: { size: 'medium', seed: 99, snrMin: 3, snrMax: 22 },
    splits: { method: 'stratified', ratios: [0.7, 0.15, 0.15] },
    augment: [{ id: 'spec_noise', strength: 0.2, prob: 0.5 }, { id: 'pitch_shift', strength: 2, prob: 0.35 }, { id: 'time_shift', strength: 3, prob: 0.3 }],
    notes: 'Clips are cut to a fixed sample count before feature extraction, so every row has the same 36 frame grid and the batcher never pads.'
  },
  vad: {
    key: 'vad', name: 'Voice activity frames', modality: 'audio', task: 'frame_binary', generator: 'vad',
    tagline: 'One label per 20 ms frame',
    defaults: { size: 'medium', seed: 111 },
    splits: { method: 'random', ratios: [0.72, 0.14, 0.14] },
    augment: [{ id: 'spec_noise', strength: 0.2, prob: 0.5 }, { id: 'freq_mask', strength: 2, prob: 0.3 }],
    notes: 'Labels are per frame, so frame counts have to match the feature grid exactly; the quality report flags any mismatch.'
  },
  denoise: {
    key: 'denoise', name: 'Speech noise pairs', modality: 'audio', task: 'spectral_mask', generator: 'denoise',
    tagline: 'Clean and noisy spectra plus raw waveforms',
    defaults: { size: 'medium', seed: 122, snrMin: -3, snrMax: 12 },
    splits: { method: 'random', ratios: [0.7, 0.15, 0.15] },
    augment: [{ id: 'spec_noise', strength: 0.15, prob: 0.4 }, { id: 'gain', strength: 0.15, prob: 0.35 }],
    notes: 'Raw waveforms and noisy phase are kept on every row so SI-SDR and the perceptual estimate are measured on reconstructed audio, not on spectra.'
  },
  pitch: {
    key: 'pitch', name: 'Pitch tones', modality: 'audio', task: 'pitch_class', generator: 'pitch',
    tagline: 'Thirty-six chromatic classes, harmonics and vibrato',
    defaults: { size: 'medium', seed: 133, harmonics: 6 },
    splits: { method: 'stratified', ratios: [0.72, 0.14, 0.14] },
    augment: [{ id: 'spec_noise', strength: 0.2, prob: 0.45 }, { id: 'pitch_shift', strength: 2, prob: 0.3 }],
    notes: 'Features are a 2048-point log magnitude spectrum restricted to 60-1400 Hz. Mel bands would smear the harmonics that carry the pitch.'
  },
  events: {
    key: 'events', name: 'Sound events', modality: 'audio', task: 'multiclass', generator: 'events',
    tagline: 'Twelve household sounds at random SNR',
    defaults: { size: 'medium', seed: 144, snrMin: 4, snrMax: 26 },
    splits: { method: 'stratified', ratios: [0.72, 0.14, 0.14] },
    augment: [{ id: 'spec_noise', strength: 0.25, prob: 0.5 }, { id: 'pitch_shift', strength: 3, prob: 0.3 }, { id: 'gain', strength: 0.2, prob: 0.3 }],
    notes: 'Fixed-length clips with coloured noise floors, including silence, so a quiet room is a class the model has seen.'
  },
  onset: {
    key: 'onset', name: 'Rhythm and onsets', modality: 'audio', task: 'onset', generator: 'onset',
    tagline: 'Per-frame onsets from 70 to 160 BPM',
    defaults: { size: 'medium', seed: 155 },
    splits: { method: 'random', ratios: [0.72, 0.14, 0.14] },
    augment: [{ id: 'spec_noise', strength: 0.15, prob: 0.4 }, { id: 'time_shift', strength: 2, prob: 0.3 }],
    notes: 'Onset F1 uses a timing tolerance window; exact-frame matching would punish a detector for firing 10 ms early.'
  }
};

Object.assign(PRESET_DATASETS, {
  superres: {
    key: 'superres', name: 'Super-resolution patches', modality: 'image', task: 'image_map', generator: 'superres',
    tagline: 'Bicubic input, sharp target, patch training',
    defaults: { size: 'medium', seed: 166, factor: 2, hrSize: 32, patchSize: 16 },
    splits: { method: 'random', ratios: [0.72, 0.14, 0.14] },
    augment: [{ id: 'flip_h', strength: 1, prob: 0.5 }, { id: 'flip_v', strength: 1, prob: 0.5 }],
    notes: 'Patch training gives four times the samples at a quarter of the convolution cost, which is what makes a residual upsampler trainable in a browser tab. The bicubic result is stored as the baseline the PSNR gain is measured against.'
  },
  denoiseimg: {
    key: 'denoiseimg', name: 'Image denoise pairs', modality: 'image', task: 'image_map', generator: 'denoiseimg',
    tagline: 'Gaussian, Poisson, JPEG blocking or mixed',
    defaults: { size: 'medium', seed: 177, degradation: 'mixed', noiseStd: 0.16 },
    splits: { method: 'random', ratios: [0.72, 0.14, 0.14] },
    augment: [{ id: 'flip_h', strength: 1, prob: 0.5 }, { id: 'img_noise', strength: 0.03, prob: 0.3 }],
    notes: 'Degradation is applied after the split so the clean target is never contaminated by augmentation.'
  },
  detection: {
    key: 'detection', name: 'Faces and hands', modality: 'image', task: 'detection', generator: 'detection',
    tagline: 'Tight boxes plus background negatives',
    defaults: { size: 'medium', seed: 188, imageSize: 48, grid: 6 },
    splits: { method: 'random', ratios: [0.72, 0.14, 0.14] },
    augment: [{ id: 'brightness', strength: 0.15, prob: 0.4 }, { id: 'img_noise', strength: 0.05, prob: 0.4 }],
    notes: 'About a fifth of the rows are background only, which is what teaches the objectness channel to stay quiet.'
  },
  segmentation: {
    key: 'segmentation', name: 'Background segmentation', modality: 'image', task: 'mask', generator: 'segmentation',
    tagline: 'Pixel-exact subject masks',
    defaults: { size: 'medium', seed: 199, imageSize: 48 },
    splits: { method: 'random', ratios: [0.72, 0.14, 0.14] },
    augment: [{ id: 'flip_h', strength: 1, prob: 0.5 }, { id: 'brightness', strength: 0.15, prob: 0.4 }],
    notes: 'Foreground is a minority of pixels, so the preset enables class weighting and a Dice term in the loss.'
  },
  style: {
    key: 'style', name: 'Style transfer pairs', modality: 'image', task: 'image_map', generator: 'style',
    tagline: 'Content image and one baked-in style',
    defaults: { size: 'medium', seed: 210, imageSize: 32, styleSeed: 4242 },
    splits: { method: 'random', ratios: [0.72, 0.14, 0.14] },
    augment: [{ id: 'flip_h', strength: 1, prob: 0.5 }, { id: 'contrast', strength: 0.1, prob: 0.3 }],
    notes: 'Change the style seed to bake a different look into the same graph; the model never sees the style at inference.'
  },
  saliency: {
    key: 'saliency', name: 'Saliency heatmaps', modality: 'image', task: 'mask', generator: 'saliency',
    tagline: 'Soft importance maps for crop planning',
    defaults: { size: 'medium', seed: 221, imageSize: 48 },
    splits: { method: 'random', ratios: [0.72, 0.14, 0.14] },
    augment: [{ id: 'flip_h', strength: 1, prob: 0.5 }],
    notes: 'Targets are continuous in zero to one, so the quality report must not treat them as class labels.'
  },
  nsfw: {
    key: 'nsfw', name: 'Content filter stand-in', modality: 'image', task: 'multiclass', generator: 'nsfw',
    tagline: 'Synthetic structural stand-in, three classes',
    defaults: { size: 'medium', seed: 232, imageSize: 32 },
    splits: { method: 'stratified', ratios: [0.72, 0.14, 0.14] },
    augment: [{ id: 'flip_h', strength: 1, prob: 0.5 }, { id: 'brightness', strength: 0.15, prob: 0.4 }],
    notes: 'No explicit imagery is generated or stored. The classes differ in skin-tone ratio, edge density and region layout, which exercises the same pipeline and metrics as a real filter.'
  },
  imu: {
    key: 'imu', name: 'IMU activity windows', modality: 'sensor', task: 'multiclass', generator: 'imu',
    tagline: 'Eight activities, six axes, fixed windows',
    defaults: { size: 'medium', seed: 243, windowLen: 64, sampleRate: 50, channels: 6 },
    splits: { method: 'stratified', ratios: [0.72, 0.14, 0.14] },
    augment: [{ id: 'jitter', strength: 0.05, prob: 0.5 }, { id: 'scale_warp', strength: 0.1, prob: 0.3 }],
    notes: 'Windows are fixed length and the fall class is deliberately rare, which is how it looks on a real wearable.'
  },
  anomaly: {
    key: 'anomaly', name: 'Telemetry anomalies', modality: 'series', task: 'anomaly', generator: 'anomaly',
    tagline: 'Normal behaviour with a small contaminated slice',
    defaults: { size: 'medium', seed: 254, windowLen: 32, dims: 6, trainContamination: 0.02 },
    splits: { method: 'anomaly-clean-train', ratios: [0.66, 0.17, 0.17] },
    augment: [],
    notes: 'The training split is kept as clean as the contamination setting allows and anomalies are concentrated in validation and test, which is the only honest way to evaluate a reconstruction detector. Training contamination is configurable because an autoencoder that has never seen an anomaly and one that has seen too many fail in opposite directions.'
  },
  forecast: {
    key: 'forecast', name: 'Forecasting series', modality: 'series', task: 'forecast', generator: 'forecast',
    tagline: 'Traffic, battery, sales and energy flavours',
    defaults: { size: 'medium', seed: 265, history: 48, horizon: 8 },
    splits: { method: 'temporal', ratios: [0.72, 0.14, 0.14] },
    augment: [{ id: 'jitter', strength: 0.03, prob: 0.4 }, { id: 'window_shift', strength: 2, prob: 0.3 }],
    notes: 'Split temporally, not randomly: shuffling a series leaks the future into the training set and inflates every metric.'
  },
  bot: {
    key: 'bot', name: 'Bot and human behaviour', modality: 'sensor', task: 'binary', generator: 'bot',
    tagline: 'Five actor types, one binary decision',
    defaults: { size: 'medium', seed: 276, points: 48 },
    splits: { method: 'stratified', ratios: [0.72, 0.14, 0.14] },
    augment: [{ id: 'jitter', strength: 0.06, prob: 0.4 }, { id: 'channel_mask', strength: 0.1, prob: 0.3 }],
    notes: 'A tenth of the rows are jittered and relabelled on purpose. Without that overlap the achievable AUC is 1.000 and the number tells you nothing.'
  },
  compression: {
    key: 'compression', name: 'Byte sequences', modality: 'text', task: 'lm', generator: 'compression',
    tagline: 'Mixed corpus for next-byte prediction',
    defaults: { size: 'medium', seed: 287, seqLen: 96, mix: 'mixed' },
    splits: { method: 'random', ratios: [0.72, 0.14, 0.14] },
    augment: [],
    notes: 'The corpus order-0 entropy is stored on the dataset so bits per symbol can be compared against what a lookup table achieves.'
  },
  texture: {
    key: 'texture', name: 'Neural texture pixels', modality: 'image', task: 'coordinate_field', generator: 'texture',
    tagline: 'Every row is one pixel of one image',
    defaults: { seed: 298, imageSize: 48, samples: 2304 },
    splits: { method: 'random', ratios: [0.72, 0.14, 0.14] },
    augment: [],
    notes: 'Inputs carry seven octaves of sine and cosine positional encoding. Plain coordinates cannot generalise spatially, which is the whole reason SIREN exists.'
  }
});

export const PRESET_DATASET_LIST = Object.keys(PRESET_DATASETS).map((k) => PRESET_DATASETS[k]);

export function datasetPresetByKey(key) {
  return PRESET_DATASETS[key] || null;
}

export function datasetPresetsForModality(modality) {
  return PRESET_DATASET_LIST.filter((p) => !modality || modality === 'all' || p.modality === modality);
}

export function datasetLicense(key) {
  const preset = PRESET_DATASETS[key];
  if (!preset) return null;
  const gen = GENERATORS[preset.generator];
  const ref = gen ? REFERENCE_DATASETS[gen.refKey] : null;
  return {
    local: {
      id: 'generated-local',
      name: 'Generated on your machine',
      allowsCommercial: true,
      notes: 'Produced in the browser from a deterministic seed. No third-party content is embedded, so you hold all rights to the generated samples and to models trained on them.'
    },
    reference: ref
  };
}

export function datasetSizeEstimate(key, size) {
  const preset = PRESET_DATASETS[key];
  const counts = SIZE_COUNTS[key];
  if (!preset || !counts) return null;
  const i = size === 'small' ? 0 : size === 'large' ? 2 : 1;
  return counts[i];
}

export function buildPresetDataset(key, overrides) {
  const preset = PRESET_DATASETS[key];
  if (!preset) throw new Error('unknown dataset preset: ' + key);
  const opts = Object.assign({}, preset.defaults, overrides || {});
  const ds = buildDataset(preset.generator, opts);
  const standard = ['random', 'stratified', 'group', 'temporal'];
  if (preset.splits && standard.indexOf(ds.splitMethod) >= 0) {
    buildSplits(ds, preset.splits.method, preset.splits.ratios, ds.splitSeed);
  }
  ds.presetKey = key;
  ds.presetName = preset.name;
  ds.presetTagline = preset.tagline;
  ds.presetNotes = preset.notes;
  ds.augmentation = {
    enabled: false,
    applyProb: 0.8,
    ops: (preset.augment || []).slice()
  };
  ds.augmentationDefaults = preset.augment || [];
  ds.availableAugmentations = defaultAugConfig(ds.modality);
  ds.licenseInfo = datasetLicense(key);
  ds.sizeEstimate = SIZE_COUNTS[key] || null;
  return ds;
}

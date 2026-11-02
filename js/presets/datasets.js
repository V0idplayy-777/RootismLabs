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
  texture: [1024, 2304, 4096],
  tickets: [380, 900, 2200],
  addresses: [300, 700, 1700],
  reviews: [420, 1000, 2400],
  voicemail: [120, 300, 700],
  fall: [280, 640, 1500],
  smartmeter: [180, 420, 1000],
  doctypes: [300, 700, 1700],
  receiptfields: [220, 520, 1300],
  skew: [300, 700, 1700],
  signatures: [260, 620, 1500],
  cardfraud: [900, 2200, 5000],
  creditrisk: [600, 1500, 3600],
  merchants: [420, 1000, 2600],
  latepayment: [380, 900, 2200],
  injection: [400, 950, 2200],
  phishing: [450, 1100, 2600],
  dga: [450, 1100, 2600],
  secrets: [260, 620, 1500],
  bearing: [260, 620, 1500],
  rul: [260, 620, 1500],
  surface: [220, 520, 1200],
  hvac: [300, 720, 1800],
  demand: [240, 560, 1300],
  products: [420, 1000, 2600],
  churn: [700, 1600, 4000],
  columntype: [380, 900, 2200],
  linkage: [240, 560, 1400],
  loganomaly: [240, 560, 1300],
  ecg: [300, 720, 1800]
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

Object.assign(PRESET_DATASETS, {
  tickets: {
    key: 'tickets', name: 'Support tickets', modality: 'text', task: 'multilabel', generator: 'tickets',
    tagline: 'Five teams, four priorities, eight bits',
    defaults: { size: 'medium', seed: 3230, dim: 768 },
    splits: { method: 'stratified', ratios: [0.72, 0.14, 0.14] },
    augment: [{ id: 'typos', strength: 0.05, prob: 0.4 }, { id: 'word_drop', strength: 0.1, prob: 0.25 }],
    notes: 'Team and priority are sampled independently, so neither can be read off the other. Per-label precision and recall are the numbers a queue owner cares about; exact match punishes a correct team with the wrong priority.'
  },
  addresses: {
    key: 'addresses', name: 'Free-text addresses', modality: 'text', task: 'seq_tag', generator: 'addresses',
    tagline: 'Unit, street, city and postcode as BIO',
    defaults: { size: 'medium', seed: 3240, maxLen: 80 },
    splits: { method: 'stratified', ratios: [0.72, 0.14, 0.14] },
    augment: [],
    notes: 'No text augmentation: the inputs are character ids, so rewriting the text would desynchronise the tags. Real checkout data has the same problem, which is why address parsers are usually trained on aligned OCR output.'
  },
  reviews: {
    key: 'reviews', name: 'Review aspects', modality: 'text', task: 'multilabel', generator: 'reviews',
    tagline: 'Price, delivery, quality and support',
    defaults: { size: 'medium', seed: 3250, dim: 640 },
    splits: { method: 'stratified', ratios: [0.72, 0.14, 0.14] },
    augment: [{ id: 'typos', strength: 0.05, prob: 0.4 }, { id: 'word_shuffle', strength: 2, prob: 0.2 }],
    notes: 'One to three aspects per review, so the label set is sparse. A model that predicts the two frequent aspects and ignores support still scores well on micro F1; macro F1 is the honest view.'
  },
  voicemail: {
    key: 'voicemail', name: 'Answering machine openings', modality: 'audio', task: 'binary', generator: 'voicemail',
    tagline: 'First three seconds, human or machine',
    defaults: { size: 'medium', seed: 3260, snrMin: 8, snrMax: 26 },
    splits: { method: 'stratified', ratios: [0.7, 0.15, 0.15] },
    augment: [{ id: 'spec_noise', strength: 0.18, prob: 0.45 }, { id: 'time_shift', strength: 2, prob: 0.3 }],
    notes: 'Machine greetings are flat and start immediately; humans pause and vary. A cheap pre-filter on the first three seconds decides whether to hand the call to an agent, so the cost of a false accept is a wasted agent minute.'
  },
  fall: {
    key: 'fall', name: 'Wearable fall windows', modality: 'sensor', task: 'binary', generator: 'fall',
    tagline: 'Fifteen percent falls, triaxial IMU',
    defaults: { size: 'medium', seed: 3270, windowLen: 96, sampleRate: 50, fallRate: 0.15 },
    splits: { method: 'stratified', ratios: [0.7, 0.15, 0.15] },
    augment: [{ id: 'jitter', strength: 0.04, prob: 0.5 }, { id: 'scale_warp', strength: 0.08, prob: 0.3 }, { id: 'window_shift', strength: 3, prob: 0.3 }],
    notes: 'A fall rate of fifteen percent is already generous compared with a real elder-care population. Recall on the fall class, not accuracy, is the number to read, and every extra point of recall usually costs false alarms at night.'
  },
  smartmeter: {
    key: 'smartmeter', name: 'Smart meter load', modality: 'series', task: 'forecast', generator: 'smartmeter',
    tagline: 'Hourly kWh plus calendar channels',
    defaults: { size: 'medium', seed: 3280, history: 48, horizon: 24 },
    splits: { method: 'temporal', ratios: [0.72, 0.14, 0.14] },
    augment: [],
    notes: 'Split temporally. The persistence baseline is reported alongside the model because last-value forecasting is genuinely strong on load data, and a model that cannot beat it has learned nothing useful.'
  },
  doctypes: {
    key: 'doctypes', name: 'Document types', modality: 'image', task: 'multiclass', generator: 'doctypes',
    tagline: 'Invoice, receipt, ID, contract, other',
    defaults: { size: 'medium', seed: 3001, imageSize: 32 },
    splits: { method: 'stratified', ratios: [0.72, 0.14, 0.14] },
    augment: [{ id: 'brightness', strength: 0.12, prob: 0.4 }, { id: 'img_noise', strength: 0.04, prob: 0.4 }, { id: 'blur', strength: 1, prob: 0.25 }],
    notes: 'Pages carry a few degrees of rotation and scan noise so the classifier cannot key on exact pixel positions. Swap in real scans and keep the same graph and metrics.'
  },
  receiptfields: {
    key: 'receiptfields', name: 'Receipt and invoice fields', modality: 'text', task: 'seq_tag', generator: 'receiptfields',
    tagline: 'OCR tokens to vendor, date, total, tax id',
    defaults: { size: 'medium', seed: 3010, maxLen: 40 },
    splits: { method: 'stratified', ratios: [0.72, 0.14, 0.14] },
    augment: [],
    notes: 'Token ids, not characters, so augmentation would desynchronise the labels and is left off. Entity F1 counts a whole field span, which is closer to what a downstream accounting system needs than token accuracy.'
  },
  skew: {
    key: 'skew', name: 'Page skew angles', modality: 'image', task: 'regression', generator: 'skew',
    tagline: 'Rotation angle in degrees',
    defaults: { size: 'medium', seed: 3020, imageSize: 32, maxAngle: 12 },
    splits: { method: 'random', ratios: [0.72, 0.14, 0.14] },
    augment: [{ id: 'img_noise', strength: 0.03, prob: 0.4 }, { id: 'brightness', strength: 0.1, prob: 0.3 }],
    notes: 'Rotation is applied after rendering, so the target is exact. Do not enable flip or rotate augmentation here: it changes the label, and the model would be trained against a constantly wrong answer.'
  },
  signatures: {
    key: 'signatures', name: 'Signature and stamp boxes', modality: 'image', task: 'detection', generator: 'signatures',
    tagline: 'Signed, stamped and blank pages',
    defaults: { size: 'medium', seed: 3030, imageSize: 32, grid: 4 },
    splits: { method: 'random', ratios: [0.72, 0.14, 0.14] },
    augment: [{ id: 'brightness', strength: 0.12, prob: 0.4 }, { id: 'img_noise', strength: 0.03, prob: 0.35 }],
    notes: 'Box localisation is the weak part of a small detector. A compliance check usually only needs objectness plus a human eye, so report mAP50 and the operating point together.'
  },
  cardfraud: {
    key: 'cardfraud', name: 'Card transaction features', modality: 'tabular', task: 'binary', generator: 'cardfraud',
    tagline: 'Cost-weighted threshold, rare positives',
    defaults: { size: 'medium', seed: 3040 },
    splits: { method: 'stratified', ratios: [0.7, 0.15, 0.15] },
    augment: [{ id: 'jitter', strength: 0.02, prob: 0.3 }],
    notes: 'The dataset carries the cost ratio between a missed fraud and a false alert; the cost-optimal threshold for a calibrated scorer is that ratio, not 0.5. Any threshold chosen here still has to be re-checked on real traffic.'
  },
  creditrisk: {
    key: 'creditrisk', name: 'Loan application features', modality: 'tabular', task: 'binary', generator: 'creditrisk',
    tagline: 'Applicant features, eighteen percent default',
    defaults: { size: 'medium', seed: 3050 },
    splits: { method: 'stratified', ratios: [0.7, 0.15, 0.15] },
    augment: [],
    notes: 'Regulated domain. Probabilities need calibration before an acceptance policy uses them, and the model card flags adverse-action reasons, fairness testing and audit trails as requirements rather than extras.'
  },
  merchants: {
    key: 'merchants', name: 'Merchant descriptors', modality: 'text', task: 'multiclass', generator: 'merchants',
    tagline: 'Raw descriptors to spend categories',
    defaults: { size: 'medium', seed: 3060, dim: 640 },
    splits: { method: 'stratified', ratios: [0.72, 0.14, 0.14] },
    augment: [{ id: 'typos', strength: 0.05, prob: 0.4 }, { id: 'case_flip', strength: 0.2, prob: 0.3 }],
    notes: 'Descriptors arrive with processor prefixes, store numbers and city suffixes. Normalise the obvious noise before feature hashing, but keep the raw string for auditing a prediction.'
  },
  latepayment: {
    key: 'latepayment', name: 'Invoice payment history', modality: 'tabular', task: 'regression', generator: 'latepayment',
    tagline: 'Days late from invoice features',
    defaults: { size: 'medium', seed: 3070 },
    splits: { method: 'random', ratios: [0.72, 0.14, 0.14] },
    augment: [{ id: 'jitter', strength: 0.02, prob: 0.3 }],
    notes: 'A large share of invoices are paid on time, so MAE is dominated by zeros. Ranking by predicted lateness and the tail error are better guides for a collections queue.'
  },
  injection: {
    key: 'injection', name: 'Prompt injection prompts', modality: 'text', task: 'binary', generator: 'injection',
    tagline: 'Benign requests and override attempts',
    defaults: { size: 'medium', seed: 3080, dim: 768, positiveRate: 0.32 },
    splits: { method: 'stratified', ratios: [0.7, 0.15, 0.15] },
    augment: [{ id: 'typos', strength: 0.04, prob: 0.3 }, { id: 'word_drop', strength: 0.08, prob: 0.2 }],
    notes: 'The attack templates are public, so a held-out score here measures pattern recall, not robustness. Treat the filter as one layer next to output-side controls and rate limits.'
  },
  phishing: {
    key: 'phishing', name: 'Phishing URLs', modality: 'text', task: 'binary', generator: 'phishing',
    tagline: 'Hashed character n-grams of a URL',
    defaults: { size: 'medium', seed: 3090, dim: 768, positiveRate: 0.35 },
    splits: { method: 'stratified', ratios: [0.7, 0.15, 0.15] },
    augment: [{ id: 'typos', strength: 0.03, prob: 0.3 }],
    notes: 'URLs are adversarial and short-lived. Retrain on a rolling window, evaluate on a later time slice, and keep a blocklist alongside the model.'
  },
  dga: {
    key: 'dga', name: 'Domain names', modality: 'text', task: 'binary', generator: 'dga',
    tagline: 'Four DGA families against real names',
    defaults: { size: 'medium', seed: 3100, dim: 512, positiveRate: 0.35 },
    splits: { method: 'stratified', ratios: [0.7, 0.15, 0.15] },
    augment: [{ id: 'typos', strength: 0.03, prob: 0.25 }],
    notes: 'DGA families rotate. A model that has only seen random-letter generators will miss a dictionary-based family, so hold out whole families rather than random rows when you can.'
  },
  secrets: {
    key: 'secrets', name: 'Secrets in code and logs', modality: 'text', task: 'seq_tag', generator: 'secrets',
    tagline: 'API keys, tokens, passwords, PEM headers',
    defaults: { size: 'medium', seed: 3110, maxLen: 96, leakRate: 0.72 },
    splits: { method: 'stratified', ratios: [0.72, 0.14, 0.14] },
    augment: [],
    notes: 'Character ids, so text augmentation is off. A hit is a real credential: wire the output to revocation, not only to a dashboard.'
  },
  bearing: {
    key: 'bearing', name: 'Bearing vibration spectra', modality: 'sensor', task: 'multiclass', generator: 'bearing',
    tagline: 'Healthy, inner race, outer race, ball',
    defaults: { size: 'medium', seed: 3120, sampleRate: 2000, windowLen: 512 },
    splits: { method: 'stratified', ratios: [0.72, 0.14, 0.14] },
    augment: [{ id: 'jitter', strength: 0.03, prob: 0.4 }, { id: 'scale_warp', strength: 0.06, prob: 0.3 }],
    notes: 'Features are already spectra, so the model starts where the FFT ends. Field data has varying speed and load, which this generator holds fixed; expect the accuracy to drop.'
  },
  rul: {
    key: 'rul', name: 'Degrading machine windows', modality: 'sensor', task: 'regression', generator: 'rul',
    tagline: 'Cycles left from four sensors',
    defaults: { size: 'medium', seed: 3130, windowLen: 32, maxCycles: 120 },
    splits: { method: 'random', ratios: [0.72, 0.14, 0.14] },
    augment: [{ id: 'jitter', strength: 0.02, prob: 0.4 }, { id: 'scale_warp', strength: 0.05, prob: 0.3 }],
    notes: 'Split by unit, not by window, when you use real run-to-failure data: windows from one machine leak its whole trajectory into the training set.'
  },
  surface: {
    key: 'surface', name: 'Surface defect patches', modality: 'image', task: 'anomaly', generator: 'surface',
    tagline: 'Good parts only, defect heatmap',
    defaults: { size: 'medium', seed: 3140, imageSize: 24, defectRate: 0.15, trainContamination: 0.01 },
    splits: { method: 'anomaly-clean-train', ratios: [0.66, 0.17, 0.17] },
    augment: [],
    notes: 'The training split is almost entirely good parts, so the reconstruction baseline is not polluted. The per-pixel error is the heatmap an inspector sees; a single window-level score is much easier to over-trust.'
  },
  hvac: {
    key: 'hvac', name: 'HVAC fault windows', modality: 'sensor', task: 'multiclass', generator: 'hvac',
    tagline: 'Six conditions across eight channels',
    defaults: { size: 'medium', seed: 3150, windowLen: 32 },
    splits: { method: 'stratified', ratios: [0.72, 0.14, 0.14] },
    augment: [{ id: 'jitter', strength: 0.03, prob: 0.4 }, { id: 'channel_mask', strength: 0.08, prob: 0.25 }],
    notes: 'Each fault moves a different subset of channels, which is what makes a small temporal model plausible. Sensor bias is the hard class: it looks like a fault in one channel and nothing in the rest.'
  },
  demand: {
    key: 'demand', name: 'Demand with covariates', modality: 'series', task: 'forecast', generator: 'demand',
    tagline: 'Sales plus promotion and holiday flags',
    defaults: { size: 'medium', seed: 3160, history: 48, horizon: 14 },
    splits: { method: 'temporal', ratios: [0.72, 0.14, 0.14] },
    augment: [],
    notes: 'Covariates are only supplied for the history window, so the model predicts the baseline. A deployment that knows future promotions should append them and retrain; feeding future flags into a random split would leak the answer.'
  },
  products: {
    key: 'products', name: 'Product titles', modality: 'text', task: 'multiclass', generator: 'products',
    tagline: 'Titles to twelve taxonomy nodes',
    defaults: { size: 'medium', seed: 3170, dim: 768 },
    splits: { method: 'stratified', ratios: [0.72, 0.14, 0.14] },
    augment: [{ id: 'typos', strength: 0.04, prob: 0.35 }, { id: 'word_drop', strength: 0.08, prob: 0.25 }],
    notes: 'Taxonomy labels drift with merchandising decisions. Version the taxonomy next to the model, and evaluate on titles from brands the model has never seen.'
  },
  churn: {
    key: 'churn', name: 'Subscription usage', modality: 'tabular', task: 'binary', generator: 'churn',
    tagline: 'Thirty-day cancellation risk',
    defaults: { size: 'medium', seed: 3180 },
    splits: { method: 'stratified', ratios: [0.7, 0.15, 0.15] },
    augment: [{ id: 'jitter', strength: 0.02, prob: 0.3 }],
    notes: 'The label is only known after the thirty-day horizon, so the newest data is always unlabelled. Split by time when you deploy, or the training set will look easier than the live problem.'
  },
  columntype: {
    key: 'columntype', name: 'Column samples', modality: 'text', task: 'multiclass', generator: 'columntype',
    tagline: 'Six values to email, phone, date, currency, ID, text',
    defaults: { size: 'medium', seed: 3190, dim: 512, sampleValues: 6 },
    splits: { method: 'stratified', ratios: [0.72, 0.14, 0.14] },
    augment: [{ id: 'typos', strength: 0.03, prob: 0.3 }],
    notes: 'Sampling a column for PII discovery is itself a privacy-sensitive action. Log which columns were sampled, by whom, and keep the sample out of the training export if it contains real values.'
  },
  linkage: {
    key: 'linkage', name: 'Entity resolution pairs', modality: 'text', task: 'embedding', generator: 'linkage',
    tagline: 'Two CRM records, one entity',
    defaults: { size: 'medium', seed: 3200, dim: 512, rowsPerEntity: 4 },
    splits: { method: 'group', ratios: [0.7, 0.15, 0.15] },
    augment: [{ id: 'typos', strength: 0.04, prob: 0.3 }],
    notes: 'Split by entity. If two records of the same person land on both sides of the boundary, retrieval scores are pure leakage. The match score is cosine similarity, so the threshold is a product decision made on validation data.'
  },
  loganomaly: {
    key: 'loganomaly', name: 'Log template counts', modality: 'series', task: 'anomaly', generator: 'loganomaly',
    tagline: 'Eight templates, four anomaly kinds',
    defaults: { size: 'medium', seed: 3210, windowLen: 32, trainContamination: 0.01 },
    splits: { method: 'anomaly-clean-train', ratios: [0.66, 0.17, 0.17] },
    augment: [],
    notes: 'Train almost entirely on normal windows and keep the anomalies in validation and test. Template extraction is a model of its own: a log-format change silently redefines every feature.'
  },
  ecg: {
    key: 'ecg', name: 'ECG beat windows', modality: 'series', task: 'multiclass', generator: 'ecg',
    tagline: 'Normal, premature and other morphology',
    defaults: { size: 'medium', seed: 3220, windowLen: 128, sampleRate: 125 },
    splits: { method: 'stratified', ratios: [0.7, 0.15, 0.15] },
    augment: [{ id: 'jitter', strength: 0.02, prob: 0.4 }, { id: 'scale_warp', strength: 0.04, prob: 0.25 }],
    notes: 'Beat-level splits leak patient identity: the same heart appears across splits. A real evaluation splits by patient, and the model card for this preset says the output is regulated if it guides care.'
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

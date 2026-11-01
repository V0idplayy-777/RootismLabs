import { analyze, blankGraph } from '../core/model.js';
import { defaultConfig, isLayerType } from '../core/layers.js';

export const PRESET_MODELS = {
  langid: {
    key: 'langid',
    name: 'Language identifier',
    family: 'text',
    task: 'multiclass',
    dataset: 'langid',
    tagline: 'Twelve languages from hashed character n-grams',
    description: 'A flat classifier over hashed character n-grams. No vocabulary ships with the model, so an unseen word degrades gracefully instead of collapsing to unknown tokens. The dropout before the penultimate layer is what keeps twelve languages from being memorised off four hundred training sentences.',
    input: [640],
    inputName: 'char_ngrams',
    head: 'softmax',
    nodes: [
      ['dense', { units: 128 }],
      ['dropout', { rate: 0.25 }],
      ['dense', { units: 64 }],
      ['dense', { units: 12, activation: 'linear' }]
    ],
    train: { epochs: 18, batchSize: 32, optimizer: { kind: 'adam', lr: 0.008 } },
    budget: { target: 95000, min: 20000, max: 400000 },
    reference: { params: [200000, 2000000], note: 'Production language id runs on a fastText-style shallow net or a distilled transformer; both are larger than what trains comfortably in a tab.' }
  },
  intent: {
    key: 'intent',
    name: 'Intent classifier',
    family: 'text',
    task: 'multiclass',
    dataset: 'intent',
    tagline: 'Eighteen assistant intents from one utterance',
    description: 'The routing model at the front of an assistant. Hashed word and character n-grams feed a two-layer classifier, and per-intent precision and recall matter more than overall accuracy because a single misrouted intent costs a whole conversation turn.',
    input: [900],
    inputName: 'utterance_features',
    head: 'softmax',
    nodes: [
      ['dense', { units: 128 }],
      ['dropout', { rate: 0.2 }],
      ['dense', { units: 64 }],
      ['dense', { units: 18, activation: 'linear' }]
    ],
    train: { epochs: 18, batchSize: 32, optimizer: { kind: 'adam', lr: 0.008 } },
    budget: { target: 130000, min: 20000, max: 500000 },
    reference: { params: [500000, 5000000], note: 'Shipped assistants usually pair this with a slot tagger; the classifier alone is rarely the largest model in the stack.' }
  },
  spam: {
    key: 'spam',
    name: 'Spam and toxicity filter',
    family: 'text',
    task: 'multiclass',
    dataset: 'spam',
    tagline: 'Three-way moderation head with label noise',
    description: 'Ham, spam, toxic. Moderation data is noisy by nature, so the preset dataset carries a configurable label-noise rate and the evaluation shows per-class recall: missing toxic content and deleting legitimate messages are different failures with different costs.',
    input: [768],
    inputName: 'message_features',
    head: 'softmax',
    nodes: [
      ['dense', { units: 96 }],
      ['dropout', { rate: 0.2 }],
      ['dense', { units: 48 }],
      ['dense', { units: 3, activation: 'linear' }]
    ],
    train: { epochs: 18, batchSize: 32, optimizer: { kind: 'adam', lr: 0.008 } },
    budget: { target: 80000, min: 15000, max: 400000 },
    reference: { params: [1000000, 10000000], note: 'Real moderation stacks are large multilingual encoders; this is the shape of the final classifier head.' }
  },
  pii: {
    key: 'pii',
    name: 'PII redactor',
    family: 'text',
    task: 'seq_tag',
    dataset: 'pii',
    tagline: 'Character-level BIO tagging for seven entity types',
    description: 'A character-input sequence labeller that marks emails, phone numbers, card numbers, API keys, names, addresses and dates so they can be redacted before text reaches a log. Character input means it still fires on tokens no word vocabulary has ever seen, which is most API keys.',
    input: [96],
    inputName: 'char_ids',
    head: 'softmax-seq',
    nodes: [
      ['reshape', { shape: '96,1' }],
      ['embedding', { vocab: 160, dim: 24 }],
      ['conv1d', { filters: 48, kernel: 3 }],
      ['dropout', { rate: 0.2 }],
      ['conv1d', { filters: 48, kernel: 3 }],
      ['dense', { units: 15, activation: 'linear' }]
    ],
    train: { epochs: 30, batchSize: 32, optimizer: { kind: 'adamw', lr: 0.006, weightDecay: 0.0005 }, labelSmoothing: 0.05 },
    budget: { target: 30000, min: 8000, max: 150000 },
    reference: { params: [10000000, 350000000], note: 'Production PII pipelines use a fine-tuned transformer; the character-window convolution here is the small-model alternative that still catches malformed identifiers.' }
  },
  code: {
    key: 'code',
    name: 'Code language detector',
    family: 'text',
    task: 'multiclass',
    dataset: 'code',
    tagline: 'Sixteen languages from syntax, not identifiers',
    description: 'Snippets are mutated so identifiers get renamed and fragments get recombined, which forces the model onto syntax: brace style, keyword density, indentation rhythm. Useful for routing a code block to the right highlighter or linter.',
    input: [1024],
    inputName: 'token_ngrams',
    head: 'softmax',
    nodes: [
      ['dense', { units: 160 }],
      ['dropout', { rate: 0.25 }],
      ['dense', { units: 80 }],
      ['dense', { units: 16, activation: 'linear' }]
    ],
    train: { epochs: 18, batchSize: 32, optimizer: { kind: 'adam', lr: 0.008 } },
    budget: { target: 180000, min: 30000, max: 600000 },
    reference: { params: [1000000, 50000000], note: 'Repository-scale language detection usually rides on an existing code embedding; a standalone head is far smaller.' }
  },
  password: {
    key: 'password',
    name: 'Password strength',
    family: 'text',
    task: 'multiclass',
    dataset: 'password',
    tagline: 'Five bands from structure and guess rank',
    description: 'Strength as an ordinal five-band classification from character-class runs, dictionary hits, keyboard walks and an estimated guess rank. Every reference corpus for this task is a leaked password dump and is not usable commercially, so the preset dataset is synthesised from structure alone.',
    input: [320],
    inputName: 'password_features',
    head: 'softmax',
    nodes: [
      ['dense', { units: 96 }],
      ['dropout', { rate: 0.15 }],
      ['dense', { units: 48 }],
      ['dense', { units: 5, activation: 'linear' }]
    ],
    train: { epochs: 20, batchSize: 32, optimizer: { kind: 'adam', lr: 0.008 } },
    budget: { target: 40000, min: 8000, max: 200000 },
    reference: { params: [50000, 2000000], note: 'zxcvbn is a hand-written matcher, not a network; a learned band classifier sits in the same size range.' }
  },
  embeddings: {
    key: 'embeddings',
    name: 'Sentence embeddings',
    family: 'text',
    task: 'embedding',
    dataset: 'embeddings',
    tagline: 'Contrastive vectors for dedupe and retrieval',
    description: 'A projection head trained with a group-aware contrastive loss: paraphrases of the same request pull together, everything else pushes apart. The output is L2-normalised so cosine similarity is a dot product, which is what a vector index wants.',
    input: [512],
    inputName: 'sentence_features',
    head: 'embedding',
    nodes: [
      ['dense', { units: 96 }],
      ['dropout', { rate: 0.15 }],
      ['dense', { units: 48, activation: 'linear' }],
      ['l2normalize', {}]
    ],
    train: { epochs: 40, batchSize: 32, optimizer: { kind: 'adamw', lr: 0.003, weightDecay: 0.0005 }, temperature: 0.07, loss: 'supcon' },
    budget: { target: 55000, min: 10000, max: 300000 },
    reference: { params: [20000000, 120000000], note: 'MiniLM-class encoders are the usual commercial choice; a projection head over hashed features is the sub-100K alternative.' }
  },
  autocomplete: {
    key: 'autocomplete',
    name: 'Character autocomplete',
    family: 'text',
    task: 'lm',
    dataset: 'autocomplete',
    tagline: 'Next-character GRU over a 128 symbol vocabulary',
    description: 'The smallest model that can power inline suggestion: an embedding, one recurrent layer and a softmax over the symbol table. Perplexity and bits per symbol are reported next to each other because bits per symbol is what an inline suggestion actually costs to transmit.',
    input: [64],
    inputName: 'symbol_ids',
    head: 'softmax-seq',
    nodes: [
      ['embedding', { vocab: 128, dim: 24 }],
      ['gru', { units: 48, returnSeq: true }],
      ['dense', { units: 128, activation: 'linear' }]
    ],
    train: { epochs: 12, batchSize: 32, optimizer: { kind: 'adam', lr: 0.006 } },
    budget: { target: 22000, min: 5000, max: 200000 },
    reference: { params: [1000000, 100000000], note: 'GPT-2 small is 124M parameters; on-device autocomplete usually ships a pruned GRU or a quantised transformer under 10M.' }
  }
};

Object.assign(PRESET_MODELS, {
  wake: {
    key: 'wake',
    name: 'Wake word',
    family: 'audio',
    task: 'binary',
    dataset: 'wake',
    tagline: 'Keyword spotting on a fixed 36 × 32 log-mel grid',
    description: 'Two strided convolutions over log-mel frames, then a recurrent layer that reads the whole clip and a single decision unit. The numbers that matter are false accepts and false rejects per hour of audio, because a wake word runs continuously and a single false accept per night is enough to make a device feel broken.',
    input: [36, 32],
    inputName: 'log_mel',
    head: 'sigmoid',
    nodes: [
      ['conv1d', { filters: 24, kernel: 3 }],
      ['conv1d', { filters: 32, kernel: 3, stride: 2 }],
      ['gru', { units: 32, returnSeq: false }],
      ['dense', { units: 1, activation: 'linear' }]
    ],
    train: { epochs: 16, batchSize: 32, optimizer: { kind: 'adam', lr: 0.006 } },
    budget: { target: 12000, min: 4000, max: 120000 },
    reference: { params: [100000, 1000000], note: 'The 100K to 1M range in the brief is what ships on a DSP; widen the convolution and recurrent layers here to walk towards it.' }
  },
  vad: {
    key: 'vad',
    name: 'Voice activity detection',
    family: 'audio',
    task: 'frame_binary',
    dataset: 'vad',
    tagline: 'One speech decision per 20 ms frame',
    description: 'A Silero-shaped problem: the label is per frame, not per clip, so the recurrent layer returns its whole sequence and the head emits one logit per frame. Frame AUC plus false-accept and false-reject rates at the chosen threshold are what a real-time pipeline is tuned against.',
    input: [31, 32],
    inputName: 'log_mel',
    head: 'sigmoid-seq',
    nodes: [
      ['conv1d', { filters: 24, kernel: 3 }],
      ['gru', { units: 32, returnSeq: true }],
      ['dense', { units: 1, activation: 'linear' }]
    ],
    train: { epochs: 14, batchSize: 32, optimizer: { kind: 'adam', lr: 0.006 } },
    budget: { target: 9000, min: 3000, max: 200000 },
    reference: { params: [1000000, 2000000], note: 'Silero VAD is about 1M parameters and runs on a single thread; this preset is the same topology one hundred times narrower.' }
  },
  denoise: {
    key: 'denoise',
    name: 'Noise suppression',
    family: 'audio',
    task: 'spectral_mask',
    dataset: 'denoise',
    tagline: 'RNNoise-shaped spectral mask estimator',
    description: 'Predicts an ideal magnitude mask from the noisy log power spectrum, then rebuilds a waveform with an inverse STFT using the noisy phase. SI-SDR improvement over the unprocessed input is the headline, with segmental SNR and a perceptual estimate alongside; the perceptual number is a documented surrogate, not ITU-T P.862.',
    input: [30, 129],
    inputName: 'noisy_log_power',
    head: 'mask',
    nodes: [
      ['conv1d', { filters: 32, kernel: 3 }],
      ['gru', { units: 32, returnSeq: true }],
      ['dense', { units: 129, activation: 'linear' }]
    ],
    train: { epochs: 25, batchSize: 16, optimizer: { kind: 'adam', lr: 0.004 } },
    budget: { target: 24000, min: 8000, max: 100000 },
    reference: { params: [85000, 100000], note: 'RNNoise is under 100K parameters by design; this preset sits in the same bracket.' }
  },
  pitch: {
    key: 'pitch',
    name: 'Pitch classifier',
    family: 'audio',
    task: 'pitch_class',
    dataset: 'pitch',
    tagline: 'Thirty-six chromatic classes from a linear spectrum',
    description: 'Frequency resolution decides this task, so the preset dataset uses a 2048-point linear spectrum restricted to 60-1400 Hz instead of mel bands, which would smear the harmonics together. Time is pooled before the classifier because pitch is stationary inside a 320 ms window. The headline metric is cent error, not accuracy: a quarter-tone mistake and an octave mistake are different failures.',
    input: [7, 174],
    inputName: 'log_spectrum',
    head: 'softmax',
    nodes: [
      ['temporalpool', { mode: 'mean' }],
      ['dense', { units: 64 }],
      ['dropout', { rate: 0.3 }],
      ['dense', { units: 36, activation: 'linear' }]
    ],
    train: { epochs: 40, batchSize: 32, optimizer: { kind: 'adamw', lr: 0.004, weightDecay: 0.0004 }, labelSmoothing: 0.1 },
    budget: { target: 14000, min: 4000, max: 150000 },
    reference: { params: [50000, 1000000], note: 'CREPE is about 24M parameters; a chromatic classifier over a narrow band is two orders of magnitude smaller.' }
  },
  events: {
    key: 'events',
    name: 'Sound event classifier',
    family: 'audio',
    task: 'multiclass',
    dataset: 'events',
    tagline: 'Twelve household sounds at random SNR',
    description: 'Doorbell, smoke alarm, glass break, baby cry, dog bark and seven more, mixed into coloured noise at random signal-to-noise ratios and cut to a fixed frame grid. Strided convolutions handle the time axis and a recurrent layer pools the clip, which is the standard small-footprint acoustic event shape.',
    input: [38, 32],
    inputName: 'log_mel',
    head: 'softmax',
    nodes: [
      ['conv1d', { filters: 24, kernel: 5 }],
      ['conv1d', { filters: 32, kernel: 3, stride: 2 }],
      ['gru', { units: 32, returnSeq: false }],
      ['dense', { units: 12, activation: 'linear' }]
    ],
    train: { epochs: 16, batchSize: 32, optimizer: { kind: 'adam', lr: 0.006 } },
    budget: { target: 14000, min: 5000, max: 200000 },
    reference: { params: [1000000, 7000000], note: 'YAMNet and PANNs sit in this range and cover 521 classes; twelve classes needs far less.' }
  },
  onset: {
    key: 'onset',
    name: 'Beat and onset detector',
    family: 'audio',
    task: 'onset',
    dataset: 'onset',
    tagline: 'Per-frame onsets plus a tempo estimate',
    description: 'Onset detection as per-frame binary classification with a timing tolerance in the F1, because punishing a model for being ten milliseconds early produces a detector that never fires. A tempo estimate in BPM is reported next to it, which is what a beat tracker is actually judged on.',
    input: [98, 32],
    inputName: 'log_mel',
    head: 'sigmoid-seq',
    nodes: [
      ['conv1d', { filters: 24, kernel: 5 }],
      ['conv1d', { filters: 24, kernel: 3 }],
      ['gru', { units: 24, returnSeq: true }],
      ['dense', { units: 1, activation: 'linear' }]
    ],
    train: { epochs: 14, batchSize: 32, optimizer: { kind: 'adam', lr: 0.006 } },
    budget: { target: 10000, min: 4000, max: 150000 },
    reference: { params: [200000, 3000000], note: 'Madmom-style onset detectors are a few hundred thousand parameters; the tolerance window matters more than the size.' }
  },
  superres: {
    key: 'superres',
    name: 'Super-resolution',
    family: 'vision',
    task: 'image_map',
    dataset: 'superres',
    tagline: 'FSRCNN-shaped residual patch upsampler',
    description: 'Three convolutions that predict a residual on top of the bicubic input, added back with a skip connection whose output layer starts at zero. Zero initialisation matters: the network begins as an exact copy of the baseline and only moves where it can prove an improvement, which is why the reported PSNR gain is never negative at epoch one.',
    input: [3, 16, 16],
    inputName: 'bicubic_patch',
    head: 'mask',
    nodes: [
      ['conv2d', { filters: 12, kernel: 5 }],
      ['conv2d', { filters: 10, kernel: 3 }],
      ['conv2d', { filters: 3, kernel: 3, activation: 'linear', kernelInit: 'zeros' }],
      ['add', {}, { from: ['prev', 'input'] }]
    ],
    train: { epochs: 40, batchSize: 32, optimizer: { kind: 'adam', lr: 0.01 } },
    budget: { target: 3000, min: 1000, max: 60000 },
    reference: { params: [12000, 1500000], note: 'FSRCNN is about 12K parameters, which is the bracket this preset targets; EDSR and RCAN are far larger.' }
  },
  denoiseimg: {
    key: 'denoiseimg',
    name: 'Image denoiser',
    family: 'vision',
    task: 'image_map',
    dataset: 'denoiseimg',
    tagline: 'Residual cleanup for noise and JPEG blocking',
    description: 'The same residual trick as super-resolution applied to degradation removal: predict what to add back to the degraded input rather than the whole image. Works across Gaussian noise, Poisson-like noise and coarse quantisation because the residual is small and mostly local.',
    input: [3, 32, 32],
    inputName: 'degraded_image',
    head: 'mask',
    nodes: [
      ['conv2d', { filters: 16, kernel: 3 }],
      ['conv2d', { filters: 16, kernel: 3 }],
      ['conv2d', { filters: 3, kernel: 3, activation: 'linear', kernelInit: 'zeros' }],
      ['add', {}, { from: ['prev', 'input'] }]
    ],
    train: { epochs: 20, batchSize: 24, optimizer: { kind: 'adam', lr: 0.008 } },
    budget: { target: 5000, min: 1500, max: 200000 },
    reference: { params: [500000, 26000000], note: 'DnCNN is about 556K and Restormer is 26M; a two-layer residual net is the sub-10K end of the same idea.' }
  },
  detection: {
    key: 'detection',
    name: 'Face and hand detector',
    family: 'vision',
    task: 'detection',
    dataset: 'detection',
    tagline: 'BlazeFace-style single-shot grid head',
    description: 'A strided convolution stack that reduces a 48 × 48 image to a 6 × 6 grid, where channel zero is objectness and the remaining channels are box centre and size. Objectness uses a logistic loss over every cell and box coordinates are supervised over every cell of an image that contains an object, which gives the regression four times the gradient signal of positive-cell-only supervision.',
    input: [3, 48, 48],
    inputName: 'image',
    head: 'det',
    nodes: [
      ['conv2d', { filters: 16, kernel: 3, stride: 2 }],
      ['conv2d', { filters: 24, kernel: 3, stride: 2 }],
      ['conv2d', { filters: 24, kernel: 3, stride: 2 }],
      ['conv2d', { filters: 5, kernel: 3, activation: 'linear' }]
    ],
    train: { epochs: 30, batchSize: 24, optimizer: { kind: 'adam', lr: 0.005 }, coordScale: 4, confThreshold: 0.3 },
    budget: { target: 11000, min: 4000, max: 500000 },
    reference: { params: [100000, 500000], note: 'BlazeFace is about 100K to 500K parameters for two anchors per cell; widen the stack here to approach it.' }
  },
  segmentation: {
    key: 'segmentation',
    name: 'Background segmenter',
    family: 'vision',
    task: 'mask',
    dataset: 'segmentation',
    tagline: 'Encoder-decoder with one upsample stage',
    description: 'Pool down, widen, upsample back, refine. The loss is half binary cross-entropy and half Dice, because foreground is a minority of pixels and cross-entropy alone settles for predicting background everywhere. Boundary AUC is reported separately from IoU: a model can find the object and still have soft edges.',
    input: [3, 32, 32],
    inputName: 'image',
    head: 'heatmap',
    nodes: [
      ['conv2d', { filters: 16, kernel: 3 }],
      ['maxpool2d', { kernel: 2, stride: 2 }],
      ['conv2d', { filters: 24, kernel: 3 }],
      ['upsample2d', { factor: 2 }],
      ['conv2d', { filters: 16, kernel: 3 }],
      ['conv2d', { filters: 1, kernel: 3, activation: 'linear' }]
    ],
    train: { epochs: 25, batchSize: 24, optimizer: { kind: 'adam', lr: 0.006 }, loss: 'bce_dice' },
    budget: { target: 8000, min: 2000, max: 1000000 },
    reference: { params: [100000, 1000000], note: 'MobileNetV3-based segmenters land in the 100K to 1M range; add depth or width here to reach it.' }
  },
  style: {
    key: 'style',
    name: 'Single-pass style transfer',
    family: 'vision',
    task: 'image_map',
    dataset: 'style',
    tagline: 'One feed-forward pass, one baked-in style',
    description: 'A feed-forward transformer for a single fixed style, trained on content and stylised pairs so inference is one forward pass instead of hundreds of optimisation steps. The baseline is the unstyled content image, so the PSNR gain says how much of the target look the network actually reproduced.',
    input: [3, 32, 32],
    inputName: 'content_image',
    head: 'mask',
    nodes: [
      ['conv2d', { filters: 16, kernel: 3 }],
      ['conv2d', { filters: 16, kernel: 3 }],
      ['conv2d', { filters: 3, kernel: 3, activation: 'linear' }]
    ],
    train: { epochs: 20, batchSize: 24, optimizer: { kind: 'adam', lr: 0.008 } },
    budget: { target: 4000, min: 1500, max: 2000000 },
    reference: { params: [1000000, 2000000], note: 'Johnson perceptual-loss transformers are 1M to 2M parameters and need instance norm plus residual blocks; fork this graph and add them to scale up.' }
  },
  saliency: {
    key: 'saliency',
    name: 'Smart crop saliency',
    family: 'vision',
    task: 'mask',
    dataset: 'saliency',
    tagline: 'Importance heatmap for automatic cropping',
    description: 'Predicts a soft importance map that a crop planner integrates to choose a bounding box. The same encoder-decoder shape as segmentation, evaluated with IoU on the thresholded map and a boundary AUC that rewards getting the subject outline right rather than just its mass.',
    input: [3, 48, 48],
    inputName: 'image',
    head: 'heatmap',
    nodes: [
      ['conv2d', { filters: 16, kernel: 3 }],
      ['maxpool2d', { kernel: 2, stride: 2 }],
      ['conv2d', { filters: 24, kernel: 3 }],
      ['upsample2d', { factor: 2 }],
      ['conv2d', { filters: 12, kernel: 3 }],
      ['conv2d', { filters: 1, kernel: 3, activation: 'linear' }]
    ],
    train: { epochs: 20, batchSize: 24, optimizer: { kind: 'adam', lr: 0.006 }, loss: 'bce_dice' },
    budget: { target: 7000, min: 2000, max: 500000 },
    reference: { params: [500000, 4000000], note: 'U2-Net is about 44M and a distilled saliency net is around 500K; the crop planner only needs the map, not the finest edges.' }
  },
  nsfw: {
    key: 'nsfw',
    name: 'NSFW filter stand-in',
    family: 'vision',
    task: 'multiclass',
    dataset: 'nsfw',
    tagline: 'MobileNet-class shape, synthetic content',
    description: 'The architecture and evaluation of a content filter without any explicit imagery: a strided convolution trunk with global average pooling and a three-way head, trained on synthetic images that differ in skin-tone ratio, edge density and region layout. Swap in a licensed real dataset and the graph, the metrics and the export path are unchanged.',
    input: [3, 32, 32],
    inputName: 'image',
    head: 'softmax',
    nodes: [
      ['conv2d', { filters: 12, kernel: 3, stride: 2 }],
      ['conv2d', { filters: 16, kernel: 3, stride: 2 }],
      ['conv2d', { filters: 24, kernel: 3 }],
      ['globalavgpool2d', {}],
      ['dense', { units: 3, activation: 'linear' }]
    ],
    train: { epochs: 18, batchSize: 24, optimizer: { kind: 'adam', lr: 0.008 } },
    budget: { target: 5000, min: 2000, max: 3000000 },
    reference: { params: [2000000, 3000000], note: 'NudeNet and MobileNet-class filters are 2M to 3M parameters; this trunk is the same shape at browser scale.' }
  }
});

Object.assign(PRESET_MODELS, {
  imu: {
    key: 'imu',
    name: 'IMU activity recognition',
    family: 'sensor',
    task: 'multiclass',
    dataset: 'imu',
    tagline: 'Eight activities from a six-axis window',
    description: 'Accelerometer and gyroscope windows classified by a strided convolution over time followed by global average pooling, which makes the model invariant to where inside the window the gesture happened. At under two thousand parameters this is the preset that shows a small model is not automatically a weak one.',
    input: [64, 6],
    inputName: 'imu_window',
    head: 'softmax',
    nodes: [
      ['conv1d', { filters: 16, kernel: 5, stride: 2 }],
      ['conv1d', { filters: 24, kernel: 3 }],
      ['globalavgpool1d', {}],
      ['dense', { units: 8, activation: 'linear' }]
    ],
    train: { epochs: 20, batchSize: 32, optimizer: { kind: 'adam', lr: 0.008 } },
    budget: { target: 2000, min: 600, max: 100000 },
    reference: { params: [20000, 500000], note: 'Wearable activity models ship between 20K and 500K parameters; this one is deliberately at the extreme low end to make the budget indicator interesting.' }
  },
  anomaly: {
    key: 'anomaly',
    name: 'Anomaly autoencoder',
    family: 'series',
    task: 'anomaly',
    dataset: 'anomaly',
    tagline: 'Reconstruction error as an anomaly score',
    description: 'A bottleneck autoencoder over telemetry windows. It is trained on normal behaviour only and scores a window by how badly it reconstructs, so it can flag failure modes nobody wrote a label for. The evaluation reports recall at a 5 percent false-positive rate, which is how an on-call threshold is actually chosen.',
    input: [32, 6],
    inputName: 'telemetry_window',
    head: 'linear',
    nodes: [
      ['flatten', {}],
      ['dense', { units: 32 }],
      ['dense', { units: 12 }],
      ['dense', { units: 32 }],
      ['dense', { units: 192, activation: 'linear' }],
      ['reshape', { shape: '32,6' }]
    ],
    train: { epochs: 26, batchSize: 32, optimizer: { kind: 'adam', lr: 0.006 } },
    budget: { target: 14000, min: 3000, max: 300000 },
    reference: { params: [10000, 1000000], note: 'Autoencoders scale with input width; an LSTM variant for longer windows lands in the same bracket.' }
  },
  forecast: {
    key: 'forecast',
    name: 'Tiny forecaster',
    family: 'series',
    task: 'forecast',
    dataset: 'forecast',
    tagline: 'Forty-eight steps in, eight steps out',
    description: 'A single recurrent layer over a history window emitting the whole horizon at once, rather than feeding its own predictions back in, which is what keeps a small forecaster from drifting. A naive last-value baseline is reported next to the model so it is obvious whether the parameters earned their keep.',
    input: [48, 1],
    inputName: 'history',
    head: 'linear',
    nodes: [
      ['gru', { units: 24, returnSeq: false }],
      ['dense', { units: 8, activation: 'linear' }]
    ],
    train: { epochs: 30, batchSize: 32, optimizer: { kind: 'adam', lr: 0.008 }, loss: 'huber', huberDelta: 0.4 },
    budget: { target: 2200, min: 500, max: 100000 },
    reference: { params: [10000, 100000], note: 'The brief puts tiny forecasters at 10K to 100K parameters; N-BEATS and DeepAR are orders of magnitude larger again.' }
  },
  bot: {
    key: 'bot',
    name: 'Bot detector',
    family: 'sensor',
    task: 'binary',
    dataset: 'bot',
    tagline: 'Pointer and keystroke statistics',
    description: 'Behavioural biometrics from summary statistics of pointer traces and keystroke timing. A tenth of the preset dataset is deliberately jittered and relabelled so the reported ceiling stays below perfect; a bot detector that scores 1.000 AUC on your own synthetic data is telling you about the data, not the model.',
    input: [96],
    inputName: 'behaviour_features',
    head: 'sigmoid',
    nodes: [
      ['dense', { units: 64 }],
      ['dropout', { rate: 0.2 }],
      ['dense', { units: 32 }],
      ['dense', { units: 1, activation: 'linear' }]
    ],
    train: { epochs: 26, batchSize: 32, optimizer: { kind: 'adamw', lr: 0.006, weightDecay: 0.0004 } },
    budget: { target: 9000, min: 2000, max: 200000 },
    reference: { params: [50000, 2000000], note: 'Fraud stacks combine this with rules and graph features; the sequence model over raw events is the larger half.' }
  },
  compression: {
    key: 'compression',
    name: 'Learned byte compressor',
    family: 'wildcard',
    task: 'lm',
    dataset: 'compression',
    tagline: 'Next-byte distribution for an arithmetic coder',
    description: 'The prediction half of a learned compressor: a byte-level recurrent model whose output distribution drives an arithmetic coder. Bits per symbol converts straight into compressed size, so the honest comparison is against order-0 entropy of the same corpus, which is what a lookup-table compressor achieves.',
    input: [96],
    inputName: 'byte_ids',
    head: 'softmax-seq',
    nodes: [
      ['embedding', { vocab: 259, dim: 16 }],
      ['gru', { units: 40, returnSeq: true }],
      ['dense', { units: 259, activation: 'linear' }]
    ],
    train: { epochs: 12, batchSize: 32, optimizer: { kind: 'adam', lr: 0.006 } },
    budget: { target: 24000, min: 6000, max: 500000 },
    reference: { params: [1000000, 200000000], note: 'NNCP and CMIX are tens to hundreds of megabytes and win every compression benchmark; they also take hours per megabyte.' }
  },
  texture: {
    key: 'texture',
    name: 'Neural texture',
    family: 'wildcard',
    task: 'coordinate_field',
    dataset: 'texture',
    tagline: 'A SIREN where the weights are the image',
    description: 'Coordinates in, colour out, with sine and cosine positional encoding at seven octaves so a small multilayer perceptron can hold high-frequency detail. After training the network is the image, so the comparison that matters is bytes of parameters against bytes of raw pixels, both reported by the evaluation.',
    input: [31],
    inputName: 'encoded_coordinate',
    head: 'linear',
    nodes: [
      ['dense', { units: 128 }],
      ['dense', { units: 128 }],
      ['dense', { units: 128 }],
      ['dense', { units: 128 }],
      ['dense', { units: 3, activation: 'linear' }]
    ],
    train: { epochs: 80, batchSize: 128, optimizer: { kind: 'adam', lr: 0.003 } },
    budget: { target: 55000, min: 10000, max: 500000 },
    reference: { params: [50000, 500000], note: 'The brief calls for a 50K-parameter network that is the image; that is exactly what this preset lands on.' }
  }
});

export const PRESET_FAMILIES = [
  { key: 'text', label: 'Text' },
  { key: 'audio', label: 'Audio' },
  { key: 'vision', label: 'Vision' },
  { key: 'sensor', label: 'Sensors' },
  { key: 'series', label: 'Series' },
  { key: 'wildcard', label: 'Wildcard' }
];

export const PRESET_MODEL_LIST = Object.keys(PRESET_MODELS).map((k) => PRESET_MODELS[k]);

export function modelPresetByKey(key) {
  return PRESET_MODELS[key] || null;
}

export function presetsForFamily(family) {
  return PRESET_MODEL_LIST.filter((p) => !family || family === 'all' || p.family === family);
}

export function buildPresetGraph(preset, overrides) {
  const src = overrides ? Object.assign({}, preset, overrides) : preset;
  const nodes = [];
  const edges = [];
  const inputId = 'input';
  nodes.push({ id: inputId, type: 'input', cfg: {}, pos: { x: 40, y: 180 } });
  let prev = inputId;
  let n = 0;
  for (const entry of src.nodes) {
    const type = entry[0];
    const cfg = Object.assign(defaultConfig(type), entry[1] || {});
    const meta = entry[2] || {};
    const id = type + '_' + n;
    n++;
    const residual = Array.isArray(meta.from) && meta.from.indexOf('input') >= 0;
    nodes.push({ id, type, cfg, pos: { x: 40 + n * 190, y: residual ? 320 : 180 } });
    const sources = meta.from
      ? meta.from.map((f) => (f === 'input' ? inputId : f === 'prev' ? prev : f))
      : [prev];
    sources.forEach((s, i) => edges.push({ from: s, to: id, toPort: i }));
    prev = id;
  }
  return {
    id: 'preset-' + src.key,
    name: src.name,
    version: 1,
    input: { name: src.inputName || 'x', shape: src.input.slice(), dtype: 'float32' },
    nodes,
    edges,
    output: prev,
    head: src.head,
    meta: { presetKey: src.key, family: src.family, task: src.task, dataset: src.dataset, tagline: src.tagline }
  };
}

const SCALEABLE_KEYS = ['units', 'filters'];

function scaleOnce(spec, factor, keepTail) {
  const out = JSON.parse(JSON.stringify(spec));
  const parametric = out.nodes.filter((n) => n.type !== 'input' && n.type !== 'add' && n.type !== 'reshape' && n.type !== 'flatten');
  const frozen = new Set(parametric.slice(Math.max(0, parametric.length - keepTail)).map((n) => n.id));
  for (const node of out.nodes) {
    if (frozen.has(node.id)) continue;
    for (const k of SCALEABLE_KEYS) {
      if (typeof node.cfg[k] === 'number' && node.cfg[k] > 1) node.cfg[k] = Math.max(2, Math.round(node.cfg[k] * factor));
    }
  }
  return out;
}

export function scalePresetGraph(spec, factor) {
  const f = Math.max(0.25, Math.min(8, factor || 1));
  if (Math.abs(f - 1) < 1e-9) return spec;
  const parametric = spec.nodes.filter((n) => n.type !== 'input' && n.type !== 'add' && n.type !== 'reshape' && n.type !== 'flatten').length;
  for (let keepTail = 1; keepTail <= parametric; keepTail++) {
    const candidate = scaleOnce(spec, f, keepTail);
    if (analyze(candidate).valid) return candidate;
  }
  return spec;
}

export function scaleToFit(spec, targetParams) {
  const base = analyze(spec).totalParams;
  if (!base || !targetParams || targetParams <= 0) return { spec, params: base, factor: 1 };
  let lo = 0.25;
  let hi = 8;
  let best = spec;
  let bestParams = base;
  let bestFactor = 1;
  for (let i = 0; i < 14; i++) {
    const mid = Math.sqrt(lo * hi);
    const candidate = scalePresetGraph(spec, mid);
    const p = analyze(candidate).totalParams;
    if (p <= targetParams) {
      lo = mid;
      if (p > bestParams || bestParams > targetParams) {
        best = candidate;
        bestParams = p;
        bestFactor = mid;
      }
    } else {
      hi = mid;
    }
  }
  if (bestParams > targetParams) {
    const small = scalePresetGraph(spec, lo);
    const sp = analyze(small).totalParams;
    if (sp <= targetParams) {
      best = small;
      bestParams = sp;
      bestFactor = lo;
    }
  }
  return { spec: best, params: bestParams, factor: bestFactor };
}

export function validatePreset(preset) {
  const spec = buildPresetGraph(preset);
  const info = analyze(spec);
  const problems = [];
  for (const e of info.errors) problems.push(e);
  for (const entry of preset.nodes) {
    if (!isLayerType(entry[0])) problems.push('unknown layer type ' + entry[0]);
  }
  return { spec, info, ok: info.valid && problems.length === 0, problems };
}

export function budgetPosition(preset, params) {
  const b = preset.budget || { target: params, min: 0, max: params * 2 };
  const ratio = b.target > 0 ? params / b.target : 1;
  let label = 'on budget';
  if (ratio < 0.5) label = 'under budget';
  else if (ratio > 1.5) label = 'over budget';
  return { target: b.target, min: b.min, max: b.max, params, ratio, label, reference: preset.reference || null };
}

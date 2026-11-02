const pct = (label, direction, group, explain) => ({ label, direction, format: 'percent', group, explain });
const rat = (label, direction, group, explain) => ({ label, direction, format: 'ratio', group, explain });
const db = (label, direction, group, explain) => ({ label, direction, format: 'db', group, explain });
const cnt = (label, direction, group, explain) => ({ label, direction, format: 'count', group, explain });
const num = (label, direction, group, explain) => ({ label, direction, format: 'number', group, explain });

export const METRIC_META = {
  train_loss: { label: 'Train loss', direction: 'down', format: 'number', group: 'loss', explain: 'Mean loss over the training batches of that epoch, measured while augmentation and dropout were active.' },
  val_loss: { label: 'Val loss', direction: 'down', format: 'number', group: 'loss', explain: 'Mean loss over the whole validation split with augmentation and dropout off. This is the curve that decides when to stop.' },
  loss: num('Loss', 'down', 'loss', 'Loss on the evaluated split.'),
  lr: { label: 'Learning rate', direction: 'flat', format: 'scientific', group: 'optimisation', explain: 'Effective learning rate after the schedule and any warmup.' },
  grad_norm: { label: 'Grad norm', direction: 'flat', format: 'number', group: 'optimisation', explain: 'L2 norm of the gradient before clipping. A sudden spike is usually a bad batch, a steady climb is usually a learning rate that is too high.' },
  seconds: num('Epoch time', 'flat', 'optimisation', 'Wall clock seconds for that epoch, including evaluation.'),
  samples_per_sec: num('Throughput', 'up', 'optimisation', 'Training samples processed per second.'),

  accuracy: pct('Accuracy', 'up', 'classification', 'Share of samples whose predicted class matches the label. Misleading on its own when classes are unbalanced, which is why macro F1 sits next to it.'),
  accuracy_ci_lower: pct('Accuracy 95% interval · lower', 'flat', 'classification', 'Wilson interval assuming independent representative examples. It excludes distribution shift, clustering and selection uncertainty.'),
  accuracy_ci_upper: pct('Accuracy 95% interval · upper', 'flat', 'classification', 'Wilson interval assuming independent representative examples. It excludes distribution shift, clustering and selection uncertainty.'),
  macro_f1: pct('Macro F1', 'up', 'classification', 'F1 averaged over classes with equal weight, so a class with twenty samples counts as much as one with two thousand.'),
  weighted_f1: pct('Weighted F1', 'up', 'classification', 'F1 averaged over classes weighted by support. Reads better than macro F1 on imbalanced data, which is exactly when to be suspicious of it.'),
  macro_precision: pct('Macro precision', 'up', 'classification', 'Equal-weight average of per-class precision: of everything predicted as a class, how much was right.'),
  macro_recall: pct('Macro recall', 'up', 'classification', 'Equal-weight average of per-class recall: of everything that was a class, how much was found.'),
  log_loss: num('Log loss', 'down', 'classification', 'Negative log likelihood of the true class under the predicted distribution. Punishes confident wrong answers far harder than accuracy does.'),
  error_rate: pct('Error rate', 'down', 'classification', 'One minus accuracy.'),
  ece: num('ECE', 'down', 'calibration', 'Expected calibration error: the mean gap between predicted confidence and observed accuracy across confidence bins. Under 0.05 is well calibrated.'),
  brier: num('Brier score', 'down', 'calibration', 'Mean squared error between predicted probabilities and the one-hot label. Combines discrimination and calibration in one number.'),
  macro_auroc: rat('Macro AUROC', 'up', 'classification', 'One-versus-rest ROC AUC averaged over classes.'),
  mean_auc: rat('Mean AUC', 'up', 'detection', 'Per-image area under the objectness ROC, averaged over images that contain at least one object. A high mean AUC with a low mAP means the objectness channel ranks correctly while the box regression does not.'),

  auc_roc: rat('ROC AUC', 'up', 'binary', 'Probability that a randomly chosen positive scores above a randomly chosen negative. Threshold-free, so it describes ranking rather than the operating point.'),
  average_precision: rat('Average precision', 'up', 'binary', 'Area under the precision-recall curve. The right headline when positives are rare, because ROC AUC stays flattering there.'),
  f1: pct('F1', 'up', 'binary', 'Harmonic mean of precision and recall at the chosen threshold.'),
  precision: pct('Precision', 'up', 'binary', 'Of everything flagged, how much was really positive. The cost of a false alarm.'),
  recall: pct('Recall', 'up', 'binary', 'Of everything positive, how much was flagged. The cost of a miss.'),
  far: pct('False accept rate', 'down', 'binary', 'Share of negatives accepted at the operating threshold. For a wake word this is how often the device wakes up for nothing.'),
  frr: pct('False reject rate', 'down', 'binary', 'Share of positives rejected at the operating threshold. For a wake word this is how often the user has to repeat themselves.'),
  mcc: rat('Matthews correlation', 'up', 'binary', 'Correlation between prediction and label over all four confusion cells. Stays informative when the classes are very unbalanced.'),
  best_threshold: rat('Decision threshold', 'flat', 'binary', 'Fixed configured score threshold used for this split. Select it on validation data, never on the test set.'),
  micro_precision: pct('Micro precision', 'up', 'multilabel', 'Precision pooled over every label decision rather than averaged per label.'),
  micro_recall: pct('Micro recall', 'up', 'multilabel', 'Recall pooled over every label decision.'),
  micro_f1: pct('Micro F1', 'up', 'multilabel', 'F1 pooled over every label decision; dominated by the frequent labels.'),
  exact_match: pct('Exact match', 'up', 'multilabel', 'Share of samples where the whole predicted label set equals the true set.'),
  false_accepts_per_hour: { label: 'False accepts / hour', direction: 'down', format: 'perhour', group: 'binary', explain: 'False accepts extrapolated to one hour of continuous audio. The number a wake-word team actually ships against.' },
  false_rejects_per_hour: { label: 'False rejects / hour', direction: 'down', format: 'perhour', group: 'binary', explain: 'Missed keywords extrapolated to one hour of audio at the same threshold.' },
  detections_per_hour: { label: 'Detections / hour', direction: 'flat', format: 'perhour', group: 'binary', explain: 'Positive detections extrapolated to one hour of audio, useful for sanity-checking the threshold.' },
  event_recall: pct('Event recall', 'up', 'binary', 'Recall over detected events rather than over frames.'),
  event_precision: pct('Event precision', 'up', 'binary', 'Precision over detected events rather than over frames.'),

  token_accuracy: pct('Token accuracy', 'up', 'sequence', 'Share of tagged positions with the correct label, padding excluded.'),
  entity_f1: pct('Entity F1', 'up', 'sequence', 'F1 over whole entities with an exact span and type match. The headline for a redactor: a half-caught email address still leaks.'),
  entity_precision: pct('Entity precision', 'up', 'sequence', 'Share of predicted entities whose span and type are exactly right.'),
  entity_recall: pct('Entity recall', 'up', 'sequence', 'Share of true entities recovered with an exact span and type.'),

  perplexity: num('Perplexity', 'down', 'language model', 'Exponentiated mean negative log likelihood. Read it as the effective branching factor the model faces at each step.'),
  nll: num('NLL per symbol', 'down', 'language model', 'Mean negative log likelihood per predicted symbol, in nats.'),
  top1_accuracy: pct('Top-1 accuracy', 'up', 'language model', 'Share of positions where the highest-probability symbol was the actual next one.'),
  bits_per_symbol: { label: 'Bits per symbol', direction: 'down', format: 'bits', group: 'language model', explain: 'NLL converted to base two. Multiplied by the symbol count this is the compressed size an arithmetic coder would produce.' },
  bytes_per_symbol: { label: 'Bytes per symbol', direction: 'down', format: 'bits', group: 'language model', explain: 'Bits per symbol divided by eight, for reading directly against a file size.' },

  pair_auc: rat('Pair AUC', 'up', 'embedding', 'Probability that a same-group pair scores closer than a different-group pair.'),
  pair_ap: rat('Pair average precision', 'up', 'embedding', 'Average precision over the same-group versus different-group pair decision.'),
  duplicate_f1: pct('Duplicate F1', 'up', 'embedding', 'F1 for a near-duplicate decision at the best cosine threshold.'),
  duplicate_threshold: rat('Duplicate threshold', 'flat', 'embedding', 'Cosine threshold that maximises duplicate F1. Ship it with the model.'),
  recall_at_1: pct('Recall@1', 'up', 'embedding', 'Share of queries whose nearest neighbour in the same split belongs to the same group, self excluded.'),
  recall_at_5: pct('Recall@5', 'up', 'embedding', 'Share of queries with a same-group item in their five nearest neighbours.'),
  recall_at_10: pct('Recall@10', 'up', 'embedding', 'Share of queries with a same-group item in their ten nearest neighbours.'),
  paraphrase_recall_at_1: pct('Paraphrase R@1', 'up', 'embedding', 'Share of sentences whose nearest neighbour among the held-out paraphrase view is in the same meaning group. The closest thing to a real retrieval benchmark here.'),
  paraphrase_recall_at_5: pct('Paraphrase R@5', 'up', 'embedding', 'Same as paraphrase recall at one, over five neighbours.'),
  map: rat('Mean average precision', 'up', 'embedding', 'Average precision of the full ranked list per query, averaged over queries.'),
  mrr: rat('MRR', 'up', 'embedding', 'Mean reciprocal rank of the first relevant result.'),
  alignment: num('Alignment', 'down', 'embedding', 'Mean squared distance between positive pairs. Lower means paraphrases sit closer together.'),
  uniformity: num('Uniformity', 'down', 'embedding', 'Log of the mean pairwise Gaussian potential over the sphere. More negative means the embedding space is better spread out rather than collapsed.'),
  dimensions: cnt('Embedding dims', 'flat', 'embedding', 'Width of the output vector, which is what a vector index has to store per document.'),

  mae: num('MAE', 'down', 'regression', 'Mean absolute error in the units of the target. Robust to the occasional large miss.'),
  rmse: num('RMSE', 'down', 'regression', 'Root mean squared error. Larger than MAE whenever the errors are unevenly spread, and the gap tells you how heavy the tail is.'),
  mse: num('MSE', 'down', 'regression', 'Mean squared error.'),
  medae: num('Median AE', 'down', 'regression', 'Median absolute error: the mistake the model makes on a typical sample, ignoring the tail.'),
  r2: rat('R²', 'up', 'regression', 'Share of target variance explained. Zero means no better than predicting the mean, negative means worse.'),
  mape: pct('MAPE', 'down', 'regression', 'Mean absolute percentage error over targets far enough from zero to divide by. Coverage is reported next to it because near-zero targets are excluded.'),
  mape_coverage: pct('MAPE coverage', 'flat', 'regression', 'Share of samples included in MAPE. Low coverage means the target crosses zero and MAPE should be ignored in favour of WAPE.'),
  wape: pct('WAPE', 'down', 'regression', 'Total absolute error divided by total absolute target. The percentage error that survives targets crossing zero.'),
  max_error: num('Max error', 'down', 'regression', 'Worst single error on this split. The number that decides whether a model is allowed near a safety-relevant path.'),
  pearson: rat('Pearson r', 'up', 'regression', 'Linear correlation between prediction and target, ignoring scale and offset.'),
  naive_mae: num('Baseline MAE', 'down', 'regression', 'MAE of a naive last-value forecast. If the model does not beat this it has learned nothing worth shipping.'),
  horizon: cnt('Horizon', 'flat', 'regression', 'Number of future steps predicted at once.'),

  auroc: rat('AUROC', 'up', 'anomaly', 'Probability a random anomaly scores above a random normal window by reconstruction error.'),
  p_at_5: pct('Precision@5', 'up', 'anomaly', 'Share of the five highest-scoring windows that are genuinely anomalous. What an on-call queue sees first.'),
  p_at_10: pct('Precision@10', 'up', 'anomaly', 'Same over the ten highest-scoring windows.'),
  recall_at_5fpr: pct('Recall at 5% FPR', 'up', 'anomaly', 'Share of anomalies caught when the threshold is set so that only five percent of normal windows are flagged. This is how the threshold gets chosen in practice.'),
  threshold: num('Score threshold', 'flat', 'anomaly', 'Reconstruction error threshold at the reported operating point.'),
  recon_mse_normal: num('Recon MSE normal', 'down', 'anomaly', 'Mean squared reconstruction error on normal windows, which is what the model was trained to minimise.'),
  recon_mse_anomaly: num('Recon MSE anomaly', 'flat', 'anomaly', 'Mean squared reconstruction error on anomalous windows. The gap to the normal figure is the entire signal.'),
  separation: num('Score separation', 'up', 'anomaly', 'Anomaly reconstruction error minus normal reconstruction error.'),
  contamination: pct('Contamination', 'flat', 'anomaly', 'Share of the evaluated split that is labelled anomalous.'),
  normal: cnt('Normal windows', 'flat', 'anomaly', 'Number of normal windows in this split.'),
  anomaly: cnt('Anomalies', 'flat', 'anomaly', 'Number of anomalous windows in this split.'),

  psnr: db('PSNR', 'up', 'image', 'Peak signal-to-noise ratio of the output against the target. Every 6 dB is roughly half the remaining error.'),
  ssim: rat('SSIM', 'up', 'image', 'Structural similarity over luminance, contrast and structure. Tracks perceived quality better than PSNR on edges and texture.'),
  baseline_psnr: db('Baseline PSNR', 'up', 'image', 'PSNR of the unprocessed input against the same target. The bar the model has to clear.'),
  baseline_ssim: rat('Baseline SSIM', 'up', 'image', 'SSIM of the unprocessed input against the same target.'),
  psnr_gain: db('PSNR gain', 'up', 'image', 'Model PSNR minus baseline PSNR. The only image number that says whether the parameters earned their keep.'),
  psnr_render: db('Render PSNR', 'up', 'image', 'PSNR of the network evaluated across the whole coordinate grid, which for a neural texture is the image the weights encode.'),

  iou: pct('IoU', 'up', 'segmentation', 'Intersection over union of the thresholded mask. The standard segmentation headline.'),
  dice: pct('Dice', 'up', 'segmentation', 'Dice coefficient, which reads higher than IoU for the same mask. Both are reported so nobody quotes the flattering one by accident.'),
  pixel_accuracy: pct('Pixel accuracy', 'up', 'segmentation', 'Share of pixels with the right label. On a mask where foreground is eighteen percent of pixels, predicting background everywhere scores eighty-two.'),
  boundary_auc: rat('Boundary AUC', 'up', 'segmentation', 'ROC AUC over pixels near the true boundary. Separates finding the object from finding its edges.'),

  frame_accuracy: pct('Frame accuracy', 'up', 'frame', 'Share of frames with the right label.'),
  frame_f1: pct('Frame F1', 'up', 'frame', 'F1 over frames at the operating threshold.'),
  onset_f1: pct('Onset F1', 'up', 'frame', 'F1 over onsets matched within a timing tolerance window. Exact-frame matching would punish a detector for firing ten milliseconds early.'),
  tempo_mae_bpm: { label: 'Tempo MAE', direction: 'down', format: 'bpm', group: 'frame', explain: 'Mean absolute tempo error in beats per minute, derived from the predicted onset train.' },

  mAP50: rat('mAP@0.50', 'up', 'detection', 'Mean average precision at an IoU threshold of 0.50, integrated over confidence so no threshold choice flatters it.'),
  mAP: rat('mAP@0.50 to 0.95', 'up', 'detection', 'Mean average precision averaged over IoU thresholds from 0.50 to 0.95. The strict number, and the one that drops fastest when boxes are loose.'),
  f1_at_best_conf: pct('F1 at best conf', 'up', 'detection', 'Detection F1 at the confidence threshold that maximises it on this split.'),
  best_conf_threshold: rat('Best conf threshold', 'flat', 'detection', 'The confidence threshold that maximises detection F1. Ship it with the model.'),
  tp: cnt('True positives', 'up', 'detection', 'Detections matched to a ground-truth box above the IoU threshold.'),
  fp: cnt('False positives', 'down', 'detection', 'Detections that matched nothing.'),
  fn: cnt('False negatives', 'down', 'detection', 'Ground-truth boxes that no detection matched.'),

  si_sdr: db('SI-SDR', 'up', 'audio', 'Scale-invariant signal-to-distortion ratio of the enhanced waveform against the clean reference, in dB.'),
  si_sdr_input: db('SI-SDR input', 'up', 'audio', 'SI-SDR of the unprocessed noisy waveform. The bar the mask has to clear.'),
  si_sdr_improvement: db('SI-SDR improvement', 'up', 'audio', 'Enhanced SI-SDR minus input SI-SDR. The headline for noise suppression.'),
  segmental_snr: db('Segmental SNR', 'up', 'audio', 'Signal-to-noise ratio averaged over short segments, which tracks perceived quality better than a whole-file figure.'),
  pesq_approx: num('PESQ (approx.)', 'up', 'audio', 'A documented surrogate on the 1 to 4.5 MOS scale built from SI-SDR and segmental SNR improvement. It is not ITU-T P.862 and the interface labels it as an estimate.'),
  mask_mae: num('Mask MAE', 'down', 'audio', 'Mean absolute error against the ideal magnitude mask, before any reconstruction.'),

  raw_pitch_accuracy: pct('Pitch accuracy (50¢)', 'up', 'pitch', 'Share of samples whose predicted pitch class is within fifty cents, the standard raw pitch accuracy window.'),
  strict_pitch_accuracy: pct('Pitch accuracy (20¢)', 'up', 'pitch', 'Share within twenty cents, which is a quarter of a semitone and about where a tuning application needs to be.'),
  mean_abs_cent_error: { label: 'Mean cent error', direction: 'down', format: 'cents', group: 'pitch', explain: 'Mean absolute error in cents. One hundred cents is one semitone.' },
  median_abs_cent_error: { label: 'Median cent error', direction: 'down', format: 'cents', group: 'pitch', explain: 'Median absolute error in cents, which ignores the octave mistakes the mean is dominated by.' },
  octave_error_rate: pct('Octave error rate', 'down', 'pitch', 'Share of predictions wrong by close to one or two octaves. The characteristic failure of harmonic pitch trackers.'),

  pixel_coverage: pct('Pixel coverage', 'flat', 'coordinate field', 'Share of image pixels present as samples in the dataset.'),
  weight_bytes: { label: 'Weight bytes', direction: 'flat', format: 'bytes', group: 'coordinate field', explain: 'Size of the trained network in bytes at four bytes per parameter. For a neural texture this is the size of the image.' },
  raw_image_bytes: { label: 'Raw image bytes', direction: 'flat', format: 'bytes', group: 'coordinate field', explain: 'Size of the same image stored as raw eight-bit pixels, for comparison.' },
  compression_ratio: rat('Compression (int8)', 'up', 'coordinate field', 'Raw image bytes divided by the network size at one byte per parameter.'),
  compression_ratio_fp32: rat('Compression (fp32)', 'up', 'coordinate field', 'Raw image bytes divided by the network size at four bytes per parameter. The honest figure for an unquantised export.'),
  steps: cnt('Steps', 'flat', 'optimisation', 'Number of optimizer steps taken.'),
  epochs_run: cnt('Epochs run', 'flat', 'optimisation', 'Epochs completed before stopping.'),
  duration_ms: { label: 'Duration', direction: 'flat', format: 'duration', group: 'optimisation', explain: 'Total wall clock time for the run.' },
  samples_seen: cnt('Samples seen', 'flat', 'optimisation', 'Training samples processed across the whole run, counting repeated epochs.'),
  params: cnt('Parameters', 'flat', 'model', 'Trainable parameter count.'),
  macs: cnt('MACs', 'flat', 'model', 'Multiply-accumulate operations for one forward pass at batch size one.'),
  inference_ms: num('Inference ms', 'down', 'model', 'Median wall clock time for a single-sample forward pass on this machine.'),
  size_bytes: { label: 'Model size', direction: 'down', format: 'bytes', group: 'model', explain: 'Serialised weight size for the chosen export precision.' }
};

export const TASK_METRIC_GROUPS = {
  multiclass: [
    { title: 'Headline', metrics: ['accuracy', 'accuracy_ci_lower', 'accuracy_ci_upper', 'macro_f1', 'log_loss'] },
    { title: 'Per class', metrics: ['macro_precision', 'macro_recall', 'weighted_f1', 'macro_auroc', 'error_rate'] },
    { title: 'Calibration', metrics: ['ece', 'brier'] }
  ],
  binary: [
    { title: 'Headline', metrics: ['auc_roc', 'average_precision', 'f1'] },
    { title: 'Operating point', metrics: ['precision', 'recall', 'far', 'frr', 'best_threshold', 'accuracy', 'accuracy_ci_lower', 'accuracy_ci_upper', 'mcc'] },
    { title: 'Continuous audio', metrics: ['false_accepts_per_hour', 'false_rejects_per_hour', 'detections_per_hour'] },
    { title: 'Calibration', metrics: ['ece', 'brier'] }
  ],
  multilabel: [
    { title: 'Headline', metrics: ['micro_f1', 'macro_f1', 'exact_match'] },
    { title: 'Pooled', metrics: ['micro_precision', 'micro_recall', 'macro_auroc'] },
    { title: 'Calibration', metrics: ['ece', 'brier'] }
  ],
  seq_tag: [
    { title: 'Headline', metrics: ['entity_f1', 'token_accuracy', 'macro_f1'] },
    { title: 'Entities', metrics: ['entity_precision', 'entity_recall'] },
    { title: 'Tokens', metrics: ['weighted_f1', 'ece'] }
  ],
  lm: [
    { title: 'Headline', metrics: ['perplexity', 'bits_per_symbol', 'top1_accuracy'] },
    { title: 'Compression', metrics: ['bytes_per_symbol', 'nll'] }
  ],
  embedding: [
    { title: 'Headline', metrics: ['paraphrase_recall_at_1', 'pair_auc', 'map'] },
    { title: 'Retrieval', metrics: ['recall_at_1', 'recall_at_5', 'recall_at_10', 'paraphrase_recall_at_5', 'mrr'] },
    { title: 'Duplicates', metrics: ['duplicate_f1', 'duplicate_threshold', 'pair_ap'] },
    { title: 'Geometry', metrics: ['alignment', 'uniformity', 'dimensions'] }
  ],
  regression: [
    { title: 'Headline', metrics: ['mae', 'rmse', 'r2'] },
    { title: 'Relative', metrics: ['wape', 'mape', 'mape_coverage', 'pearson'] },
    { title: 'Tail', metrics: ['max_error', 'medae', 'mse'] }
  ],
  forecast: [
    { title: 'Headline', metrics: ['mae', 'naive_mae', 'r2'] },
    { title: 'Relative', metrics: ['wape', 'rmse', 'pearson'] },
    { title: 'Tail', metrics: ['max_error', 'medae'] }
  ],
  anomaly: [
    { title: 'Headline', metrics: ['auroc', 'recall_at_5fpr', 'average_precision'] },
    { title: 'Top of the queue', metrics: ['p_at_5', 'p_at_10'] },
    { title: 'Reconstruction', metrics: ['recon_mse_normal', 'recon_mse_anomaly', 'separation', 'threshold'] },
    { title: 'Split', metrics: ['contamination', 'normal', 'anomaly'] }
  ],
  image_map: [
    { title: 'Headline', metrics: ['psnr_gain', 'psnr', 'ssim'] },
    { title: 'Baseline', metrics: ['baseline_psnr', 'baseline_ssim'] },
    { title: 'Error', metrics: ['mae'] }
  ],
  mask: [
    { title: 'Headline', metrics: ['iou', 'dice', 'boundary_auc'] },
    { title: 'Pixels', metrics: ['pixel_accuracy'] }
  ],
  spectral_mask: [
    { title: 'Headline', metrics: ['si_sdr_improvement', 'si_sdr', 'pesq_approx'] },
    { title: 'Reference', metrics: ['si_sdr_input', 'segmental_snr'] },
    { title: 'Mask', metrics: ['mask_mae'] }
  ],
  frame_binary: [
    { title: 'Headline', metrics: ['auc_roc', 'frame_f1', 'average_precision'] },
    { title: 'Frames', metrics: ['frame_accuracy', 'far', 'frr'] }
  ],
  onset: [
    { title: 'Headline', metrics: ['onset_f1', 'auc_roc', 'tempo_mae_bpm'] },
    { title: 'Frames', metrics: ['frame_f1', 'frame_accuracy', 'far', 'frr'] }
  ],
  detection: [
    { title: 'Headline', metrics: ['mAP50', 'mAP', 'f1_at_best_conf'] },
    { title: 'Operating point', metrics: ['precision', 'recall', 'best_conf_threshold'] },
    { title: 'Counts', metrics: ['tp', 'fp', 'fn'] }
  ],
  pitch_class: [
    { title: 'Headline', metrics: ['raw_pitch_accuracy', 'mean_abs_cent_error', 'octave_error_rate'] },
    { title: 'Strict', metrics: ['strict_pitch_accuracy', 'median_abs_cent_error'] },
    { title: 'As a classifier', metrics: ['accuracy', 'macro_f1', 'log_loss'] }
  ],
  coordinate_field: [
    { title: 'Headline', metrics: ['psnr_render', 'psnr', 'ssim'] },
    { title: 'As the image', metrics: ['weight_bytes', 'raw_image_bytes', 'compression_ratio', 'compression_ratio_fp32'] },
    { title: 'Held out pixels', metrics: ['r2', 'mae', 'rmse', 'pixel_coverage'] }
  ]
};

export const HEADLINE_METRIC = {
  multiclass: 'accuracy',
  binary: 'auc_roc',
  multilabel: 'micro_f1',
  seq_tag: 'entity_f1',
  lm: 'perplexity',
  embedding: 'paraphrase_recall_at_1',
  regression: 'mae',
  forecast: 'mae',
  anomaly: 'auroc',
  image_map: 'psnr_gain',
  mask: 'iou',
  spectral_mask: 'si_sdr_improvement',
  frame_binary: 'auc_roc',
  onset: 'onset_f1',
  detection: 'mAP50',
  pitch_class: 'raw_pitch_accuracy',
  coordinate_field: 'psnr_render'
};

export function metricMeta(key) {
  return METRIC_META[key] || { label: humanize(key), direction: 'flat', format: 'number', group: 'other', explain: '' };
}

export function humanize(key) {
  return String(key).replace(/_/g, ' ').replace(/^./, (c) => c.toUpperCase());
}

export function metricDirection(key) {
  return metricMeta(key).direction;
}

export function isBetter(key, a, b) {
  if (!Number.isFinite(a) || !Number.isFinite(b)) return false;
  const d = metricDirection(key);
  if (d === 'up') return a > b;
  if (d === 'down') return a < b;
  return false;
}

export function formatBytes(v) {
  const a = Math.abs(v);
  if (a >= 1048576) return (v / 1048576).toFixed(2) + ' MB';
  if (a >= 1024) return (v / 1024).toFixed(1) + ' kB';
  return Math.round(v) + ' B';
}

export function formatDuration(ms) {
  if (!Number.isFinite(ms)) return '-';
  if (ms < 1000) return Math.round(ms) + ' ms';
  const s = ms / 1000;
  if (s < 60) return s.toFixed(1) + ' s';
  const m = Math.floor(s / 60);
  return m + 'm ' + Math.round(s - m * 60) + 's';
}

export function formatMetric(key, value) {
  if (value === null || value === undefined || !Number.isFinite(value)) return '-';
  const m = metricMeta(key);
  switch (m.format) {
    case 'percent': return (value * 100).toFixed(1) + '%';
    case 'db': return value.toFixed(2) + ' dB';
    case 'count': return String(Math.round(value));
    case 'bytes': return formatBytes(value);
    case 'duration': return formatDuration(value);
    case 'scientific': return value.toExponential(2);
    case 'ratio': return value.toFixed(3);
    case 'cents': return value.toFixed(1) + ' cents';
    case 'bpm': return value.toFixed(2) + ' BPM';
    case 'perhour': return value.toFixed(2) + ' / h';
    case 'bits': return value.toFixed(3);
    default: return Math.abs(value) >= 1000 ? value.toFixed(0) : value.toFixed(4);
  }
}

export function metricsForTask(task) {
  return TASK_METRIC_GROUPS[task] || [];
}

export function headlineFor(task) {
  return HEADLINE_METRIC[task] || 'val_loss';
}

export function orderScalars(task, scalars) {
  const groups = metricsForTask(task);
  const known = [];
  for (const g of groups) for (const k of g.metrics) if (scalars[k] !== undefined && known.indexOf(k) < 0) known.push(k);
  const rest = Object.keys(scalars).filter((k) => known.indexOf(k) < 0);
  return known.concat(rest);
}

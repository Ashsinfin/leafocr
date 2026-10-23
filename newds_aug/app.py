"""
LeafOCR — end-to-end demo app
====================================
Pipeline (all in memory, single image in -> text out):

  1. Take one uploaded palm-leaf image.
  2. Resize it up to a multiple of PATCH_SIZE and cut it into 512x512
     patches (row-major), mirroring how your training patches were made.
  3. Run the DeepLabv3+ segmentation model on every patch to get a
     foreground (leaf/text) mask.
  4. Reconstruct the full-size image + mask, filling the background with
     the manuscript's own average leaf color.
  5. Split the reconstructed mask into text lines (peak/valley detection
     on the row ink profile).
  6. Split each line into ~200px-wide chunks, cutting at low-ink columns.
  7. Run the multi-scale CRNN model on every chunk, CTC-decode, and
     re-assemble chunks -> lines -> full manuscript text.

Before running, fill in the three paths in the CONFIG section below:
  - DEEPLAB_MODEL_PATH : your trained DeepLabv3+ .h5 weights file
  - CRNN_MODEL_PATH    : your trained CRNN inference_model.keras file
  - VOCAB_PATH         : the vocabulary.json saved during CRNN training

Run with:  python app.py
"""

import os
import math
import json
import hashlib
import numpy as np
import cv2

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow import keras

from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

import gradio as gr

# ============================================================
# CONFIG — UPDATE THESE THREE PATHS FOR YOUR MACHINE
# ============================================================

_HERE = os.path.dirname(os.path.abspath(__file__))
DEEPLAB_MODEL_PATH = os.environ.get("DEEPLAB_MODEL_PATH", os.path.join(_HERE, "compariosn_best_deeplabv3plus_model_model.h5"))
CRNN_MODEL_PATH     = os.path.join(_HERE, "training_model_best.keras")
VOCAB_PATH           = os.path.join(_HERE, "vocabulary (6) (1).json")


# Segmentation
PATCH_SIZE = 512  # must match what the DeepLab model was trained on
SEG_BATCH  = 4    # patches per DeepLab call; the UI can update/stop between batches

# Live scan display
LIVE_MAX_SIDE  = 1600          # downscale the live preview of big images (final view is full-res)
CURRENT_COLOR  = (0, 200, 0)   # BGR green  — chunk being recognized right now
SELECTED_COLOR = (0, 140, 255) # BGR orange — chunk picked from the transcript
REGION_PAD     = 12            # px of context around the "source region" crop

# Line splitting (row profile of the mask)
SMOOTH_WINDOW   = 9
MIN_PEAK_DIST   = 15
PEAK_PROMINENCE = 0.08
MIN_LINE_ROWS   = 30
ROW_PAD         = 3
TRIM_COLUMNS    = True

# Line -> chunk splitting
TARGET_WIDTH   = 200
TOLERANCE      = 30
INK_MARGIN     = 20
COL_SMOOTH     = 5
MIN_LAST_CHUNK = 50

# CRNN input size (must match training)
CRNN_IMG_WIDTH  = 200
CRNN_IMG_HEIGHT = 50


# ============================================================
# 1. DEEPLABV3+ ARCHITECTURE (must match training exactly so the
#    .h5 weights load correctly)
# ============================================================

def depthwise_separable_conv(x, filters, kernel_size=3, strides=1, dilation_rate=1, name_prefix=''):
    x = layers.DepthwiseConv2D(
        kernel_size=kernel_size, strides=strides, dilation_rate=dilation_rate,
        padding='same', use_bias=False, name=f'{name_prefix}_depthwise'
    )(x)
    x = BatchNormalization(name=f'{name_prefix}_depthwise_bn')(x)
    x = Activation('relu', name=f'{name_prefix}_depthwise_relu')(x)

    x = Conv2D(filters, kernel_size=1, padding='same', use_bias=False, name=f'{name_prefix}_pointwise')(x)
    x = BatchNormalization(name=f'{name_prefix}_pointwise_bn')(x)
    x = Activation('relu', name=f'{name_prefix}_pointwise_relu')(x)
    return x


def atrous_spatial_pyramid_pooling(x, output_stride=16):
    b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
    b0 = BatchNormalization(name='aspp0_bn')(b0)
    b0 = Activation('relu', name='aspp0_activation')(b0)

    atrous_rates = (6, 12, 18) if output_stride != 8 else (12, 24, 36)

    b1 = Conv2D(256, (3, 3), padding='same', dilation_rate=atrous_rates[0], use_bias=False, name='aspp1')(x)
    b1 = BatchNormalization(name='aspp1_bn')(b1)
    b1 = Activation('relu', name='aspp1_activation')(b1)

    b2 = Conv2D(256, (3, 3), padding='same', dilation_rate=atrous_rates[1], use_bias=False, name='aspp2')(x)
    b2 = BatchNormalization(name='aspp2_bn')(b2)
    b2 = Activation('relu', name='aspp2_activation')(b2)

    b3 = Conv2D(256, (3, 3), padding='same', dilation_rate=atrous_rates[2], use_bias=False, name='aspp3')(x)
    b3 = BatchNormalization(name='aspp3_bn')(b3)
    b3 = Activation('relu', name='aspp3_activation')(b3)

    b4 = GlobalAveragePooling2D()(x)
    b4 = tf.keras.layers.Lambda(lambda t: tf.expand_dims(t, 1))(b4)
    b4 = tf.keras.layers.Lambda(lambda t: tf.expand_dims(t, 1))(b4)
    b4 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='image_pooling')(b4)
    b4 = BatchNormalization(name='image_pooling_bn')(b4)
    b4 = Activation('relu', name='image_pooling_activation')(b4)

    def resize_to_feature_map(inputs):
        feature_map, pooled_features = inputs
        shape = tf.shape(feature_map)
        return tf.image.resize(pooled_features, [shape[1], shape[2]])

    b4 = tf.keras.layers.Lambda(resize_to_feature_map)([x, b4])

    x = layers.Concatenate()([b0, b1, b2, b3, b4])
    x = Conv2D(256, (1, 1), padding='same', use_bias=False, name='concat_projection')(x)
    x = BatchNormalization(name='concat_projection_bn')(x)
    x = Activation('relu', name='concat_projection_activation')(x)
    x = Dropout(0.1)(x)
    return x


def deeplabv3plus_model(input_size=(512, 512, 3), output_stride=16):
    inputs = Input(input_size)

    x = Conv2D(32, (3, 3), strides=2, padding='same', use_bias=False, name='entry_flow_conv1_1')(inputs)
    x = BatchNormalization(name='entry_flow_conv1_1_bn')(x)
    x = Activation('relu', name='entry_flow_conv1_1_relu')(x)

    x = Conv2D(64, (3, 3), padding='same', use_bias=False, name='entry_flow_conv1_2')(x)
    x = BatchNormalization(name='entry_flow_conv1_2_bn')(x)
    x = Activation('relu', name='entry_flow_conv1_2_relu')(x)

    # Block 1
    residual = Conv2D(128, (1, 1), strides=2, padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = depthwise_separable_conv(x, 128, name_prefix='entry_flow_block1_1')
    x = depthwise_separable_conv(x, 128, name_prefix='entry_flow_block1_2')
    x = MaxPooling2D((3, 3), strides=2, padding='same')(x)
    x = layers.Add()([x, residual])

    # Block 2
    residual = Conv2D(256, (1, 1), strides=2, padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = Activation('relu')(x)
    x = depthwise_separable_conv(x, 256, name_prefix='entry_flow_block2_1')
    x = depthwise_separable_conv(x, 256, name_prefix='entry_flow_block2_2')
    x = MaxPooling2D((3, 3), strides=2, padding='same')(x)
    x = layers.Add()([x, residual])

    low_level_features = x

    # Block 3
    residual = Conv2D(728, (1, 1), strides=2, padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = Activation('relu')(x)
    x = depthwise_separable_conv(x, 728, name_prefix='entry_flow_block3_1')
    x = depthwise_separable_conv(x, 728, name_prefix='entry_flow_block3_2')
    x = MaxPooling2D((3, 3), strides=2, padding='same')(x)
    x = layers.Add()([x, residual])

    # Middle flow
    for i in range(8):
        residual = x
        prefix = f'middle_flow_unit_{i + 1}'
        x = Activation('relu')(x)
        x = depthwise_separable_conv(x, 728, name_prefix=f'{prefix}_1')
        x = depthwise_separable_conv(x, 728, name_prefix=f'{prefix}_2')
        x = depthwise_separable_conv(x, 728, name_prefix=f'{prefix}_3')
        x = layers.Add()([x, residual])

    # Exit flow
    residual = Conv2D(1024, (1, 1), strides=1, padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = Activation('relu')(x)
    x = depthwise_separable_conv(x, 728, name_prefix='exit_flow_block1_1')
    x = depthwise_separable_conv(x, 1024, name_prefix='exit_flow_block1_2')
    x = layers.Add()([x, residual])

    x = depthwise_separable_conv(x, 1536, name_prefix='exit_flow_block2_1')
    x = depthwise_separable_conv(x, 1536, name_prefix='exit_flow_block2_2')
    x = depthwise_separable_conv(x, 2048, name_prefix='exit_flow_block2_3')

    x = atrous_spatial_pyramid_pooling(x, output_stride)

    low_level_features = Conv2D(48, (1, 1), padding='same', use_bias=False, name='feature_projection0')(low_level_features)
    low_level_features = BatchNormalization(name='feature_projection0_bn')(low_level_features)
    low_level_features = Activation('relu', name='feature_projection0_relu')(low_level_features)

    def upsample_to_match(inputs):
        encoder_features, low_level_features = inputs
        low_level_shape = tf.shape(low_level_features)
        return tf.image.resize(encoder_features, [low_level_shape[1], low_level_shape[2]])

    x = tf.keras.layers.Lambda(upsample_to_match)([x, low_level_features])
    x = layers.Concatenate()([x, low_level_features])

    x = depthwise_separable_conv(x, 256, name_prefix='decoder_conv0')
    x = depthwise_separable_conv(x, 256, name_prefix='decoder_conv1')

    def upsample_to_input_size(t):
        return tf.image.resize(t, [512, 512])

    x = tf.keras.layers.Lambda(upsample_to_input_size)(x)
    outputs = Conv2D(1, kernel_size=1, activation='sigmoid', name='output')(x)

    return Model(inputs=[inputs], outputs=[outputs])


# ============================================================
# 2. CRNN CUSTOM LAYERS (needed to load the saved .keras model)
# ============================================================

class ChannelMeanPool(layers.Layer):
    def call(self, x):
        return tf.reduce_mean(x, axis=-1, keepdims=True)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (1,)


class ChannelMaxPool(layers.Layer):
    def call(self, x):
        return tf.reduce_max(x, axis=-1, keepdims=True)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (1,)


class MainCTCLayer(layers.Layer):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, inputs):
        y_true, y_pred = inputs
        bl = tf.cast(tf.shape(y_true)[0], dtype='int64')
        il = tf.cast(tf.shape(y_pred)[1], dtype='int64') * tf.ones((bl, 1), dtype='int64')
        ll = tf.cast(tf.shape(y_true)[1], dtype='int64') * tf.ones((bl, 1), dtype='int64')
        self.add_loss(self.loss_fn(y_true, y_pred, il, ll))
        return y_pred

    def get_config(self):
        return super().get_config()


class PathCTCLayer(layers.Layer):
    def __init__(self, path_name, initial_weight=0.1, min_weight=0.02, decay_epochs=15, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.path_name = path_name
        self.initial_weight = initial_weight
        self.min_weight = min_weight
        self.decay_epochs = decay_epochs
        self.loss_fn = keras.backend.ctc_batch_cost
        self.epoch = tf.Variable(0.0, trainable=False, dtype=tf.float32)

    def call(self, inputs):
        y_true, y_pred = inputs
        bl = tf.cast(tf.shape(y_true)[0], dtype='int64')
        il = tf.cast(tf.shape(y_pred)[1], dtype='int64') * tf.ones((bl, 1), dtype='int64')
        ll = tf.cast(tf.shape(y_true)[1], dtype='int64') * tf.ones((bl, 1), dtype='int64')
        loss = self.loss_fn(y_true, y_pred, il, ll)
        ne = tf.minimum(self.epoch / self.decay_epochs, 1.0)
        weight = self.min_weight + (self.initial_weight - self.min_weight) * ((1.0 - ne) ** 2)
        self.add_loss(weight * loss)
        return y_pred

    def get_config(self):
        c = super().get_config()
        c.update({'path_name': self.path_name, 'initial_weight': self.initial_weight,
                  'min_weight': self.min_weight, 'decay_epochs': self.decay_epochs})
        return c


# ============================================================
# 3. MODEL LOADING (lazy — happens once, on first request)
# ============================================================

_deeplab_model = None
_crnn_model = None
_num_to_char = None


def get_deeplab_model():
    global _deeplab_model
    if _deeplab_model is None:
        print("Loading DeepLabv3+ segmentation model...")
        m = deeplabv3plus_model(input_size=(PATCH_SIZE, PATCH_SIZE, 3))
        m.load_weights(DEEPLAB_MODEL_PATH)
        _deeplab_model = m
        print("DeepLabv3+ model loaded.")
    return _deeplab_model


# ============================================================
# 3. CRNN MODEL LOADING
# ============================================================

_crnn_training_model = None
_crnn_inference_model = None
_num_to_char = None


def get_crnn_model():

    global _crnn_training_model
    global _crnn_inference_model
    global _num_to_char

    # --------------------------------------------------------
    # Load only once
    # --------------------------------------------------------

    if _crnn_inference_model is None:

        print("Loading CRNN training model...")

        # ----------------------------------------------------
        # Load vocabulary
        # ----------------------------------------------------

        with open(
            VOCAB_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            vocabulary = json.load(f)

        _num_to_char = layers.StringLookup(
            vocabulary=vocabulary,
            mask_token=None,
            invert=True
        )

        # ----------------------------------------------------
        # Allow Lambda/custom objects if required
        # ----------------------------------------------------

        keras.config.enable_unsafe_deserialization()

        # ----------------------------------------------------
        # Load the COMPLETE 5-output training model
        # ----------------------------------------------------

        _crnn_training_model = keras.models.load_model(

            CRNN_MODEL_PATH,

            custom_objects={

                "MainCTCLayer": MainCTCLayer,

                "PathCTCLayer": PathCTCLayer,

                "ChannelMeanPool": ChannelMeanPool,

                "ChannelMaxPool": ChannelMaxPool,

            },

            safe_mode=False
        )

        print(
            "Training model loaded."
        )

        print(
            "Training model inputs:"
        )

        for x in _crnn_training_model.inputs:

            print(
                "  ",
                x,
                x.shape
            )

        print(
            "\nTraining model outputs:"
        )

        for y in _crnn_training_model.outputs:

            print(
                "  ",
                y,
                y.shape
            )

        # ====================================================
        # BUILD INFERENCE MODEL
        # ====================================================

        # IMPORTANT:
        #
        # We take ONLY main_softmax.
        #
        # But we KEEP BOTH inputs:
        #
        # image
        # label
        #
        # because your new training graph has both inputs.
        # ====================================================

        softmax_layer = (
            _crnn_training_model
            .get_layer("main_softmax")
        )

        _crnn_inference_model = keras.Model(

            inputs=_crnn_training_model.inputs,

            outputs=softmax_layer.output,

            name="CRNN_Inference_Model"
        )

        print(
            "\nInference model created."
        )

        print(
            "Inference inputs:"
        )

        for x in _crnn_inference_model.inputs:

            print(
                "  ",
                x.name,
                x.shape
            )

        print(
            "\nInference output:"
        )

        print(
            _crnn_inference_model.output_shape
        )

    return (
        _crnn_inference_model,
        _num_to_char
    )

# ============================================================
# 4. PATCHING — resize the input image up to a multiple of
#    PATCH_SIZE and cut it into a row-major grid, mirroring how
#    the training patches were generated.
# ============================================================

def resize_to_multiple(img, patch_size=PATCH_SIZE):
    h, w = img.shape[:2]
    new_h = math.ceil(h / patch_size) * patch_size
    new_w = math.ceil(w / patch_size) * patch_size
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    rows = new_h // patch_size
    cols = new_w // patch_size
    return resized, rows, cols


def cut_patches(resized_img, rows, cols, patch_size=PATCH_SIZE):
    """Row-major patches: patch_idx increments across the whole grid."""
    patches = {}
    idx = 0
    for r in range(rows):
        for c in range(cols):
            patch = resized_img[r*patch_size:(r+1)*patch_size, c*patch_size:(c+1)*patch_size]
            patches[idx] = patch
            idx += 1
    return patches


# ============================================================
# 5. SEGMENTATION + RECONSTRUCTION
# ============================================================

def predict_patches_batch(model, patches_bgr):
    """patches_bgr: list of PATCH_SIZExPATCH_SIZEx3 uint8 BGR arrays.
    Runs ALL patches through the model in a single batched call instead of
    one at a time — looping single-image model.predict() calls is very slow
    on CPU because of per-call overhead. Returns list of (patch_rgb, mask)."""
    rgb_list = [cv2.cvtColor(p, cv2.COLOR_BGR2RGB) for p in patches_bgr]
    batch = np.stack([rgb.astype(np.float32) / 255.0 for rgb in rgb_list], axis=0)
    preds = model.predict(batch, verbose=0)
    masks = (preds.squeeze(-1) > 0.5).astype(np.uint8)
    return [(rgb_list[i], masks[i]) for i in range(len(rgb_list))]


def segment_patches_iter(model, orig_bgr, patch_masks, batch_size=SEG_BATCH):
    """Segments the patches that are not yet in `patch_masks` (dict idx -> mask,
    filled in place), SEG_BATCH at a time, yielding (done, total) after every
    batch. Small batches give the UI a chance to update and to stop/resume
    in the middle of segmenting a long manuscript."""
    resized, rows, cols = resize_to_multiple(orig_bgr)
    patches = cut_patches(resized, rows, cols)
    pending = [i for i in sorted(patches.keys()) if i not in patch_masks]
    for start in range(0, len(pending), batch_size):
        ids = pending[start:start + batch_size]
        results = predict_patches_batch(model, [patches[i] for i in ids])
        for i, (_rgb, mask) in zip(ids, results):
            patch_masks[i] = mask
        yield len(patch_masks), len(patches)


def assemble_reconstruction(orig_bgr, patch_masks):
    """Builds the full-size (reconstructed image, mask) from per-patch masks."""
    h, w = orig_bgr.shape[:2]
    resized, rows, cols = resize_to_multiple(orig_bgr)
    patches = cut_patches(resized, rows, cols)
    patch_results = {
        i: (cv2.cvtColor(patches[i], cv2.COLOR_BGR2RGB), patch_masks[i])
        for i in sorted(patches.keys())
    }

    fg_sum = np.zeros(3, dtype=np.float64)
    fg_count = 0
    for patch_rgb, patch_mask in patch_results.values():
        fg_pixels = patch_rgb[patch_mask == 1]
        if fg_pixels.size > 0:
            fg_sum += fg_pixels.sum(axis=0)
            fg_count += fg_pixels.shape[0]

    avg_color = (fg_sum / fg_count).astype(np.uint8) if fg_count > 0 else np.array([255, 255, 255], dtype=np.uint8)

    rgb_canvas = np.empty((rows * PATCH_SIZE, cols * PATCH_SIZE, 3), dtype=np.uint8)
    rgb_canvas[:] = avg_color
    mask_canvas = np.zeros((rows * PATCH_SIZE, cols * PATCH_SIZE), dtype=np.uint8)

    for idx, (patch_rgb, patch_mask) in patch_results.items():
        r, c = idx // cols, idx % cols
        masked_patch = np.where(patch_mask[..., None] == 1, patch_rgb, avg_color)
        rgb_canvas[r*PATCH_SIZE:(r+1)*PATCH_SIZE, c*PATCH_SIZE:(c+1)*PATCH_SIZE] = masked_patch
        mask_canvas[r*PATCH_SIZE:(r+1)*PATCH_SIZE, c*PATCH_SIZE:(c+1)*PATCH_SIZE] = patch_mask

    rgb_canvas = cv2.resize(rgb_canvas, (w, h), interpolation=cv2.INTER_LINEAR)
    mask_canvas = cv2.resize(mask_canvas, (w, h), interpolation=cv2.INTER_NEAREST)

    bgr_canvas = cv2.cvtColor(rgb_canvas, cv2.COLOR_RGB2BGR)
    return bgr_canvas, mask_canvas


def reconstruct(model, orig_bgr):
    """One-shot segmentation + reconstruction (no progress / stopping)."""
    patch_masks = {}
    for _ in segment_patches_iter(model, orig_bgr, patch_masks):
        pass
    return assemble_reconstruction(orig_bgr, patch_masks)


# ============================================================
# 6. LINE SPLITTING
# ============================================================

def _tighten_band(row_sum, top, bot):
    band = row_sum[top:bot]
    nonzero = np.where(band > 0)[0]
    if len(nonzero) == 0:
        return top, bot
    return top + int(nonzero[0]), top + int(nonzero[-1]) + 1


def split_into_lines(mask):
    row_sum = mask.sum(axis=1).astype(float)
    if row_sum.max() == 0:
        return []

    smooth = uniform_filter1d(row_sum, size=SMOOTH_WINDOW)
    peaks, _ = find_peaks(smooth, distance=MIN_PEAK_DIST, prominence=smooth.max() * PEAK_PROMINENCE)

    if len(peaks) == 0:
        if row_sum.max() < MIN_LINE_ROWS:
            return []
        return [_tighten_band(row_sum, 0, len(row_sum))]

    h = len(row_sum)
    bounds = [0]
    for i in range(len(peaks) - 1):
        a, b = peaks[i], peaks[i + 1]
        valley = a + int(np.argmin(smooth[a:b + 1]))
        bounds.append(valley)
    bounds.append(h)

    lines = []
    for i in range(len(bounds) - 1):
        top, bot = bounds[i], bounds[i + 1]
        if row_sum[top:bot].max() < MIN_LINE_ROWS:
            continue
        lines.append(_tighten_band(row_sum, top, bot))
    return lines


def crop_line(img, mask, top, bot):
    """Returns (crop, top_abs, bot_abs, left_abs, right_abs) — the offsets let
    callers map anything found inside `crop` back to absolute coordinates on
    the full reconstructed image (used for drawing the annotated overlay)."""
    h, w = img.shape[0], img.shape[1]
    top_p = max(0, top - ROW_PAD)
    bot_p = min(h, bot + ROW_PAD)
    crop = img[top_p:bot_p, :]
    mask_band = mask[top_p:bot_p, :]

    left, right = 0, w
    if TRIM_COLUMNS:
        col_fg = mask_band.any(axis=0)
        if col_fg.any():
            left = int(np.argmax(col_fg))
            right = len(col_fg) - int(np.argmax(col_fg[::-1]))
            crop = crop[:, left:right]
    return crop, top_p, bot_p, left, right


# ============================================================
# 7. LINE -> CHUNK SPLITTING
# ============================================================

def ink_column_profile(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    counts = np.bincount(gray.flatten(), minlength=256)
    bg_val = int(np.argmax(counts))
    thresh = bg_val - INK_MARGIN
    ink = (gray < thresh).astype(np.uint8)
    col_sum = ink.sum(axis=0).astype(float)
    return uniform_filter1d(col_sum, size=COL_SMOOTH)


def find_chunk_bounds(col_profile, target=TARGET_WIDTH, tol=TOLERANCE, min_last=MIN_LAST_CHUNK):
    w = len(col_profile)
    if w <= target + tol:
        return [(0, w)]

    bounds = [0]
    pos = 0
    while True:
        remaining = w - pos
        if remaining <= target + tol:
            bounds.append(w)
            break
        lo = max(pos + target - tol, pos + 1)
        hi = min(pos + target + tol, w - 1)
        window = col_profile[lo:hi + 1]
        min_val = window.min()
        candidates = np.where(window == min_val)[0]
        ideal = (pos + target) - lo
        best = candidates[np.argmin(np.abs(candidates - ideal))]
        cut = lo + int(best)
        if cut <= pos:
            cut = pos + target
        bounds.append(cut)
        pos = cut

    if len(bounds) >= 3 and (bounds[-1] - bounds[-2]) < min_last:
        bounds.pop(-2)
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def split_line_into_chunks(line_img):
    if line_img.shape[1] == 0 or line_img.shape[0] == 0:
        return []
    profile = ink_column_profile(line_img)
    bounds = find_chunk_bounds(profile)
    return [line_img[:, left:right] for left, right in bounds]


def split_line_into_chunks_with_bounds(line_img):
    """Same as split_line_into_chunks, but also returns each chunk's
    (left, right) column bounds *within line_img* — needed to compute the
    chunk's absolute rectangle on the full manuscript for the annotated
    overlay."""
    if line_img.shape[1] == 0 or line_img.shape[0] == 0:
        return [], []
    profile = ink_column_profile(line_img)
    bounds = find_chunk_bounds(profile)
    imgs = [line_img[:, left:right] for left, right in bounds]
    return bounds, imgs


# ============================================================
# 8. CRNN RECOGNITION (operates directly on in-memory BGR arrays)
# ============================================================

def _as_rgb_uint8(image):
    """Normalize Gradio image variants before passing them to OpenCV."""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    if image.dtype != np.uint8:
        peak = float(np.nanmax(image)) if image.size else 0.0
        scale = 255.0 if peak <= 1.0 else 1.0
        image = np.clip(image * scale, 0, 255).astype(np.uint8)
    return image


# ============================================================
# CRNN RECOGNITION
# ============================================================

def _preprocess_chunk(chunk_bgr):

    # --------------------------------------------------------
    # BGR → RGB
    # --------------------------------------------------------

    img_rgb = cv2.cvtColor(
        chunk_bgr,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------------
    # Normalize to [0, 1]
    # --------------------------------------------------------

    img = (
        img_rgb.astype(np.float32)
        / 255.0
    )

    # --------------------------------------------------------
    # Resize exactly as training
    # --------------------------------------------------------

    img = tf.image.resize(
        img,
        (
            CRNN_IMG_HEIGHT,
            CRNN_IMG_WIDTH
        )
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Training:
    # (50, 200, 3)
    #
    # transpose:
    # (200, 50, 3)
    # --------------------------------------------------------

    img = tf.transpose(
        img,
        perm=[1, 0, 2]
    )

    return img


def predict_chunks_batch(
        model,
        num_to_char,
        chunks_bgr
    ):

    """
    Predict all chunks using the NEW
    two-input inference model.

    Model inputs:

        image : (None, 200, 50, 3)
        label : (None, None)

    Model output:

        main_softmax : (None, 300, 81)
    """

    if not chunks_bgr:

        return []


    # ========================================================
    # PREPROCESS ALL CHUNKS
    # ========================================================

    batch = tf.stack(
        [
            _preprocess_chunk(c)
            for c in chunks_bgr
        ],
        axis=0
    )

    print(
        f"CRNN batch shape: {batch.shape}"
    )


    # ========================================================
    # CREATE DUMMY LABEL
    # ========================================================

    batch_size = tf.shape(
        batch
    )[0]

    # The actual label is NOT needed for inference.
    #
    # It is supplied only because the saved training model
    # has two inputs.
    #
    # One dummy value per image is sufficient for this
    # inference graph because we are extracting main_softmax.
    # ========================================================

    dummy_label = tf.zeros(
        (
            batch_size,
            1
        ),
        dtype=tf.float32
    )


    # ========================================================
    # PREDICT
    # ========================================================

    preds = model.predict(

        [
            batch,
            dummy_label
        ],

        verbose=0
    )


    # ========================================================
    # SAFETY CHECK
    # ========================================================

    if isinstance(
        preds,
        list
    ):

        print(
            "WARNING: inference model returned a list."
        )

        preds = preds[0]


    print(
        f"CRNN prediction shape: {preds.shape}"
    )


    # Expected:

    # (batch_size, 300, 81)


    # ========================================================
    # CTC DECODE
    # ========================================================

    input_len = (
        np.ones(
            preds.shape[0]
        )
        * preds.shape[1]
    )


    decoded = keras.backend.ctc_decode(

        preds,

        input_length=input_len,

        greedy=True

    )[0][0].numpy()


    # ========================================================
    # CONVERT INDICES → CHARACTERS
    # ========================================================

    texts = []


    for result in decoded:

        text = ""

        for idx in result:

            if idx != -1:

                char = num_to_char(

                    tf.constant(
                        [idx]
                    )

                )

                text += (
                    char.numpy()[0]
                    .decode("utf-8")
                )

        texts.append(text)


    return texts


def draw_one_chunk(annotated_bgr, rect, in_place=True, color=CURRENT_COLOR):
    """Draws a single chunk's highlight box onto annotated_bgr (BGR, mutated
    in place by default)."""
    x1, y1, x2, y2 = rect
    if x2 <= x1 or y2 <= y1:
        return annotated_bgr
    target = annotated_bgr if in_place else annotated_bgr.copy()
    region = target[y1:y2, x1:x2].astype(np.float32)
    tint = region * 0.6 + np.array(color, dtype=np.float32) * 0.4
    target[y1:y2, x1:x2] = tint.clip(0, 255).astype(np.uint8)
    cv2.rectangle(target, (x1, y1), (x2, y2), color, 2)
    return target


def draw_annotations(recon_bgr, chunk_rects_and_texts):
    """Draws every chunk's highlight in one pass. Returns an RGB image."""
    annotated = recon_bgr.copy()
    for rect, _text in chunk_rects_and_texts:
        draw_one_chunk(annotated, rect, in_place=True)
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)


def render_scan(recon_bgr, rect=None, color=CURRENT_COLOR, max_side=None):
    """The reconstructed image with at most ONE highlighted chunk (the one
    being recognized, or the one selected in the transcript). Returns RGB,
    optionally downscaled so live updates of big manuscripts stay cheap."""
    annotated = recon_bgr.copy()
    if rect is not None:
        draw_one_chunk(annotated, rect, in_place=True, color=color)
    h, w = annotated.shape[:2]
    if max_side and max(h, w) > max_side:
        s = max_side / max(h, w)
        annotated = cv2.resize(annotated, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)


# ============================================================
# 9. FULL PIPELINE — generators that yield as each line/chunk is
#    processed, so the UI updates live. Progress lives in a per-user
#    session dict (gr.State) so a run can be stopped and resumed.
#    The CRNN is still called once per LINE (batched across that line's
#    chunks) for speed — only the on-screen update happens per chunk.
# ============================================================

# Order of the values every pipeline/handler function returns; must match
# the `outputs=` list wired up in the UI section.
OUT_KEYS = ("mask", "scan", "lines", "chunks", "transcript", "plain", "status", "region", "session",
            "stop_btn", "resume_btn", "skeleton")

# Placeholder shown in the transcript panel (and a sweep over the live scan)
# while models load and the page is segmented — i.e. until the first chunk of
# text arrives. An empty string hides it again.
SKELETON_HTML = (
    '<div class="ocr-skeleton" role="status" aria-label="Recognizing manuscript">'
    + '<div class="sk-lines" aria-hidden="true">'
    + ''.join(f'<span class="sk-line" style="width:{w}%;--sk-delay:{i * 0.11:.2f}s"></span>'
              for i, w in enumerate((92, 78, 86, 64, 88, 72, 54)))
    + '</div><div class="sk-caption"><span class="sk-dots" aria-hidden="true">'
    + '<i></i><i></i><i></i></span>Reading the manuscript…</div></div>'
)


def _out(**values):
    """Builds an output tuple; anything not given is left unchanged in the UI."""
    return tuple(values[k] if k in values else gr.skip() for k in OUT_KEYS)


def _buttons(paused):
    """Stop and Resume share one slot: Stop normally, Resume while paused."""
    return {"stop_btn": gr.update(visible=not paused), "resume_btn": gr.update(visible=paused)}


def new_session():
    return {
        "key": None,          # fingerprint of the image this progress belongs to
        "orig_rgb": None,
        "patch_masks": {},    # segmentation progress: patch idx -> mask
        "recon": None,        # reconstructed BGR image (set once segmentation is done)
        "line_crops": [],
        "line_offsets": [],   # (top, bottom, left) of each line on the full image
        "line_gallery": [],
        "overlay": None,
        "next_line": 0,       # index of the first line not fully recognized yet
                              # (its finished chunks are kept; resume skips them)
        "records": [],        # one dict per recognized chunk: line, chunk, text, rect, img
        "done": False,
    }


def _image_key(image):
    return f"{image.shape}:{hashlib.md5(image.tobytes()).hexdigest()}"


def plain_text(records):
    lines = {}
    for r in records:
        lines.setdefault(r["line"], []).append(r["text"])
    return '\n'.join(' '.join(lines[i]) for i in sorted(lines))


def build_transcript(records, current=None, selected=None):
    """HighlightedText value: one clickable token per chunk, with a plain
    separator token between chunks — so chunk i is always token 2*i."""
    tokens = []
    for i, r in enumerate(records):
        if i > 0:
            tokens.append(("\n" if r["line"] != records[i - 1]["line"] else " ", None))
        label = "selected" if i == selected else "current" if i == current else "chunk"
        tokens.append((r["text"] or "·", label))
    return tokens


def chunk_gallery(records):
    return [(r["img"], f"L{r['line']:02d} C{r['chunk']:02d}: {r['text']}") for r in records]


def _last_rect(records):
    """Where the scan stopped: the most recently recognized chunk, if any."""
    return records[-1]["rect"] if records else None


def _progress_status(session):
    total = len(session["line_crops"])
    return f"{session['next_line']}/{total} line(s), {len(session['records'])} chunk(s)"


def run_pipeline(image, session=None):
    """Recognize: start from scratch on the uploaded image."""
    yield from _guarded(_pipeline(image, session, resume=False))


def resume_pipeline(image, session=None):
    """Resume: continue a stopped run on the same image."""
    yield from _guarded(_pipeline(image, session, resume=True))


def _guarded(gen):
    """Clears the loading skeleton if the pipeline fails, then re-raises."""
    try:
        yield from gen
    except Exception:
        yield _out(skeleton="", **_buttons(paused=False))
        raise


def _pipeline(image, session, resume):
    if image is None:
        yield _out(status="Upload an image to begin.")
        return

    image = _as_rgb_uint8(image)
    key = _image_key(image)

    if resume:
        if session is None or session["key"] != key:
            yield _out(status="Nothing to resume for this image · press Recognize to start")
            return
        if session["done"]:
            yield _out(status=f"Already complete · {_progress_status(session)}")
            return
        yield _out(status="Resuming…", skeleton="" if session["records"] else SKELETON_HTML,
                   **_buttons(paused=False))
    else:
        # A NEW dict (not cleared in place): a previously stopped run that is
        # still finishing its last step can't write into this one.
        session = new_session()
        session["key"] = key
        session["orig_rgb"] = image
        # Clear the previous transcript before expensive model loading/inference.
        yield _out(mask=None, scan=image, lines=[], chunks=[], transcript=[], plain="", region=None,
                   status="Preparing · loading recognition model", session=session,
                   skeleton=SKELETON_HTML, **_buttons(paused=False))

    orig_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    crnn, num_to_char = get_crnn_model()

    # 1. Segment (in small batches) + reconstruct — skipped when resuming
    #    after segmentation already finished.
    if session["recon"] is None:
        yield _out(status="Scanning · loading segmentation model")
        deeplab = get_deeplab_model()
        for done, total in segment_patches_iter(deeplab, orig_bgr, session["patch_masks"]):
            yield _out(status=f"Scanning · finding text · patch {done}/{total}")

        recon_bgr, mask = assemble_reconstruction(orig_bgr, session["patch_masks"])

        # Mask overlay for display (red tint over detected foreground)
        overlay = recon_bgr.copy()
        overlay[mask == 1] = (0.5 * overlay[mask == 1] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
        session["overlay"] = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        # 2. Split into lines — keep each line's absolute (top, bottom, left)
        #    offset in the full image so chunk positions can be mapped back.
        for top, bot in split_into_lines(mask):
            crop, top_p, bot_p, left, _right = crop_line(recon_bgr, mask, top, bot)
            session["line_crops"].append(crop)
            session["line_offsets"].append((top_p, bot_p, left))
        session["line_gallery"] = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
                                   for c in session["line_crops"] if c.size > 0]
        session["recon"] = recon_bgr

    recon_bgr = session["recon"]
    if not session["line_crops"]:
        session["done"] = True
        yield _out(mask=session["overlay"], scan=render_scan(recon_bgr), lines=[], skeleton="",
                   status="No text lines detected. Try a clearer image with complete text lines.")
        return

    records = session["records"]
    total_lines = len(session["line_crops"])

    # On resume, keep the green box on the chunk where the scan stopped.
    yield _out(
        mask=session["overlay"],
        scan=render_scan(recon_bgr, _last_rect(records), CURRENT_COLOR, max_side=LIVE_MAX_SIDE),
        lines=session["line_gallery"],
        transcript=build_transcript(records, current=len(records) - 1 if records else None),
        plain=plain_text(records),
        status=f"Scanning · {total_lines} text line(s) detected",
    )

    # 3. Walk line by line. Within each line, batch all of that line's
    #    chunks through the CRNN in ONE call (fast), then reveal them on
    #    screen one at a time (live). Only the scan image + transcript are
    #    sent per chunk; the galleries are sent once at the end.
    for li in range(session["next_line"], total_lines):
        line_img = session["line_crops"][li]
        top_p, bot_p, left = session["line_offsets"][li]
        bounds, imgs = split_line_into_chunks_with_bounds(line_img)
        # A line interrupted half-way continues after its last finished chunk.
        finished = {r["chunk"] for r in records if r["line"] == li + 1}
        todo = [(k, b, img) for k, (b, img) in enumerate(zip(bounds, imgs), start=1) if k not in finished]
        texts = predict_chunks_batch(crnn, num_to_char, [img for _k, _b, img in todo]) if todo else []

        for (chunk_idx, (cl, cr), chunk_img), text in zip(todo, texts):
            if chunk_img.size == 0:
                continue
            rect = (left + cl, top_p, left + cr, bot_p)
            records.append({
                "line": li + 1, "chunk": chunk_idx, "text": text, "rect": rect,
                "img": cv2.cvtColor(chunk_img, cv2.COLOR_BGR2RGB),
            })
            yield _out(
                scan=render_scan(recon_bgr, rect, CURRENT_COLOR, max_side=LIVE_MAX_SIDE),
                transcript=build_transcript(records, current=len(records) - 1),
                plain=plain_text(records),
                status=f"Scanning · line {li + 1}/{total_lines} · {len(records)} chunk(s) recognized",
                **({"skeleton": ""} if len(records) == 1 else {}),  # first text replaces the skeleton
            )
        session["next_line"] = li + 1

    session["done"] = True
    n_lines = len({r["line"] for r in records})
    yield _out(
        scan=render_scan(recon_bgr),
        chunks=chunk_gallery(records),
        transcript=build_transcript(records),
        plain=plain_text(records),
        skeleton="",
        status=f"Complete · recognized {len(records)} chunk(s) across {n_lines} line(s) · "
               f"click any text to see where it came from",
    )


def stop_pipeline(session):
    """Runs after the Stop button cancels the running generator."""
    if session is None or session["key"] is None:
        return _out(status="Stopped", skeleton="")
    if session["done"]:  # nothing was running; keep the finished result as is
        return _out()
    if session["recon"] is None:
        done = len(session["patch_masks"])
        return _out(status=f"Paused while finding text ({done} patch(es) done) · press Resume to continue",
                    skeleton="", **_buttons(paused=True))
    records = session["records"]
    return _out(
        scan=render_scan(session["recon"], _last_rect(records), CURRENT_COLOR),
        chunks=chunk_gallery(records),
        transcript=build_transcript(records, current=len(records) - 1 if records else None),
        plain=plain_text(records),
        status=f"Paused · {_progress_status(session)} done · press Resume to continue",
        skeleton="",
        **_buttons(paused=True),
    )


def reset_session(_session=None):
    return _out(mask=None, scan=None, lines=[], chunks=[], transcript=[], plain="", region=None,
                status="Ready · upload an image to begin", session=new_session(),
                skeleton="", **_buttons(paused=False))


def _show_source(session, idx):
    """Highlights chunk `idx` on the scan and shows its region of the input."""
    if session is None or session["recon"] is None or not (0 <= idx < len(session["records"])):
        return _out()
    records = session["records"]
    r = records[idx]
    x1, y1, x2, y2 = r["rect"]
    orig = session["orig_rgb"]
    h, w = orig.shape[:2]
    region = orig[max(0, y1 - REGION_PAD):min(h, y2 + REGION_PAD),
                  max(0, x1 - REGION_PAD):min(w, x2 + REGION_PAD)]
    return _out(
        scan=render_scan(session["recon"], r["rect"], SELECTED_COLOR),
        transcript=build_transcript(records, selected=idx),
        region=region,
        status=f"Line {r['line']}, chunk {r['chunk']} · “{r['text']}” · "
               f"x {x1}–{x2}, y {y1}–{y2} on the source image",
    )


def select_from_transcript(session, evt: gr.SelectData):
    idx = evt.index
    if isinstance(idx, (list, tuple)):
        idx = idx[0]
    if not isinstance(idx, int) or idx % 2:  # odd tokens are separators
        return _out()
    return _show_source(session, idx // 2)


def select_from_gallery(session, evt: gr.SelectData):
    return _show_source(session, evt.index)



# ============================================================
# 10. GRADIO UI
# ============================================================
import csv as _csv
import json as _json
import io
import tempfile
import datetime

# ── Export helpers ───────────────────────────────────────────────────────────

def _timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def export_txt(session):
    """Export plain text transcript."""
    if not session or not session.get("records"):
        return None
    text = plain_text(session["records"])
    path = os.path.join(tempfile.gettempdir(), f"leafocr_{_timestamp()}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def export_json(session):
    """Export structured JSON with per-chunk metadata."""
    if not session or not session.get("records"):
        return None
    records = session["records"]
    data = {
        "exported_at": datetime.datetime.now().isoformat(),
        "total_lines": len({r["line"] for r in records}),
        "total_chunks": len(records),
        "chunks": [
            {
                "line": r["line"],
                "chunk": r["chunk"],
                "text": r["text"],
                "bbox": {"x1": r["rect"][0], "y1": r["rect"][1],
                         "x2": r["rect"][2], "y2": r["rect"][3]},
            }
            for r in records
        ],
        "full_text": plain_text(records),
    }
    path = os.path.join(tempfile.gettempdir(), f"leafocr_{_timestamp()}.json")
    with open(path, "w", encoding="utf-8") as f:
        _json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def export_csv(session):
    """Export tabular CSV: line, chunk, text, x1, y1, x2, y2."""
    if not session or not session.get("records"):
        return None
    path = os.path.join(tempfile.gettempdir(), f"leafocr_{_timestamp()}.csv")
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["line", "chunk", "text", "x1", "y1", "x2", "y2"])
        for r in session["records"]:
            x1, y1, x2, y2 = r["rect"]
            w.writerow([r["line"], r["chunk"], r["text"], x1, y1, x2, y2])
    return path


def export_docx(session):
    """Export a Word document with one paragraph per line."""
    if not session or not session.get("records"):
        return None
    from docx import Document
    from docx.shared import Pt, RGBColor, Cm
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    doc = Document()
    # Page margins
    for sec in doc.sections:
        sec.top_margin = Cm(2.5)
        sec.bottom_margin = Cm(2.5)
        sec.left_margin = Cm(3)
        sec.right_margin = Cm(2.5)

    title = doc.add_heading("LeafOCR Transcript", level=1)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.runs[0].font.color.rgb = RGBColor(0x25, 0x3e, 0x34)

    ts = doc.add_paragraph(
        f"Exported: {datetime.datetime.now().strftime('%d %B %Y, %H:%M')}"
    )
    ts.alignment = WD_ALIGN_PARAGRAPH.CENTER
    ts.runs[0].font.size = Pt(9)
    ts.runs[0].font.color.rgb = RGBColor(0x7b, 0x86, 0x7b)
    doc.add_paragraph()   # spacer

    records = session["records"]
    lines_map = {}
    for r in records:
        lines_map.setdefault(r["line"], []).append(r["text"])

    for li in sorted(lines_map):
        para = doc.add_paragraph(" ".join(lines_map[li]))
        para.runs[0].font.size = Pt(14)
        para.paragraph_format.space_after = Pt(8)

    path = os.path.join(tempfile.gettempdir(), f"leafocr_{_timestamp()}.docx")
    doc.save(path)
    return path


def export_pdf(session):
    """Export a PDF using fpdf2 with the Nirmala TTC font (supports Malayalam)."""
    if not session or not session.get("records"):
        return None
    from fpdf import FPDF

    FONT_PATH = r"C:\Windows\Fonts\Nirmala.ttc"
    FALLBACK  = None
    # Check if Malayalam font is available
    has_font = os.path.exists(FONT_PATH)

    class PDF(FPDF):
        pass

    pdf = PDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    pdf.set_margins(25, 20, 20)

    if has_font:
        pdf.add_font("Nirmala", style="", fname=FONT_PATH)
        pdf.add_font("Nirmala", style="B", fname=FONT_PATH)
        body_font = "Nirmala"
    else:
        body_font = "Helvetica"

    # Title
    pdf.set_font(body_font, style="B", size=18)
    pdf.set_text_color(37, 62, 52)
    pdf.cell(0, 10, "LeafOCR Transcript", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    # Timestamp
    pdf.set_font(body_font, size=9)
    pdf.set_text_color(123, 134, 123)
    pdf.cell(0, 6,
             f"Exported: {datetime.datetime.now().strftime('%d %B %Y, %H:%M')}",
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    # Divider
    pdf.set_draw_color(217, 223, 212)
    pdf.line(25, pdf.get_y(), 190, pdf.get_y())
    pdf.ln(8)

    # Body text
    pdf.set_font(body_font, size=14)
    pdf.set_text_color(52, 69, 54)

    records = session["records"]
    lines_map = {}
    for r in records:
        lines_map.setdefault(r["line"], []).append(r["text"])

    for li in sorted(lines_map):
        line_text = " ".join(lines_map[li])
        pdf.multi_cell(0, 9, line_text, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

    # Footer
    pdf.set_y(-18)
    pdf.set_font(body_font, size=8)
    pdf.set_text_color(160, 170, 155)
    pdf.cell(0, 6, "Generated by LeafOCR · Palm-leaf manuscript recognition", align="C")

    path = os.path.join(tempfile.gettempdir(), f"leafocr_{_timestamp()}.pdf")
    pdf.output(path)
    return path


# Dispatcher — called by the export button
_EXPORT_FNS = {
    "Plain text (.txt)":   export_txt,
    "JSON (.json)":        export_json,
    "CSV (.csv)":          export_csv,
    "Word document (.docx)": export_docx,
    "PDF (.pdf)":          export_pdf,
}

def do_export(fmt, session):
    fn = _EXPORT_FNS.get(fmt)
    if fn is None or not session or not session.get("records"):
        return None
    try:
        return fn(session)
    except Exception as e:
        print(f"[Export error] {fmt}: {e}")
        return None


# ── CSS ──────────────────────────────────────────────────────────────────────

APP_CSS = """
body, .dark, gradio-app { background: #f3f2ec !important; }
.gradio-container {
    max-width: 1500px !important; margin: 0 auto !important; padding: 24px 32px !important;
    font-family: 'Segoe UI', sans-serif !important; color: #253e34 !important;
    --body-background-fill: #f3f2ec; --body-text-color: #253e34;
    --body-text-color-subdued: #69766d; --background-fill-primary: #fffefa;
    --background-fill-secondary: #f1f3ed; --block-background-fill: #fffefa;
    --block-border-color: #dce2d8; --border-color-primary: #dce2d8;
    --border-color-accent: #537963; --input-background-fill: #f7f8f3;
    --input-border-color: #dce2d8; --input-placeholder-color: #7b877e;
    --block-label-background-fill: #fffefa; --block-label-text-color: #52675a;
    --block-title-text-color: #253e34; --input-text-color: #253e34;
    --color-accent: #346b50; --checkbox-background-color-selected: #346b50;
    --button-secondary-background-fill: #eff3ed; --button-secondary-text-color: #253e34;
}
.gradio-container [data-testid="block-info"] { background:transparent !important; color:#65765f !important; font-size:11px !important; font-weight:600 !important; box-shadow:none !important; }
.brandbar { display:flex; align-items:center; justify-content:space-between; padding:0 0 20px; border-bottom:1px solid #d9dfd4; }
.brand { display:flex; align-items:center; gap:10px; font-weight:700; font-size:23px; letter-spacing:-1px; }
.brand-mark { display:grid; place-items:center; width:38px; height:38px; background:#214d38; color:#e7efd9; border-radius:12px 12px 12px 3px; }
.brand-note { color:#728071; font-size:11px; text-transform:uppercase; letter-spacing:2px; }
.brand-right { display:flex; align-items:center; gap:20px; }
.leaf-seal { width:76px; height:76px; flex:none; border:1px solid #d1dac9; border-radius:50%; display:grid; place-content:center; text-align:center; color:#708368; font:italic 30px Georgia,serif; background:radial-gradient(circle,#e8edde,#f3f2ec); }
.leaf-seal small { font:7px 'Segoe UI',sans-serif; letter-spacing:1.1px; margin-top:4px; }

/* ── Scanner toggle toolbar ── */
#scanner-toolbar { display:flex; align-items:center; justify-content:flex-end; gap:10px; margin-bottom:10px; }
#scanner-toolbar > div { flex:none !important; width:auto !important; }
#toggle-scanner {
    display:inline-flex !important; align-items:center !important; gap:6px !important;
    background:#eff3ed !important; border:1px solid #cfd8c7 !important; border-radius:9px !important;
    padding:6px 14px !important; font-size:12px !important; font-weight:600 !important;
    color:#3d5c44 !important; cursor:pointer !important;
    transition:background .15s, box-shadow .15s !important;
    white-space:nowrap !important; min-height:0 !important; height:34px !important;
    width:auto !important;
}
#toggle-scanner:hover { background:#e3ead9 !important; box-shadow:0 2px 6px #2a4a3510 !important; }

/* ── Stage cards ── */
.stage-card {
    background:#fffefa !important; border:1px solid #dce2d8 !important;
    border-radius:18px !important; padding:20px !important;
    box-shadow:0 5px 20px #233e2f06; gap:14px !important;
}
.stage-title { height:78px; flex:none !important; }
.stage-heading { display:flex; align-items:center; gap:10px; margin-bottom:9px; }
.step { display:grid; place-items:center; width:28px; height:28px; border-radius:9px; background:#edf2e7; color:#57734d; font-size:11px; font-weight:700; }
.stage-heading h2 { font-size:17px; font-weight:650; color:#253e34; margin:0; letter-spacing:-.3px; }
.stage-title p { color:#7b867b; font-size:12px; line-height:1.65; margin:0; }
.status-bar { background:#e9efe2; border:1px solid #d7e1ce; border-radius:10px; padding:9px 14px !important; margin-bottom:6px; }
.status-bar p { font-size:12px !important; color:#50674a !important; margin:0 !important; }
.status-bar p:before { content:'●'; color:#598152; margin-right:9px; font-size:10px; }

/* ── Workspace row ── */
#workspace { gap:20px; align-items:stretch !important; }

/* ── Rectangular image panel ── */
.image-stage {
    height:360px !important; min-height:320px !important; flex:none !important;
    border:1px solid #cfd8c7 !important; border-radius:12px !important;
    background:#f3f5ee !important; overflow:hidden !important;
    display:flex !important; align-items:center !important; justify-content:center !important;
}
#ms-upload .image-stage { border-style:dashed !important; }
.image-stage img { object-fit:contain !important; object-position:center !important; width:100% !important; height:100% !important; }
.image-stage .image-container { background:transparent !important; width:100% !important; height:100% !important; }

/* ── Rectangular transcript panel ── */
#transcript {
    height:300px !important; overflow-y:auto !important; flex:none !important;
    background:#faf8f0 !important; border:1px solid #e8e3d4 !important;
    border-radius:12px !important; box-sizing:border-box !important;
}
#transcript, #transcript * { font-family:'Nirmala UI','Noto Sans Malayalam',sans-serif; }
#transcript .textfield { font-size:19px; line-height:1.85; color:#344536; }
#transcript .token-container { white-space:nowrap; }
#transcript .token-container:empty { flex-basis:100%; height:0; }
#transcript .token { margin:2px 0 !important; padding:0 4px !important; cursor:pointer; border-radius:5px; }
#transcript .token:not(.highlighted) { padding:0 !important; cursor:default; }
#transcript .token.highlighted:hover { filter:brightness(.92); outline:1px solid #b9c7ad; }
#transcript > label { background:transparent !important; box-shadow:none !important; border:0 !important; }

/* ── Source region panel ── */
#region { height:110px !important; flex:none !important; border-radius:12px !important; }

/* ── Controls ── */
#run-controls { gap:8px; }
#copy-text { align-self:flex-end; flex:none !important; min-height:0 !important; height:30px; width:auto !important; padding:0 12px !important; border-radius:9px !important; font-size:12px !important; }
#run-controls button { border-radius:9px !important; font-size:12px !important; }
#recognize { background:#28563e !important; color:#fffef8 !important; border:1px solid #28563e !important; border-radius:10px !important; font-size:14px !important; font-weight:600 !important; min-height:46px; box-shadow:0 3px 8px #28563e20; transition:background .15s,transform .15s; }
#recognize:hover { background:#183e2a !important; transform:translateY(-1px); }
.panel-note { color:#81907e; font-size:11px; line-height:1.7; padding-top:3px; }
#details { margin-top:10px; background:#fffefa !important; border:1px solid #dce2d8 !important; border-radius:12px !important; }
#details button, #details button span { color:#53654d !important; }
.app-foot { display:flex; justify-content:space-between; gap:16px; color:#8a9485; font-size:10px; letter-spacing:.7px; padding:14px 2px 4px; }
footer { display:none !important; }

/* ── Export bar ── */
#export-bar {
    display:flex; align-items:center; gap:8px;
    background:#f3f5ee; border:1px solid #dce2d8; border-radius:11px;
    padding:8px 12px; margin-top:4px;
}
#export-bar > * { margin:0 !important; }
#export-bar label { display:none !important; }        /* hide Gradio label */
#export-fmt { flex:1 1 auto; min-width:0 !important; }
#export-fmt select { font-size:12px !important; height:32px !important; border-radius:8px !important; padding:0 8px !important; }
#export-btn {
    flex:none !important; min-width:0 !important; width:auto !important;
    height:32px !important; min-height:0 !important;
    padding:0 14px !important; font-size:12px !important;
    border-radius:8px !important; font-weight:600 !important;
    background:#28563e !important; color:#fffef8 !important;
    border:1px solid #28563e !important;
    transition:background .15s !important;
}
#export-btn:hover { background:#183e2a !important; }
/* Download link that Gradio renders after export */
#export-file { flex:none; }
#export-file a { font-size:11px !important; color:#346b50 !important; font-weight:600; text-decoration:none; }
#export-file a:hover { text-decoration:underline; }

/* ── Loading skeleton ── */
#ocr-loading:not(:has(.ocr-skeleton)) { display:none !important; }
#workspace:has(.ocr-skeleton) #transcript { display:none !important; }
.ocr-skeleton { height:300px; box-sizing:border-box; padding:22px 18px; display:flex; flex-direction:column; gap:18px; background:#faf8f0; border:1px solid #e8e3d4; border-radius:12px; overflow:hidden; animation:sk-enter .3s ease-out both; }
.sk-lines { display:flex; flex-direction:column; gap:11px; }
.sk-line { position:relative; display:block; height:10px; flex:none; border-radius:5px; background:#e8e6d9; overflow:hidden; }
.sk-line::after { content:''; position:absolute; inset:0; background:linear-gradient(100deg,transparent 15%,#fffdf4b3 48%,transparent 80%); transform:translateX(-110%); animation:sk-shimmer 2.8s cubic-bezier(.4,0,.2,1) infinite; animation-delay:var(--sk-delay,0s); }
.sk-caption { margin-top:auto; display:flex; align-items:center; gap:9px; color:#71816b; font-size:11px; letter-spacing:.25px; line-height:18px; }
.sk-dots { display:inline-flex; gap:4px; align-items:center; }
.sk-dots i { width:4px; height:4px; border-radius:50%; background:#779369; animation:sk-breathe 1.8s ease-in-out infinite; }
.sk-dots i:nth-child(2) { animation-delay:.18s; }
.sk-dots i:nth-child(3) { animation-delay:.36s; }
.live-scan { position:relative; overflow:hidden; }
#workspace:has(.ocr-skeleton) .live-scan::after { content:''; position:absolute; inset:0; pointer-events:none; z-index:5;
    background:linear-gradient(180deg,transparent 28%,#7fc98a08 40%,#7fc98a26 49.5%,#59815299 50%,transparent 50.5%); animation:sk-sweep 3.6s cubic-bezier(.4,0,.2,1) infinite; }
@keyframes sk-enter { from { opacity:0; } to { opacity:1; } }
@keyframes sk-shimmer { 0%,12% { transform:translateX(-110%); } 72%,100% { transform:translateX(110%); } }
@keyframes sk-breathe { 0%,70%,100% { opacity:.4; transform:translateY(0); } 35% { opacity:1; transform:translateY(-2px); } }
@keyframes sk-sweep { 0% { transform:translateY(-65%); opacity:0; } 12%,80% { opacity:1; } 100% { transform:translateY(65%); opacity:0; } }
@media(prefers-reduced-motion:reduce) {
    .ocr-skeleton, .sk-line::after, .sk-dots i, #workspace:has(.ocr-skeleton) .live-scan::after { animation:none !important; }
    .sk-line::after, #workspace:has(.ocr-skeleton) .live-scan::after { display:none !important; }
}

/* ── Responsive ── */
.gradio-container.gradio-container { width:100% !important; min-width:0 !important; box-sizing:border-box; padding:clamp(12px,3vw,32px) !important; }
.gradio-container .main { width:100% !important; min-width:0 !important; padding:0 !important; box-sizing:border-box; }

/* Tablet 601-960px */
@media(min-width:601px) and (max-width:960px) {
    #workspace { flex-wrap:wrap; gap:16px; }
    #workspace > div:nth-child(1),
    #workspace > div:nth-child(3) { flex:1 1 calc(50% - 10px) !important; min-width:240px !important; box-sizing:border-box; }
    #scanner-col { flex:1 1 100% !important; min-width:0 !important; order:99; }
    #scanner-col .image-stage { height:240px !important; min-height:200px !important; }
    .image-stage { height:300px !important; min-height:260px !important; }
    #transcript { height:250px !important; }
    #workspace .stage-title { height:auto; min-height:64px; }
    #export-bar { flex-wrap:wrap; }
}

/* Mobile <=600px */
@media(max-width:600px) {
    #workspace { flex-direction:column !important; gap:14px; }
    #workspace .stage-card { min-width:0 !important; width:100% !important; box-sizing:border-box; padding:16px !important; }
    #scanner-col { order:99; }
    .brandbar .brand-note { display:none; }
    .brandbar .leaf-seal { width:60px; height:60px; font-size:25px; }
    .brandbar { gap:12px; }
    #workspace .stage-title { height:auto; min-height:64px; }
    #workspace .image-stage { height:260px !important; min-height:220px !important; }
    #transcript { height:220px !important; }
    .app-foot { flex-wrap:wrap; gap:6px; }
    #scanner-toolbar { justify-content:flex-start; }
    #export-bar { flex-wrap:wrap; }
    #export-fmt { min-width:100% !important; }
}
"""


def _toggle_scanner(is_visible):
    """Toggle the Live Scanner column visibility."""
    new_visible = not is_visible
    label = "\u25be Hide Live Scanner" if new_visible else "\u25b4 Show Live Scanner"
    return new_visible, gr.update(visible=new_visible), gr.update(value=label)


with gr.Blocks(title="Palm Leaf OCR") as demo:
    session_state   = gr.State(new_session())
    scanner_visible = gr.State(False)   # scanner starts collapsed

    gr.HTML(
        '<div class="brandbar"><div class="brand"><span class="brand-mark">\u2381</span>LeafOCR</div>'
        '<div class="brand-right"><div class="brand-note">Palm-leaf manuscript studio</div>'
        '<div class="leaf-seal" aria-label="Leaf to letter emblem">\u0d05<small>LEAF TO LETTER</small></div></div></div>'
    )

    status_output = gr.Markdown("Ready \u00b7 upload an image to begin", elem_classes="status-bar")

    # ── Scanner toggle toolbar ──────────────────────────────────────────────
    with gr.Row(elem_id="scanner-toolbar"):
        toggle_btn = gr.Button(
            "\u25b4 Show Live Scanner",
            size="sm",
            variant="secondary",
            elem_id="toggle-scanner",
            scale=0,
        )

    # ── Three-column workspace ─────────────────────────────────────────────
    with gr.Row(equal_height=True, elem_id="workspace"):

        # Column 1 — Your manuscript
        with gr.Column(scale=2, min_width=280, elem_classes="stage-card", elem_id="ms-upload"):
            gr.HTML(
                '<div class="stage-heading"><span class="step">01</span>'
                '<h2>Your manuscript</h2></div>'
                '<p>Start with a leaf image or a cropped text segment.</p>',
                elem_classes="stage-title",
            )
            input_image = gr.Image(
                label="Source image", type="numpy", height=360,
                scale=0, elem_classes="image-stage",
                buttons=["fullscreen"], sources=["upload", "clipboard"],
            )
            run_btn = gr.Button(
                "Recognize manuscript  \u2192",
                variant="primary", size="lg", elem_id="recognize",
            )
            with gr.Row(elem_id="run-controls"):
                stop_btn   = gr.Button("\u25a0 Stop",   size="sm", min_width=60)
                resume_btn = gr.Button("\u25b6 Resume", size="sm", min_width=60, visible=False)
                reset_btn  = gr.Button("\u21ba Reset",  size="sm", min_width=60)

        # Column 2 — Live scanner (starts hidden)
        with gr.Column(
            scale=2, min_width=260,
            elem_classes="stage-card", elem_id="scanner-col",
            visible=False,
        ) as scanner_col:
            gr.HTML(
                '<div class="stage-heading"><span class="step">02</span>'
                '<h2>Live scanner</h2></div>'
                '<p>Follow each text region as it is recognized.</p>',
                elem_classes="stage-title",
            )
            annotated_output = gr.Image(
                label="Live scan", height=360, interactive=False,
                scale=0, elem_classes=["image-stage", "live-scan"],
                buttons=["fullscreen", "download"],
            )
            gr.HTML(
                '<div class="panel-note">'
                '<b style="color:#2f8a3c">Green</b> marks the text being read right now; '
                '<b style="color:#d9822b">orange</b> marks text you picked in the transcript.<br>'
                'Open full screen to inspect the finer details.</div>'
            )

        # Column 3 — The transcript
        with gr.Column(scale=2, min_width=280, elem_classes="stage-card"):
            gr.HTML(
                '<div class="stage-heading"><span class="step">03</span>'
                '<h2>The transcript</h2></div>'
                '<p>Click any word to see where it came from.</p>',
                elem_classes="stage-title",
            )
            skeleton_output = gr.HTML("", elem_id="ocr-loading")
            output_text = gr.HighlightedText(
                label="Recognized text",
                color_map={"chunk": "#f1efe3", "current": "#7fc98a", "selected": "#f5a54a"},
                show_legend=False,
                show_inline_category=False,
                interactive=False,
                elem_id="transcript",
            )

            # ── Transcript action row: Copy + Export ──────────────────────
            with gr.Row(elem_classes="transcript-actions"):
                copy_btn = gr.Button(
                    "\u29c9 Copy text", size="sm", variant="secondary",
                    scale=0, elem_id="copy-text",
                )

            # ── Export bar ────────────────────────────────────────────────
            with gr.Row(elem_id="export-bar"):
                export_fmt = gr.Dropdown(
                    choices=list(_EXPORT_FNS.keys()),
                    value="Plain text (.txt)",
                    label="Format",
                    container=False,
                    scale=3,
                    elem_id="export-fmt",
                )
                export_btn = gr.Button(
                    "\u2913 Export",
                    size="sm",
                    scale=1,
                    elem_id="export-btn",
                )
                export_file = gr.File(
                    label="",
                    visible=False,
                    elem_id="export-file",
                    scale=2,
                )

            region_output = gr.Image(
                label="Source region", height=110, interactive=False,
                elem_id="region", buttons=["fullscreen"],
            )

    with gr.Accordion(
        "Behind the scan \u00b7 lines & text regions",
        open=False, elem_id="details",
    ):
        plain_output  = gr.Textbox(label="Plain text", lines=4, buttons=["copy"], interactive=False)
        mask_output   = gr.Image(label="Segmentation analysis")
        line_output   = gr.Gallery(label="Detected lines", columns=1, object_fit="contain")
        chunk_output  = gr.Gallery(
            label="Recognized chunks (click one to locate it)",
            columns=4, object_fit="contain",
        )

    gr.HTML(
        '<div class="app-foot">'
        '<span>LeafOCR \u00b7 Manuscript recognition</span>'
        '<span>From fragile leaves to lasting words.</span>'
        '</div>'
    )

    # ── Output list (must match OUT_KEYS order) ────────────────────────────
    all_outputs = [
        mask_output, annotated_output, line_output, chunk_output, output_text,
        plain_output, status_output, region_output, session_state, stop_btn, resume_btn,
        skeleton_output,
    ]

    # ── Pipeline events (unchanged) ────────────────────────────────────────
    run_evt    = run_btn.click(fn=run_pipeline,    inputs=[input_image, session_state], outputs=all_outputs)
    resume_evt = resume_btn.click(fn=resume_pipeline, inputs=[input_image, session_state], outputs=all_outputs)
    running    = [run_evt, resume_evt]

    stop_btn.click(fn=stop_pipeline, inputs=[session_state], outputs=all_outputs,
                   cancels=running, concurrency_limit=None)
    reset_btn.click(fn=reset_session, inputs=[session_state], outputs=all_outputs,
                    cancels=running, concurrency_limit=None)
    input_image.upload(fn=reset_session, inputs=[session_state], outputs=all_outputs,
                       cancels=running, concurrency_limit=None)
    input_image.clear(fn=reset_session, inputs=[session_state], outputs=all_outputs,
                      cancels=running, concurrency_limit=None)

    output_text.select(fn=select_from_transcript, inputs=[session_state], outputs=all_outputs,
                       concurrency_limit=None)
    chunk_output.select(fn=select_from_gallery, inputs=[session_state], outputs=all_outputs,
                        concurrency_limit=None)
    copy_btn.click(fn=None, inputs=[plain_output],
                   js="(t) => { navigator.clipboard.writeText(t || ''); }")

    # ── Export event ───────────────────────────────────────────────────────
    def _do_export_ui(fmt, session):
        path = do_export(fmt, session)
        if path:
            return gr.update(value=path, visible=True)
        return gr.update(value=None, visible=False)

    export_btn.click(
        fn=_do_export_ui,
        inputs=[export_fmt, session_state],
        outputs=[export_file],
        concurrency_limit=None,
    )

    # ── Scanner toggle ─────────────────────────────────────────────────────
    toggle_btn.click(
        fn=_toggle_scanner,
        inputs=[scanner_visible],
        outputs=[scanner_visible, scanner_col, toggle_btn],
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        theme=gr.themes.Soft(primary_hue="green", neutral_hue="stone"),
        css=APP_CSS,
        share=True,
    )
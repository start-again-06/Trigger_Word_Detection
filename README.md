# Trigger Word Detection with Chime Overlay

This project detects a spoken trigger word (like "activate") in 10-second audio clips using a deep learning model, and overlays a chime sound wherever the trigger word is detected.

---

## 💡 Highlights

- Preprocessing includes:
  - Creating training examples by inserting activate and negative words into random background noise.
  - Converting audio to spectrograms.
  - Labeling time steps where the trigger word occurs.

- Model architecture:
  - 1D Convolution → BatchNorm → ReLU → GRUs → TimeDistributed Dense with sigmoid.
  - Trained to detect the presence of the trigger word in short time windows.

- Inference pipeline:
  - Predict trigger word activations in new audio files.
  - Overlay a chime sound at locations with high activation.

---

## 🛠️ Main Components

- **Data Augmentation**:
  - `insert_audio_clip()`: Inserts audio snippets into background at non-overlapping positions.
  - `insert_ones()`: Labels the output with `1`s for 50 time steps after each activation.

- **Model Definition**:
  - CNN + GRU-based sequence model with `TimeDistributed(Dense)` output.
  - Loss: `binary_crossentropy`, Optimizer: Adam.

- **Training**:
  - Input: Spectrogram of shape `(5511, 101)`
  - Output: `(1375, 1)` binary activations.

- **Evaluation**:
  - Evaluate model on dev set.
  - Detect and visualize activations.
  - Overlay `chime.wav` if 20+ consecutive positive predictions occur.

---

## 📦 Output Example

- Input: `"audio_examples/my_audio.wav"`  
- Output: `"chime_output.wav"` with chimes at detected trigger positions.

---

## ▶️ Usage

1. Preprocess your audio:
   - Pads or trims to 10 seconds.
   - Converts to spectrograms.

2. Run inference:
   - `detect_triggerword(filename)` → generates probability curve.
   - `chime_on_activate(filename, prediction, threshold)` → overlays chime.

---

## 📂 Files Required

- `td_utils.py`: Utility functions for audio loading, spectrogram, amplitude matching, etc.
- `models/model.h5`: Pre-trained weights.
- `models/model.json`: Model architecture.

---

## 🎧 Demos

- Example predictions:  
  - `raw_data/dev/1.wav`  
  - `raw_data/dev/2.wav`  
  - `audio_examples/my_audio.wav` (custom recording)

---

## 🧪 Test Functions

- `is_overlapping_test()`: Verifies correct non-overlapping logic.
- `insert_audio_clip_test()`: Ensures audio is inserted correctly.
- `insert_ones_test()`: Validates labeling logic.
- `create_training_example_test()`: Tests full example generation pipeline.
- `modelf_test()`: Validates model structure.

---

## 📈 Dev Accuracy

- Evaluated using `model.evaluate(X_dev, Y_dev)`
- Typical accuracy: ~0.98+ depending on model capacity and data size.

---

## 🚀 Next Steps

- Add more training data to improve generalization.
- Use real-world noisy audio for robustness.
- Experiment with Transformer-based audio models.

---

# Signvoice AI

Real-time **Indian Sign Language (ISL) to text and speech** — project demo by Team 9, Dattakala Group of Institutions (2026–27).

> **Prototype status:** The web demo's camera preview and sample-phrase speech work. The trained ISL recognition model is **not yet connected**; the translations shown use sample phrases and are not validated ISL translations. The new Python **input module** (below) captures signing video, extracts and normalises landmarks and builds the pilot dataset that the model will be trained on.

## Components

| Part | What it does | Where |
|---|---|---|
| Web demo | Camera preview, sample phrases, text-to-speech | `index.html`, `script.js`, `styles.css` |
| **Input module** (Python) | Camera/video → MediaPipe Holistic → signer-adaptive features → ISL dataset | `signvoice/`, `scripts/` |

## Input module

```
camera / video file
      │  un-mirrored frames
      ▼
MediaPipe Holistic ── hands (2×21) + upper-body pose (13) + face subset (132)   signvoice/landmarks.py
      │  raw landmarks, NaN = not detected
      ▼
signer-adaptive normalisation                                                   signvoice/normalize.py
      │  691-d feature vector per frame
      ▼
pilot dataset: ~100 signs + 40 sentences × several signers                      signvoice/dataset.py
```

**Normalisation (what makes it signer-adaptive)**

1. *Body frame:* origin = mid-shoulder, scale = shoulder width, rotation = shoulder-line roll → removes body size, camera distance, framing and camera tilt.
2. *Local frames:* hand shape relative to its own wrist and hand size; face relative to nose and face width (head tilt is kept, it carries meaning).
3. *Handedness:* left-dominant signers are mirrored so the dominant hand is always the "right" stream.
4. *Per-signer statistics:* `SignerStats` z-scores features per signer (also usable online for a new user).
5. Short detection dropouts (≤5 frames) are interpolated; longer ones stay zero with a presence flag in the feature vector.

### Setup

Python 3.9–3.12 (MediaPipe 0.10.14 has no 3.13 wheels).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python -m pytest tests -q       # 13 tests, no camera needed
```

### Typical workflow

```bash
python scripts/live_preview.py                                   # check landmarks + live features
python scripts/record_dataset.py register --signer S01 --dominant right --experience native --consent-confirmed
python scripts/record_dataset.py record --signer S01 --session 2026-11-20-lab
python scripts/record_dataset.py status                          # progress / what is missing
python scripts/dataset_report.py                                 # health + detection-rate report
python scripts/build_features.py                                 # → dataset/processed/*.npz
```

Already have recordings? `python scripts/extract_from_videos.py --videos videos_in` (see script header for layout).
Full recording protocol, consent and camera setup: [`docs/DATA_COLLECTION.md`](docs/DATA_COLLECTION.md).

### Using it from your own code

```python
from signvoice.capture import FeatureStream
with FeatureStream(0, dominant="right") as stream:       # webcam 0
    for item in stream:                                  # one item per frame
        if item.features is not None:
            window = stream.window(64)                   # (64, 691) → feed an LSTM/GRU
```

```python
import numpy as np
d = np.load("dataset/processed/signs.npz")
X, y, split = d["X"], d["y"], d["split"]                 # (N, 64, 691), label index (1..100), train/val/test by signer
sent = np.load("dataset/processed/sentences.npz")        # X padded + lengths, y_seq/y_len = CTC-ready gloss targets
```

Feature layout of the 691 dims (`FeatureSpec.slices()`): `pose 0–39 · hands_body 39–165 · hands_shape 165–291 · face 291–687 · masks 687–691`. Drop groups with `FeatureSpec(face=False)` for a smaller, faster model.

### Vocabulary

`data/vocab/signs.csv` (100 signs) and `data/vocab/sentences.csv` (40 sentences built from those signs).
**These are a draft for the demo, not a validated ISL lexicon** — have fluent ISL signers verify every item against the ISLRTC dictionary before recording (see the protocol doc). `python scripts/check_vocab_against_include.py` cross-checks them against the published INCLUDE lexicon.

## Run the web demo locally

Requires [Node.js](https://nodejs.org) (no dependencies to install).

```bash
npm start
```

Then open http://localhost:4173. Use another port with `PORT=8080 npm start`.

> Camera access needs `localhost` or an **HTTPS** address (important on mobile Chrome).

## Project structure

```
index.html, styles.css, script.js, demo-server.cjs   Web demo
signvoice/
  config.py          landmark subsets, feature layout (single source of truth)
  landmarks.py       MediaPipe Holistic wrapper, RawSequence container
  normalize.py       signer-adaptive normalisation, streaming mode, SignerStats
  capture.py         camera/video source, overlay, live FeatureStream, quality checks
  recorder.py        interactive guided recorder
  dataset.py         dataset store, manifest, signer registry, signer-independent splits, export
  vocabulary.py      loads/validates data/vocab/*.csv
scripts/             record_dataset, extract_from_videos, build_features, dataset_report, live_preview
data/vocab/          100 signs + 40 sentences (draft)
tests/               synthetic-data tests (invariance, handedness, gaps, dataset, recorder flow)
docs/                data-collection protocol
examples/            Python practice snippets
```

## Pipeline & roadmap

1. ✅ Capture continuous signing video (webcam or files).
2. ✅ MediaPipe Holistic landmarks (hands, pose, face) → signer-normalised features.
3. ✅ Pilot dataset tooling: 100 signs + 40 sentences, several signers, signer-independent splits.
4. ⏳ Record the pilot dataset with real signers; verify the vocabulary with ISL experts.
5. ⏳ Train an LSTM/GRU (isolated signs), then CTC for continuous sentences; segment signs and combine into sentences with text-to-speech.
6. ⏳ Lightweight model in a Flutter Android app with TensorFlow Lite.

## Deploy the web demo on GitHub Pages (HTTPS, works on phones)

Repo **Settings → Pages → Deploy from branch → `main` / root**. Your site will be at `https://<username>.github.io/<repo>/`.

## License

MIT — see [LICENSE](LICENSE).

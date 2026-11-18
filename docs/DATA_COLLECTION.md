# Pilot ISL dataset - collection protocol

Target: **100 isolated signs + 40 sentences, several signers** (aim for 5-8).
Vocabulary lives in `data/vocab/signs.csv` and `data/vocab/sentences.csv`.

> **The vocabulary is a draft.** The glosses are common everyday words chosen for this demo
> (greetings, questions, health, travel, numbers...). They are **not** a validated ISL lexicon and
> the gloss order of the sentences is a plan, not authoritative ISL grammar. Before recording, have a
> fluent / native ISL signer (ideally a Deaf signer or an ISL teacher) check every sign and sentence
> against the ISLRTC dictionary, and edit the CSVs (keep ids stable once recording has started).

## Published ISL resources (to verify and extend the vocabulary)

* **INCLUDE** (AI4Bharat / IIT Madras): 263 isolated ISL word signs, 15 categories, 4,287+ videos recorded with
  experienced signers, CC-BY-4.0. Labels: Hugging Face `ai4bharat/INCLUDE`; videos: Zenodo record 4010759;
  code: github.com/AI4Bharat/INCLUDE. Videos were shot in Chennai - ISL varies by region, so check
  signs with signers from *your* region. It has no verbs, question words or sentences.
  `python scripts/check_vocab_against_include.py --write` marks which of your 100 signs exist in INCLUDE
  so your expert can compare against its reference videos.
* **ISLRTC** (islrtc.nic.in): the government ISL research and training centre - the authority for the official dictionary.
* **iSign** (exploration-lab.github.io/iSign): 118k+ ISL video-sentence pairs, useful for sentence-level work.
* Also published: ISL-CSLRT (continuous sentences) and CISLR (4,765-word corpus; access needs agreement).

None of these replace your own consent-based recordings of your own signers; they help you verify the
vocabulary and could later pre-train a model.

## Before you record

1. **Consent.** Every signer reads and signs a consent form covering: what is recorded (landmarks, and
   video only if they agree to `--save-video`), how it is stored, who can access it, and that they can
   withdraw and have their clips deleted. Face landmarks and video identify a person. Register them
   only afterwards:
   `python scripts/record_dataset.py register --signer S01 --dominant right --experience native --hearing deaf --consent-confirmed`
   Use anonymous ids (S01, S02...), never names. The recorder refuses signers without recorded consent.
2. **Signer mix.** Different ages, body sizes, and at least one left-dominant signer if possible -
   the normalisation is designed to cope with this, but only if the data contains it.
3. **Camera.** Fixed camera at upper-chest height, 1.5-2 m away, landscape 1280x720 @ 30 fps. Frame
   the head, both shoulders and the full signing space (hands raised above the waist and at both sides).
   Plain background, even light from the front, no strong backlight (windows behind the signer).
   Plain, non-skin-coloured, long-sleeved clothing helps hand detection.
4. **Rest pose.** Hands down at the sides before and after every clip. The recorder trims the idle
   frames automatically (non-destructively: full landmarks are kept, the trim span is stored).

## Recording

```bash
python scripts/live_preview.py                       # 1. check the camera sees head, shoulders, both hands
python scripts/record_dataset.py record --signer S01 --session 2026-11-20-lab
python scripts/record_dataset.py record --signer S01 --types sentence      # or only sentences
python scripts/record_dataset.py status
```

Default: **5 repetitions per sign, 3 per sentence** = 620 clips per signer, roughly 1.5 hours
(estimate: ~8 s per clip including countdown and review). Do it in 2-3 sessions with breaks; the
recorder is resumable and saves after every clip. Repetitions are ordered in rounds (one pass through
all signs, then the next pass) so fatigue is not tied to particular signs.

* **Signs** are recorded one at a time, starting and ending in the rest pose.
* **Sentences** are signed **continuously** at natural speed, without pausing between signs.
* If a take is wrong press `R`; if you noticed a mistake after accepting press `U` to undo it.
* Quality warnings (face/hands not detected, low frame-rate) are shown on screen - retake those.

Already have phone videos? Put them in `videos_in/<signer>/<sign|sentence>/<label_id>/take1.mp4` and run
`python scripts/extract_from_videos.py --videos videos_in`.

## Evaluating honestly

`scripts/build_features.py` assigns **whole signers** to train / val / test. Report accuracy only on
held-out signers; clip-level random splits leak the signer's style into the test set and inflate results.
With 5 signers x 5 repetitions you will have ~25 examples per sign - enough for a pilot baseline, not for
a production model.

## What gets stored

```
dataset/
  manifest.csv        one row per clip (label, signer, frames, fps, detection rates, status ...)
  signers.csv         anonymous id, dominant hand, experience, consent record
  clips/<signer>/<sign|sentence>/<label_id>/<clip_id>.npz   raw landmarks (NaN = not detected)
  videos/...          only with --save-video
  processed/          signs.npz, sentences.npz, splits.json, signer_stats/  (build_features.py)
```
`dataset/` is git-ignored. Keep it out of public repositories: it contains biometric data.

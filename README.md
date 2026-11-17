# Signvoice AI

Real-time **Indian Sign Language (ISL) to text and speech** — project demo by Team 9, Dattakala Group of Institutions (2026–27).

> **Prototype status:** The camera preview and sample-phrase speech work. The trained ISL recognition model is **not yet connected**; the translations shown use sample phrases and are not validated ISL translations.

## Features

- Live camera preview (stays on your device, nothing is uploaded)
- Sample sign phrases converted into a sentence
- Text-to-speech output (browser Speech Synthesis, `en-IN`)
- Copy translated text
- Phrasebook and "How it works" pages

## Run locally

Requires [Node.js](https://nodejs.org) (no dependencies to install).

```bash
npm start
```

Then open http://localhost:4173. Use another port with `PORT=8080 npm start`.

> Camera access needs `localhost` or an **HTTPS** address (important on mobile Chrome).

## Project structure

```
index.html          App markup
styles.css          Styling
script.js           UI logic, camera, speech
demo-server.cjs     Tiny static file server for local demo
examples/           Python practice snippets
```

## Proposed pipeline

1. Capture gesture frames from a smartphone camera.
2. MediaPipe extracts 21 hand landmarks (X, Y, Z → 63 features per hand).
3. An LSTM or GRU learns motion across frame sequences.
4. Segment signs, combine words into sentences, and use text-to-speech.
5. Deploy a lightweight model in a Flutter Android app with TensorFlow Lite.

## Roadmap

Demo → collect & label ISL sequences → train & evaluate model → TensorFlow Lite → Android app.

## Deploy on GitHub Pages (HTTPS, works on phones)

Repo **Settings → Pages → Deploy from branch → `main` / root**. Your site will be at `https://<username>.github.io/<repo>/`.

## License

MIT — see [LICENSE](LICENSE).

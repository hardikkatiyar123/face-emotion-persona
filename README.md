# Mood Mirror: real-time emotion detection

Detects facial expressions (happy, sad, angry, surprised, scared, disgusted, neutral) from a webcam. The model runs **in the browser**, so the site is plain static files that deploy to Vercel for free, and no video ever leaves the user's computer.

## How it works

```
Webcam (getUserMedia) → face detection (Tiny Face Detector)
      → crop face → emotion model → 7 emotion probabilities → smoothing → draw on screen
```

There are two parts:

| Folder      | What it is                                   | Runs where           |
|-------------|----------------------------------------------|----------------------|
| `training/` | Python: trains a CNN on FER-2013, exports it to TensorFlow.js | Your computer        |
| `web/`      | The website (HTML/CSS/JS + your exported model) | The visitor's browser, hosted on Vercel |

The website works even before you train anything. The "Model" menu has three emotion models:

| Model | What it is | Download | Speed (one face) |
|---|---|---|---|
| **HSEmotion** (default when fast enough) | EfficientNet-B0 trained on AffectNet, run with onnxruntime-web | 16 MB model + ~5 MB runtime | ~20–25 ms with WebGPU, ~85–110 ms on CPU only |
| **Built-in** | face-api.js' small expression model | ~0.3 MB | a few ms |
| **My trained model** | your CNN from `training/train.py` (48×48 grayscale, FER-2013) | ~7 MB | a few ms |

HSEmotion loads in the background and runs a quick speed test. If one face takes under 50 ms on that device, it becomes the default; otherwise Built-in stays the default and HSEmotion is marked "slow here". The page shows "Uncertain 🤔" instead of a label when the top emotion is below 40% or within 12 points of the runner-up. HSEmotion is © its authors, Apache-2.0; see [web/models/hsemotion/README.md](web/models/hsemotion/README.md).

**Which is most accurate?** HSEmotion, most likely. Its published accuracy is 64.6% on AffectNet (7 classes), a large dataset of real-world photos, much closer to webcam images than FER-2013's tiny 48×48 grayscale crops. face-api.js publishes no accuracy figures for its expression model. Your own model scores ~60–68% on FER-2013, but that's a different, easier-to-overfit test set, so the numbers can't be compared directly.

## Project structure

```
emotion-detection/
├── README.md
├── vercel.json           # tells Vercel to publish the web/ folder
├── .gitignore            # keeps the dataset, venv and *.h5 out of Git
├── training/
│   ├── requirements.txt
│   ├── train.py          # train + evaluate + export to web/model/
│   └── local_demo.py     # test your trained model with OpenCV
└── web/                  # ← this folder is what Vercel deploys
    ├── index.html
    ├── style.css
    ├── app.js
    ├── model/            # train.py writes model.json, *.bin, labels.json, manifest.json
    └── models/hsemotion/ # HSEmotion ONNX model + its Apache-2.0 LICENSE
```

## Step 1: run the website locally

The camera only works on `https://` or `localhost`, so don't just double-click `index.html`. From the `web` folder:

```bash
cd web
python -m http.server 8000
```

Open http://localhost:8000, press **Start camera**, and allow camera access.

## Step 2: train your own model (optional, but this is the "real" project)

1. Download FER-2013 from Kaggle: https://www.kaggle.com/datasets/msambare/fer2013
2. Unzip it into `training/data/` so you have `training/data/train/angry/...` and `training/data/test/angry/...`.
3. Create an environment with **Python 3.10 or 3.11** and install the pinned versions:

   ```bash
   cd training
   python -m venv venv
   venv\Scripts\activate          # Windows
   source venv/bin/activate       # macOS / Linux
   pip install -r requirements.txt
   ```

4. Train:

   ```bash
   python train.py --data data --epochs 50
   ```

   On a CPU this takes a while (roughly 1–3 hours). With a GPU it's much faster. Expect about **60–68% test accuracy**; FER-2013 is a hard, noisy dataset, and human accuracy on it is around 65%.

   15% of `train/` is held out as a validation set, which decides when to stop and which epoch to keep. The `test/` set is only used once at the end, so the test accuracy it prints is a fair score. Change the split with `--val-split 0.1`.

5. The script saves `emotion_model.h5`, prints a per-class report, and exports the browser model to `web/model/`.
6. Try it locally with OpenCV: `python local_demo.py`.
7. Reload the website. "My trained model" is now available in the Model menu (HSEmotion stays the default when it runs fast enough).

## Step 3: deploy to Vercel

1. Push the project to a GitHub repository. The `.gitignore` already leaves out the dataset, your virtual environment and `*.h5` files. Do commit `web/model/` if you trained a model.
2. Go to https://vercel.com, choose **Add New → Project**, and import the repository.
3. Leave all the settings at their defaults (Framework Preset: Other, Root Directory: `./`). The `vercel.json` in the repository tells Vercel to publish the `web/` folder; no build step is needed.
4. Click **Deploy**. Vercel gives you an `https://…vercel.app` link, and HTTPS is what allows the camera to work.

Or, with the Vercel CLI: `cd web` then `npx vercel`. Run it from `web/` so the CLI uploads only the website, not the dataset in `training/`.

## Troubleshooting

- **Camera doesn't start:** check that the site is on `https://` or `localhost`, that you allowed camera access, and that no other app (Zoom, Teams) is using the camera.
- **Export fails in train.py:** this is almost always a version problem. Use Python 3.10/3.11 and exactly the versions in `requirements.txt` (TensorFlow 2.16+ uses Keras 3, which the TensorFlow.js converter doesn't handle well). Your `emotion_model.h5` is saved before the export, so you can fix the versions and convert without retraining:
  `tensorflowjs_converter --input_format keras emotion_model.h5 ../web/model`, then copy `labels.json` into `web/model/` and change `web/model/manifest.json` to `{"customModel": true}` so the website loads it.
- **Custom model predicts the wrong emotions:** make sure `web/model/labels.json` came from the same training run as `model.json`.
- **Slow or laggy:** use a laptop with a decent GPU/browser (Chrome or Edge), and close other heavy tabs. HSEmotion is fast only with WebGPU (Chrome/Edge on most laptops); without it, pick Built-in. The FPS pill on the video shows the frame rate and the face detector size, which drops automatically when the frame rate falls below 15.
- **HSEmotion stays on "loading…":** it's a 16 MB model plus a ~5 MB runtime, so the first visit takes a moment on slow connections; after that the browser caches it.

## Ideas to extend it

Emotion timeline chart over a session, screenshot button, transfer learning with MobileNet for better accuracy, or a "make this face" game.

# HSEmotion model

`enet_b0_8_best_vgaf.onnx` is an EfficientNet-B0 facial emotion model trained on AffectNet by
Andrey Savchenko, from the HSEmotion / EmotiEffLib project:
https://github.com/sb-ai-lab/EmotiEffLib (formerly HSE-asavchenko/face-emotion-recognition).

- License: Apache License 2.0, see `LICENSE` in this folder (copied unchanged from the project).
  The authors state: "There is no limitation for both academic and commercial usage."
  Note that the training data, AffectNet, has its own non-commercial research terms; if you plan
  commercial use, check whether that matters for you.
- Source file: `models/affectnet_emotions/onnx/enet_b0_8_best_vgaf.onnx` (16,039,595 bytes), not modified.
- Input: `input`, float32 `[1, 3, 224, 224]`, RGB face crop scaled to 0–1 and normalised with
  ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.
- Output: `output`, 8 raw scores (logits) in this order:
  Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise.
  The app applies softmax over the 7 emotions it shows and ignores Contempt.
- Published accuracy: 61.3% on AffectNet (8 classes), 64.6% (7 classes).

If you use the model in research, the authors ask you to cite:

- A. Savchenko, "Facial Expression Recognition with Adaptive Frame Rate based on Multiple Testing
  Correction," ICML 2023, PMLR 202, pp. 30119–30129. https://proceedings.mlr.press/v202/savchenko23a.html
- A. V. Savchenko, L. V. Savchenko, I. Makarov, "Classifying emotions and engagement in online
  learning based on a single facial expression recognition neural network," IEEE Transactions on
  Affective Computing, 2022. https://ieeexplore.ieee.org/document/9815154

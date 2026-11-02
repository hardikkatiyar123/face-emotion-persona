`train.py` writes your trained model here (model.json, *.bin, labels.json) and sets
`"customModel": true` in manifest.json.
The website only loads model.json when manifest.json says so; otherwise "My trained model"
stays unavailable and the other models are used.

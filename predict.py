
import argparse
import os

import keras
import numpy as np
from PIL import Image, ImageOps

ap = argparse.ArgumentParser()
ap.add_argument("image")
ap.add_argument("--dataset", choices=["mnist", "emnist"], default="mnist")
ap.add_argument("--model_dir", default="outputs")
args = ap.parse_args()

model_path = os.path.join(args.model_dir, f"{args.dataset}_cnn.keras")
if not os.path.exists(model_path):
    raise SystemExit(f"Model not found: {model_path}\n"
                     f"Train it first: python handwritten_recognition.py --dataset {args.dataset}")
model = keras.models.load_model(model_path)
classes = np.load(os.path.join(args.model_dir, f"{args.dataset}_classes.npy"))

img = ImageOps.autocontrast(Image.open(args.image).convert("L"))
arr = np.asarray(img, dtype="float32")
if arr.mean() > 127:                      # dark ink on light paper -> invert to MNIST style
    arr = 255 - arr
mask = arr > 0.3 * arr.max()
ys, xs = np.where(mask)
arr = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]   # crop to the character

h, w = arr.shape                           # scale longest side to 20 px, centre in 28x28
scale = 20.0 / max(h, w)
arr = np.asarray(Image.fromarray(arr.astype("uint8")).resize(
    (max(1, int(w * scale)), max(1, int(h * scale)))))
canvas = np.zeros((28, 28), dtype="float32")
top, left = (28 - arr.shape[0]) // 2, (28 - arr.shape[1]) // 2
canvas[top:top + arr.shape[0], left:left + arr.shape[1]] = arr

probs = model.predict((canvas / 255.0)[None, ..., None], verbose=0)[0]
for i in probs.argsort()[::-1][:5]:
    print(f"{classes[i]:>3}: {probs[i]:.1%}")
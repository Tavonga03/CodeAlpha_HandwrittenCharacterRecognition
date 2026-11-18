"""
CodeAlpha ML Internship - Task 3: Handwritten Character Recognition
Datasets: MNIST (digits, 10 classes) or EMNIST Balanced (digits + letters, 47 classes)
Model   : CNN (Keras) with data augmentation

Usage:
    python handwritten_recognition.py --dataset mnist
    python handwritten_recognition.py --dataset emnist --epochs 30
"""
import argparse
import os

import keras
import matplotlib
matplotlib.use("Agg")  # save plots to files; no window needed
import matplotlib.pyplot as plt
import numpy as np
from keras import callbacks, layers, models
from sklearn.metrics import classification_report, confusion_matrix

# EMNIST Balanced label order (47 classes)
EMNIST_BALANCED = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt")


def load_data(name):
    if name == "mnist":
        (x_tr, y_tr), (x_te, y_te) = keras.datasets.mnist.load_data()
        classes = [str(i) for i in range(10)]
    else:
        import tensorflow_datasets as tfds
        ds = tfds.load("emnist/balanced", batch_size=-1, as_supervised=True)
        x_tr, y_tr = tfds.as_numpy(ds["train"])
        x_te, y_te = tfds.as_numpy(ds["test"])
        # EMNIST images are stored transposed -> fix orientation
        x_tr = np.transpose(x_tr[..., 0], (0, 2, 1))
        x_te = np.transpose(x_te[..., 0], (0, 2, 1))
        classes = EMNIST_BALANCED
    x_tr = (x_tr.astype("float32") / 255.0)[..., None]
    x_te = (x_te.astype("float32") / 255.0)[..., None]
    return x_tr, y_tr.astype("int64"), x_te, y_te.astype("int64"), classes


def build_model(n_classes):
    aug = keras.Sequential([
        layers.RandomRotation(0.08),
        layers.RandomTranslation(0.08, 0.08),
        layers.RandomZoom(0.1),
    ], name="augmentation")
    inp = layers.Input(shape=(28, 28, 1))
    x = aug(inp)
    for filters in (32, 64):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    model = models.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def plot_history(h, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(h.history["loss"], label="train"); ax[0].plot(h.history["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].legend()
    ax[1].plot(h.history["accuracy"], label="train"); ax[1].plot(h.history["val_accuracy"], label="val")
    ax[1].set_title("Accuracy"); ax[1].legend()
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_confusion(cm, classes, path):
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, fontsize=7)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion matrix")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_errors(x, y_true, y_pred, classes, path, n=16):
    idx = np.where(y_true != y_pred)[0][:n]
    if len(idx) == 0:
        return
    fig, axes = plt.subplots(4, 4, figsize=(7, 7))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), idx):
        ax.imshow(x[i, ..., 0], cmap="gray")
        ax.set_title(f"true {classes[y_true[i]]} / pred {classes[y_pred[i]]}", fontsize=8)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mnist", "emnist"], default="mnist")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--out_dir", default="outputs")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    keras.utils.set_random_seed(42)

    x_tr, y_tr, x_te, y_te, classes = load_data(args.dataset)
    print(f"train {x_tr.shape} | test {x_te.shape} | classes {len(classes)}")

    model = build_model(len(classes))
    model.summary()
    cbs = [
        callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_accuracy"),
        callbacks.ReduceLROnPlateau(patience=2, factor=0.5, monitor="val_loss"),
    ]
    hist = model.fit(x_tr, y_tr, validation_split=0.1, epochs=args.epochs,
                     batch_size=args.batch_size, callbacks=cbs)

    loss, acc = model.evaluate(x_te, y_te, verbose=0)
    print(f"\nTest accuracy: {acc:.4f}")
    pred = model.predict(x_te, batch_size=512).argmax(axis=1)
    print(classification_report(y_te, pred, target_names=classes, digits=3))

    plot_history(hist, os.path.join(args.out_dir, "training_curves.png"))
    plot_confusion(confusion_matrix(y_te, pred), classes,
                   os.path.join(args.out_dir, "confusion_matrix.png"))
    plot_errors(x_te, y_te, pred, classes, os.path.join(args.out_dir, "misclassified.png"))
    model.save(os.path.join(args.out_dir, f"{args.dataset}_cnn.keras"))
    np.save(os.path.join(args.out_dir, f"{args.dataset}_classes.npy"), np.array(classes))
    print(f"Saved model and plots to {args.out_dir}/")


if __name__ == "__main__":
    main()
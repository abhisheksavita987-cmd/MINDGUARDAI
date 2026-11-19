"""
train_model.py
Trains a lightweight LSTM text classifier and evaluates it on a held-out test set.
Nothing here is hard-coded: every number shown in the dashboard comes from
the run that produced it.
"""
import os
import json
import time
from datetime import datetime

import numpy as np
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from preprocessing import (SimpleTokenizer, pad_sequences, save_tokenizer,
                           build_training_frame, RISKY_NAME, NORMAL_NAME)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")


class _Progress(keras.callbacks.Callback):
    def __init__(self, notify, max_epochs):
        super().__init__()
        self.notify, self.max_epochs, self.history = notify, max_epochs, []
        self.steps = 1

    def on_train_begin(self, logs=None):
        self.steps = self.params.get("steps", 1)

    def on_epoch_begin(self, epoch, logs=None):
        self.notify(epoch=epoch + 1, epoch_fraction=0.0)

    def on_train_batch_end(self, batch, logs=None):
        if batch % 5 == 0:
            self.notify(epoch_fraction=min(1.0, (batch + 1) / self.steps))

    def on_epoch_end(self, epoch, logs=None):
        row = {"epoch": epoch + 1}
        row.update({k: float(v) for k, v in (logs or {}).items()})
        self.history.append(row)
        self.notify(epoch_fraction=1.0, history=list(self.history))


def train(df, text_col, label_col, title_col=None, risky_classes=None,
          max_epochs=12, vocab_size=10000, max_len=200, embed_dim=64,
          lstm_units=64, test_size=0.2, batch_size=64, filter_noise=True,
          proxy_labels=False, dataset_name="", on_progress=None,
          models_dir=MODELS_DIR, seed=42):
    notify = on_progress or (lambda **kw: None)
    t0 = time.time()
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.makedirs(models_dir, exist_ok=True)

    # ---- 1. prepare ------------------------------------------------------
    notify(stage="Cleaning text and removing duplicates")
    frame, report = build_training_frame(
        df, text_col, label_col, title_col, risky_classes, filter_noise)

    if risky_classes:
        classes = [NORMAL_NAME, RISKY_NAME]
        mode = "binary"
    else:
        classes = sorted(frame["label"].unique().tolist())
        mode = "multiclass"
    counts = frame["label"].value_counts()
    too_small = [c for c in classes if counts.get(c, 0) < 10]
    if len(classes) < 2 or too_small:
        raise ValueError(
            "Each class needs at least 10 usable posts after cleaning and at least "
            f"2 classes are required. Usable counts: {counts.to_dict()}")
    cls_index = {c: i for i, c in enumerate(classes)}
    y = frame["label"].map(cls_index).to_numpy()
    texts = frame["clean"].to_numpy()

    # ---- 2. split (stratified; duplicates already removed -> no leakage) --
    notify(stage="Splitting into train / validation / test")
    idx = np.arange(len(frame))
    tr_idx, te_idx = train_test_split(idx, test_size=test_size, stratify=y,
                                      random_state=seed)
    tr_idx, va_idx = train_test_split(tr_idx, test_size=0.1, stratify=y[tr_idx],
                                      random_state=seed)

    # ---- 3. tokenizer is fitted on TRAINING text only ----------------------
    notify(stage="Building vocabulary")
    tok = SimpleTokenizer(vocab_size).fit(texts[tr_idx])
    def encode(ix):
        return pad_sequences(tok.texts_to_sequences(texts[ix]), max_len)
    X_tr, X_va, X_te = encode(tr_idx), encode(va_idx), encode(te_idx)
    y_tr, y_va, y_te = y[tr_idx], y[va_idx], y[te_idx]

    # ---- 4. model ---------------------------------------------------------
    notify(stage="Training LSTM", total_epochs=max_epochs)
    keras.backend.clear_session()
    model = keras.Sequential([
        keras.layers.Input(shape=(max_len,)),
        keras.layers.Embedding(vocab_size, embed_dim, mask_zero=True),
        keras.layers.LSTM(lstm_units, dropout=0.2),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(len(classes), activation="softmax"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    cw = compute_class_weight("balanced", classes=np.arange(len(classes)), y=y_tr)
    prog = _Progress(notify, max_epochs)
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=2,
                                       restore_best_weights=True)
    hist = model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=max_epochs,
                     batch_size=batch_size, class_weight=dict(enumerate(cw)),
                     callbacks=[es, prog], verbose=2)
    epochs_run = len(hist.history["loss"])
    best_epoch = int(np.argmin(hist.history["val_loss"])) + 1

    # ---- 5. evaluate on data the model has never seen ------------------------
    notify(stage="Evaluating on the test set")
    def predict_classes(X):
        return np.argmax(model.predict(X, batch_size=256, verbose=0), axis=1)
    pred_tr, pred_te = predict_classes(X_tr), predict_classes(X_te)
    train_acc = float((pred_tr == y_tr).mean())
    test_acc = float((pred_te == y_te).mean())
    majority = float(np.bincount(y_te).max() / len(y_te))
    rep = classification_report(y_te, pred_te, labels=list(range(len(classes))),
                                target_names=classes, output_dict=True,
                                zero_division=0)
    cm = confusion_matrix(y_te, pred_te, labels=list(range(len(classes))))

    # ---- 6. save -----------------------------------------------------------
    notify(stage="Saving model")
    model.save(os.path.join(models_dir, "lstm_model.keras"))
    save_tokenizer(tok, os.path.join(models_dir, "tokenizer.pkl"))
    meta = {"classes": classes, "mode": mode, "max_len": max_len,
            "risky_classes": [str(c) for c in (risky_classes or [])],
            "proxy_labels": bool(proxy_labels)}
    with open(os.path.join(models_dir, "labels.json"), "w") as f:
        json.dump(meta, f, indent=2)

    seconds = time.time() - t0
    metrics = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": dataset_name,
        "mode": mode, "classes": classes, "proxy_labels": bool(proxy_labels),
        "label_column": label_col, "text_column": text_col, "title_column": title_col,
        "train_accuracy": train_acc, "test_accuracy": test_acc,
        "majority_baseline": majority,
        "precision_macro": rep["macro avg"]["precision"],
        "recall_macro": rep["macro avg"]["recall"],
        "f1_macro": rep["macro avg"]["f1-score"],
        "per_class": [{"name": c, "precision": rep[c]["precision"],
                       "recall": rep[c]["recall"], "f1": rep[c]["f1-score"],
                       "support": int(rep[c]["support"])} for c in classes],
        "confusion_matrix": cm.tolist(),
        "training_seconds": round(seconds, 1),
        "epochs_run": epochs_run, "best_epoch": best_epoch, "max_epochs": max_epochs,
        "history": prog.history,
        "split": {"train": int(len(tr_idx)), "validation": int(len(va_idx)),
                  "test": int(len(te_idx))},
        "data_report": report,
        "config": {"vocab_size": vocab_size, "max_len": max_len,
                   "embed_dim": embed_dim, "lstm_units": lstm_units,
                   "batch_size": batch_size, "test_size": test_size,
                   "filter_noise": filter_noise},
    }
    with open(os.path.join(models_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics

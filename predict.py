"""
predict.py
Loads the saved model ONCE and keeps it in memory, so every prediction is fast.
"""
import os
import json
import threading

import numpy as np
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from preprocessing import clean_text, load_tokenizer, pad_sequences, RISKY_NAME

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")


class Predictor:
    def __init__(self, models_dir=MODELS_DIR):
        self.dir = models_dir
        self.model = self.tok = self.meta = None
        self.lock = threading.Lock()
        self.load()

    def available(self):
        return self.model is not None

    def load(self):
        paths = [os.path.join(self.dir, n) for n in
                 ("lstm_model.keras", "tokenizer.pkl", "labels.json")]
        if not all(os.path.exists(p) for p in paths):
            self.model = self.tok = self.meta = None
            return False
        from tensorflow import keras
        with self.lock:
            self.model = keras.models.load_model(paths[0])
            self.tok = load_tokenizer(paths[1])
            with open(paths[2]) as f:
                self.meta = json.load(f)
        return True

    def predict(self, text):
        if not self.available():
            raise RuntimeError("No trained model found. Train a model first.")
        cleaned = clean_text(text)
        if len(cleaned.split()) < 3:
            raise ValueError("Please enter at least a few words (3 or more).")
        x = pad_sequences(self.tok.texts_to_sequences([cleaned]), self.meta["max_len"])
        with self.lock:
            probs = self.model(x, training=False).numpy()[0]
        classes = self.meta["classes"]
        i = int(np.argmax(probs))
        return {
            "label": classes[i],
            "score": float(probs[i]),
            "mode": self.meta["mode"],
            "is_risky": (classes[i] == RISKY_NAME) if self.meta["mode"] == "binary" else None,
            "proxy_labels": self.meta.get("proxy_labels", False),
            "probabilities": [{"name": c, "p": float(p)} for c, p in zip(classes, probs)],
        }

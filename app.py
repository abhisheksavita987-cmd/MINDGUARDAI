"""
app.py - Flask backend for MINDGUARD AI
Run with:  python app.py   then open http://127.0.0.1:5000
"""
import os
import json
import sqlite3
import threading
from datetime import datetime

from flask import Flask, jsonify, request, render_template
from werkzeug.exceptions import RequestEntityTooLarge

from preprocessing import load_dataset, dataset_info, class_counts, DERIVED_LABEL
import train_model
from predict import Predictor

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
DATASET_PATH = os.path.join(UPLOAD_DIR, "dataset.csv")
DB_PATH = os.path.join(DATA_DIR, "history.db")
for d in (UPLOAD_DIR, MODELS_DIR, DATA_DIR):
    os.makedirs(d, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB upload limit

predictor = Predictor(MODELS_DIR)
_lock = threading.Lock()
TRAIN = {"state": "idle", "stage": "", "epoch": 0, "total_epochs": 0,
         "epoch_fraction": 0.0, "history": [], "error": None}
_dataset_cache = {"mtime": None, "df": None, "derived": False}


# ----------------------------------------------------------------- helpers
def get_dataset():
    if not os.path.exists(DATASET_PATH):
        return None, False
    m = os.path.getmtime(DATASET_PATH)
    if _dataset_cache["mtime"] != m:
        df, meta = load_dataset(DATASET_PATH)
        _dataset_cache.update(mtime=m, df=df, derived=meta["derived"])
    return _dataset_cache["df"], _dataset_cache["derived"]


def read_metrics():
    p = os.path.join(MODELS_DIR, "metrics.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS predictions(
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, text TEXT,
        label TEXT, score REAL)""")
    return con


def err(msg, code=400):
    return jsonify({"error": msg}), code


@app.errorhandler(RequestEntityTooLarge)
def too_large(_):
    return err("File is too large (limit 100 MB).", 413)


@app.errorhandler(Exception)
def unexpected(e):
    if hasattr(e, "code") and isinstance(e.code, int) and e.code < 500:
        return err(getattr(e, "description", str(e)), e.code)
    app.logger.exception(e)
    return err(f"Unexpected server error: {e}", 500)


# ------------------------------------------------------------------- pages
@app.route("/")
def index():
    return render_template("index.html")


# --------------------------------------------------------------------- API
@app.route("/api/status")
def status():
    df, derived = get_dataset()
    m = read_metrics()
    return jsonify({
        "dataset_loaded": df is not None,
        "dataset_rows": int(len(df)) if df is not None else 0,
        "model_ready": predictor.available(),
        "training_state": TRAIN["state"],
        "metrics_summary": None if not m else {
            "test_accuracy": m["test_accuracy"], "trained_at": m["trained_at"],
            "rows_final": m["data_report"]["rows_final"], "mode": m["mode"],
            "proxy_labels": m["proxy_labels"]},
    })


@app.route("/api/upload", methods=["POST"])
def upload():
    f = request.files.get("file")
    if not f or not f.filename:
        return err("No file received. Choose a .csv file.")
    if not f.filename.lower().endswith(".csv"):
        return err("Only .csv files are supported.")
    if TRAIN["state"] == "running":
        return err("Training is running. Wait until it finishes before uploading.", 409)
    f.save(DATASET_PATH)
    try:
        df, meta = load_dataset(DATASET_PATH)
        if len(df) < 20 or df.shape[1] < 1:
            raise ValueError("The file has too few rows to train on.")
    except Exception as e:
        os.remove(DATASET_PATH)
        _dataset_cache["mtime"] = None
        return err(f"Could not read this CSV: {e}")
    _dataset_cache["mtime"] = None
    df, derived = get_dataset()
    return jsonify(dataset_info(df, f.filename, derived))


@app.route("/api/dataset")
def dataset():
    df, derived = get_dataset()
    if df is None:
        return jsonify({"loaded": False})
    info = dataset_info(df, "dataset.csv", derived)
    info["loaded"] = True
    return jsonify(info)


@app.route("/api/dataset/classes")
def dataset_classes():
    df, _ = get_dataset()
    col = request.args.get("column")
    if df is None:
        return err("Upload a dataset first.")
    if col not in df.columns:
        return err(f"Column '{col}' does not exist.")
    counts = class_counts(df, col)
    if len(counts) > 50:
        return err(f"'{col}' has {len(counts)} distinct values - it does not look like a label column.")
    return jsonify({"classes": counts})


def _run_training(kwargs):
    def notify(**kw):
        with _lock:
            TRAIN.update(kw)
    try:
        train_model.train(on_progress=notify, **kwargs)
        predictor.load()
        with _lock:
            TRAIN.update(state="done", stage="Finished", epoch_fraction=1.0)
    except Exception as e:
        app.logger.exception(e)
        with _lock:
            TRAIN.update(state="error", error=str(e))


@app.route("/api/train", methods=["POST"])
def train():
    if TRAIN["state"] == "running":
        return err("Training is already running.", 409)
    df, derived = get_dataset()
    if df is None:
        return err("Upload a dataset first.")
    p = request.get_json(silent=True) or {}
    text_col, label_col = p.get("text_col"), p.get("label_col")
    title_col = p.get("title_col") or None
    for name, col in (("text", text_col), ("label", label_col)):
        if col not in df.columns:
            return err(f"Select a valid {name} column.")
    if title_col and title_col not in df.columns:
        return err("Selected title column does not exist.")
    if label_col in (text_col, title_col):
        return err("The label column must differ from the text columns.")
    try:
        epochs = max(1, min(30, int(p.get("epochs", 12))))
    except (TypeError, ValueError):
        return err("Epochs must be a whole number.")
    kwargs = dict(
        df=df, text_col=text_col, label_col=label_col, title_col=title_col,
        risky_classes=[str(c) for c in p.get("risky_classes", [])] or None,
        max_epochs=epochs, filter_noise=bool(p.get("filter_noise", True)),
        proxy_labels=(label_col == DERIVED_LABEL), dataset_name=os.path.basename(DATASET_PATH))
    with _lock:
        TRAIN.update(state="running", stage="Starting", epoch=0, total_epochs=epochs,
                     epoch_fraction=0.0, history=[], error=None)
    threading.Thread(target=_run_training, args=(kwargs,), daemon=True).start()
    return jsonify({"started": True})


@app.route("/api/train/status")
def train_status():
    with _lock:
        return jsonify(dict(TRAIN))


@app.route("/api/metrics")
def metrics():
    m = read_metrics()
    return jsonify(m) if m else jsonify({"available": False})


@app.route("/api/predict", methods=["POST"])
def predict():
    if TRAIN["state"] == "running":
        return err("Training is running. Try again when it finishes.", 409)
    if not predictor.available():
        return err("No trained model yet. Go to Model Training and train one first.", 409)
    text = ((request.get_json(silent=True) or {}).get("text") or "").strip()
    if not text:
        return err("Enter some text to analyse.")
    if len(text) > 10000:
        return err("Text is too long (limit 10,000 characters).")
    try:
        result = predictor.predict(text)
    except ValueError as e:
        return err(str(e))
    with db() as con:
        con.execute("INSERT INTO predictions(ts,text,label,score) VALUES(?,?,?,?)",
                    (datetime.now().isoformat(timespec="seconds"), text,
                     result["label"], result["score"]))
    return jsonify(result)


@app.route("/api/history", methods=["GET", "DELETE"])
def history():
    with db() as con:
        if request.method == "DELETE":
            con.execute("DELETE FROM predictions")
            return jsonify({"cleared": True})
        rows = con.execute(
            "SELECT id,ts,text,label,score FROM predictions ORDER BY id DESC LIMIT 200").fetchall()
    return jsonify([{"id": r[0], "time": r[1], "text": r[2], "label": r[3], "score": r[4]}
                    for r in rows])


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)

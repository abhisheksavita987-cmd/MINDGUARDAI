"""
smoke_test.py - runs the whole pipeline through the Flask API without a browser.
Usage:  python smoke_test.py path/to/your.csv
"""
import io
import sys
import time

from app import app


def main(csv_path):
    c = app.test_client()
    with open(csv_path, "rb") as f:
        r = c.post("/api/upload", data={"file": (io.BytesIO(f.read()), "data.csv")},
                   content_type="multipart/form-data")
    info = r.get_json()
    assert r.status_code == 200, info
    print("UPLOAD ok  rows:", info["rows"], "| text:", info["suggested_text"],
          "| title:", info["suggested_title"], "| label:", info["suggested_label"])
    cls = c.get("/api/dataset/classes?column=" + info["suggested_label"]).get_json()
    print("CLASSES", cls["classes"])

    r = c.post("/api/train", json={
        "text_col": info["suggested_text"], "title_col": info["suggested_title"],
        "label_col": info["suggested_label"], "epochs": 8, "filter_noise": True})
    assert r.status_code == 200, r.get_json()
    while True:
        s = c.get("/api/train/status").get_json()
        print(f"  [{s['state']}] {s['stage']} epoch {s['epoch']}/{s['total_epochs']}")
        if s["state"] in ("done", "error"):
            break
        time.sleep(3)
    assert s["state"] == "done", s["error"]

    m = c.get("/api/metrics").get_json()
    print("METRICS train_acc=%.3f test_acc=%.3f baseline=%.3f macroF1=%.3f time=%ss"
          % (m["train_accuracy"], m["test_accuracy"], m["majority_baseline"],
             m["f1_macro"], m["training_seconds"]))
    print("DATA REPORT", m["data_report"])
    print("SPLIT", m["split"], "| epochs run", m["epochs_run"], "best", m["best_epoch"])

    for t in ["I feel so empty and nothing matters anymore, I cannot sleep at all",
              "Today was a good day, I went for a walk and met some friends",
              "ok"]:
        r = c.post("/api/predict", json={"text": t})
        print("PREDICT", r.status_code, {k: v for k, v in r.get_json().items()
                                          if k in ("label", "score", "error")})
    print("HISTORY rows:", len(c.get("/api/history").get_json()))
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "surveydata.csv")

# MINDGUARD AI – Mental Health Risk Analysis System

Flask + LSTM (TensorFlow/Keras, CPU only) + HTML/CSS/JavaScript dashboard.
Everything runs on your own computer. No internet is needed once packages are installed.

## 1. One-time setup (Windows + VS Code)

1. Install **Python 3.10, 3.11 or 3.12** from python.org. Tick **"Add python.exe to PATH"**.
2. Unzip this folder, then in VS Code choose **File > Open Folder** and open `mindguard_ai`.
3. Open a terminal in VS Code (**Terminal > New Terminal**) and run:

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

If PowerShell blocks `activate`, run `Set-ExecutionPolicy -Scope Process Bypass` first and try again.
The TensorFlow download is large (a few hundred MB); this only happens once.

## 2. Run the app

```
python app.py
```

Open **http://127.0.0.1:5000** in your browser. Keep the terminal open; it shows the technical training logs.

## 3. Use it

1. **Dataset Management** – upload your CSV. The app reports rows, columns, missing values and repeats.
2. **Model Training** – check the text, title and label columns, then press **Train model**.
   Progress, epochs and a live loss chart update while it trains. The model saves itself into `models/`.
3. **Model Performance** – training and testing accuracy, macro precision / recall / F1, confusion matrix,
   per-class table, training time and a log of what was removed from the data.
4. **Live Text Prediction** – paste new text and press **Analyse text**.
5. **Prediction History** – every analysis is stored locally in `data/history.db`.

## 4. About the labels (important)

A classifier can only learn what the label column teaches it. Use a CSV that has a real target column
(for example 0/1 or `suicidal` / `non-suicidal`). Tick the classes that mean "risky" in the Training screen
and you get the **Potentially Risky / Normal** result. Leave them unticked and you get a multi-class model.

If the file has no label column (like the `surveydata.csv` you started with), the app labels stacked exports
`block_1…block_N` and clearly marks every result as a **proxy** (which export a post came from), not a risk
assessment. The target column is never used as an input feature.

## 5. Test without the browser

```
python smoke_test.py path\to\your.csv
```

Uploads, trains, evaluates and predicts through the API, and prints real numbers. Expect it to take a few minutes.

## 6. Files

| File | Purpose |
|---|---|
| `app.py` | Flask server and JSON API |
| `preprocessing.py` | Text cleaning, tokenizer, dataset inspection, de-duplication (shared by training and prediction) |
| `train_model.py` | Split, LSTM training, early stopping, evaluation, saving |
| `predict.py` | Loads the saved model once and predicts |
| `templates/index.html`, `static/` | Dashboard (Chart.js and the font are bundled, so it works offline) |
| `models/` | `lstm_model.keras`, `tokenizer.pkl`, `labels.json`, `metrics.json` (created by training) |

## 7. Troubleshooting

- **`pip install` fails on tensorflow-cpu:** check `python --version`. Python 3.13+ may lack wheels on older TensorFlow releases; use 3.11 or 3.12.
- **Port 5000 busy:** edit the last line of `app.py` and change `port=5000`.
- **Training is slow:** lower *Maximum epochs* to 5, or use a smaller CSV. Early stopping already ends training when validation loss stops improving.
- **"No trained model yet":** train once in Model Training; the saved model is reloaded automatically on the next start.

## 8. Likely viva questions

- *Why was the target column needed?* LSTM training needs a loss computed against ground truth; clustering cannot say which group means "risky".
- *How is leakage prevented?* Repeated texts are removed before the split; the tokenizer is fitted on the training split only; the label never enters the input.
- *Why report a majority baseline?* To show whether the model beats guessing the most common class.
- *Why is the score not a probability?* A softmax output is uncalibrated, and the UI says so.

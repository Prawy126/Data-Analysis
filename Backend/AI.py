# run_classifier.py
import os, sys, itertools
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

# Jeżeli chcesz użyć własnego skalera, odkomentuj poniżej:
# from Backend.Skalowanie import standard_scaler

# === PARAMETRY ===
CSV_PATH   = "student-mat.csv"   # ← podaj ścieżkę do pliku CSV
TARGET_COL = "Pstatus"                     # ← podaj nazwę kolumny z etykietą
TEST_SIZE  = 0.7
RANDOM_SEED = 42

def read_csv_auto(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        sys.exit(f"[ERR] Nie znaleziono pliku: {path}")

    encodings   = ["utf-8", "cp1250", "latin-1"]
    separators  = [",", ";", "\t", "|"]

    best_df, best_cols = None, 0
    for enc, sep in itertools.product(encodings, separators):
        try:
            df_try = pd.read_csv(path, sep=sep, encoding=enc,
                                 engine="python", on_bad_lines="skip", nrows=200)
            if df_try.shape[1] > best_cols:
                best_cols, best_df, best_enc, best_sep = df_try.shape[1], df_try, enc, sep
        except Exception:
            continue

    if best_df is None or best_cols == 1:
        sys.exit("[ERR] Nie udało się poprawnie odczytać CSV – sprawdź separator lub plik.")

    full_df = pd.read_csv(path, sep=best_sep, encoding=best_enc,
                          engine="python", on_bad_lines="skip")
    print(f"[INFO] Użyty separator '{best_sep}', kodowanie '{best_enc}', "
          f"kształt: {full_df.shape}")
    return full_df

def classify(X: pd.DataFrame, y: pd.Series,
             test_size=0.2, seed=42) -> dict:
    X_enc = pd.get_dummies(X, drop_first=True)

    # --- Skalowanie numerycznych (sklearn StandardScaler):
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_enc),
        columns=X_enc.columns,
        index=X_enc.index
    )

    # Jeśli chcesz użyć własnego skalera, zamień powyższe na:
    # X_scaled = standard_scaler(X_enc, zwroc_tylko_dane=True)

    # --- USUŃ KLASY Z JEDNYM WYSTĄPIENIEM
    value_counts = y.value_counts()
    valid_classes = value_counts[value_counts > 1].index
    mask = y.isin(valid_classes)
    y = y[mask]
    X_scaled = X_scaled.loc[y.index]

    if y.nunique() < 2:
        sys.exit("[ERR] Po usunięciu rzadkich klas pozostała tylko jedna klasa! "
                 "Nie można przeprowadzić klasyfikacji.")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y, test_size=test_size,
        random_state=seed, stratify=y
    )

    results = {}

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_tr, y_tr)
    y_pred = lr.predict(X_te)
    results["logreg"] = {
        "accuracy": round(accuracy_score(y_te, y_pred), 4),
        "f1":       round(f1_score(y_te, y_pred, average="weighted"), 4)
    }

    dt = DecisionTreeClassifier(random_state=seed)
    dt.fit(X_tr, y_tr)
    y_pred = dt.predict(X_te)
    results["dtree"] = {
        "accuracy": round(accuracy_score(y_te, y_pred), 4),
        "f1":       round(f1_score(y_te, y_pred, average="weighted"), 4)
    }
    return results

# ---------------- MAIN ----------------
if __name__ == "__main__":
    df = read_csv_auto(CSV_PATH)

    if TARGET_COL not in df.columns:
        sys.exit(f"[ERR] Kolumna etykiety '{TARGET_COL}' nie istnieje.\n"
                 f"Dostępne kolumny: {list(df.columns)[:20]} …")

    df = df.dropna(subset=[TARGET_COL])
    y  = df[TARGET_COL]
    X  = df.drop(columns=[TARGET_COL])

    # Zakoduj etykiety tekstowe na liczby (np. "A", "B", "C" → 0,1,2)
    if y.dtype == object or str(y.dtype).startswith("category"):
        y = y.astype("category").cat.codes

    res = classify(X, y, test_size=TEST_SIZE, seed=RANDOM_SEED)

    print("\n=== WYNIKI KLASYFIKACJI ===")
    for m, met in res.items():
        print(f"{m:6s} ➜ acc: {met['accuracy']:.4f}, f1: {met['f1']:.4f}")

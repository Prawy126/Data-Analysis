import sys
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, silhouette_score

from Backend.Skalowanie import standard_scaler
from Dane.Dane import wczytaj_csv

CSV_PATH = "student-mat.csv"
TARGET_COL = "Pstatus"
RANDOM_SEED = 42


def classify_and_return_predictions(X, y, test_size=0.3, random_state=42):
    X_enc = pd.get_dummies(X, drop_first=True)
    X_scaled = standard_scaler(X_enc, zwroc_tylko_dane=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )

    lr = LogisticRegression(max_iter=1000)
    dt = DecisionTreeClassifier(random_state=random_state)

    lr.fit(X_train, y_train)
    dt.fit(X_train, y_train)

    y_pred_lr = lr.predict(X_scaled)
    y_pred_dt = dt.predict(X_scaled)

    y_test_pred_lr = lr.predict(X_test)
    y_test_pred_dt = dt.predict(X_test)

    print("\n=== WYNIKI KLASYFIKACJI (test set) ===")
    print(f"logreg ➜ acc: {accuracy_score(y_test, y_test_pred_lr):.4f}, f1: {f1_score(y_test, y_test_pred_lr, average='weighted'):.4f}")
    print(f"dtree  ➜ acc: {accuracy_score(y_test, y_test_pred_dt):.4f}, f1: {f1_score(y_test, y_test_pred_dt, average='weighted'):.4f}")

    preds_df = X.copy()
    preds_df["true_label"] = y
    preds_df["logreg_pred"] = y_pred_lr
    preds_df["dtree_pred"] = y_pred_dt

    return preds_df


def cluster_kmeans(X: pd.DataFrame, n_clusters: int = 4, seed: int = RANDOM_SEED):
    """
    Zwraca:
      - DataFrame z kolumną 'cluster' (etykieta przypisana przez KMeans)
      - słownik metryk (inertia, silhouette)
    """
    X_enc = pd.get_dummies(X, drop_first=True)
    X_scaled = standard_scaler(X_enc, zwroc_tylko_dane=True)

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
    labels = pd.Series(km.fit_predict(X_scaled), index=X.index, name="cluster")

    metrics = {
        "inertia": km.inertia_,
        "silhouette": silhouette_score(X_scaled, labels) if n_clusters > 1 else None
    }

    df_with_clusters = X.copy()
    df_with_clusters["cluster"] = labels

    return df_with_clusters, metrics


# ===== MAIN TEST =====
if __name__ == "__main__":
    df_full = wczytaj_csv(CSV_PATH, wyswietlaj_informacje=True)

    if TARGET_COL not in df_full.columns:
        sys.exit(f"[ERR] Kolumna '{TARGET_COL}' nie istnieje w danych.")

    df_full = df_full.dropna(subset=[TARGET_COL])
    y = df_full[TARGET_COL]
    X = df_full.drop(columns=[TARGET_COL])

    if y.dtype == object or str(y.dtype).startswith("category"):
        y = y.astype("category").cat.codes

    # --- klasyfikacja
    preds = classify_and_return_predictions(X, y)

    # --- klasteryzacja
    df_with_clusters, metrics = cluster_kmeans(X, n_clusters=4)
    preds["cluster"] = df_with_clusters["cluster"]

    print("\n=== KMEANS (k=4) metryki ===")
    print(f"Inertia:    {metrics['inertia']:.2f}")
    print(f"Silhouette: {metrics['silhouette']:.4f}" if metrics["silhouette"] else "Silhouette: brak")

    print("\n=== PODGLĄD WYNIKOWEGO DATAFRAME (z predykcjami i klastrem) ===")
    print(preds.head(10).to_string(index=False))

from Dane.Dane import wczytaj_csv
from Backend.Kodowanie import jedno_gorace_kodowanie
from Backend.Skalowanie import standard_scaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd


def klasyfikuj_dane(
    X: pd.DataFrame,
    y: pd.Series,
    metody: list = ["logreg", "dtree"],
    test_size: float = 0.2,
    wyswietlaj_informacje: bool = True
) -> dict:
    """
    Klasyfikacja danych za pomocą różnych metod.
    """
    try:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Parametr 'X' musi być typu pd.DataFrame")

        # Podział na zbiory
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        wyniki = {}

        # Regresja logistyczna
        if "logreg" in metody:
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            wyniki["logreg"] = {
                "accuracy": accuracy_score(y_test, model.predict(X_test)),
                "f1": f1_score(y_test, model.predict(X_test), average="weighted")
            }

        # Drzewo decyzyjne
        if "dtree" in metody:
            model = DecisionTreeClassifier()
            model.fit(X_train, y_train)
            wyniki["dtree"] = {
                "accuracy": accuracy_score(y_test, model.predict(X_test)),
                "f1": f1_score(y_test, model.predict(X_test), average="weighted")
            }

        if wyswietlaj_informacje:
            print("[INFO] Wyniki klasyfikacji:")
            for metoda, wartosci in wyniki.items():
                print(f"{metoda}: {wartosci}")

        return wyniki

    except Exception as e:
        print(f"[BŁĄD] Błąd podczas klasyfikacji: {str(e)}")
        return {}


# --- Główna część skryptu ---
if __name__ == "__main__":
    # Wczytaj dane
    df = wczytaj_csv("student-mat.csv")

    # Wyświetlanie informacji o danych
    print("[INFO] Pierwotne dane:", df.shape)

    # Kodowanie kategorii (jeśli potrzebne, ale nie dla G3!)
    kolumny_do_zakodowania = [col for col in df.select_dtypes(include='object').columns if col != 'G3']
    if kolumny_do_zakodowania:
        df = jedno_gorace_kodowanie(df, *kolumny_do_zakodowania)

    # Skalowanie zmiennych numerycznych - teraz zwraca bezpośrednio DataFrame
    df = standard_scaler(df, zwroc_tylko_dane=True)

    # Podział na cechy i etykiety
    if 'G3' not in df.columns:
        raise ValueError("Brak kolumny 'G3' w danych po przetworzeniu!")

    X = df.drop(columns=['G3'])
    y = df['G3']

    # Klasyfikacja
    wyniki = klasyfikuj_dane(
        X=X,
        y=y,
        metody=["logreg", "dtree"],
        test_size=0.2,
        wyswietlaj_informacje=True
    )

    print("\n[Wyniki końcowe]:")
    print(wyniki)
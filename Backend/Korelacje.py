from typing import Optional, Union, List
import numpy as np
import pandas as pd
from Dane.Dane import wczytaj_csv


def oblicz_korelacje_pearsona(dane):
    """
    Oblicza korelacje Pearsona dla kolumn numerycznych.

    Parametry:
    ---------
    dane : Union[str, pd.DataFrame]
        Ścieżka do pliku CSV lub DataFrame z danymi

    Zwraca:
    -------
    pd.DataFrame
        Macierz korelacji Pearsona
    """
    if isinstance(dane, str):
        # Jeśli podano ścieżkę do pliku
        df = wczytaj_csv(dane)
    else:
        # Jeśli podano DataFrame
        df = dane

    if df is None or df.empty:
        print("Brak danych do analizy!")
        return pd.DataFrame()

    # Wybierz tylko kolumny numeryczne
    df_num = df.select_dtypes(include=[np.number])

    if df_num.empty:
        print("Brak kolumn numerycznych w danych!")
        return pd.DataFrame()

    # Oblicz i zwróć macierz korelacji
    return df_num.corr(method='pearson')


def oblicz_korelacje_spearmana(dane):
    """
    Oblicza korelacje Spearmana dla kolumn numerycznych.

    Parametry:
    ---------
    dane : Union[str, pd.DataFrame]
        Ścieżka do pliku CSV lub DataFrame z danymi

    Zwraca:
    -------
    pd.DataFrame
        Macierz korelacji Spearmana
    """
    if isinstance(dane, str):
        # Jeśli podano ścieżkę do pliku
        df = wczytaj_csv(dane)
    else:
        # Jeśli podano DataFrame
        df = dane

    if df is None or df.empty:
        print("Brak danych do analizy!")
        return pd.DataFrame()

    # Wybierz tylko kolumny numeryczne
    df_num = df.select_dtypes(include=[np.number])

    if df_num.empty:
        print("Brak kolumn numerycznych w danych!")
        return pd.DataFrame()

    # Oblicz i zwróć macierz korelacji
    return df_num.corr(method='spearman')
def oblicz_korelacje_spearmana(dane) -> pd.DataFrame:
    """
    Oblicza korelacje Spearmana pomiędzy kolumnami numerycznymi.

    Parametry:
    ---------
    dane : pd.DataFrame lub str
        DataFrame zawierający dane lub ścieżka do pliku CSV

    Zwraca:
    -------
    pd.DataFrame
        Macierz korelacji
    """
    # Sprawdź czy dane to ścieżka do pliku i wczytaj jeśli tak
    if isinstance(dane, str):
        df = wczytaj_csv(dane)
    else:
        df = dane

    if df is None or df.empty:
        return pd.DataFrame()

    # Wybierz tylko kolumny numeryczne
    df_num = df.select_dtypes(include=[np.number])

    # Sprawdź czy są kolumny numeryczne
    if df_num.empty or len(df_num.columns) < 2:
        return pd.DataFrame()

    # Oblicz macierz korelacji
    corr_matrix = df_num.corr(method='spearman')
    return corr_matrix


def oblicz_korelacje_spearmana(dane) -> pd.DataFrame:
    """
    Oblicza korelacje Spearmana pomiędzy kolumnami numerycznymi.

    Parametry:
    ---------
    dane : pd.DataFrame lub str
        DataFrame zawierający dane lub ścieżka do pliku CSV

    Zwraca:
    -------
    pd.DataFrame
        Macierz korelacji
    """
    # Sprawdź czy dane to ścieżka do pliku i wczytaj jeśli tak
    if isinstance(dane, str):
        df = wczytaj_csv(dane)
    else:
        df = dane

    if df is None or df.empty:
        return pd.DataFrame()

    # Wybierz tylko kolumny numeryczne
    df_num = df.select_dtypes(include=[np.number])

    # Sprawdź czy są kolumny numeryczne
    if df_num.empty or len(df_num.columns) < 2:
        return pd.DataFrame()

    # Oblicz macierz korelacji
    corr_matrix = df_num.corr(method='spearman')
    return corr_matrix

def oblicz_korelacje_spearmana(
        df: pd.DataFrame,
        wyswietlaj_informacje: bool = False
) -> Optional[pd.DataFrame]:
    """
    Oblicza macierz korelacji Spearmana dla numerycznych kolumn w DataFrame

    Parametry:
    ---------
    df : pd.DataFrame
        DataFrame zawierający dane do analizy
    wyswietlaj_informacje : bool
        Czy wyświetlać informacje diagnostyczne

    Zwraca:
    -------
    Optional[pd.DataFrame]
        Macierz korelacji Spearmana lub None w przypadku błędu
    """
    try:
        if not isinstance(df, pd.DataFrame):
            if wyswietlaj_informacje:
                print("[UWAGA] Przekazany obiekt nie jest DataFrame")
            return None

        # Wybierz tylko kolumny numeryczne
        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.empty:
            if wyswietlaj_informacje:
                print("[UWAGA] Brak kolumn numerycznych do obliczenia korelacji")
            return None

        # Oblicz macierz korelacji Spearmana
        korelacja = numeric_df.corr(method='spearman')

        if wyswietlaj_informacje:
            print("\n[INFO] Macierz korelacji Spearmana:")
            print(korelacja)

        return korelacja

    except Exception as e:
        print(f"[BŁĄD] Nie można obliczyć korelacji: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None
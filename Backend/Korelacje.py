from typing import Optional, Union, List
import numpy as np
import pandas as pd
from Dane.Dane import wczytaj_csv

def oblicz_korelacje_pearsona(
        df: pd.DataFrame,
        wyswietlaj_informacje: bool = False
) -> Optional[pd.DataFrame]:
    """
    Oblicza macierz korelacji Pearsona dla numerycznych kolumn w DataFrame

    Parametry:
    ---------
    df : pd.DataFrame
        DataFrame zawierający dane do analizy
    wyswietlaj_informacje : bool
        Czy wyświetlać informacje diagnostyczne

    Zwraca:
    -------
    Optional[pd.DataFrame]
        Macierz korelacji Pearsona lub None w przypadku błędu
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

        # Oblicz macierz korelacji Pearsona
        korelacja = numeric_df.corr(method='pearson')

        if wyswietlaj_informacje:
            print("\n[INFO] Macierz korelacji Pearsona:")
            print(korelacja)

        return korelacja

    except Exception as e:
        print(f"[BŁĄD] Nie można obliczyć korelacji: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None


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
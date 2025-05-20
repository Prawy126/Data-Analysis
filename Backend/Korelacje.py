from typing import Optional, Union, List
import numpy as np
import pandas as pd
from Dane.Dane import wczytaj_csv

def oblicz_korelacje_pearsona(
        sciezka_pliku: str,
        separator: Union[str, List[str]] = None,
        kolumny_daty: List[str] = None,
        format_daty: str = None,
        wymagane_kolumny: List[str] = None,
        wyswietlaj_informacje: bool = False
) -> Optional[pd.DataFrame]:
    """
    Oblicza macierz korelacji Pearsona dla numerycznych kolumn w pliku CSV

    Parametry:
    ---------
    sciezka_pliku : str
        Ścieżka do pliku CSV
    separator : Union[str, List[str]], opcjonalnie
        Separator kolumn (jeśli None, zostanie wykryty automatycznie)
    kolumny_daty : List[str], opcjonalnie
        Lista nazw kolumn daty
    format_daty : str, opcjonalnie
        Format daty (np. "%Y-%m-%d %H:%M:%S")
    wymagane_kolumny : List[str], opcjonalnie
        Lista wymaganych kolumn
    wyswietlaj_informacje : bool
        Czy wyświetlać informacje diagnostyczne

    Zwraca:
    -------
    Optional[pd.DataFrame]
        Macierz korelacji Pearsona lub None w przypadku błędu
    """
    try:
        # Wczytaj dane z użyciem istniejącej funkcji
        df = wczytaj_csv(
            sciezka_pliku=sciezka_pliku,
            separator=separator,
            kolumny_daty=kolumny_daty,
            format_daty=format_daty,
            wymagane_kolumny=wymagane_kolumny,
            wyswietlaj_informacje=wyswietlaj_informacje
        )

        if df is None:
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


def oblicz_korelacje_spearmana(sciezka_pliku: str,
                               separator: Union[str, List[str]] = None,
                               kolumny_daty: List[str] = None,
                               format_daty: str = None,
                               wymagane_kolumny: List[str] = None,
                               wyswietlaj_informacje: bool = False) -> Optional[pd.DataFrame]:
    try:
        df = wczytaj_csv(
            sciezka_pliku=sciezka_pliku,
            separator=separator,
            kolumny_daty=kolumny_daty,
            format_daty=format_daty,
            wymagane_kolumny=wymagane_kolumny,
            wyswietlaj_informacje=wyswietlaj_informacje
        )

        if df is None:
            return None

        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.empty:
            if wyswietlaj_informacje:
                print("[UWAGA] Brak kolumn numerycznych do obliczenia korelacji")
            return None

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

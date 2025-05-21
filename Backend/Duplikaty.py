from typing import Optional, List, Union, Dict

import pandas as pd

from Dane.Dane import wczytaj_csv, _optymalizuj_pamiec


def usun_duplikaty(
    df: pd.DataFrame,
    kolumny: Optional[List[str]] = None,
    tryb: str = 'pierwszy',
    wyswietlaj_info: bool = True
) -> Dict[str, Union[pd.DataFrame, int]]:
    """
    Usuwa powtarzające się wiersze z DataFrame'a.

    Parametry:
    ---------
    df : pd.DataFrame
        Wejściowy DataFrame.
    kolumny : List[str], opcjonalnie
        Lista kolumn, na podstawie których sprawdzamy duplikaty.
        Jeśli None, sprawdzane są wszystkie kolumny.
    tryb : str, opcjonalnie
        'pierwszy' — zachowuje pierwsze wystąpienie,
        'ostatni' — zachowuje ostatnie wystąpienie,
        'wszystkie' — usuwa wszystkie duplikaty.
    wyswietlaj_info : bool
        Czy wyświetlać informacje diagnostyczne.

    Zwraca:
    -------
    Dict[str, Union[pd.DataFrame, int]]
        - 'df_cleaned': DataFrame po usunięciu duplikatów
        - 'liczba_duplikatow': liczba usuniętych duplikatów
    """
    try:
        # Walidacja trybu działania
        if tryb not in ['pierwszy', 'ostatni', 'wszystkie']:
            raise ValueError("Parametr 'tryb' musi być 'pierwszy', 'ostatni' lub 'wszystkie'.")

        # Sprawdzenie, czy wybrane kolumny istnieją
        if kolumny:
            nieistniejace = [col for col in kolumny if col not in df.columns]
            if nieistniejace:
                raise ValueError(f"Brakujące kolumny: {nieistniejace}")
            klucze = kolumny
        else:
            klucze = df.columns.tolist()

        # Wykrywanie duplikatów
        duplikaty_maska = df.duplicated(subset=klucze, keep=False)
        liczba_duplikatow = duplikaty_maska.sum()

        # Wybór trybu zachowania duplikatów
        if tryb == 'pierwszy':
            keep = 'first'
        elif tryb == 'ostatni':
            keep = 'last'
        else:
            keep = False

        # Usuwanie duplikatów
        df_wynikowy = df.drop_duplicates(subset=klucze, keep=keep)

        if wyswietlaj_info:
            print(f"[INFO] Liczba znalezionych duplikatów: {liczba_duplikatow}")
            print(f"[INFO] Tryb usuwania: {tryb}")
            print(f"[INFO] Liczba wierszy przed: {len(df)}, po: {len(df_wynikowy)}")
            print(f"[INFO] Usunięto {liczba_duplikatow} duplikatów")

        return {
            'df_cleaned': df_wynikowy,
            'liczba_duplikatow': liczba_duplikatow
        }

    except Exception as e:
        print(f"[BŁĄD] Błąd podczas usuwania duplikatów: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return {
            'df_cleaned': df,
            'liczba_duplikatow': 0
        }

df = wczytaj_csv(
    sciezka_pliku="online_retail_II.csv",
    kolumny_daty=["InvoiceDate"],
    format_daty="%d-%m-%Y %H:%M",
    wyswietlaj_informacje=True
)

# Zakomentowałem testy może się jeszcze przydadzą nie wiem

"""

# 1. Usuń duplikaty
wynik = usun_duplikaty(df, kolumny=["Invoice", "StockCode"], wyswietlaj_info=True)
df = wynik['df_cleaned']

# 2. Optymalizuj pamięć po usunięciu duplikatów
df = _optymalizuj_pamiec(df)

"""
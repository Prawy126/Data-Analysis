from typing import Optional, List

import pandas as pd

from Dane.Dane import wczytaj_csv


def ekstrakcja_podtablicy(
        df: pd.DataFrame,
        rows: Optional[List[int]] = None,
        cols: Optional[List[str]] = None,
        mode: str = 'keep',
        wyswietlaj_informacje: bool = False
) -> Optional[pd.DataFrame]:
    """
    Ekstrakcja podtablicy poprzez usuwanie lub zachowywanie wskazanych wierszy/kolumn.

    Parametry:
    ---------
    df : pd.DataFrame
        Wejściowy DataFrame.
    rows : List[int], opcjonalnie
        Lista numerów wierszy (indeksów) do zachowania/usuwania.
    cols : List[str], opcjonalnie
        Lista nazw kolumn do zachowania/usuwania.
    mode : str, opcjonalnie
        Tryb działania: 'keep' (zachowaj podane) lub 'remove' (usuń podane).
    wyswietlaj_informacje : bool
        Czy wyświetlać informacje diagnostyczne.

    Zwraca:
    -------
    Optional[pd.DataFrame]
        Przefiltrowany DataFrame lub None w przypadku błędu.
    """
    try:
        if wyswietlaj_informacje:
            print("[INFO] Rozpoczynam ekstrakcję podtablicy...")

        if mode not in ('keep', 'remove'):
            raise ValueError("Tryb musi być 'keep' lub 'remove'.")

        # Kopiujemy DataFrame, by nie modyfikować oryginału
        wynik_df = df.copy()

        # Obsługa wierszy
        if rows is not None:
            if any(i >= len(wynik_df) or i < 0 for i in rows):
                raise ValueError(f"Nieprawidłowe numery wierszy: {rows}")
            if mode == 'keep':
                wynik_df = wynik_df.iloc[rows]
            elif mode == 'remove':
                wynik_df = wynik_df.drop(wynik_df.index[rows])

        # Obsługa kolumn
        if cols is not None:
            if not set(cols).issubset(wynik_df.columns):
                brakujace = set(cols) - set(wynik_df.columns)
                raise ValueError(f"Brakujące kolumny: {brakujace}")
            if mode == 'keep':
                wynik_df = wynik_df[cols]
            elif mode == 'remove':
                wynik_df = wynik_df.drop(columns=cols)

        if wyswietlaj_informacje:
            print(f"[SUCCES] Podtablica została wygenerowana. Wymiary: {wynik_df.shape}")
            print(wynik_df.head())

        return wynik_df

    except Exception as e:
        print(f"[BŁĄD] Ekstrakcja podtablicy nie powiodła się: {str(e)}")
        return None

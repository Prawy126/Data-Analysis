from typing import Union, Dict, Optional

import pandas as pd

from Dane.Dane import _optymalizuj_pamiec, wczytaj_csv


def zamien_wartosci(
        df: pd.DataFrame,
        kolumna: str = None,
        stara_wartosc: Union[str, int, float, pd.Timestamp] = None,
        nowa_wartosc: Union[str, int, float, pd.Timestamp] = None,
        reguly: Dict[str, Dict[Union[str, int, float], Union[str, int, float]]] = None,
        wyswietlaj_informacje: bool = True
) -> Optional[pd.DataFrame]:
    """
    Zastępuje wartości w DataFrame:
    - Ręcznie: Zamiana konkretnej wartości w konkretnej kolumnie
    - Automatycznie: Zamiana wielu wartości w wielu kolumnach na podstawie słownika reguł

    Parametry:
    ---------
    df : pd.DataFrame
        Wejściowy DataFrame wczytany przez `wczytaj_csv`.
    kolumna : str, opcjonalnie
        Nazwa kolumny do ręcznej zmiany wartości.
    stara_wartosc : Union[str, int, float, pd.Timestamp], opcjonalnie
        Wartość do zamiany (dla trybu ręcznego).
    nowa_wartosc : Union[str, int, float, pd.Timestamp], opcjonalnie
        Nowa wartość (dla trybu ręcznego).
    reguly : Dict[str, Dict[...]], opcjonalnie
        Słownik reguł: `{kolumna: {stara: nowa}}`.
    wyswietlaj_informacje : bool
        Czy wyświetlać szczegółowe informacje diagnostyczne.

    Zwraca:
    -------
    Optional[pd.DataFrame]
        Zmodyfikowany DataFrame lub None w przypadku błędu.
    """
    try:
        wynik_df = df.copy()

        # Walidacja: Ręczna zamiana
        if kolumna and stara_wartosc is not None and nowa_wartosc is not None:
            if kolumna not in wynik_df.columns:
                raise ValueError(f"Kolumna '{kolumna}' nie istnieje w DataFrame.")

            # Obsługa kategorii: jeśli kolumna jest typu category, konwertujemy na object
            if pd.api.types.is_categorical_dtype(wynik_df[kolumna]):
                wynik_df[kolumna] = wynik_df[kolumna].astype(object)

            # Zamiana wartości
            wynik_df[kolumna] = wynik_df[kolumna].replace(stara_wartosc, nowa_wartosc)

            if wyswietlaj_informacje:
                print(f"[INFO] 🔧 Zamieniono '{stara_wartosc}' na '{nowa_wartosc}' w kolumnie '{kolumna}'.")

        # Walidacja: Automatyczna zamiana
        elif reguly:
            for kolumna, zmiany in reguly.items():
                if kolumna not in wynik_df.columns:
                    raise ValueError(f"Kolumna '{kolumna}' nie istnieje w DataFrame.")

                if pd.api.types.is_categorical_dtype(wynik_df[kolumna]):
                    wynik_df[kolumna] = wynik_df[kolumna].astype(object)

                for stara, nowa in zmiany.items():
                    wynik_df[kolumna] = wynik_df[kolumna].replace(stara, nowa)

                if wyswietlaj_informacje:
                    print(f"[INFO] 🤖 Automatycznie zaktualizowano kolumnę '{kolumna}': {zmiany}")

        else:
            raise ValueError(
                "Podaj albo parametry ręczne (kolumna, stara_wartosc, nowa_wartosc), albo reguły automatyczne.")

        # Optymalizacja pamięci po zmianach
        wynik_df = _optymalizuj_pamiec(wynik_df)

        return wynik_df

    except Exception as e:
        print(f"[BŁĄD] Nie udało się zastąpić wartości: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None

# Załaduj dane
df = wczytaj_csv("online_retail_II.csv", separator=";", kolumny_daty=["InvoiceDate"], wyswietlaj_informacje=True)

# Zamień wartość 2.55 na 3.0 w kolumnie "Price"
df = zamien_wartosci(
    df=df,
    kolumna="Price",
    stara_wartosc=2.55,
    nowa_wartosc=3.0,
    wyswietlaj_informacje=True
)
print(df[["Price"]].head())
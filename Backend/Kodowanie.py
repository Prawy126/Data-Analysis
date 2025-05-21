from typing import Union, List, Dict

import numpy as np
import pandas as pd

from Dane.Dane import _optymalizuj_pamiec, wczytaj_csv


def jedno_gorace_kodowanie(
        df: pd.DataFrame,
        kolumny: Union[str, List[str]],
        usun_pierwsza: bool = False,
        wyswietl_informacje: bool = True
) -> Dict[str, Union[pd.DataFrame, Dict]]:
    """
    Kodowanie One-Hot dla wybranych kolumn kategorycznych.

    Parametry:
    ---------
    df : pd.DataFrame
        Wejściowy DataFrame z danymi.
    kolumny : Union[str, List[str]]
        Nazwa kolumny lub lista kolumn do zakodowania.
    usuń_pierwszą : bool, opcjonalnie
        Czy usunąć pierwszą kolumnę (dla uniknięcia multicollinearity).
    wyświetl_informacje : bool
        Czy wyświetlać informacje diagnostyczne.

    Zwraca:
    -------
    Dict[str, Union[pd.DataFrame, Dict]]
        - 'df_zakodowany': DataFrame po kodowaniu
        - 'mapowania': słownik z mapowaniami dla każdej zakodowanej kolumny
    """
    try:
        # Walidacja wejścia
        if isinstance(kolumny, str):
            kolumny = [kolumny]

        # Sprawdzenie, czy kolumny istnieją i są kategoryczne
        nieistniejace_kolumny = [col for col in kolumny if col not in df.columns]
        if nieistniejace_kolumny:
            raise ValueError(f"Brakujące kolumny: {nieistniejace_kolumny}")

        niekategoryczne_kolumny = [col for col in kolumny if not pd.api.types.is_categorical_dtype(df[col])]
        if niekategoryczne_kolumny:
            raise ValueError(f"Te kolumny nie są kategoryczne: {niekategoryczne_kolumny}")

        # Kopiowanie DataFrame
        df_zakodowany = df.copy()
        mapowania = {}

        # Kodowanie dla każdej kolumny
        for kolumna in kolumny:
            # Zastosowanie pd.get_dummies
            df_zakodowany = pd.get_dummies(
                df_zakodowany,
                columns=[kolumna],
                drop_first=usun_pierwsza,
                dtype=np.int8  # Optymalizacja typu danych
            )

            # Zapisz mapowanie (dla ewentualnego odwrócenia)
            wartosci_unikalne = df[kolumna].cat.categories.tolist()
            mapowania[kolumna] = {wartosc: indeks for indeks, wartosc in enumerate(wartosci_unikalne)}

            if wyswietl_informacje:
                print(f"[INFO] Zakodowano kolumnę '{kolumna}' metodą One-Hot")
                print(f"  Wartości unikalne: {wartosci_unikalne}")

        # Optymalizacja pamięci po kodowaniu
        df_zakodowany = _optymalizuj_pamiec(df_zakodowany)

        if wyswietl_informacje:
            print(f"[INFO] Nowy rozmiar danych: {df_zakodowany.shape}")

        return {
            'df_zakodowany': df_zakodowany,
            'mapowania': mapowania
        }

    except Exception as e:
        print(f"[BŁĄD] Błąd podczas kodowania One-Hot: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return {
            'df_zakodowany': df,
            'mapowania': {}
        }

# Nie wiem dlaczego ale z tymi parametrami funkcja one_hot_encoding sypła błędem że nie mamy takiej kolumny
# df = wczytaj_csv("online_retail_II.csv", separator=",", kolumny_daty=["InvoiceDate"])
"""
df = wczytaj_csv("online_retail_II.csv")
wynik = jedno_gorace_kodowanie(
    df=df,
    kolumny="Country",
    usun_pierwsza=True,
    wyswietl_informacje=True
)
"""

def binarne_kodowanie(
        df: pd.DataFrame,
        kolumny: Union[str, List[str]],
        wyswietlaj_informacje: bool = True
) -> Dict[str, Union[pd.DataFrame, Dict]]:
    """
    Kodowanie binarne (Binary Encoding) dla wybranych kolumn kategorycznych.

    Parametry:
    ---------
    df : pd.DataFrame
        Wejściowy DataFrame z danymi.
    kolumny : Union[str, List[str]]
        Nazwa kolumny lub lista kolumn do zakodowania.
    wyswietlaj_informacje : bool
        Czy wyświetlać informacje diagnostyczne.

    Zwraca:
    -------
    Dict[str, Union[pd.DataFrame, Dict]]
        - 'df_zakodowany': DataFrame po Binary Encoding
        - 'mapowania': słownik z mapowaniami {kolumna: {wartosc: kod_binarny}}
    """
    try:
        # Walidacja wejscia
        if isinstance(kolumny, str):
            kolumny = [kolumny]

        # Sprawdzenie, czy kolumny istnieją i są kategoryczne/tekstowe
        nieistniejace_kolumny = [col for col in kolumny if col not in df.columns]
        if nieistniejace_kolumny:
            raise ValueError(f"Brakujące kolumny: {nieistniejace_kolumny}")

        niekategoryczne_kolumny = [
            col for col in kolumny
            if not (pd.api.types.is_categorical_dtype(df[col]) or
                    pd.api.types.is_object_dtype(df[col]))
        ]
        if niekategoryczne_kolumny:
            raise ValueError(f"Te kolumny nie są kategoryczne ani tekstowe: {niekategoryczne_kolumny}")

        # Kopiowanie DataFrame
        df_zakodowany = df.copy()
        mapowania = {}

        # Kodowanie dla każdej kolumny
        for kolumna in kolumny:
            # Utworzenie słownika wartości -> liczba porządkowa
            unikalne_wartosci = df[kolumna].astype(str).str.strip().unique()
            max_dlugosc = len(unikalne_wartosci) - 1
            liczba_bitow = max_dlugosc.bit_length()

            # Stworzenie mapowania wartości na liczby, potem na bity
            mapping = {}
            for indeks, wartosc in enumerate(unikalne_wartosci):
                # Konwersja liczby porządkowej na kod binarny
                binarnie = format(indeks, f'0{liczba_bitow}b')
                mapping[wartosc] = list(map(int, binarnie))

            # Dodanie do mapowań
            mapowania[kolumna] = mapping

            # Usunięcie oryginalnej kolumny
            df_zakodowany = df_zakodowany.drop(columns=[kolumna])

            # Dodanie nowych kolumn binarnych
            for bit in range(liczba_bitow):
                nowa_kolumna = f"{kolumna}_bin_{bit}"
                df_zakodowany[nowa_kolumna] = df[kolumna].apply(
                    lambda x: mapping.get(str(x).strip(), [np.nan] * liczba_bitow)[bit]
                )

            if wyswietlaj_informacje:
                print(f"[INFO] Zakodowano kolumnę '{kolumna}' metodą Binary Encoding")
                print(f"  Liczba bitów: {liczba_bitow}")
                print(f"  Przykładowe mapowania: {dict(list(mapping.items())[:3])}...")

        # Optymalizacja pamięci po kodowaniu
        df_zakodowany = _optymalizuj_pamiec(df_zakodowany)

        if wyswietlaj_informacje:
            print(f"[INFO] Nowy rozmiar danych: {df_zakodowany.shape}")

        return {
            'df_zakodowany': df_zakodowany,
            'mapowania': mapowania
        }

    except Exception as e:
        print(f"[BŁĄD] Błąd podczas kodowania Binary Encoding: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return {
            'df_zakodowany': df,
            'mapowania': {}
        }

# Przykład użycia
"""
df = wczytaj_csv("online_retail_II.csv", separator=";", kolumny_daty=["InvoiceDate"])
wynik = binarne_kodowanie(df, kolumny="Country", wyswietlaj_informacje=True)
df_encoded = wynik['df_zakodowany']
print(df_encoded.head())
"""


def kodowanie_docelowe(
        df: pd.DataFrame,
        kolumny: Union[str, List[str]],
        target: str,
        smoothing: float = 1.0,
        wyswietlaj_informacje: bool = True
) -> Dict[str, Union[pd.DataFrame, Dict]]:
    """
    Kodowanie Target Encoding dla wybranych kolumn kategorycznych.

    Parametry:
    ---------
    df : pd.DataFrame
        Wejściowy DataFrame z danymi.
    kolumny : Union[str, List[str]]
        Nazwa kolumny lub lista kolumn do zakodowania.
    target : str
        Nazwa kolumny docelowej (target) do obliczenia średnich.
    smoothing : float, opcjonalnie
        Współczynnik wygładzania (większa wartość = większy wpływ średniej globalnej).
    wyswietlaj_informacje : bool
        Czy wyświetlać informacje diagnostyczne.

    Zwraca:
    -------
    Dict[str, Union[pd.DataFrame, Dict]]
        - 'df_encoded': DataFrame po Target Encoding
        - 'mapowania': słownik z mapowaniami {kolumna: {wartosc: kod_targetowy}}
    """
    try:
        # Walidacja wejscia
        if isinstance(kolumny, str):
            kolumny = [kolumny]

        # Sprawdzenie, czy kolumna target istnieje i jest numeryczna
        if target not in df.columns:
            raise ValueError(f"Brak kolumny docelowej '{target}'")
        if not pd.api.types.is_numeric_dtype(df[target]):
            raise ValueError(f"Kolumna docelowa '{target}' musi być numeryczna")

        # Sprawdzenie, czy kolumny do kodowania istnieją i są kategoryczne/tekstowe
        nieistniejace_kolumny = [col for col in kolumny if col not in df.columns]
        if nieistniejace_kolumny:
            raise ValueError(f"Brakujące kolumny: {nieistniejace_kolumny}")

        niekategoryczne_kolumny = [
            col for col in kolumny
            if not (pd.api.types.is_categorical_dtype(df[col]) or
                    pd.api.types.is_object_dtype(df[col]))
        ]
        if niekategoryczne_kolumny:
            raise ValueError(f"Te kolumny nie są kategoryczne ani tekstowe: {niekategoryczne_kolumny}")

        # Kopiowanie DataFrame
        df_encoded = df.copy()
        mapowania = {}
        globalna_srednia = df[target].mean()

        # Kodowanie dla każdej kolumny
        for kolumna in kolumny:
            # Konwersja kolumny kategorycznej na typ tekstowy
            df_encoded[kolumna] = df_encoded[kolumna].astype(str)

            # Obliczenie średnich dla każdej kategorii
            unikalne_wartosci = df_encoded[kolumna].str.strip().unique()
            liczba_kategorii = len(unikalne_wartosci)

            # Obliczenie średnich z wygładzaniem (smoothed mean)
            agregacja = df.groupby(kolumna)[target].agg(['count', 'mean'])
            agregacja['smoothed_mean'] = (
                (agregacja['count'] * agregacja['mean'] + smoothing * globalna_srednia) /
                (agregacja['count'] + smoothing)
            )
            mapping = dict(zip(agregacja.index, agregacja['smoothed_mean']))

            # Dodaj do mapowań
            mapowania[kolumna] = mapping

            # Zamieniamy wartości kategoryczne na zakodowane średnie
            df_encoded[f"{kolumna}_target"] = df_encoded[kolumna].map(mapping).fillna(globalna_srednia)

            if wyswietlaj_informacje:
                print(f"[INFO] Zakodowano kolumne '{kolumna}' metoda Target Encoding")
                print(f"  Liczba unikalnych wartosci: {liczba_kategorii}")
                print(f"  Srednia globalna: {globalna_srednia:.4f}")

        # Usun oryginalne kolumny kategoryczne
        df_encoded = df_encoded.drop(columns=kolumny)

        # Optymalizacja pamięci po kodowaniu
        df_encoded = _optymalizuj_pamiec(df_encoded)

        if wyswietlaj_informacje:
            print(f"[INFO] Nowy rozmiar danych: {df_encoded.shape}")

        return {
            'df_encoded': df_encoded,
            'mapowania': mapowania
        }

    except Exception as e:
        print(f"[BŁĄD] Blad podczas kodowania Target Encoding: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return {
            'df_encoded': df,
            'mapowania': {}
        }
# Załóżmy, że mamy wczytany DataFrame z kolumną 'Price' jako target
df = wczytaj_csv("online_retail_II.csv")

# Kodowanie kolumny "Country" z wygładzaniem
wynik = kodowanie_docelowe(
    df=df,
    kolumny="Country",
    target="Price",
    smoothing=10.0,
    wyswietlaj_informacje=True
)

df_encoded = wynik['df_encoded']
mapowania = wynik['mapowania']

print("\nPrzykładowe dane po kodowaniu:")
print(df_encoded[[col for col in df_encoded.columns if 'target' in col]].head())
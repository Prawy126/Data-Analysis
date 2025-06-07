import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Union, Tuple, Any
import os
from Dane.Dane import wczytaj_csv


def oblicz_statystyki_nie_numeryczne(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Oblicza statystyki dla kolumn nie-numerycznych.

    Parametry:
    ---------
    df : pd.DataFrame
        DataFrame z danymi

    Zwraca:
    ------
    Dict[str, Dict[str, Any]]
        Słownik ze statystykami dla każdej kolumny nie-numerycznej
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Parametr df musi być typu pandas.DataFrame")

    statystyki = {}

    for kolumna in df.columns:
        if not pd.api.types.is_numeric_dtype(df[kolumna]):
            # Pomiń puste kolumny
            if df[kolumna].isna().all():
                continue

            # Obliczanie statystyk
            wartosci = df[kolumna].dropna()
            statystyki[kolumna] = {
                'liczba_wystapien': len(wartosci),
                'wartosci_unikalne': wartosci.nunique(),
                'najczestsza_wartosc': wartosci.mode().iloc[0] if not wartosci.empty else None,
                'czestotliwosc_najczestszej': wartosci.value_counts().max() / len(wartosci),
                'procent_wypelnienia': (len(wartosci) / len(df)) * 100,
                'dlugosc_min': wartosci.astype(str).str.len().min() if not wartosci.empty else 0,
                'dlugosc_max': wartosci.astype(str).str.len().max() if not wartosci.empty else 0,
                'dlugosc_srednia': wartosci.astype(str).str.len().mean() if not wartosci.empty else 0
            }

    return statystyki


def znajdz_kolumny_numeryczne(df: pd.DataFrame) -> List[str]:
    """
    Znajduje wszystkie kolumny numeryczne w DataFrame.

    Parametry:
    ---------
    df : pd.DataFrame
        DataFrame z danymi

    Zwraca:
    ------
    List[str]
        Lista nazw kolumn numerycznych
    """
    kolumny_numeryczne = []

    for kolumna in df.columns:
        if pd.api.types.is_numeric_dtype(df[kolumna]):
            kolumny_numeryczne.append(kolumna)

    return kolumny_numeryczne


def wydobadz_wartosci_numeryczne(df: pd.DataFrame, wybrane_kolumny: Optional[List[str]] = None) -> Dict[
    str, np.ndarray]:
    """
    Wydobywa wartości numeryczne z wybranych kolumn DataFrame.

    Parametry:
    ---------
    df : pd.DataFrame
        DataFrame z danymi
    wybrane_kolumny : Optional[List[str]]
        Lista wybranych kolumn do analizy. Jeśli None, analizuje wszystkie kolumny numeryczne.

    Zwraca:
    ------
    Dict[str, np.ndarray]
        Słownik z nazwami kolumn jako kluczami i tablicami wartości numerycznych
    """
    wartosci_numeryczne = {}

    if wybrane_kolumny is None:
        kolumny_do_analizy = znajdz_kolumny_numeryczne(df)
    else:
        kolumny_do_analizy = [k for k in wybrane_kolumny if k in df.columns]

        # Sprawdź czy wszystkie wybrane kolumny istnieją i są numeryczne
        for kolumna in kolumny_do_analizy:
            if not pd.api.types.is_numeric_dtype(df[kolumna]):
                print(f"[UWAGA] Kolumna {kolumna} nie jest numeryczna - zostanie pominięta.")
                kolumny_do_analizy.remove(kolumna)

    for kolumna in kolumny_do_analizy:
        wartosci = df[kolumna].dropna().values
        if len(wartosci) > 0:
            wartosci_numeryczne[kolumna] = wartosci

    return wartosci_numeryczne


def analizuj_dane_numeryczne(
        df: pd.DataFrame,
        wybrane_kolumny: Optional[List[str]] = None
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, float]]]:
    """
    Wyodrębnia wartości numeryczne z DataFrame i oblicza podstawowe statystyki.

    Parametry:
    ---------
    df : pd.DataFrame
        DataFrame zawierający dane do analizy
    wybrane_kolumny : Optional[List[str]], opcjonalnie
        Lista kolumn do analizy. Jeśli None, analizuje wszystkie kolumny numeryczne.

    Zwraca:
    ------
    Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, float]]]
        Krotka zawierająca:
        - Słownik z wartościami numerycznymi dla każdej kolumny
        - Słownik ze statystykami dla każdej kolumny
    """
    print("\n[INFO] Rozpoczynam analizę danych numerycznych...")

    if not isinstance(df, pd.DataFrame) or df.empty:
        print("[BŁĄD] Przekazano pusty DataFrame lub obiekt nie jest DataFrame.")
        return {}, {}

    # Znalezienie kolumn numerycznych
    kolumny_numeryczne = znajdz_kolumny_numeryczne(df)
    print(f"\n[INFO] Dostępne kolumny numeryczne: {kolumny_numeryczne}")

    # Wydobycie wartości numerycznych
    wartosci_numeryczne = wydobadz_wartosci_numeryczne(df, wybrane_kolumny)

    # Obliczenie statystyk
    statystyki = oblicz_statystyki(wartosci_numeryczne)

    # Wyświetlenie wyników
    print("\n[WYNIKI] Statystyki dla kolumn numerycznych:")
    for kolumna, stats in statystyki.items():
        print(f"\nKolumna: {kolumna}")
        for nazwa_stat, wartosc in stats.items():
            print(f"  - {nazwa_stat}: {wartosc}")

    return wartosci_numeryczne, statystyki


def analizuj_dane_numeryczne(
        df: pd.DataFrame,
        wybrane_kolumny: Optional[List[str]] = None
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, float]]]:
    """
    Wyodrębnia wartości numeryczne z DataFrame i oblicza podstawowe statystyki.

    Parametry:
    ---------
    df : pd.DataFrame
        DataFrame zawierający dane do analizy
    wybrane_kolumny : Optional[List[str]], opcjonalnie
        Lista kolumn do analizy. Jeśli None, analizuje wszystkie kolumny numeryczne.

    Zwraca:
    ------
    Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, float]]]
        Krotka zawierająca:
        - Słownik z wartościami numerycznymi dla każdej kolumny
        - Słownik ze statystykami dla każdej kolumny
    """
    if 'sciezka_pliku' in df.attrs:
        nazwa_pliku = os.path.basename(df.attrs['sciezka_pliku'])
        print(f"\n[INFO] Rozpoczynam analizę danych numerycznych z pliku {nazwa_pliku}...")
    else:
        print("\n[INFO] Rozpoczynam analizę danych numerycznych...")

    if not isinstance(df, pd.DataFrame) or df.empty:
        print("[BŁĄD] Przekazano pusty DataFrame lub obiekt nie jest DataFrame.")
        return {}, {}

    # Znalezienie kolumn numerycznych
    kolumny_numeryczne = znajdz_kolumny_numeryczne(df)
    print(f"\n[INFO] Dostępne kolumny numeryczne: {kolumny_numeryczne}")

    # Wydobycie wartości numerycznych
    wartosci_numeryczne = wydobadz_wartosci_numeryczne(df, wybrane_kolumny)

    # Obliczenie statystyk
    statystyki = oblicz_statystyki(wartosci_numeryczne)

    # Wyświetlenie wyników
    print("\n[WYNIKI] Statystyki dla kolumn numerycznych:")
    for kolumna, stats in statystyki.items():
        print(f"\nKolumna: {kolumna}")
        for nazwa_stat, wartosc in stats.items():
            print(f"  - {nazwa_stat}: {wartosc}")

    return wartosci_numeryczne, statystyki

def oblicz_statystyki(wartosci_numeryczne: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    """
    Oblicza podstawowe statystyki dla każdej kolumny numerycznej.

    Parametry:
    ---------
    wartosci_numeryczne : Dict[str, np.ndarray]
        Słownik z nazwami kolumn i wartościami numerycznymi

    Zwraca:
    ------
    Dict[str, Dict[str, float]]
        Słownik ze statystykami dla każdej kolumny
    """
    statystyki = {}

    for kolumna, wartosci in wartosci_numeryczne.items():
        statystyki[kolumna] = {
            'średnia': float(np.mean(wartosci)),
            'mediana': float(np.median(wartosci)),
            'min': float(np.min(wartosci)),
            'max': float(np.max(wartosci)),
            'odchylenie_std': float(np.std(wartosci)),
            'liczba_wartości': len(wartosci)
        }

    return statystyki

def srednia_wszystkich_wartosci_numerycznych(wartosci_numeryczne: Dict[str, np.ndarray]) -> float:
    """
    Oblicza średnią wszystkich wartości numerycznych ze wszystkich kolumn.

    Parametry:
    ---------
    wartosci_numeryczne : Dict[str, np.ndarray]
        Słownik z wartościami numerycznymi dla każdej kolumny

    Zwraca:
    ------
    float
        Średnia wszystkich wartości numerycznych
    """
    wszystkie_wartosci = []

    for wartosci in wartosci_numeryczne.values():
        wszystkie_wartosci.extend(wartosci)

    if not wszystkie_wartosci:
        return 0.0

    return float(np.mean(wszystkie_wartosci))


if __name__ == "__main__":
    sciezka_pliku = "online_retail_II.csv"

    # Załaduj dane raz
    df = wczytaj_csv(sciezka_pliku, separator=None, wyswietlaj_informacje=True)

    if df is not None:
        # Analiza danych numerycznych
        wybrane_kolumny = ["Quantity", "Price"]
        wartosci, statystyki = analizuj_dane_numeryczne(sciezka_pliku, wybrane_kolumny=wybrane_kolumny)

        # Oblicz średnią ogólną
        if wartosci:
            srednia_ogolna = srednia_wszystkich_wartosci_numerycznych(wartosci)
            print(f"\n[PODSUMOWANIE] Średnia wszystkich wartości numerycznych: {srednia_ogolna}")

        # Analiza danych nie-numerycznych (przekazujemy załadowany wcześniej DataFrame)
        statystyki_nie_numeryczne = oblicz_statystyki_nie_numeryczne(df)
        print("\n[WYNIKI] Statystyki dla kolumn nie-numerycznych:")
        for kolumna, stats in statystyki_nie_numeryczne.items():
            print(f"\nKolumna: {kolumna}")
            for nazwa_stat, wartosc in stats.items():
                print(f"  - {nazwa_stat}: {wartosc}")
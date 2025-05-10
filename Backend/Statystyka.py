import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Union, Tuple
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Dane.Dane import wczytaj_csv


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


def wydobadz_wartosci_numeryczne(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Wydobywa wszystkie wartości numeryczne z DataFrame.

    Parametry:
    ---------
    df : pd.DataFrame
        DataFrame z danymi

    Zwraca:
    ------
    Dict[str, np.ndarray]
        Słownik z nazwami kolumn jako kluczami i tablicami wartości numerycznych
    """
    wartosci_numeryczne = {}
    kolumny_numeryczne = znajdz_kolumny_numeryczne(df)

    for kolumna in kolumny_numeryczne:
        # Usuwamy wartości NaN i konwertujemy do tablicy numpy
        wartosci = df[kolumna].dropna().values
        if len(wartosci) > 0:
            wartosci_numeryczne[kolumna] = wartosci

    return wartosci_numeryczne


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


def analizuj_dane_numeryczne(sciezka_pliku: str, separator: str = None) -> Tuple[
    Dict[str, np.ndarray], Dict[str, Dict[str, float]]]:
    """
    Wczytuje dane z pliku CSV, wyodrębnia wartości numeryczne i oblicza podstawowe statystyki.

    Parametry:
    ---------
    sciezka_pliku : str
        Ścieżka do pliku CSV
    separator : str, opcjonalnie
        Separator kolumn (jeśli None, zostanie wykryty automatycznie)

    Zwraca:
    ------
    Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, float]]]
        Krotka zawierająca:
        - Słownik z wartościami numerycznymi dla każdej kolumny
        - Słownik ze statystykami dla każdej kolumny
    """
    print(f"\n[INFO] Rozpoczynam analizę danych numerycznych z pliku {os.path.basename(sciezka_pliku)}...")

    # Wczytanie danych z wykorzystaniem istniejącej funkcji
    df = wczytaj_csv(sciezka_pliku, separator=separator, wyswietlaj_informacje=True)

    if df is None:
        print("[BŁĄD] Nie udało się wczytać danych.")
        return {}, {}

    # Znalezienie kolumn numerycznych
    kolumny_numeryczne = znajdz_kolumny_numeryczne(df)
    print(f"\n[INFO] Znaleziono {len(kolumny_numeryczne)} kolumn numerycznych: {kolumny_numeryczne}")

    if not kolumny_numeryczne:
        print("[UWAGA] Nie znaleziono żadnych kolumn numerycznych w danych.")
        return {}, {}

    # Wydobycie wartości numerycznych
    wartosci_numeryczne = wydobadz_wartosci_numeryczne(df)

    # Obliczenie statystyk
    statystyki = oblicz_statystyki(wartosci_numeryczne)

    # Wyświetlenie wyników
    print("\n[WYNIKI] Statystyki dla kolumn numerycznych:")
    for kolumna, stats in statystyki.items():
        print(f"\nKolumna: {kolumna}")
        for nazwa_stat, wartosc in stats.items():
            print(f"  - {nazwa_stat}: {wartosc}")

    return wartosci_numeryczne, statystyki


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

    wartosci, statystyki = analizuj_dane_numeryczne(sciezka_pliku)

    # Usuń niepożądane klucze
    wartosci.pop('InvoiceNo', None)
    wartosci.pop('StockCode', None)
    wartosci.pop("CustomerID", None)

    # Oblicz średnią po usunięciu kluczy
    if wartosci:
        srednia_ogolna = srednia_wszystkich_wartosci_numerycznych(wartosci)
        print(f"\n[PODSUMOWANIE] Średnia wszystkich wartości numerycznych: {srednia_ogolna}")
import pandas as pd
import csv
from typing import Optional

# Wspólna funkcja pomocnicza
def _wczytaj_csv(
        sciezka_pliku: str,
        separator: str,
        separator_dziesietny: str,
        kolumny_daty: list = None,
        kolumny_numeryczne: list = None,
        kolumny_kategorialne: dict = None,
        wymagane_kolumny: list = None,
        kodowanie: str = "utf-8",
        wyswietlaj_informacje: bool = False
) -> Optional[pd.DataFrame]:
    """
    Główna funkcja do wczytywania plików CSV
    """
    try:
        if wyswietlaj_informacje:
            print(f"\n[INFO] Wczytywanie danych z {sciezka_pliku}...")

        # Wczytywanie danych
        df = pd.read_csv(
            sciezka_pliku,
            sep=separator,
            decimal=separator_dziesietny,
            parse_dates=kolumny_daty,
            dayfirst=True,
            encoding=kodowanie,
            on_bad_lines="warn",
            engine="python",
            dtype=kolumny_kategorialne
        )

        # Naprawiony fragment z wyrażeniem regularnym
        df.columns = df.columns.str.strip().str.replace(r'[\'\"\\]', '', regex=True)

        # Konwersja kolumn numerycznych
        if kolumny_numeryczne:
            for kolumna in kolumny_numeryczne:
                df[kolumna] = pd.to_numeric(
                    df[kolumna].astype(str).str.replace(',', '.'),
                    errors="coerce"
                )

        # Weryfikacja wymaganych kolumn
        if wymagane_kolumny:
            brakujace = set(wymagane_kolumny) - set(df.columns)
            if brakujace:
                raise ValueError(f"Brakujące kluczowe kolumny: {brakujace}")

        # Usuwanie brakujących wartości
        if wymagane_kolumny:
            liczba_poczatkowa = len(df)
            df.dropna(subset=wymagane_kolumny, inplace=True)
            if wyswietlaj_informacje:
                print(f"Usunięto {liczba_poczatkowa - len(df)} wierszy z brakującymi danymi")

        if wyswietlaj_informacje:
            print("\n[SUKCES] Dane zostały poprawnie wczytane")
            print(f"Liczba wierszy: {len(df)}")
            print(f"Kolumny: {df.columns.tolist()}")
            print("\nPrzykładowe rekordy:")
            print(df.head(2))

        return df

    except FileNotFoundError:
        print(f"\n[BŁĄD] Plik nie istnieje: {sciezka_pliku}")
        return None
    except Exception as e:
        print(f"\n[BŁĄD] Błąd podczas wczytywania danych: {str(e)}")
        return None

# Funkcje dla danych sklepowych
def wczytaj_dane_sklepu(
        sciezka_pliku: str = "online_retail_II.csv",
        wyswietlaj_informacje: bool = False
) -> Optional[pd.DataFrame]:
    return _wczytaj_csv(
        sciezka_pliku=sciezka_pliku,
        separator=";",
        separator_dziesietny=",",
        kolumny_daty=["InvoiceDate"],
        kolumny_numeryczne=["Price", "Quantity"],
        kolumny_kategorialne={
            "Invoice": str,
            "StockCode": str,
            "Customer ID": str
        },
        wymagane_kolumny=["InvoiceDate", "Price"],
        kodowanie="utf-8-sig",
        wyswietlaj_informacje=wyswietlaj_informacje
    )

# Funkcje dla danych szkolnych
def wczytaj_dane_szkolne(
        przedmiot: str = "math",
        sciezka_pliku: str = None,
        wyswietlaj_informacje: bool = False
) -> Optional[pd.DataFrame]:
    if not sciezka_pliku:
        sciezka_pliku = f"student-{przedmiot}.csv"

    kolumny_numeryczne = ["G1", "G2", "G3", "age", "Medu", "Fedu"]
    if przedmiot == "por":
        kolumny_numeryczne.append("absences")

    return _wczytaj_csv(
        sciezka_pliku=sciezka_pliku,
        separator=";",
        separator_dziesietny=".",
        kolumny_kategorialne={
            "school": "category",
            "sex": "category",
            "address": "category",
            "famsize": "category",
            "Pstatus": "category",
            "Mjob": "category",
            "Fjob": "category",
            "reason": "category",
            "guardian": "category",
            "paid": "category"
        },
        kolumny_numeryczne=kolumny_numeryczne,
        wymagane_kolumny=["G3"],
        kodowanie="utf-8-sig",
        wyswietlaj_informacje=wyswietlaj_informacje
    )
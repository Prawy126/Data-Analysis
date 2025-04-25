import pandas as pd
import csv
from typing import Optional

# TODO: zmienić ścieżkę do pliku na względną
def load_retail_data(
        file_path: str = r"C:\Users\Lenovo\Desktop\nauka\HurtownieDanych\online_retail_II.csv",
        encoding: str = "utf-8-sig",
        verbose: bool = False
) -> Optional[pd.DataFrame]:
    """
    Wczytuje i przetwarza dane sprzedażowe z pliku CSV.

    Parametry:
    file_path (str): Ścieżka do pliku CSV
    encoding (str): Kodowanie pliku (domyślnie 'utf-8-sig')
    verbose (bool): Czy wyświetlać szczegółowe informacje (domyślnie False)

    Zwraca:
    pd.DataFrame: DataFrame z przetworzonymi danymi lub None w przypadku błędu
    """
    try:
        if verbose:
            print("\n[INFO] Rozpoczynanie wczytywania danych...")

        # Wykrywanie dialektu CSV
        with open(file_path, "r", encoding=encoding) as f:
            dialect = csv.Sniffer().sniff(f.read(1024))

        # Wczytywanie danych
        df = pd.read_csv(
            file_path,
            sep=";",
            decimal=",",
            parse_dates=["InvoiceDate"],
            dayfirst=True,
            encoding=encoding,
            on_bad_lines="warn",
            dtype={
                "Invoice": str,
                "StockCode": str,
                "Customer ID": str
            },
            engine="python"
        )

        # Czyszczenie nazw kolumn
        df.columns = df.columns.str.strip()

        # Podstawowe czyszczenie danych
        df.dropna(subset=["InvoiceDate", "Price"], inplace=True)

        # Konwersja typów
        df["Customer ID"] = df["Customer ID"].astype(str)
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

        if verbose:
            print("\n[SUKCES] Dane zostały poprawnie wczytane")
            print(f"Liczba wierszy: {len(df)}")
            print(f"Kolumny: {df.columns.tolist()}")
            print("\nPrzykładowe rekordy:")
            print(df.head(2))

        return df

    except FileNotFoundError:
        print(f"\n[BŁĄD] Plik nie istnieje: {file_path}")
        return None
    except Exception as e:
        print(f"\n[BŁĄD] Problem podczas wczytywania danych: {str(e)}")
        return None

# TODO: zmienić siecżkę na względną
def load_student_math_data(
        file_path: str = r"C:\Users\Lenovo\Desktop\nauka\HurtownieDanych\student-mat.csv",
        encoding: str = "utf-8-sig",
        verbose: bool = False
) -> Optional[pd.DataFrame]:
    """
    Wczytuje i przetwarza dane uczniów z pliku CSV.

    Parametry:
    file_path (str): Ścieżka do pliku CSV
    encoding (str): Kodowanie pliku (domyślnie 'utf-8-sig')
    verbose (bool): Czy wyświetlać szczegółowe informacje (domyślnie False)

    Zwraca:
    pd.DataFrame: DataFrame z przetworzonymi danymi lub None w przypadku błędu
    """
    try:
        if verbose:
            print("\n[INFO] Rozpoczynanie wczytywania danych...")

        # Wykrywanie dialektu CSV
        with open(file_path, "r", encoding=encoding) as f:
            sample = f.read(1024)
            dialect = csv.Sniffer().sniff(sample)
            has_header = csv.Sniffer().has_header(sample)

        # Wczytywanie danych
        df = pd.read_csv(
            file_path,
            sep=";",
            decimal=".",  # Kropka jako separator dziesiętny
            quotechar='"',  # Obsługa cudzysłowów w danych
            dtype={
                "school": "category",
                "sex": "category",
                "address": "category",
                "famsize": "category",
                "Pstatus": "category",
                "Mjob": "category",
                "Fjob": "category",
                "reason": "category",
                "guardian": "category",
                "paid": "category"  # Przykładowa konwersja kolumn kategorycznych
            },
            encoding=encoding,
            on_bad_lines="warn",
            engine="python"
        )

        # Czyszczenie nazw kolumn (usuwanie białych znaków i cudzysłowów)
        df.columns = df.columns.str.strip().str.replace('"', '')

        # Konwersja kolumn numerycznych (G1, G2, G3 mogą mieć błędne formaty)
        numeric_cols = ["G1", "G2", "G3", "age", "Medu", "Fedu"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace('"', ''), errors="coerce")

        # Usuwanie wierszy z brakującymi ocenami (G3 jest kluczowe)
        initial_count = len(df)
        df.dropna(subset=["G3"], inplace=True)

        if verbose:
            print("\n[SUKCES] Dane zostały poprawnie wczytane")
            print(f"Liczba wierszy: {len(df)} (usunięto {initial_count - len(df)} rekordów)")
            print(f"Kolumny: {df.columns.tolist()}")
            print("\nPrzykładowe rekordy:")
            print(df.head(2))

        return df

    except FileNotFoundError:
        print(f"\n[BŁĄD] Plik nie istnieje: {file_path}")
        return None
    except Exception as e:
        print(f"\n[BŁĄD] Problem podczas wczytywania danych: {str(e)}")
        return None

# TODO: Zmienić ścieżkę do pliku na względną
def load_student_por_data(
        file_path: str = r"C:\Users\Lenovo\Desktop\nauka\HurtownieDanych\student-por.csv",
        encoding: str = "utf-8-sig",
        verbose: bool = False
) -> Optional[pd.DataFrame]:
    """
    Wczytuje i przetwarza dane uczniów z przedmiotu portugalski z pliku CSV.

    Parametry:
    file_path (str): Ścieżka do pliku CSV
    encoding (str): Kodowanie pliku (domyślnie 'utf-8-sig')
    verbose (bool): Czy wyświetlać szczegółowe informacje (domyślnie False)

    Zwraca:
    pd.DataFrame: DataFrame z przetworzonymi danymi lub None w przypadku błędu
    """
    try:
        if verbose:
            print("\n[INFO] Rozpoczynanie wczytywania danych...")

        # Wykrywanie dialektu CSV
        with open(file_path, "r", encoding=encoding) as f:
            sample = f.read(1024)
            dialect = csv.Sniffer().sniff(sample)
            has_header = csv.Sniffer().has_header(sample)

        # Wczytywanie danych
        df = pd.read_csv(
            file_path,
            sep=";",
            decimal=".",
            quotechar='"',
            dtype={
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
            encoding=encoding,
            on_bad_lines="warn",
            engine="python"
        )

        # Czyszczenie nazw kolumn
        df.columns = df.columns.str.strip().str.replace('"', '')

        # Konwersja kolumn numerycznych
        numeric_cols = ["G1", "G2", "G3", "age", "Medu", "Fedu", "absences"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace('"', '').str.replace(',', '.'),  # Dodatkowe zabezpieczenie
                errors="coerce"
            )

        # Usuwanie wierszy z brakującymi ocenami
        initial_count = len(df)
        df.dropna(subset=["G3"], inplace=True)

        if verbose:
            print("\n[SUKCES] Dane zostały poprawnie wczytane")
            print(f"Liczba wierszy: {len(df)} (usunięto {initial_count - len(df)} rekordów)")
            print(f"Kolumny: {df.columns.tolist()}")
            print("\nPrzykładowe rekordy:")
            print(df.head(2))

        return df

    except FileNotFoundError:
        print(f"\n[BŁĄD] Plik nie istnieje: {file_path}")
        return None
    except Exception as e:
        print(f"\n[BŁĄD] Problem podczas wczytywania danych: {str(e)}")
        return None
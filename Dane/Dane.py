import pandas as pd
from typing import Optional, List, Union
import re
import os
from datetime import datetime


def wczytaj_csv(
        sciezka_pliku: str,
        separator: Union[str, List[str]] = None,
        kolumny_daty: List[str] = None,
        format_daty: str = None,
        wymagane_kolumny: List[str] = None,
        wyswietlaj_informacje: bool = False
) -> Optional[pd.DataFrame]:
    """
    Ulepszona funkcja do automatycznego wczytywania CSV.
    """
    if wyswietlaj_informacje:
        print(f"[INFO] Wczytywanie {os.path.basename(sciezka_pliku)}...")

    if not os.path.exists(sciezka_pliku):
        raise FileNotFoundError(f"Plik nie istnieje: {sciezka_pliku}")

    # 1) detekcja kodowania i separatora (pomijam tu implementację _wykryj_kodowanie/_wykryj_separator)
    kodowanie = _wykryj_kodowanie(sciezka_pliku)
    if separator is None:
        separator = _wykryj_separator(sciezka_pliku, kodowanie)

    if wyswietlaj_informacje:
        print(f"Kodowanie: {kodowanie}, Separator: '{separator}'")

    filesize = os.path.getsize(sciezka_pliku)

    # 2) czytanie pliku
    def _read(engine, on_bad):
        return pd.read_csv(
            sciezka_pliku,
            sep=separator,
            encoding=kodowanie,
            engine=engine,
            on_bad_lines=on_bad
        )

    if filesize > 100 * 1024**2:
        # chunking
        chunks = []
        for eng, bad in [('c', 'warn'), ('python', 'skip')]:
            try:
                reader = pd.read_csv(
                    sciezka_pliku, sep=separator, encoding=kodowanie,
                    engine=eng, on_bad_lines=bad, chunksize=100_000
                )
                for c in reader:
                    c = _automatyczna_detekcja_typow(c, kolumny_daty, format_daty, wyswietlaj_informacje)
                    c = _optymalizuj_pamiec(c)
                    chunks.append(c)
                break
            except Exception:
                continue
        df = pd.concat(chunks, ignore_index=True)
    else:
        # całość na raz
        for eng, bad in [('c', 'warn'), ('python', 'skip')]:
            try:
                df = _read(eng, bad)
                break
            except Exception:
                continue

        # nagłówki, wymagane kolumny, typy i pamięć
        df.columns = [col.strip().replace('\ufeff', '') for col in df.columns]
        if wymagane_kolumny:
            brak = set(wymagane_kolumny) - set(df.columns)
            if brak:
                raise ValueError(f"Brak wymaganych kolumn: {brak}")
            df = df.dropna(subset=wymagane_kolumny)

        df = _automatyczna_detekcja_typow(df, kolumny_daty, format_daty, wyswietlaj_informacje)
        df = _optymalizuj_pamiec(df)

    if wyswietlaj_informacje:
        print(f"[SUKCES] Wczytano {len(df)} wierszy x {df.shape[1]} kolumn")

    return df

def _automatyczna_detekcja_typow(df: pd.DataFrame,
                                 kolumny_daty: List[str] = None,
                                 format_daty: str = None,
                                 wyswietlaj: bool = False) -> pd.DataFrame:
    """
    Ulepszona automatyczna detekcja typów z próbkowaniem.
    """
    df_copy = df.copy()
    sample = df_copy.sample(n=min(len(df_copy), 1000), random_state=42) if len(df_copy) > 0 else df_copy

    # Najpierw konwertuj podane kolumny dat
    if kolumny_daty:
        for col in kolumny_daty:
            if col in df_copy.columns:
                try:
                    fmt = format_daty or _wykryj_format_daty(sample[col])
                    df_copy[col] = pd.to_datetime(df_copy[col], format=fmt, errors='coerce')
                    if wyswietlaj:
                        print(f"Data [{col}] użyty format: {fmt}")
                except Exception as e:
                    if wyswietlaj:
                        print(f"Błąd daty {col}: {e}")

    for col in df_copy.columns:
        if kolumny_daty and col in kolumny_daty:
            continue
        s = sample[col].dropna().astype(str)
        # Numeryczne
        if _czy_kolumna_numeryczna(s):
            sep = _wykryj_separator_dziesietny(s)
            df_copy[col] = _konwertuj_na_liczbe(df_copy[col], sep)
            if wyswietlaj:
                print(f"Numeryczna: {col}")
            continue
        # Daty
        if _czy_kolumna_zawiera_daty(s):
            fmt = _wykryj_format_daty(s)
            df_copy[col] = pd.to_datetime(df_copy[col], format=fmt, errors='coerce')
            if wyswietlaj:
                print(f"Data wykryta: {col} (format: {fmt})")
            continue
        # Kategorie
        if _czy_kolumna_kategorialna(s):
            df_copy[col] = df_copy[col].astype('category')
            if wyswietlaj:
                print(f"Kategoria: {col}")
    return df_copy




def _wykryj_kodowanie(sciezka_pliku: str) -> str:
    """Automatyczne wykrywanie kodowania pliku"""
    try:
        # Próba importu modułu chardet
        try:
            import chardet
            with open(sciezka_pliku, 'rb') as f:
                raw_data = f.read(10000)
            result = chardet.detect(raw_data)
            return result['encoding'] if result['encoding'] else 'utf-8'
        except ImportError:
            # Jeśli nie ma modułu chardet, próbujemy bez niego
            encodings = ['utf-8', 'latin-1', 'cp1250', 'windows-1250']
            for enc in encodings:
                try:
                    with open(sciezka_pliku, 'r', encoding=enc) as f:
                        f.read(1000)
                    return enc
                except UnicodeDecodeError:
                    continue
            return 'utf-8'
    except Exception:
        return 'utf-8'


def _wykryj_separator(sciezka_pliku: str, kodowanie: str = 'utf-8') -> str:
    """Ulepszone automatyczne wykrywanie separatora w CSV"""
    try:
        with open(sciezka_pliku, 'r', encoding=kodowanie, errors='replace') as f:
            # Czytamy więcej linii dla lepszej detekcji
            pierwsze_linie = [f.readline() for _ in range(5)]

        separatory = {',': 0, ';': 0, '\t': 0, '|': 0}

        for linia in pierwsze_linie:
            if not linia.strip():
                continue

            for sep, count in separatory.items():
                # Liczymy separatory tylko jeśli są otoczone danymi
                if sep == '\t':
                    # Specjalna obsługa dla tabulatorów
                    separatory[sep] += linia.count(sep)
                else:
                    # Dla innych separatorów liczymy tylko te, które rozdzielają dane
                    potencjalne_pola = linia.split(sep)
                    if len(potencjalne_pola) > 1:
                        separatory[sep] += len(potencjalne_pola) - 1

        # Wybierz separator z najwyższą liczbą wystąpień
        najlepszy_separator = max(separatory.items(), key=lambda x: x[1])

        # Jeśli żaden separator nie został znaleziony, użyj przecinka
        if najlepszy_separator[1] == 0:
            return ','

        return najlepszy_separator[0]
    except Exception as e:
        print(f"Błąd wykrywania separatora: {str(e)}")
        return ","  # Domyślnie przecinek w przypadku problemu


def _wykryj_format_daty(sample: pd.Series) -> str:
    """Wykrywanie formatu daty na podstawie próbki danych"""
    # Lista popularnych formatów dat do przetestowania
    formaty = [
        "%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y",
        "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y",
        "%Y.%m.%d", "%d.%m.%Y", "%m.%d.%Y",
        "%Y-%m-%d %H:%M:%S", "%d-%m-%Y %H:%M:%S",
        "%Y/%m/%d %H:%M:%S", "%d/%m/%Y %H:%M:%S",
        "%d.%m.%Y %H:%M:%S", "%Y.%m.%d %H:%M:%S",
        "%H:%M:%S %d-%m-%Y", "%H:%M:%S %Y-%m-%d"
    ]

    # Pobierz kilka przykładowych wartości jako string
    sample_vals = sample.dropna().astype(str).head(10).tolist()

    # Sprawdź każdy format dla każdej wartości próbki
    wyniki = {}
    for format_daty in formaty:
        poprawnych = 0
        for val in sample_vals:
            try:
                datetime.strptime(val.strip(), format_daty)
                poprawnych += 1
            except:
                pass

        if poprawnych > 0:
            wyniki[format_daty] = poprawnych

    # Wybierz format z największą liczbą poprawnych konwersji
    if wyniki:
        najlepszy_format = max(wyniki.items(), key=lambda x: x[1])[0]
        return najlepszy_format

    # Jeśli nie znaleziono żadnego dopasowania, zwróć None
    return None


def _automatyczna_detekcja_typow(df: pd.DataFrame, kolumny_daty: List[str] = None,
                                 format_daty: str = None, wyswietlaj: bool = False) -> pd.DataFrame:
    """Ulepszona automatyczna detekcja typów danych dla wszystkich kolumn"""
    # Utwórz kopię DataFrame, aby uniknąć ostrzeżeń o modyfikacji podczas iteracji
    df_copy = df.copy()

    # Najpierw przetwarzamy kolumny dat (jeśli są określone)
    if kolumny_daty:
        for kolumna in kolumny_daty:
            if kolumna in df_copy.columns:
                try:
                    # Jeżeli format daty jest podany, używamy go
                    if format_daty:
                        df_copy[kolumna] = pd.to_datetime(df_copy[kolumna], format=format_daty, errors='coerce')
                    else:
                        # Wykrywamy format daty na podstawie danych
                        wykryty_format = _wykryj_format_daty(df_copy[kolumna])
                        if wykryty_format:
                            df_copy[kolumna] = pd.to_datetime(df_copy[kolumna], format=wykryty_format, errors='coerce')
                        else:
                            df_copy[kolumna] = pd.to_datetime(df_copy[kolumna], errors='coerce')

                    if wyswietlaj:
                        print(f"Konwersja na datę (z listy): {kolumna}")
                except Exception as e:
                    if wyswietlaj:
                        print(f"Błąd konwersji daty {kolumna}: {str(e)}")

    # Przetwarzanie pozostałych kolumn
    for kolumna in df_copy.columns:
        # Pomijamy już przetworzone daty
        if kolumny_daty and kolumna in kolumny_daty:
            continue

        # Pomijamy puste kolumny
        if df_copy[kolumna].isna().all():
            if wyswietlaj:
                print(f"Pominięto pustą kolumnę: {kolumna}")
            continue

        try:
            # Próbujemy wykryć liczby (przed datami!)
            if _czy_kolumna_numeryczna(df_copy[kolumna]):
                separator_dziesietny = _wykryj_separator_dziesietny(df_copy[kolumna])
                try:
                    df_copy[kolumna] = _konwertuj_na_liczbe(df_copy[kolumna], separator_dziesietny)
                    if wyswietlaj:
                        print(f"Wykryto kolumnę numeryczną: {kolumna}")
                    continue
                except Exception as e:
                    if wyswietlaj:
                        print(f"Błąd konwersji liczby {kolumna}: {str(e)}")

            # Wykrywanie dat (tylko jeśli nie są liczbami)
            if _czy_kolumna_zawiera_daty(df_copy[kolumna]):
                try:
                    # Wykrywamy format daty i używamy go
                    wykryty_format = _wykryj_format_daty(df_copy[kolumna])
                    if wykryty_format:
                        df_copy[kolumna] = pd.to_datetime(df_copy[kolumna], format=wykryty_format, errors='coerce')
                    else:
                        df_copy[kolumna] = pd.to_datetime(df_copy[kolumna], errors='coerce')

                    if wyswietlaj:
                        print(f"Wykryto kolumnę daty: {kolumna}")
                    continue
                except Exception as e:
                    if wyswietlaj:
                        print(f"Błąd konwersji daty {kolumna}: {str(e)}")

            # Wykrywanie kategorii (jeśli nie są ani liczbami, ani datami)
            if _czy_kolumna_kategorialna(df_copy[kolumna]):
                df_copy[kolumna] = df_copy[kolumna].astype('category')
                if wyswietlaj:
                    print(f"Wykryto kolumnę kategoryczną: {kolumna}")
        except Exception as e:
            if wyswietlaj:
                print(f"Błąd podczas przetwarzania kolumny {kolumna}: {str(e)}")

    return df_copy


def _czy_kolumna_numeryczna(kolumna: pd.Series) -> bool:
    """Ulepszone sprawdzanie czy kolumna zawiera liczby"""
    # Jeśli już jest typem numerycznym
    if pd.api.types.is_numeric_dtype(kolumna):
        return True

    # Pobieramy próbkę niepustych wartości
    sample = kolumna.dropna().astype(str).head(20)
    if len(sample) < 3:
        return False

    # Usuwamy białe znaki i zamieniamy przecinki na kropki
    sample_clean = sample.str.strip().str.replace(',', '.')

    # Sprawdzamy wzór liczby (dopuszczamy liczby ujemne i zmiennoprzecinkowe)
    pattern = r'^-?\d+(\.\d+)?$'
    matches = sample_clean.str.match(pattern)

    # Jeśli większość wartości pasuje do wzorca liczby
    return matches.mean() > 0.7


def _czy_kolumna_zawiera_daty(kolumna: pd.Series) -> bool:
    """Ulepszone sprawdzanie czy kolumna zawiera daty"""
    # Jeśli kolumna jest już datą
    if pd.api.types.is_datetime64_any_dtype(kolumna):
        return True

    # Konwertujemy na string dla bezpieczeństwa
    sample = kolumna.dropna().astype(str).head(10)
    if len(sample) < 3:
        return False

    # Sprawdzamy, czy możemy znaleźć format daty
    wykryty_format = _wykryj_format_daty(sample)
    if wykryty_format:
        return True

    # Jeśli nie udało się wykryć formatu, próbujemy bardziej ogólnej metody
    try:
        # Najpierw próbujemy z formatem 'mixed' dla lepszej wydajności
        converted = pd.to_datetime(sample, format='mixed', errors='coerce')
        if converted.notna().mean() > 0.7:
            return True

        # Jeśli format 'mixed' nie zadziałał, próbujemy bez określonego formatu
        converted = pd.to_datetime(sample, errors='coerce')
        return converted.notna().mean() > 0.7

    except Exception:
        return False


def _wykryj_separator_dziesietny(kolumna: pd.Series) -> str:
    """Ulepszone wykrywanie separatora dziesiętnego"""
    sample = kolumna.dropna().astype(str).head(20)

    # Liczymy wystąpienia kropek i przecinków w pozycji, która może być separatorem dziesiętnym
    kropki = 0
    przecinki = 0

    for val in sample:
        val = str(val).strip()
        if re.search(r'\d+\.\d+', val):
            kropki += 1
        if re.search(r'\d+,\d+', val):
            przecinki += 1

    # Wybieramy częściej występujący separator
    if kropki >= przecinki:
        return '.'
    else:
        return ','


def _konwertuj_na_liczbe(kolumna: pd.Series, separator: str) -> pd.Series:
    """Ulepszona konwersja na liczby"""
    try:
        # Usuwamy białe znaki
        cleaned = kolumna.astype(str).str.strip()

        # Zamieniamy separator na kropkę, jeśli jest inny
        if separator == ',':
            cleaned = cleaned.str.replace(',', '.')

        # Konwertujemy na liczby
        return pd.to_numeric(cleaned, errors='coerce')
    except Exception as e:
        print(f"Błąd konwersji na liczbę: {str(e)}")
        # W przypadku błędu, zwracamy oryginalną kolumnę
        return kolumna


def _czy_kolumna_kategorialna(kolumna: pd.Series) -> bool:
    """Ulepszone sprawdzanie czy kolumna powinna być kategoryczna"""
    # Jeśli już jest typem kategorycznym
    if pd.api.types.is_categorical_dtype(kolumna):
        return True

    # Jeśli jest tekstem, sprawdzamy unikalność
    if pd.api.types.is_object_dtype(kolumna):
        try:
            unikalne = kolumna.nunique()
            całkowita_liczba = len(kolumna)
            # Zabezpieczenie przed dzieleniem przez zero
            if całkowita_liczba > 0:
                # Mniej niż 50% unikalnych wartości i mniej niż 100 unikalnych wartości
                return (unikalne / całkowita_liczba < 0.5) and unikalne < 100
        except:
            pass
    return False


def _optymalizuj_pamiec(df: pd.DataFrame) -> pd.DataFrame:
    # Liczby zmiennoprzecinkowe
    for col in df.select_dtypes(include=['float64']).columns:
        try:
            df[col] = pd.to_numeric(df[col], downcast='float')
        except:
            pass
    # Liczby całkowite
    for col in df.select_dtypes(include=['int64']).columns:
        try:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        except:
            pass
    # Kategorie
    for col in df.select_dtypes(include=['object']).columns:
        try:
            if df[col].nunique() / df.shape[0] < 0.5 and df[col].nunique() < 100:
                df[col] = df[col].astype('category')
        except:
            pass
    return df



# Funkcja pomocnicza do analizy pliku CSV
def analizuj_csv(sciezka_pliku: str) -> None:
    """
    Analizuje strukturę pliku CSV i wyświetla informacje o zawartości

    Parametry:
    ---------
    sciezka_pliku : str
        Ścieżka do pliku CSV
    """
    try:
        print(f"\n[INFO] Analiza pliku {os.path.basename(sciezka_pliku)}...")

        # Sprawdzenie, czy plik istnieje
        if not os.path.exists(sciezka_pliku):
            raise FileNotFoundError(f"Plik nie istnieje: {sciezka_pliku}")

        # Wykrywanie kodowania
        kodowanie = _wykryj_kodowanie(sciezka_pliku)
        print(f"Wykryte kodowanie: {kodowanie}")

        # Testowanie różnych separatorów
        separatory = [',', ';', '\t', '|']
        wyniki = {}

        for sep in separatory:
            try:
                df_test = pd.read_csv(sciezka_pliku, sep=sep, encoding=kodowanie,
                                      nrows=5, engine='python')
                wyniki[sep] = len(df_test.columns)
            except:
                wyniki[sep] = 0

        najlepszy_sep = max(wyniki.items(), key=lambda x: x[1])[0]
        print(f"Testowane separatory: {wyniki}")
        print(f"Zalecany separator: '{najlepszy_sep}'")

        # Próbka danych
        try:
            df_sample = pd.read_csv(sciezka_pliku, sep=najlepszy_sep,
                                    encoding=kodowanie, nrows=5, engine='python')
            print("\nPróbka danych (pierwsze 5 wierszy):")
            print(df_sample)

            print("\nSugerowane typy danych:")
            for kolumna in df_sample.columns:
                sample = df_sample[kolumna].dropna().astype(str)

                if _czy_kolumna_numeryczna(sample):
                    print(f"  - {kolumna}: numeryczna")
                elif _czy_kolumna_zawiera_daty(sample):
                    # Sprawdź, czy możemy określić format daty
                    format_daty = _wykryj_format_daty(sample)
                    if format_daty:
                        print(f"  - {kolumna}: data (format: {format_daty})")
                    else:
                        print(f"  - {kolumna}: data (nieznany format)")
                elif _czy_kolumna_kategorialna(sample):
                    print(f"  - {kolumna}: kategorialna")
                else:
                    print(f"  - {kolumna}: tekst")
        except Exception as e:
            print(f"Błąd podczas analizy próbki: {str(e)}")

    except Exception as e:
        print(f"[BŁĄD] Nie udało się przeanalizować pliku: {str(e)}")


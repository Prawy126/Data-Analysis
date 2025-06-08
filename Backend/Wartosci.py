from typing import Union, Dict, Optional, Any

import pandas as pd

from Dane.Dane import _optymalizuj_pamiec, wczytaj_csv


def zamien_wartosci(df: pd.DataFrame, reguly: Dict[str, Dict[Any, Any]] = None,
                    wyswietlaj_informacje: bool = False) -> pd.DataFrame:
    """
    Zamienia wartości w DataFrame według podanych reguł.
    Uproszczona i bardziej niezawodna wersja.

    Parametry:
    ---------
    df : pd.DataFrame
        DataFrame do modyfikacji
    reguly : Dict[str, Dict[Any, Any]], opcjonalne
        Słownik reguł zamiany, gdzie kluczem głównym jest nazwa kolumny,
        a wartością słownik {stara_wartosc: nowa_wartosc}
    wyswietlaj_informacje : bool, opcjonalne
        Czy wyświetlać informacje o liczbie zamienionych wartości

    Zwraca:
    -------
    pd.DataFrame
        Zmodyfikowany DataFrame
    """
    if reguly is None or not reguly:
        if wyswietlaj_informacje:
            print("Brak reguł zamiany.")
        return df

    df_wynik = df.copy()
    licznik_zmian = 0

    for kolumna, zamiana in reguly.items():
        if kolumna not in df_wynik.columns:
            if wyswietlaj_informacje:
                print(f"Kolumna '{kolumna}' nie istnieje w danych.")
            continue

        # Sprawdź czy kolumna jest typu kategorycznego
        is_categorical = isinstance(df_wynik[kolumna].dtype, pd.CategoricalDtype)

        # Dla danych kategorycznych, potrzebujemy dodać nowe kategorie
        if is_categorical:
            current_categories = df_wynik[kolumna].cat.categories.tolist()
            new_categories = []
            for _, nowa_wartosc in zamiana.items():
                if nowa_wartosc not in current_categories and nowa_wartosc not in new_categories:
                    new_categories.append(nowa_wartosc)

            if new_categories:
                df_wynik[kolumna] = df_wynik[kolumna].cat.add_categories(new_categories)

        # PRZETWÓRZ REGUŁY ZAMIANY
        for stara_wartosc, nowa_wartosc in zamiana.items():
            # Zapisz oryginalny typ kolumny
            col_dtype = df_wynik[kolumna].dtype
            is_numeric = pd.api.types.is_numeric_dtype(col_dtype)

            # Obsługa NaN
            if pd.isna(stara_wartosc) or (isinstance(stara_wartosc, str) and stara_wartosc.lower() == "nan"):
                mask = df_wynik[kolumna].isna()
                ile_zmian = mask.sum()
                if ile_zmian > 0:
                    df_wynik.loc[mask, kolumna] = nowa_wartosc
                    licznik_zmian += ile_zmian
                    if wyswietlaj_informacje:
                        print(f"Zamieniono {ile_zmian} wartości NaN na '{nowa_wartosc}' w kolumnie '{kolumna}'")
                else:
                    if wyswietlaj_informacje:
                        print(f"Nie znaleziono wartości NaN w kolumnie '{kolumna}'")
                continue

            # Dla wartości liczbowych
            if is_numeric:
                try:
                    # Konwertuj stara_wartosc do odpowiedniego typu liczbowego
                    if isinstance(stara_wartosc, str):
                        if '.' in stara_wartosc:
                            stara_wartosc_num = float(stara_wartosc)
                        else:
                            stara_wartosc_num = int(stara_wartosc)
                    else:
                        stara_wartosc_num = stara_wartosc

                    # Użyj alternatywnej metody zamiany wartości liczbowych:
                    # zamiast maski i loc, użyj replace
                    przed_zmiana = df_wynik[kolumna].copy()
                    df_wynik[kolumna] = df_wynik[kolumna].replace(stara_wartosc_num, nowa_wartosc)

                    # Sprawdź ile wartości zostało zmienionych
                    ile_zmian = (df_wynik[kolumna] != przed_zmiana).sum()
                    licznik_zmian += ile_zmian

                    if wyswietlaj_informacje:
                        if ile_zmian > 0:
                            print(
                                f"Zamieniono {ile_zmian} wystąpień '{stara_wartosc}' na '{nowa_wartosc}' w kolumnie '{kolumna}'")
                        else:
                            print(f"Nie znaleziono wartości '{stara_wartosc}' w kolumnie '{kolumna}'")

                except Exception as e:
                    if wyswietlaj_informacje:
                        print(
                            f"Błąd przy zamianie wartości liczbowej '{stara_wartosc}' w kolumnie '{kolumna}': {str(e)}")
            else:
                # Dla wartości nieliczbowych
                try:
                    # Użyj standardowej metody replace
                    przed_zmiana = df_wynik[kolumna].copy()
                    df_wynik[kolumna] = df_wynik[kolumna].replace(stara_wartosc, nowa_wartosc)

                    # Sprawdź ile wartości zostało zmienionych
                    ile_zmian = (df_wynik[kolumna] != przed_zmiana).sum()
                    licznik_zmian += ile_zmian

                    if wyswietlaj_informacje:
                        if ile_zmian > 0:
                            print(
                                f"Zamieniono {ile_zmian} wystąpień '{stara_wartosc}' na '{nowa_wartosc}' w kolumnie '{kolumna}'")
                        else:
                            print(f"Nie znaleziono wartości '{stara_wartosc}' w kolumnie '{kolumna}'")

                except Exception as e:
                    if wyswietlaj_informacje:
                        print(f"Błąd przy zamianie wartości '{stara_wartosc}' w kolumnie '{kolumna}': {str(e)}")

    if wyswietlaj_informacje:
        print(f"Łącznie zamieniono {licznik_zmian} wartości.")

    return df_wynik
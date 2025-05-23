import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Union, Any

from Dane.Dane import _optymalizuj_pamiec, wczytaj_csv


def minmax_scaler(
        df: pd.DataFrame,
        kolumny: Optional[List[str]] = None,
        wyswietlaj_informacje: bool = True,
        zwroc_tylko_dane: bool = False
) -> Union[pd.DataFrame, Dict[str, Any]]:
    """
    Skaluje wybrane kolumny numeryczne do zakresu [0, 1] metodą MinMax.

    Parametry:
    ---------
    df : pd.DataFrame
        Wejściowy DataFrame z danymi.
    kolumny : List[str], opcjonalnie
        Lista kolumn do skalowania. Jeśli None, skalowane są wszystkie kolumny numeryczne.
    wyswietlaj_informacje : bool
        Czy wyświetlać informacje diagnostyczne.
    zwroc_tylko_dane : bool
        Jeśli True, zwraca tylko przeskalowany DataFrame zamiast słownika z dodatkowymi informacjami.

    Zwraca:
    -------
    Union[pd.DataFrame, Dict[str, Any]]
        Jeśli zwroc_tylko_dane=True: przeskalowany DataFrame
        W przeciwnym razie: słownik z kluczami:
        - 'df_scaled': przeskalowany DataFrame
        - 'skale': informacje o min/max użyte do skalowania {'kolumna': (min, max)}
    """
    try:
        wynik_df = df.copy()

        # Wybór kolumn do skalowania
        if kolumny is None:
            kolumny_numeryczne = wynik_df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            kolumny_numeryczne = [col for col in kolumny if col in wynik_df.columns and
                                  pd.api.types.is_numeric_dtype(wynik_df[col])]

        if not kolumny_numeryczne:
            raise ValueError("Brak kolumn numerycznych do skalowania.")

        skale = {}

        for kolumna in kolumny_numeryczne:
            min_val = wynik_df[kolumna].min()
            max_val = wynik_df[kolumna].max()

            if min_val == max_val:
                if wyswietlaj_informacje:
                    print(f"[UWAGA] Kolumna '{kolumna}' ma stałą wartość - zostanie pominięta.")
                continue

            wynik_df[kolumna] = (wynik_df[kolumna] - min_val) / (max_val - min_val)
            skale[kolumna] = (min_val, max_val)

            if wyswietlaj_informacje:
                print(f"[INFO] Skalowano kolumnę '{kolumna}' do zakresu [0, 1]")
                print(f"  Min: {min_val:.4f}, Max: {max_val:.4f}")

        # Optymalizacja pamięci
        wynik_df = _optymalizuj_pamiec(wynik_df)

        # Zwróć tylko DataFrame lub pełny słownik z informacjami
        if zwroc_tylko_dane:
            return wynik_df
        else:
            return {
                'df_scaled': wynik_df,
                'skale': skale
            }

    except Exception as e:
        print(f"[BŁĄD] Błąd podczas skalowania: {str(e)}")
        import traceback
        print(traceback.format_exc())
        if zwroc_tylko_dane:
            return df
        else:
            return {'df_scaled': df, 'skale': {}}


def standard_scaler(
        df: pd.DataFrame,
        kolumny: Optional[List[str]] = None,
        wyswietlaj_informacje: bool = True,
        zwroc_tylko_dane: bool = False
) -> Union[pd.DataFrame, Dict[str, Any]]:
    """
    Standaryzuje wybrane kolumny numeryczne do średniej 0 i wariancji 1 metodą StandardScaler.

    Parametry:
    ---------
    df : pd.DataFrame
        Wejściowy DataFrame z danymi.
    kolumny : List[str], opcjonalnie
        Lista kolumn do standaryzacji. Jeśli None, standaryzowane są wszystkie kolumny numeryczne.
    wyswietlaj_informacje : bool
        Czy wyświetlać informacje diagnostyczne.
    zwroc_tylko_dane : bool
        Jeśli True, zwraca tylko przeskalowany DataFrame zamiast słownika z dodatkowymi informacjami.

    Zwraca:
    -------
    Union[pd.DataFrame, Dict[str, Any]]
        Jeśli zwroc_tylko_dane=True: przeskalowany DataFrame
        W przeciwnym razie: słownik z kluczami:
        - 'df_scaled': przestandaryzowany DataFrame
        - 'skale': DataFrame z informacjami o średniej i odchyleniu
    """
    try:
        wynik_df = df.copy()

        # Wybór kolumn do standaryzacji
        if kolumny is None:
            kolumny_numeryczne = wynik_df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            kolumny_numeryczne = [col for col in kolumny if col in wynik_df.columns and
                                  pd.api.types.is_numeric_dtype(wynik_df[col])]

        if not kolumny_numeryczne:
            raise ValueError("Brak kolumn numerycznych do standaryzacji.")

        # Utworzenie DataFrame'a na informacje o skalowaniu
        skale = pd.DataFrame(
            columns=['mean', 'std'],
            index=kolumny_numeryczne,
            dtype='float64'
        )

        for kolumna in kolumny_numeryczne:
            srednia = wynik_df[kolumna].mean()
            std = wynik_df[kolumna].std()

            if std == 0:
                if wyswietlaj_informacje:
                    print(f"[UWAGA] Kolumna '{kolumna}' ma zerowe odchylenie — zostanie pominięta.")
                continue

            wynik_df[kolumna] = (wynik_df[kolumna] - srednia) / std
            skale.loc[kolumna] = [srednia, std]

            if wyswietlaj_informacje:
                print(f"[INFO] Standaryzowano kolumnę '{kolumna}'")
                print(f"  Średnia: {srednia:.4f}, Odchylenie: {std:.4f}")

        # Optymalizacja pamięci po standaryzacji
        wynik_df = _optymalizuj_pamiec(wynik_df)

        # Zwróć tylko DataFrame lub pełny słownik z informacjami
        if zwroc_tylko_dane:
            return wynik_df
        else:
            return {
                'df_scaled': wynik_df,
                'skale': skale
            }

    except Exception as e:
        print(f"[BŁĄD] Błąd podczas standaryzacji: {str(e)}")
        import traceback
        print(traceback.format_exc())
        if zwroc_tylko_dane:
            return df
        else:
            return {'df_scaled': df, 'skale': pd.DataFrame()}

# 1. Wczytaj dane
df = wczytaj_csv(
    sciezka_pliku="online_retail_II.csv",
    separator=";",
    kolumny_daty=["InvoiceDate"],
    wyswietlaj_informacje=True
)

# 2. Standaryzuj wybrane kolumny
wynik = standard_scaler(
    df=df,
    kolumny=["Quantity", "Price"],
    wyswietlaj_informacje=True
)

# 3. Wyświetl wynik
print("\n[INFO] Przykładowe dane po standaryzacji:")
print(wynik['df_scaled'][["Quantity", "Price"]].head())
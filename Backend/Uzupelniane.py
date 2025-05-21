from typing import Union, Dict

import numpy as np
import pandas as pd

from Dane.Dane import wczytaj_csv


def uzupelnij_braki(
    df: pd.DataFrame,
    metoda: str = 'srednia',
    wartosc_stala: Union[str, int, float] = None,
    reguly: Dict[str, str] = None,
    wyswietlaj_info: bool = True
) -> pd.DataFrame:
    """
    Wypełnia brakujące wartości różnymi strategiami.

    Parametry:
    ---------
    df : pd.DataFrame
        Wejściowy DataFrame.
    metoda : str
        Metoda wypełnienia: 'srednia', 'mediana', 'moda', 'stała'.
    wartosc_stala : Union[str, int, float]
        Stała wartość do wypełnienia (jeśli metoda='stała').
    reguly : Dict[str, str]
        Słownik z regułami dla konkretnych kolumn: {'nazwa_kolumny': 'metoda'}.
    wyswietlaj_info : bool
        Czy wyświetlać informacje diagnostyczne.

    Zwraca:
    -------
    pd.DataFrame
        DataFrame z uzupełnionymi brakami.
    """

    try:
        df_kopia = df.copy()
        metody_dozwolone = ['srednia', 'mediana', 'moda', 'stała']

        if reguly:
            for kolumna, metoda in reguly.items():
                if metoda not in metody_dozwolone:
                    raise ValueError(f"Nieznana metoda: {metoda}")
                if kolumna not in df_kopia.columns:
                    raise ValueError(f"Brak kolumny: {kolumna}")

                if metoda == 'srednia':
                    srednia = df_kopia[kolumna].mean()
                    df_kopia[kolumna] = df_kopia[kolumna].fillna(srednia)
                elif metoda == 'mediana':
                    mediana = df_kopia[kolumna].median()
                    df_kopia[kolumna] = df_kopia[kolumna].fillna(mediana)
                elif metoda == 'moda':
                    moda = df_kopia[kolumna].mode()[0] if not df_kopia[kolumna].mode().empty else np.nan
                    df_kopia[kolumna] = df_kopia[kolumna].fillna(moda)
                elif metoda == 'stała':
                    df_kopia[kolumna] = df_kopia[kolumna].fillna(wartosc_stala)

                if wyswietlaj_info:
                    print(f"[INFO] Wypełniono kolumnę '{kolumna}' metodą: {metoda}")

        else:
            if metoda not in metody_dozwolone:
                raise ValueError(f"Nieznana metoda: {metoda}")

            if metoda == 'srednia':
                df_kopia = df_kopia.fillna(df_kopia.mean())
            elif metoda == 'mediana':
                df_kopia = df_kopia.fillna(df_kopia.median())
            elif metoda == 'moda':
                df_kopia = df_kopia.fillna(df_kopia.mode().iloc[0])
            elif metoda == 'stała':
                df_kopia = df_kopia.fillna(wartosc_stala)

            if wyswietlaj_info:
                print(f"[INFO] Wypełniono brakujące wartości metodą: {metoda}")

        return df_kopia

    except Exception as e:
        print(f"[BŁĄD] Błąd podczas wypełniania braków: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return df

def usun_braki(
    df: pd.DataFrame,
    os_wiersze_kolumny: str = 'wiersze',
    liczba_min_niepustych: int = 1,
    wyswietlaj_info: bool = True
) -> pd.DataFrame:
    """
    Usuwa wiersze lub kolumny zawierające brakujące dane.

    Parametry:
    ---------
    df : pd.DataFrame
        Wejściowy DataFrame.
    os_wiersze_kolumny : str
        'wiersze' lub 'kolumny' — co chcemy usunąć.
    liczba_min_niepustych : int
        Minimalna liczba niepustych wartości w wierszu/kolumnie, aby go zachować.
    wyswietlaj_info : bool
        Czy wyświetlać informacje diagnostyczne.

    Zwraca:
    -------
    pd.DataFrame
        DataFrame po usunięciu braków.
    """

    try:
        if os_wiersze_kolumny not in ['wiersze', 'kolumny']:
            raise ValueError("Parametr 'os_wiersze_kolumny' musi być 'wiersze' lub 'kolumny'.")

        # Ustalenie osi (0 - wiersze, 1 - kolumny)
        oś = 0 if os_wiersze_kolumny == 'wiersze' else 1

        # Liczba brakujących przed operacją
        liczba_brakujacych_przed = df.isnull().sum(axis=oś).gt(0).sum()

        # Wykonanie operacji
        df_wynikowy = df.dropna(axis=oś, thresh=liczba_min_niepustych)

        if wyswietlaj_info:
            print(f"[INFO] Usunięto {liczba_brakujacych_przed} {os_wiersze_kolumny} z brakującymi wartościami.")
            print(f"[INFO] Nowy rozmiar danych: {df_wynikowy.shape}")

        return df_wynikowy

    except Exception as e:
        print(f"[BŁĄD] Nie udało się usunąć braków: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return df

# Wczytaj dane
df = wczytaj_csv("online_retail_II.csv", separator=";", kolumny_daty=["InvoiceDate"], wyswietlaj_informacje=True)
# Usuń wiersze, które mają mniej niż 3 niepuste wartości
df_bez_brakow = usun_braki(df, os_wiersze_kolumny='wiersze', liczba_min_niepustych=3)
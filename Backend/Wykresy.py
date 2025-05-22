import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from Dane.Dane import wczytaj_csv


def rysuj_wykres(
        df: pd.DataFrame,
        typ_wykresu: str = "scatter",
        kolumna_x: str = None,
        kolumna_y: str = None,
        kolumna_koloru: str = None,
        kolumna_rozmiaru: str = None,
        nazwa_wykresu: str = "Wykres",
        etykieta_x: str = None,
        etykieta_y: str = None,
        wyswietlaj_informacje: bool = True
) -> None:
    """
    Rysuje wykres na podstawie danych DataFrame.

    Parametry:
    ----------
    df : pd.DataFrame
        Wejściowy DataFrame.
    typ_wykresu : str
        Typ wykresu (opcje: 'scatter', 'bar', 'line', 'heatmap', 'pie').
    kolumna_x : str, opcjonalnie
        Nazwa kolumny dla osi X (dla scatter, bar, line).
    kolumna_y : str, opcjonalnie
        Nazwa kolumny dla osi Y (dla scatter, bar, line).
    kolumna_koloru : str, opcjonalnie
        Nazwa kolumny do mapowania na kolor punktów (dla scatter).
    kolumna_rozmiaru : str, opcjonalnie
        Nazwa kolumny do mapowania na rozmiar punktów (dla scatter).
    nazwa_wykresu : str
        Tytuł wykresu.
    etykieta_x : str, opcjonalnie
        Etykieta osi X.
    etykieta_y : str, opcjonalnie
        Etykieta osi Y.
    wyswietlaj_informacje : bool
        Czy wyświetlać informacje diagnostyczne.

    Zwraca:
    -------
    None
    """
    try:
        # Walidacja typu wykresu
        if typ_wykresu not in ["scatter", "bar", "line", "heatmap", "pie"]:
            raise ValueError(
                f"Nieznany typ wykresu: {typ_wykresu}. Dostępne opcje: 'scatter', 'bar', 'line', 'heatmap', 'pie'.")

        # Wybór odpowiedniej funkcji do rysowania wykresu
        if typ_wykresu == "scatter":
            # Wykres rozrzutu (scatter plot)
            if kolumna_x is None or kolumna_y is None:
                raise ValueError("Dla wykresu rozrzutu wymagane są kolumny x i y.")

            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=df, x=kolumna_x, y=kolumna_y, hue=kolumna_koloru, size=kolumna_rozmiaru)
            plt.title(nazwa_wykresu)
            plt.xlabel(etykieta_x or kolumna_x)
            plt.ylabel(etykieta_y or kolumna_y)
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        elif typ_wykresu == "bar":
            # Wykres słupkowy (bar plot)
            if kolumna_x is None or kolumna_y is None:
                raise ValueError("Dla wykresu słupkowego wymagane są kolumny x i y.")

            plt.figure(figsize=(10, 6))
            sns.barplot(data=df, x=kolumna_x, y=kolumna_y)
            plt.title(nazwa_wykresu)
            plt.xlabel(etykieta_x or kolumna_x)
            plt.ylabel(etykieta_y or kolumna_y)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()

        elif typ_wykresu == "line":
            # Wykres liniowy (line plot)
            if kolumna_x is None or kolumna_y is None:
                raise ValueError("Dla wykresu liniowego wymagane są kolumny x i y.")

            plt.figure(figsize=(10, 6))
            sns.lineplot(data=df, x=kolumna_x, y=kolumna_y)
            plt.title(nazwa_wykresu)
            plt.xlabel(etykieta_x or kolumna_x)
            plt.ylabel(etykieta_y or kolumna_y)
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        elif typ_wykresu == "heatmap":
            # Mapa cieplna (heatmap)
            plt.figure(figsize=(10, 8))
            sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
            plt.title(nazwa_wykresu)
            plt.tight_layout()
            plt.show()

        elif typ_wykresu == "pie":
            # Wykres kołowy (pie chart)
            if kolumna_x is None:
                raise ValueError("Dla wykresu kołowego wymagana jest kolumna x.")

            plt.figure(figsize=(8, 8))
            counts = df[kolumna_x].value_counts()
            labels = counts.index
            sizes = counts.values
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            plt.title(nazwa_wykresu)
            plt.axis('equal')  # Aby wykres był okrągły
            plt.tight_layout()
            plt.show()

        if wyswietlaj_informacje:
            print(f"[INFO] Wykres {typ_wykresu} utworzony pomyślnie.")

    except Exception as e:
        print(f"[BŁĄD] Nie udało się utworzyć wykresu: {str(e)}")
        import traceback
        print(traceback.format_exc())

# Przykładowe dane
df = wczytaj_csv("online_retail_II.csv")

# Wykres kołowy: udział poszczególnych krajów
rysuj_wykres(
    df=df,
    typ_wykresu="pie",
    kolumna_x="Country",
    nazwa_wykresu="Udział poszczególnych krajów"
)
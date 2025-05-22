import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from Dane.Dane import wczytaj_csv

def rysuj_wykres(
    df: pd.DataFrame,
    typ_wykresu: str = "pie",
    kolumna_x: str = None,
    kolumna_y: str = None,
    nazwa_wykresu: str = "Wykres",
    etykieta_x: str = None,
    etykieta_y: str = None,
    wyswietlaj_informacje: bool = True,
    maks_kategorie: int = 10,
    min_procent: float = 2.0
) -> None:
    try:
        if typ_wykresu == "pie":
            if kolumna_x is None:
                raise ValueError("Dla wykresu kołowego wymagana jest kolumna x.")

            dane = df[kolumna_x].dropna()
            wszystkie_kategorie = dane.value_counts(normalize=True) * 100

            if len(wszystkie_kategorie) > maks_kategorie:
                czeste_kategorie = wszystkie_kategorie.head(maks_kategorie - 1)
                inne_suma = wszystkie_kategorie.iloc[maks_kategorie - 1:].sum()
                czeste_kategorie = pd.concat([
                    czeste_kategorie,
                    pd.Series({f"Inne (> {min_procent}%)": inne_suma})
                ])
                wszystkie_kategorie = czeste_kategorie

            wszystkie_kategorie = wszystkie_kategorie[wszystkie_kategorie >= min_procent]

            if wszystkie_kategorie.empty:
                wszystkie_kategorie = pd.Series({f"Inne (> {min_procent}%)": 100.0})

            labels = wszystkie_kategorie.index.tolist()
            sizes = wszystkie_kategorie.values.tolist()

            explode = [0.05] * len(sizes)
            if len(sizes) > 1:
                najwiekszy_index = sizes.index(max(sizes))
                explode[najwiekszy_index] = 0.15

            plt.figure(figsize=(10, 8))
            wedges, _ = plt.pie(  # Usunięto 'labels' z argumentów
                sizes,
                startangle=90,
                colors=sns.color_palette("pastel"),
                wedgeprops=dict(width=0.4),
                textprops=dict(color="black", fontsize=10)
            )

            plt.legend(
                wedges, labels,
                title="Kategorie",
                loc="center left",
                bbox_to_anchor=(1, 0, 0.5, 1),
                prop={'size': 10}
            )

            plt.setp(_, fontsize=10)  # Usunięto tekst wycinków
            plt.title(nazwa_wykresu, fontsize=14)
            plt.axis('equal')
            plt.tight_layout()
            plt.show()

            if wyswietlaj_informacje:
                print(f"[INFO] Wykres kołowy utworzony dla kolumny: {kolumna_x}")
                print(f"  Liczba kategorii: {len(sizes)}")
                print(f"  Grupowanie: >{min_procent}% lub Top {maks_kategorie}")

    except Exception as e:
        print(f"[BŁĄD] Błąd podczas rysowania wykresu: {str(e)}")
# Wczytaj dane
df = wczytaj_csv("online_retail_II.csv")

# Przykładowe użycie
rysuj_wykres(
    df=df,
    typ_wykresu="pie",
    kolumna_x="Country",  # Kolumna do analizy
    nazwa_wykresu="Udział krajów",
    maks_kategorie=8,     # Maksymalnie 8 kategorii
    min_procent=1.0       # Kategorie poniżej 1% trafiają do 'Inne'
)
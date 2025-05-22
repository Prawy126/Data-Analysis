import pandas as pd
import numpy as np
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
    kolumna_hue: str = None,
    nazwa_wykresu: str = "Wykres",
    etykieta_x: str = None,
    etykieta_y: str = None,
    wyswietlaj_informacje: bool = True,
    figsize: tuple = (10, 6),
    palette: str = "pastel",
    styl: str = "whitegrid",
    orient: str = "v",
    maks_kategorie: int = 8,
    min_procent: float = 1.0,
    alpha: float = 0.8,
    regline: bool = False,
    fill_between: bool = False,
    ci: int = None,
    sort_values: bool = True,
    descending: bool = True,
    marker: str = "o"
) -> None:
    """
    Uniwersalna funkcja do rysowania różnych typów wykresów.
    """

    sns.set_theme(style=styl, palette=palette, context="notebook")

    # Scatter
    if typ_wykresu == "scatter":
        if kolumna_x is None or kolumna_y is None:
            raise ValueError("Dla scatter wymagane są kolumny x i y.")
        fig, ax = plt.subplots(figsize=figsize)
        sns.scatterplot(
            data=df, x=kolumna_x, y=kolumna_y,
            hue=kolumna_koloru, size=kolumna_rozmiaru,
            alpha=alpha, edgecolor="w", marker=marker, ax=ax
        )
        if regline:
            sns.regplot(
                data=df, x=kolumna_x, y=kolumna_y,
                scatter=False, ci=ci, color="grey", ax=ax
            )
        ax.grid(True)
        ax.set_xlabel(etykieta_x or kolumna_x)
        ax.set_ylabel(etykieta_y or kolumna_y)

    # Bar
    elif typ_wykresu == "bar":
        if kolumna_x is None or kolumna_y is None:
            raise ValueError("Dla bar wymagane są kolumny x i y.")
        data = df.copy()
        if sort_values:
            data = data.sort_values(kolumna_y, ascending=not descending)
        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(
            data=data,
            x=kolumna_x if orient=="v" else kolumna_y,
            y=kolumna_y if orient=="v" else kolumna_x,
            orient=orient,
            ci=ci,
            ax=ax
        )
        ax.set_xlabel(etykieta_x or kolumna_x)
        ax.set_ylabel(etykieta_y or kolumna_y)
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(45)
            lbl.set_ha('right')

    # Line
    elif typ_wykresu == "line":
        if kolumna_x is None or kolumna_y is None:
            raise ValueError("Dla line wymagane są kolumny x i y.")
        fig, ax = plt.subplots(figsize=figsize)
        sns.lineplot(
            data=df, x=kolumna_x, y=kolumna_y,
            hue=kolumna_hue, marker=marker, ax=ax
        )
        if fill_between:
            ax.fill_between(df[kolumna_x], df[kolumna_y], alpha=0.15)
        ax.grid(True)
        ax.set_xlabel(etykieta_x or kolumna_x)
        ax.set_ylabel(etykieta_y or kolumna_y)

    # Heatmap
    elif typ_wykresu == "heatmap":
        corr = df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            corr, mask=mask, annot=True,
            cmap="coolwarm", fmt=".2f",
            linewidths=0.5, vmin=-1, vmax=1,
            ax=ax
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # Pie
    elif typ_wykresu == "pie":
        if kolumna_x is None:
            raise ValueError("Dla pie wymagane jest podanie kolumny x.")
        dane = df[kolumna_x].dropna()
        pct = dane.value_counts(normalize=True) * 100

        # grupowanie
        if len(pct) > maks_kategorie:
            top = pct.head(maks_kategorie - 1)
            other = pct.iloc[maks_kategorie - 1:].sum()
            pct = pd.concat([top, pd.Series({"Inne": other})])

        pct = pct[pct >= min_procent]
        if pct.empty:
            pct = pd.Series({"Inne": 100.0})

        labels = pct.index.tolist()
        sizes  = pct.values.tolist()
        explode = [0.05]*len(sizes)
        explode[sizes.index(max(sizes))] = 0.15 if len(sizes)>1 else 0

        fig, ax = plt.subplots(figsize=(figsize[0], figsize[0]))
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, autopct='%1.1f%%',
            startangle=90, explode=explode,
            colors=sns.color_palette(palette, n_colors=len(sizes)),
            wedgeprops=dict(width=0.4)
        )
        ax.axis('equal')
        ax.legend(
            wedges, labels, title="Kategorie",
            loc="center left", bbox_to_anchor=(1, 0, 0.5, 1)
        )

    else:
        raise ValueError(f"Nieznany typ wykresu: {typ_wykresu}")

    # Tytuł i wyświetlenie
    fig.suptitle(nazwa_wykresu, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    if wyswietlaj_informacje:
        print(f"[INFO] Wykres '{typ_wykresu}' utworzony dla: {kolumna_x or ''} {kolumna_y or ''}")

df = wczytaj_csv("online_retail_II.csv")

rysuj_wykres(df, typ_wykresu="pie", kolumna_x="Country", nazwa_wykresu="Udział krajów")
rysuj_wykres(df, typ_wykresu="bar", kolumna_x="Country", kolumna_y="Quantity",
             nazwa_wykresu="Suma ilości zamówień według kraju", sort_values=True, descending=False)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.widgets import Button
import matplotlib.patches as patches

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
        marker: str = "o",
        fig=None,
        ax=None,
        show_percentages: bool = True,
        pie_style: str = "full",
        **kwargs
) -> tuple:
    """
    Uniwersalna funkcja do rysowania różnych typów wykresów.

    Argumenty są elastycznie interpretowane w zależności od typu wykresu:

    - scatter: wymaga kolumna_x i kolumna_y (punkty na płaszczyźnie)
    - line: wymaga kolumna_x i kolumna_y (linia łącząca punkty)
    - bar:
        * jeśli podano kolumna_y: tradycyjny wykres słupkowy (x = kategorie, y = wartości)
        * jeśli nie podano kolumna_y: wykres liczebności dla kolumna_x
    - pie:
        * wymaga tylko kolumna_x (zlicza wartości w tej kolumnie)
        * nie używa kolumna_y w ogóle
    """
    sns.set_theme(style=styl, palette=palette, context="notebook")

    # Jeśli nie podano figury, utwórz nową
    if fig is None:
        fig, ax = plt.subplots(figsize=figsize)
    # Jeśli podano figurę, ale nie oś, utwórz nową oś
    elif ax is None:
        ax = fig.add_subplot(111)

    # Wyczyść oś przed rysowaniem
    ax.clear()

    # Scatter - potrzebuje kolumn X i Y
    if typ_wykresu == "scatter":
        if kolumna_x is None or kolumna_y is None:
            raise ValueError("Dla scatter wymagane są kolumny x i y.")
        sns.scatterplot(
            data=df, x=kolumna_x, y=kolumna_y,
            hue=kolumna_koloru or kolumna_hue, size=kolumna_rozmiaru,
            alpha=alpha, edgecolor="w", marker=marker, ax=ax
        )
        if regline:
            sns.regplot(
                data=df, x=kolumna_x, y=kolumna_y,
                scatter=False, errorbar=None, color="grey", ax=ax
            )
        ax.grid(True)
        ax.set_xlabel(etykieta_x or kolumna_x)
        ax.set_ylabel(etykieta_y or kolumna_y)

    # Bar - elastyczny, działa na dwa sposoby
    elif typ_wykresu == "bar":
        if kolumna_x is None:
            raise ValueError("Dla bar wymagane jest podanie kolumny x.")

        data = df.copy()

        # Tryb 1: Tradycyjny - pokazuje kolumnę Y względem X
        if kolumna_y is not None:
            if sort_values:
                data = data.sort_values(kolumna_y, ascending=not descending)
            sns.barplot(
                data=data,
                x=kolumna_x if orient == "v" else kolumna_y,
                y=kolumna_y if orient == "v" else kolumna_x,
                orient=orient,
                errorbar=None,
                ax=ax,
                hue=kolumna_hue
            )
            ax.set_xlabel(etykieta_x or kolumna_x)
            ax.set_ylabel(etykieta_y or kolumna_y)

        # Tryb 2: Zliczanie - pokazuje liczebności wartości w kolumnie X
        else:
            # Zlicz wartości w kolumnie X
            counts = df[kolumna_x].value_counts()
            if sort_values:
                counts = counts.sort_values(ascending=not descending)

            # Ogranicz liczbę kategorii jeśli trzeba
            if len(counts) > maks_kategorie:
                top_counts = counts.head(maks_kategorie - 1)
                other_count = counts.iloc[maks_kategorie - 1:].sum()
                counts = pd.concat([top_counts, pd.Series({"Inne": other_count})])

            # Narysuj wykres
            if orient == "v":
                ax.bar(counts.index, counts.values, color=sns.color_palette(palette, n_colors=len(counts)))
                ax.set_xlabel(etykieta_x or kolumna_x)
                ax.set_ylabel(etykieta_y or "Liczebność")
            else:
                ax.barh(counts.index, counts.values, color=sns.color_palette(palette, n_colors=len(counts)))
                ax.set_xlabel(etykieta_x or "Liczebność")
                ax.set_ylabel(etykieta_y or kolumna_x)

            # Dodaj etykiety wartości na słupkach
            for i, v in enumerate(counts.values):
                if orient == "v":
                    ax.text(i, v, f"{v}", ha='center', va='bottom')
                else:
                    ax.text(v, i, f"{v}", ha='left', va='center')

        # Rotacja etykiet
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(45)
            lbl.set_ha('right')

    # Line - potrzebuje kolumn X i Y
    elif typ_wykresu == "line":
        if kolumna_x is None or kolumna_y is None:
            raise ValueError("Dla line wymagane są kolumny x i y.")
        sns.lineplot(
            data=df, x=kolumna_x, y=kolumna_y,
            hue=kolumna_hue, marker=marker, ax=ax
        )
        if fill_between:
            if kolumna_hue is None:
                ax.fill_between(df[kolumna_x], df[kolumna_y], alpha=0.15)
        ax.grid(True)
        ax.set_xlabel(etykieta_x or kolumna_x)
        ax.set_ylabel(etykieta_y or kolumna_y)

    # Pie - potrzebuje tylko kolumny X (liczy wartości)
    elif typ_wykresu == "pie":
        if kolumna_x is None:
            raise ValueError("Dla pie wykresu wymagane jest podanie kolumny x.")

        dane = df[kolumna_x].dropna()
        pct = dane.value_counts(normalize=True) * 100

        # Grupowanie i obsługa dużej liczby kategorii
        if len(pct) > maks_kategorie:
            top = pct.head(maks_kategorie - 1)
            other = pct.iloc[maks_kategorie - 1:].sum()
            pct = pd.concat([top, pd.Series({"Inne": other})])

        pct = pct[pct >= min_procent]
        if pct.empty:
            pct = pd.Series({"Inne": 100.0})

        labels = pct.index.tolist()
        sizes = pct.values.tolist()
        counts = [len(dane[dane == label]) if label != "Inne" else
                  len(dane) - sum([len(dane[dane == l]) for l in labels[:-1]])
                  for label in labels]

        # KONTRASTOWE KOLORY
        if palette == "pastel":
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57',
                      '#FF9FF3', '#54A0FF', '#5F27CD', '#00D2D3', '#FF9F43',
                      '#FD79A8', '#A29BFE', '#6C5CE7', '#74B9FF', '#00B894']
        else:
            colors = sns.color_palette(palette, n_colors=len(sizes))

        # Explode
        explode = [0.02] * len(sizes)
        if len(sizes) > 1:
            max_idx = sizes.index(max(sizes))
            explode[max_idx] = 0.08

        ax.clear()

        # GŁÓWNY WYKRES KOŁOWY
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=None,
            autopct=lambda pct: f'{pct:.1f}%' if show_percentages and pct > 2 else '',
            startangle=90,
            explode=explode,
            colors=colors[:len(sizes)],
            wedgeprops=dict(width=1.0 if pie_style == "full" else 0.4,
                            edgecolor='white', linewidth=3),
            textprops={'fontsize': 11, 'weight': 'bold', 'color': 'white'},
            pctdistance=0.85,
            shadow=True
        )

        ax.axis('equal')

        # PEŁNA LEGENDA + INTERAKTYWNOŚĆ - Zastosowanie nowej wersji
        _create_full_interactive_legend(fig, ax, wedges, labels, sizes, counts, colors, dane)

    else:
        raise ValueError(f"Nieznany typ wykresu: {typ_wykresu}")

    # Ustaw tytuł wykresu
    if nazwa_wykresu:
        ax.set_title(nazwa_wykresu, fontsize=16, pad=20, weight='bold')
    elif typ_wykresu == "pie" and kolumna_x:
        ax.set_title(f"Rozkład: {kolumna_x}", fontsize=16, pad=20, weight='bold')

    # Dopasuj układ wykresu
    fig.tight_layout()

    return fig, ax


def _create_full_interactive_legend(fig, ax, wedges, labels, sizes, counts, colors, dane):
    """
    Tworzy PEŁNĄ legendę z interaktywnością i automatycznym rozmieszczeniem w wielu kolumnach
    """
    total_items = len(labels)

    # Stan legendy
    legend_state = {
        'highlighted_wedge': None,
        'original_colors': [w.get_facecolor() for w in wedges],
        'legend_obj': None,
        'info_text': None,
        'hover_text': None
    }

    def create_legend_labels():
        """Tworzy etykiety dla legendy"""
        legend_labels = []
        for i, (label, size, count) in enumerate(zip(labels, sizes, counts)):
            # Skróć długie nazwy ale pokaż pełne info
            display_label = label if len(str(label)) <= 15 else str(label)[:12] + "..."
            legend_labels.append(f"{display_label} ({size:.1f}% • {count:,})")
        return legend_labels

    def update_legend():
        """Aktualizuje pełną legendę z automatycznym formatowaniem w kolumnach"""
        # Usuń poprzednią legendę
        if legend_state['legend_obj']:
            legend_state['legend_obj'].remove()

        # Usuń poprzedni tekst info
        if legend_state['info_text']:
            legend_state['info_text'].remove()

        visible_wedges = wedges
        visible_labels = create_legend_labels()

        # Automatyczne dopasowanie liczby kolumn względem ilości elementów
        ncol = 1
        if total_items > 8:
            ncol = 2
        if total_items > 16:
            ncol = 3
        if total_items > 24:
            ncol = 4

        # Zmniejszamy rozmiar czcionki przy dużej liczbie elementów
        fontsize = 9
        if total_items > 20:
            fontsize = 8

        # Utwórz legendę z wieloma kolumnami
        legend = ax.legend(
            visible_wedges, visible_labels,
            title="Kategorie (kliknij aby podświetlić)",
            loc="center left",
            bbox_to_anchor=(1.05, 0.5),
            frameon=True,
            shadow=True,
            fontsize=fontsize,
            title_fontsize=10,
            ncol=ncol  # Automatycznie rozmieszczaj w kolumnach
        )

        # Stylizuj legendę
        legend.get_frame().set_facecolor('#f8f9fa')
        legend.get_frame().set_alpha(0.95)
        legend.get_frame().set_edgecolor('#dee2e6')
        legend.get_frame().set_linewidth(1)

        # Pogrub tytuł legendy ręcznie
        legend.get_title().set_fontweight('bold')

        legend_state['legend_obj'] = legend

        # Dodaj info o liczbie kategorii
        legend_state['info_text'] = fig.text(
            0.85, 0.12,
            f"Łącznie {total_items} kategorii",
            fontsize=8, ha='center',
            style='italic', color='gray'
        )

        # Dodaj interaktywność do tekstów legendy
        for i, legend_text in enumerate(legend.get_texts()):
            legend_text.set_picker(True)
            legend_text.wedge_index = i

    def highlight_wedge(wedge_idx, highlight=True):
        """Podświetla kawałek wykresu"""
        if wedge_idx >= len(wedges):
            return

        wedge = wedges[wedge_idx]

        if highlight:
            # Podświetl wybrany kawałek
            wedge.set_edgecolor('#000000')  # Czarna ramka
            wedge.set_linewidth(5)
            wedge.set_alpha(0.9)

            # Przytmij pozostałe
            for i, other_wedge in enumerate(wedges):
                if i != wedge_idx:
                    other_wedge.set_alpha(0.3)

            # Pokaż szczegółowe info w tytule - CZARNY TEKST
            label = labels[wedge_idx]
            size = sizes[wedge_idx]
            count = counts[wedge_idx]
            percentage_of_total = count / len(dane) * 100

            ax.set_title(f" {label}\n{size:.1f}% • {count:,} rekordów • {percentage_of_total:.2f}% wszystkich danych",
                         fontsize=14, pad=25, weight='bold', color='#000000')  # CZARNY kolor

        else:
            # Przywróć normalny wygląd
            wedge.set_edgecolor('white')
            wedge.set_linewidth(3)
            wedge.set_alpha(1.0)

            # Przywróć wszystkie kawałki
            for other_wedge in wedges:
                other_wedge.set_alpha(1.0)

            # Przywróć oryginalny tytuł
            original_title = ax.get_title().split('\n')[0] if '\n' in ax.get_title() else ax.get_title()
            ax.set_title(original_title, fontsize=16, pad=20, weight='bold', color='black')

        fig.canvas.draw_idle()

    def on_pick(event):
        """Obsługa kliknięcia w legendę"""
        if hasattr(event.artist, 'wedge_index'):
            wedge_idx = event.artist.wedge_index

            # Toggle highlight
            if legend_state['highlighted_wedge'] == wedge_idx:
                highlight_wedge(wedge_idx, False)
                legend_state['highlighted_wedge'] = None
            else:
                # Usuń poprzednie podświetlenie
                if legend_state['highlighted_wedge'] is not None:
                    highlight_wedge(legend_state['highlighted_wedge'], False)

                # Dodaj nowe podświetlenie
                highlight_wedge(wedge_idx, True)
                legend_state['highlighted_wedge'] = wedge_idx

    def on_hover(event):
        """Obsługa najechania myszką - CIEMNY TEKST"""
        if event.inaxes == ax:
            # Usuń poprzedni hover text
            if legend_state['hover_text']:
                legend_state['hover_text'].remove()
                legend_state['hover_text'] = None

            # Sprawdź czy najechano na kawałek wykresu
            for i, wedge in enumerate(wedges):
                if wedge.contains_point([event.x, event.y]):
                    if legend_state['highlighted_wedge'] != i:
                        # Lekkie podświetlenie przy hover
                        wedge.set_edgecolor('#333333')  # Ciemno szary
                        wedge.set_linewidth(4)

                        # Pokaż tooltip - CZARNY TEKST
                        label = labels[i]
                        size = sizes[i]
                        count = counts[i]

                        # Tekst na dole wykresu - CZARNY i większy
                        legend_state['hover_text'] = fig.text(0.5, 0.02,
                                                              f" {label}: {size:.1f}% ({count:,} rekordów)",
                                                              ha='center', fontsize=12,
                                                              weight='bold', color='#000000',  # CZARNY
                                                              bbox=dict(boxstyle="round,pad=0.3",
                                                                        facecolor='#FFFF99', alpha=0.8))

                        fig.canvas.draw_idle()
                    break
            else:
                # Nie ma hover - przywróć normalny wygląd
                for i, wedge in enumerate(wedges):
                    if legend_state['highlighted_wedge'] != i:
                        wedge.set_edgecolor('white')
                        wedge.set_linewidth(3)

                fig.canvas.draw_idle()

    # Podłącz eventy
    fig.canvas.mpl_connect('pick_event', on_pick)
    fig.canvas.mpl_connect('motion_notify_event', on_hover)

    # Inicjalne utworzenie legendy
    update_legend()
def _add_scroll_buttons(fig, legend_state, update_callback):
    """Dodaje przyciski przewijania"""
    max_visible = 10
    total_items = len(legend_state['original_colors'])

    if total_items <= max_visible:
        return

    # Przyciski przewijania
    button_size = 0.025

    # Przycisk "w górę"
    ax_up = plt.axes([0.88, 0.7, button_size, button_size])
    ax_up.set_facecolor('#e8f4fd')
    button_up = Button(ax_up, '▲', color='#4CAF50', hovercolor='#45a049')

    # Przycisk "w dół"
    ax_down = plt.axes([0.88, 0.3, button_size, button_size])
    ax_down.set_facecolor('#e8f4fd')
    button_down = Button(ax_down, '▼', color='#f44336', hovercolor='#da190b')

    def scroll_up(event):
        if legend_state['scroll_position'] > 0:
            legend_state['scroll_position'] -= 1
            update_callback()

    def scroll_down(event):
        max_scroll = max(0, total_items - max_visible)
        if legend_state['scroll_position'] < max_scroll:
            legend_state['scroll_position'] += 1
            update_callback()

    button_up.on_clicked(scroll_up)
    button_down.on_clicked(scroll_down)
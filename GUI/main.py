# main.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy  as np

from Dane.Dane  import wczytaj_csv
from Backend.Statystyka import analizuj_dane_numeryczne
from Backend.Korelacje  import oblicz_korelacje_pearsona, oblicz_korelacje_spearmana


class MainApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Data Toolkit")
        self.geometry("1000x600")
        self.minsize(820, 480)

        # przechowujemy DF i ścieżkę
        self.df:   pd.DataFrame | None = None
        self.path: str | None          = None

        # ---------- menu ----------
        menubar = tk.Menu(self);  self.config(menu=menubar)
        akcje   = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Akcje", menu=akcje)
        akcje.add_command(label="Pre-processing", command=self._load_pre)
        akcje.add_command(label="Statystyka",     command=self._load_stats)
        akcje.add_separator();  akcje.add_command(label="Zamknij", command=self.quit)

        # ---------- notebook ----------
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=10, pady=10)

        self._load_pre()          # domyślna sekcja

    # ──────────────────────────────────────────────────────────────
    #  Wspólne: loader CSV + pomoc
    # ──────────────────────────────────────────────────────────────
    def _add_loader(self, parent: tk.Widget, on_success=None) -> None:
        """Przycisk + etykieta. on_success(df) wywoływane po udanym wczytaniu."""
        frm = ttk.Frame(parent); frm.pack(fill="x", padx=10, pady=(5, 12))
        info = tk.StringVar(value="(brak pliku)")

        def choose() -> None:
            fp = filedialog.askopenfilename(
                title="Wybierz plik CSV",
                filetypes=[("CSV", "*.csv"), ("Wszystkie", "*.*")]
            )
            if not fp:
                return
            df = wczytaj_csv(fp, separator=None, wyswietlaj_informacje=True)
            if df is None:
                messagebox.showerror("Błąd", "Nie udało się wczytać pliku.")
                return
            self.df, self.path = df, fp
            info.set(f"{fp.split('/')[-1]}  ({len(df)}×{len(df.columns)})")
            messagebox.showinfo("OK", "Plik wczytany!")
            if on_success:
                on_success(df)

        ttk.Button(frm, text="Wczytaj plik CSV", command=choose).pack(side="left")
        ttk.Label(frm,  textvariable=info).pack(side="left", padx=10)

    # ──────────────────────────────────────────────────────────────
    #  PRE-PROCESSING (placeholder)
    # ──────────────────────────────────────────────────────────────
    def _load_pre(self) -> None:
        self._clear_tabs()
        tab = ttk.Frame(self.nb);  self.nb.add(tab, text="Cleaning")
        self._add_loader(tab)
        ttk.Label(tab, text="Tu dodasz narzędzia czyszczenia…",
                         font=("Helvetica", 11)).pack(padx=20, pady=20)

    # ──────────────────────────────────────────────────────────────
    #  STATYSTYKA
    # ──────────────────────────────────────────────────────────────
    def _load_stats(self) -> None:
        self._clear_tabs()
        self._build_numeric_stats_tab()
        self._build_corr_tab()

    # ---------- Statystyki liczbowe ---------------------------------------
    def _build_numeric_stats_tab(self) -> None:
        tab = ttk.Frame(self.nb);  self.nb.add(tab, text="Statystyki liczbowe")

        # lista kolumn numerycznych (aktualizowana po wczytaniu CSV)
        lbl_cols = ttk.Label(tab, text="Wybierz kolumny (Ctrl+klik):")
        lbl_cols.pack(anchor="w", padx=10)

        listbox = tk.Listbox(tab, selectmode="multiple", height=6, exportselection=False)
        listbox.pack(fill="x", padx=10, pady=(0, 10))

        # tabela wyników
        cols = ("kolumna", "średnia", "mediana", "min", "max", "std", "liczba")
        tree, ybar, xbar = self._make_treeview(tab, cols)
        tree.pack(fill="both", expand=True, padx=10, pady=5)

        def update_selector(df: pd.DataFrame) -> None:
            listbox.delete(0, tk.END)
            numeric = df.select_dtypes(include=[np.number]).columns
            for col in numeric:
                listbox.insert(tk.END, col)

        def run() -> None:
            if not self.path:
                messagebox.showwarning("Uwaga", "Najpierw wczytaj plik.")
                return
            selection = [listbox.get(i) for i in listbox.curselection()]
            _, wyniki = analizuj_dane_numeryczne(self.path,
                                                 wybrane_kolumny=selection or None)

            # odśwież Treeview
            for row in tree.get_children():
                tree.delete(row)
            for kol, staty in wyniki.items():
                tree.insert("", "end", values=(
                    kol,
                    staty["średnia"],
                    staty["mediana"],
                    staty["min"],
                    staty["max"],
                    staty["odchylenie_std"],
                    staty["liczba_wartości"]
                ))

        ttk.Button(tab, text="Oblicz statystyki", command=run)\
            .pack(anchor="w", padx=10, pady=(0, 6))

        # loader + callback, żeby lista kolumn odświeżała się automatycznie
        self._add_loader(tab, on_success=update_selector)

    # ---------- Korelacje -------------------------------------------------
    # wewnątrz klasy MainApp
    def _build_corr_tab(self) -> None:
        """Zakładka 'Korelacje' – macierz Pearsona / Spearmana w Treeview."""
        import numpy as np
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="Korelacje")

        # ---------- wybór metody ------------
        method_var = tk.StringVar(value="pearson")
        frm_select = ttk.Frame(tab);
        frm_select.pack(anchor="w", padx=10, pady=(0, 8))
        ttk.Radiobutton(frm_select, text="Pearson", variable=method_var,
                        value="pearson").pack(side="left")
        ttk.Radiobutton(frm_select, text="Spearman", variable=method_var,
                        value="spearman").pack(side="left")

        # ---------- Treeview (początkowo puste) ------------
        tree, ybar, xbar = self._make_treeview(tab, cols=())
        tree.pack(fill="both", expand=True, padx=10, pady=5)

        # ---------- akcja 'Oblicz korelacje' ----------
        def run_corr() -> None:
            if not self.path:
                messagebox.showwarning("Uwaga", "Najpierw wczytaj plik CSV.")
                return

            if method_var.get() == "pearson":
                df_corr = oblicz_korelacje_pearsona(self.path)
            else:
                df_corr = oblicz_korelacje_spearmana(self.path)

            if df_corr is None or df_corr.empty:
                messagebox.showinfo("Info", "Brak danych numerycznych do korelacji.")
                return

            # ---- konfiguracja kolumn Treeview ----
            cols = ("Variable", *df_corr.columns)  # ← dodatkowa kolumna z lewej
            tree.config(columns=cols, show="headings")

            tree.heading("Variable", text="Variable")
            tree.column("Variable", width=130, anchor="w")

            for col in df_corr.columns:
                tree.heading(col, text=col)
                tree.column(col, width=90, anchor="center")

            # wyczyść stare wiersze
            tree.delete(*tree.get_children())

            # dodaj nowe wiersze (nazwa wiersza + wartości)
            for row_name, values in df_corr.iterrows():
                tree.insert(
                    "", "end",
                    values=(row_name, *np.round(values.values, 4))
                )

        ttk.Button(tab, text="Oblicz korelacje", command=run_corr) \
            .pack(anchor="w", padx=10, pady=(0, 6))

        # ---------- loader pliku CSV ----------
        self._add_loader(tab)  # przycisk 'Wczytaj plik CSV'

    # ──────────────────────────────────────────────────────────────
    #  Pomocnicze
    # ──────────────────────────────────────────────────────────────
    def _make_treeview(self, parent, cols):
        """Zwraca Treeview + h/v scrollbary."""
        ybar = ttk.Scrollbar(parent, orient="vertical")
        xbar = ttk.Scrollbar(parent, orient="horizontal")
        tree = ttk.Treeview(parent, columns=cols, show="headings",
                            yscrollcommand=ybar.set, xscrollcommand=xbar.set)
        ybar.config(command=tree.yview); xbar.config(command=tree.xview)
        ybar.pack(side="right", fill="y"); xbar.pack(side="bottom", fill="x")
        if cols:
            for c in cols:
                tree.heading(c, text=c)
                tree.column(c, width=100, anchor="center")
        return tree, ybar, xbar

    def _clear_tabs(self) -> None:
        for tab in self.nb.tabs():
            self.nb.forget(tab)


if __name__ == "__main__":
    MainApp().mainloop()

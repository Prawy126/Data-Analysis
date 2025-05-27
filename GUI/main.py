# main.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy  as np

from Backend.Czyszczenie import ekstrakcja_podtablicy
from Backend.Duplikaty import usun_duplikaty
from Dane.Dane  import wczytaj_csv
from Backend.Statystyka import analizuj_dane_numeryczne
from Backend.Korelacje  import oblicz_korelacje_pearsona, oblicz_korelacje_spearmana


class MainApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Data Toolkit")
        self.geometry("1000x600")
        self.minsize(820, 480)
        self.current_result_df: pd.DataFrame | None = None  # Nowy atrybut

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

    def _load_pre(self) -> None:
        self._clear_tabs()
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="Cleaning")

        # Loader CSV
        self._add_loader(tab, on_success=self._clear_preprocessing_data)

        # Kontener główny z zakładkami
        notebook = ttk.Notebook(tab)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # Zakładka 1 - Ekstrakcja podtablicy
        extract_frame = ttk.Frame(notebook)
        self._build_extraction_tab(extract_frame)
        notebook.add(extract_frame, text="Ekstrakcja")

        # Zakładka 2 - Usuwanie duplikatów
        duplicates_frame = ttk.Frame(notebook)
        self._build_duplicates_tab(duplicates_frame)
        notebook.add(duplicates_frame, text="Duplikaty")

        # Tabela wyników
        self.result_tree, ybar, xbar = self._make_treeview(tab, [])
        self.result_tree.pack(fill="both", expand=True, padx=10, pady=10)

        # Przycisk zapisu
        self.save_btn = ttk.Button(tab, text="Zapisz wynik", state="disabled", command=self._save_result)
        self.save_btn.pack(side="right", padx=10, pady=5)

    def _build_extraction_tab(self, parent):
        """Zakładka do ekstrakcji podtablicy"""
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill="x", padx=10, pady=10)

        # Wiersze do obsługi
        ttk.Label(control_frame, text="Wiersze (indeksy, np. 0,2,5):").grid(row=0, column=0, sticky="w")
        self.rows_entry = ttk.Entry(control_frame)
        self.rows_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=2)

        # Kolumny do obsługi
        ttk.Label(control_frame, text="Kolumny (nazwy, np. age,salary):").grid(row=1, column=0, sticky="w")
        self.cols_entry = ttk.Entry(control_frame)
        self.cols_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=2)

        # Tryb działania
        self.mode_var = tk.StringVar(value="keep")
        ttk.Radiobutton(control_frame, text="Zachowaj", variable=self.mode_var, value="keep").grid(row=2, column=0,
                                                                                                   sticky="w")
        ttk.Radiobutton(control_frame, text="Usuń", variable=self.mode_var, value="remove").grid(row=2, column=1,
                                                                                                 sticky="w")

        ttk.Button(control_frame, text="Ekstrahuj podtablicę", command=self._run_extraction) \
            .grid(row=3, column=0, columnspan=2, pady=5)

    def _clear_preprocessing_data(self, df: pd.DataFrame) -> None:
        """Resetuje stan po wczytaniu nowego pliku"""
        self.current_result_df = None
        self.save_btn.config(state="disabled")
        self.result_tree.delete(*self.result_tree.get_children())

        # Aktualizacja listy kolumn dla duplikatów
        self.duplicates_listbox.delete(0, tk.END)
        for col in df.columns:
            self.duplicates_listbox.insert(tk.END, col)

    def _run_extraction(self) -> None:
        """Obsługa logiki ekstrakcji"""
        if self.df is None:
            messagebox.showwarning("Brak danych", "Proszę najpierw wczytać plik CSV!")
            return

        # Parsowanie wejść
        rows = None
        if self.rows_entry.get().strip():
            try:
                rows = list(map(int, self.rows_entry.get().split(",")))
            except ValueError:
                messagebox.showerror("Błąd", "Nieprawidłowy format wierszy. Podaj indeksy oddzielone przecinkami.")
                return

        cols = None
        if self.cols_entry.get().strip():
            cols = [c.strip() for c in self.cols_entry.get().split(",")]

        # Wywołanie funkcji ekstrakcji
        result = ekstrakcja_podtablicy(
            self.df,
            rows=rows,
            cols=cols,
            mode=self.mode_var.get(),
            wyswietlaj_informacje=True
        )

        if result is not None:
            self.current_result_df = result
            self._display_dataframe(result)
            self.save_btn.config(state="normal")

    def _display_dataframe(self, df: pd.DataFrame) -> None:
        """Aktualizacja Treeview z wynikami"""
        self.result_tree.delete(*self.result_tree.get_children())

        # Konfiguracja kolumn
        self.result_tree["columns"] = list(df.columns)
        for col in df.columns:
            self.result_tree.heading(col, text=col)
            self.result_tree.column(col, width=100, anchor="center")

        # Wypełnianie danymi
        for index, row in df.iterrows():
            self.result_tree.insert("", "end", values=list(row))

    def _save_result(self) -> None:
        """Zapis wyniku do CSV"""
        if self.current_result_df is None:
            messagebox.showwarning("Brak danych", "Brak wyników do zapisania!")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )

        if file_path:
            try:
                self.current_result_df.to_csv(file_path, index=False)
                messagebox.showinfo("Sukces", f"Dane zapisano w:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Błąd", f"Błąd podczas zapisu:\n{str(e)}")

    def _build_duplicates_tab(self, parent):
        """Zakładka do usuwania duplikatów"""
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Lista kolumn
        ttk.Label(control_frame, text="Kolumny do sprawdzania:").pack(anchor="w")
        self.duplicates_listbox = tk.Listbox(control_frame, selectmode="multiple", height=6, exportselection=False)
        self.duplicates_listbox.pack(fill="x", padx=5, pady=5)

        # Tryb usuwania
        self.duplicates_mode = tk.StringVar(value="pierwszy")
        mode_frame = ttk.Frame(control_frame)
        mode_frame.pack(fill="x", pady=5)

        ttk.Radiobutton(mode_frame, text="Zachowaj pierwszy", variable=self.duplicates_mode, value="pierwszy").pack(
            side="left")
        ttk.Radiobutton(mode_frame, text="Zachowaj ostatni", variable=self.duplicates_mode, value="ostatni").pack(
            side="left")
        ttk.Radiobutton(mode_frame, text="Usuń wszystkie", variable=self.duplicates_mode, value="wszystkie").pack(
            side="left")

        # Przycisk wykonania
        ttk.Button(control_frame, text="Usuń duplikaty", command=self._run_duplicate_removal) \
            .pack(side="right", padx=5, pady=5)

    def _run_duplicate_removal(self) -> None:
        """Obsługa usuwania duplikatów"""
        if self.df is None:
            messagebox.showwarning("Brak danych", "Proszę najpierw wczytać plik CSV!")
            return

        try:
            # Pobranie wybranych kolumn
            selected_cols = [self.duplicates_listbox.get(i)
                             for i in self.duplicates_listbox.curselection()]

            # Wywołanie funkcji z backendu
            result = usun_duplikaty(
                self.df,
                kolumny=selected_cols or None,
                tryb=self.duplicates_mode.get(),
                wyswietlaj_info=True
            )

            if result['liczba_duplikatow'] > 0:
                self.current_result_df = result['df_cleaned']
                self._display_dataframe(self.current_result_df)
                self.save_btn.config(state="normal")
                messagebox.showinfo("Sukces",
                                    f"Usunięto {result['liczba_duplikatow']} duplikatów!\n"
                                    f"Nowa liczba wierszy: {len(self.current_result_df)}")
            else:
                messagebox.showinfo("Informacja", "Nie znaleziono duplikatów do usunięcia")

        except Exception as e:
            messagebox.showerror("Błąd", f"Błąd podczas usuwania duplikatów:\n{str(e)}")




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

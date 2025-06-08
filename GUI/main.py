import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy  as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from Backend.AI import RANDOM_SEED, classify_and_return_predictions, cluster_kmeans
from Backend.Czyszczenie import ekstrakcja_podtablicy
from Backend.Duplikaty import usun_duplikaty
from Backend.Kodowanie import jedno_gorace_kodowanie, binarne_kodowanie, kodowanie_docelowe
from Backend.Skalowanie import minmax_scaler, standard_scaler
from Backend.Uzupelniane import uzupelnij_braki, usun_braki
from Backend.Wartosci import zamien_wartosci
from Backend.Wykresy import rysuj_wykres
from Dane.Dane  import wczytaj_csv
from Backend.Statystyka import analizuj_dane_numeryczne
from Backend.Korelacje  import oblicz_korelacje_pearsona, oblicz_korelacje_spearmana


class MainApp(tk.Tk):
    def __init__(self) -> None:
        """Inicjalizacja głównego okna aplikacji"""
        super().__init__()

        # Konfiguracja okna
        self.title("Data Toolkit")
        self.geometry("1000x600")

        # Zmienne aplikacji
        self.current_result_df = None
        self.df = None
        self.path = None
        self._replacement_rules = {}  # Dodane: inicjalizacja słownika reguł zamiany

        # Informacje o pliku
        self.file_info_var = tk.StringVar(value="Brak wczytanego pliku")
        self.file_info_label = ttk.Label(self, textvariable=self.file_info_var, font=("Helvetica", 10))
        self.file_info_label.pack(anchor="w", padx=10, pady=(5, 0))
        # Pasek menu
        self.menubar = tk.Menu(self)
        self.config(menu=self.menubar)

        # Menu "Akcje"
        akcje_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Akcje", menu=akcje_menu)
        akcje_menu.add_command(label="Pre-processing", command=self._load_pre)
        akcje_menu.add_command(label="Statystyka", command=self._load_stats)
        akcje_menu.add_command(label="AI", command=self._build_ai_window)
        akcje_menu.add_separator()
        akcje_menu.add_command(label="Zamknij", command=self.quit)

        # Nowe menu "Opcje"
        plik_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Plik", menu=plik_menu)
        plik_menu.add_command(label="Wczytaj CSV", command=self._load_csv_from_menu)
        plik_menu.add_command(label="Zapisz wynik", command=self._save_result)


        # Notebook (główne zakładki)
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=10, pady=10)

        # Pasek stanu
        self.status_var = tk.StringVar(value="Gotowe")
        self.status_bar = ttk.Label(self, textvariable=self.status_var,
                                    relief="sunken", anchor="w", padding=(6, 2))
        self.status_bar.pack(side="bottom", fill="x")

        # Paginacja
        self.page_size = 100  # domyślny rozmiar strony
        self.current_page = 0  # numer strony (0-indeks)
        self._pagination_df_id = None  # identyfikacja DataFrame dla paginacji

        # Dodaj style dla przycisków
        style = ttk.Style()

        # Styl dla głównego przycisku
        style.configure('Accent.TButton',
                        font=('Arial', 9, 'bold'),
                        foreground='white')

        # Próba ustawienia tła (może nie działać na wszystkich systemach)
        try:
            style.map('Accent.TButton',
                      background=[('active', '#0066CC'), ('!active', '#0080FF')])
        except:
            pass
        # Załaduj pierwszą zakładkę
        self._load_pre()

    def _set_busy(self, msg: str = "Ładowanie…") -> None:
        self.status_var.set(msg)
        self.config(cursor="watch")
        self.update_idletasks()

    def _set_ready(self, msg: str = "Gotowe") -> None:
        self.status_var.set(msg)
        self.config(cursor="")
        self.update_idletasks()

    def _on_file_loaded(self, df):
        self._update_all_columns(df)
        self._display_dataframe(df)

    def _select_all(self, listbox: tk.Listbox):
        listbox.selection_set(0, tk.END)

    def _add_refresh_button(self, parent: tk.Widget) -> None:
        """Dodaje tylko przycisk 'Odśwież' do podanego kontenera."""
        frm = ttk.Frame(parent)
        frm.pack(fill="x", padx=10, pady=(5, 12))

        ttk.Button(frm, text="Odśwież", command=self._refresh_dataframe).pack(side="left", padx=5)

    def _clear_selection(self, listbox: tk.Listbox):
        listbox.selection_clear(0, tk.END)

    def _refresh_dataframe(self):
        """Odświeża aktualny widok DataFrame w zakładce Cleaning"""
        if self.current_result_df is not None:
            # Użyj wyniku przetwarzania, jeśli dostępny
            self._display_dataframe(self.current_result_df)
        elif self.df is not None:
            # Jeśli brak wyniku przetwarzania, użyj oryginalnych danych
            self._display_dataframe(self.df)
        else:
            # Brak danych całkowicie
            messagebox.showwarning("Brak danych", "Najpierw wczytaj plik CSV!")

    def _load_csv_from_menu(self):
        """Metoda wywoływana przez Plik → Wczytaj CSV"""
        fp = filedialog.askopenfilename(
            title="Wybierz plik CSV",
            filetypes=[("CSV", "*.csv"), ("Wszystkie", "*.*")]
        )
        if not fp:
            return

        self._set_busy("Wczytywanie pliku…")
        df = wczytaj_csv(fp, separator=None, wyswietlaj_informacje=True)
        if df is None:
            self._set_ready()
            messagebox.showerror("Błąd", "Nie udało się wczytać pliku.")
            return

        self.df = df
        self.path = fp
        self.file_info_var.set(f"Wczytano: {fp.split('/')[-1]} ({len(df)}×{len(df.columns)})")

        # Ustaw current_result_df na oryginalne dane
        self.current_result_df = df.copy()

        # Odśwież wszystkie sekcje GUI po wczytaniu danych
        self._update_all_columns(df)
        self._clear_preprocessing_data(df)  # Reset stanu przetwarzania

        # Aktualizuj widok danych w aktualnej zakładce
        if hasattr(self, 'result_tree'):
            self._display_dataframe(df)

        messagebox.showinfo("OK", "Plik wczytany pomyślnie!")
        self._set_ready()

    def _commit_df(self, result, *, dict_key: str | None = None) -> None:
        """
        Przyjmuje:
            •  result  – DataFrame **lub** słownik zawierający DataFrame,
            •  dict_key – jeśli wynik jest słownikiem, klucz pod którym
                        spodziewamy się DataFrame'u (np. 'df_cleaned').
        Aktualizuje:
            •  self.df i self.current_result_df,
            •  wszystkie listy/comboboxy przez _update_all_columns,
            •  tabelę z wynikami i przyciski nawigacji.
        """
        if result is None:
            return

        # 1) wyciągnij DataFrame z wyniku
        if isinstance(result, pd.DataFrame):
            new_df = result
        elif isinstance(result, dict):
            # jeżeli nie podano klucza, spróbuj odgadnąć
            if dict_key is None:
                for k in ("df_cleaned", "df_zakodowany", "df", "df_scaled"):
                    if k in result:
                        dict_key = k
                        break
            if dict_key is None or dict_key not in result:
                raise ValueError("Nie znaleziono DataFrame'u w wyniku funkcji.")
            new_df = result[dict_key]
        else:
            raise TypeError("Nieznany typ wyniku operacji.")

        # 2) zapisz jako główny DataFrame aplikacji
        self.df = new_df
        self.current_result_df = new_df

        # 3) odśwież interfejs we wszystkich zakładkach
        self._update_all_columns(new_df)  # listy kolumn, comboboxy, itp.
        self._clear_preprocessing_data(new_df)  # Reset stanu przetwarzania
        self._display_dataframe(new_df)  # widok w tabeli

        # 4) Aktualizuj również dane w zakładkach AI
        if hasattr(self, 'classification_tree'):
            self._show_page("cls", self.classification_tree)
        if hasattr(self, 'clustering_tree'):
            self._show_page("clu", self.clustering_tree)

    # ─────────────────────────────────────────────
    #  Helpers wizualizacji
    # ─────────────────────────────────────────────
    def _set_combo_state(self, combo: ttk.Combobox, active: bool) -> None:
        """Ustawia stan comboboxa; przy dezaktywacji czyści jego wartość."""
        if active:
            combo.config(state="readonly")
        else:
            combo.set("")
            combo.config(state="disabled")

    # ──────────────────────────────────────────────────────────────
    #  Wspólne: loader CSV + pomoc
    # ──────────────────────────────────────────────────────────────
    def _add_loader(self, parent: tk.Widget, on_success=None) -> None:
        """Przycisk 'Wczytaj plik CSV' + zarządzanie pulsującym paskiem stanu."""
        frm = ttk.Frame(parent)
        frm.pack(fill="x", padx=10, pady=(5, 12))

        def choose() -> None:
            fp = filedialog.askopenfilename(
                title="Wybierz plik CSV",
                filetypes=[("CSV", "*.csv"), ("Wszystkie", "*.*")]
            )
            if not fp:
                return
            self._set_busy("Wczytywanie pliku…")
            df = wczytaj_csv(fp, separator=None, wyswietlaj_informacje=True)
            if df is None:
                self._set_ready()
                messagebox.showerror("Błąd", "Nie udało się wczytać pliku.")
                return
            self.df, self.path = df, fp
            self.file_info_var.set(f"Wczytano: {fp.split('/')[-1]} ({len(df)}×{len(df.columns)})")
            self.current_result_df = df.copy()
            if on_success:
                on_success(df)
            messagebox.showinfo("OK", "Plik wczytany pomyślnie!")
            self._set_ready()


    # ─────────────────────────────────────────────
    #  Paginacja wyników
    # ─────────────────────────────────────────────
    def _setup_pagination_controls(self, parent: tk.Widget) -> None:
        nav = ttk.Frame(parent)
        nav.pack(fill="x", padx=10, pady=(0, 6))

        self.prev_btn = ttk.Button(nav, text="« Poprzednia", width=12,
                                   command=self._prev_page)
        self.prev_btn.pack(side="left")

        self.next_btn = ttk.Button(nav, text="Następna »", width=12,
                                   command=self._next_page)
        self.next_btn.pack(side="left", padx=(6, 0))

        ttk.Label(nav, text=" |  Rekordów na stronę:").pack(side="left", padx=6)

        self.page_size_cmb = ttk.Combobox(nav, width=6, state="readonly",
                                          values=["50", "100", "200", "500", "Wszystkie"])
        self.page_size_cmb.set(str(self.page_size))
        self.page_size_cmb.bind("<<ComboboxSelected>>", self._change_page_size)
        self.page_size_cmb.pack(side="left")

        self.page_info_var = tk.StringVar(value="")
        ttk.Label(nav, textvariable=self.page_info_var).pack(side="right")

    # ─────────────────────────────────────────────
    #  Paginator dla dowolnego Treeview
    # ─────────────────────────────────────────────
    def _init_paginator(self, name: str):
        """Tworzy atrybuty stanu paginacji dla wskazanego ekranu."""
        setattr(self, f"{name}_df", None)
        setattr(self, f"{name}_df_id", None)
        setattr(self, f"{name}_page", 0)
        setattr(self, f"{name}_size", 100)

    def _build_paginator_ui(self, parent: tk.Widget,
                            name: str,
                            prev_cmd, next_cmd, size_cmd):
        """Rysuje belkę nawigacyjną i zapisuje referencje w self.<name>_*"""
        bar = ttk.Frame(parent);
        bar.pack(fill="x", padx=10, pady=(0, 6))

        prev = ttk.Button(bar, text="« Poprzednia", width=12, command=prev_cmd)
        prev.pack(side="left")
        nextb = ttk.Button(bar, text="Następna »", width=12, command=next_cmd)
        nextb.pack(side="left", padx=(6, 0))

        ttk.Label(bar, text=" | Rekordów/str.:").pack(side="left", padx=6)
        size_cmb = ttk.Combobox(bar, width=6, state="readonly",
                                values=["50", "100", "200", "500", "Wszystkie"])
        size_cmb.set("100")
        size_cmb.bind("<<ComboboxSelected>>", size_cmd)
        size_cmb.pack(side="left")

        info = tk.StringVar(value="")
        ttk.Label(bar, textvariable=info).pack(side="right")

        # zachowaj referencje
        setattr(self, f"{name}_prev_btn", prev)
        setattr(self, f"{name}_next_btn", nextb)
        setattr(self, f"{name}_size_cmb", size_cmb)
        setattr(self, f"{name}_info_var", info)

    def _change_page_size(self, evt=None):
        val = self.page_size_cmb.get()
        self.page_size = -1 if val == "Wszystkie" else int(val)
        self.current_page = 0
        if self.current_result_df is not None:
            self._display_dataframe(self.current_result_df)

    def _show_page(self, name: str, tree: ttk.Treeview):
        """Odświeża wskazany Treeview wg bieżących ustawień paginacji."""
        df = getattr(self, f"{name}_df")
        if df is None:  # nic do pokazania
            return

        size = getattr(self, f"{name}_size")
        page = getattr(self, f"{name}_page")
        total = len(df)

        if size == -1:  # 'Wszystkie'
            start, end = 0, None
            pages = 1
            page_num = 1
        else:
            pages = max(1, (total - 1) // size + 1)
            page = min(page, pages - 1)
            setattr(self, f"{name}_page", page)
            start = page * size
            end = start + size
            page_num = page + 1

        view = df.iloc[start:end]

        tree.delete(*tree.get_children())
        tree["columns"] = list(view.columns)
        for col in view.columns:
            tree.heading(col, text=col)
            tree.column(col, width=90, anchor="center")
        for _, row in view.iterrows():
            tree.insert("", "end", values=list(row))

        info_var = getattr(self, f"{name}_info_var")
        info_var.set(f"Strona {page_num}/{pages}  "
                     f"({start + 1}-{min(end or total, total)} z {total})")

        # aktywność strzałek
        prev_btn = getattr(self, f"{name}_prev_btn")
        next_btn = getattr(self, f"{name}_next_btn")
        prev_btn.state(["!disabled"] if page_num > 1 else ["disabled"])
        next_btn.state(["!disabled"] if page_num < pages else ["disabled"])

    def _prev_page(self):
        if self.current_page > 0:
            self.current_page -= 1
            self._display_dataframe(self.current_result_df)

    def _next_page(self):
        if self.current_result_df is None or self.page_size == -1:
            return
        max_page = (len(self.current_result_df) - 1) // self.page_size
        if self.current_page < max_page:
            self.current_page += 1
            self._display_dataframe(self.current_result_df)

    def _change_page_size_generic(self, evt, name: str, tree: ttk.Treeview):
        cmb = getattr(self, f"{name}_size_cmb")
        val = cmb.get()
        setattr(self, f"{name}_size", -1 if val == "Wszystkie" else int(val))
        setattr(self, f"{name}_page", 0)
        self._show_page(name, tree)

    def _prev_generic(self, name: str, tree: ttk.Treeview):
        if getattr(self, f"{name}_page") > 0:
            setattr(self, f"{name}_page", getattr(self, f"{name}_page") - 1)
            self._show_page(name, tree)

    def _next_generic(self, name: str, tree: ttk.Treeview):
        size = getattr(self, f"{name}_size")
        if size == -1:  # jedna strona
            return
        df = getattr(self, f"{name}_df")
        max_page = (len(df) - 1) // size
        if getattr(self, f"{name}_page") < max_page:
            setattr(self, f"{name}_page", getattr(self, f"{name}_page") + 1)
            self._show_page(name, tree)

    # W metodzie _update_all_columns dodajemy aktualizację dla wizualizacji:
    def _update_all_columns(self, df: pd.DataFrame) -> None:
        """Odśwież wszystkie comboboxy/listboxy w całym GUI po wczytaniu nowego DataFrame."""
        # Preprocessing
        self._clear_preprocessing_data(df)

        # Wizualizacje
        if hasattr(self, 'x_col'):
            self._update_visualization_columns(df)

        # AI - klasyfikacja
        if hasattr(self, 'feature_listbox'):
            self._update_classification_columns(df)

        # AI - klasteryzacja
        if hasattr(self, 'clustering_listbox'):
            self._update_clustering_columns(df)

    def _update_visualization_columns(self, df: pd.DataFrame) -> None:
        """Aktualizuje listy kolumn w zakładce wizualizacji"""
        # Pobierz listy kolumn
        cols = df.columns.tolist()
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include=["category", "object"]).columns.tolist()

        # Aktualizuj comboboxy
        self.x_col["values"] = cols
        self.y_col["values"] = numeric_cols
        self.hue_col["values"] = ["brak"] + cat_cols

        # Ustaw domyślne wartości jeśli istnieją
        self.x_col.set(cols[0] if cols else "")
        self.y_col.set(numeric_cols[0] if numeric_cols else "")
        self.hue_col.set("brak")

    def _load_pre(self) -> None:
        """Buduje zakładkę "Cleaning" – pre-processing."""
        self._clear_tabs()
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="Cleaning")

        self._add_refresh_button(tab)

        # Kontenery z podzakładkami
        notebook = ttk.Notebook(tab)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)

        extract_frame = ttk.Frame(notebook)
        self._build_extraction_tab(extract_frame)
        notebook.add(extract_frame, text="Ekstrakcja")

        duplicates_frame = ttk.Frame(notebook)
        self._build_duplicates_tab(duplicates_frame)
        notebook.add(duplicates_frame, text="Duplikaty")

        missing_frame = ttk.Frame(notebook)
        self._build_missing_tab(missing_frame)
        notebook.add(missing_frame, text="Braki danych")

        encoding_frame = ttk.Frame(notebook)
        self._build_encoding_tab(encoding_frame)
        notebook.add(encoding_frame, text="Kodowanie")

        scaling_frame = ttk.Frame(notebook)
        self._build_scaling_tab(scaling_frame)
        notebook.add(scaling_frame, text="Skalowanie")

        replace_frame = ttk.Frame(notebook)
        self._build_value_replacement_tab(replace_frame)
        notebook.add(replace_frame, text="Zamiana wartości")

        # Tabela wyników + paginacja
        self.result_tree, ybar, xbar = self._make_treeview(tab, [])
        self.result_tree.pack(fill="both", expand=True, padx=10, pady=10)
        self._setup_pagination_controls(tab)

        # Odśwież dane jeśli są dostępne
        if self.df is not None:
            self._clear_preprocessing_data(self.df)
            self._display_dataframe(self.df)

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

    def _run_extraction(self) -> None:
        if self.df is None:
            messagebox.showwarning("Brak danych", "Proszę najpierw wczytać plik CSV!")
            return

        # Parsowanie pól
        rows = None
        if self.rows_entry.get().strip():
            try:
                rows = list(map(int, self.rows_entry.get().split(",")))
            except ValueError:
                messagebox.showerror("Błąd",
                                     "Nieprawidłowy format wierszy. Podaj indeksy oddzielone przecinkami.")
                return
        cols = None
        if self.cols_entry.get().strip():
            cols = [c.strip() for c in self.cols_entry.get().split(",")]

        self._set_busy("Ekstrakcja danych…")
        try:
            result = ekstrakcja_podtablicy(
                self.df, rows=rows, cols=cols,
                mode=self.mode_var.get(), wyswietlaj_informacje=True
            )
            self._commit_df(result)
            messagebox.showinfo("OK", "Ekstrakcja zakończona!")
        except Exception as e:
            messagebox.showerror("Błąd", str(e))
        finally:
            self._set_ready()

    def _display_dataframe(self, df: pd.DataFrame) -> None:
        """Pokazuje fragment DataFrame zależnie od paginacji."""
        # Jeśli to nowy DF → reset strony
        if id(df) != self._pagination_df_id:
            self._pagination_df_id = id(df)
            self.current_page = 0

        total_len = len(df)
        if self.page_size == -1:  # „Wszystkie”
            start, end = 0, None
            total_pages = 1
            page_num = 1
        else:
            total_pages = max(1, (total_len - 1) // self.page_size + 1)
            self.current_page = min(self.current_page, total_pages - 1)
            start = self.current_page * self.page_size
            end = start + self.page_size
            page_num = self.current_page + 1

        df_show = df.iloc[start:end]

        # Odśwież Treeview
        self.result_tree.delete(*self.result_tree.get_children())
        self.result_tree["columns"] = list(df_show.columns)
        for col in df_show.columns:
            self.result_tree.heading(col, text=col)
            self.result_tree.column(col, width=100, anchor="center")
        for _, row in df_show.iterrows():
            self.result_tree.insert("", "end", values=list(row))

        # Aktualizacja nawigacji
        self.page_info_var.set(f"Strona {page_num}/{total_pages}  "
                               f"(rekordy {start + 1}-{min(end or total_len, total_len)} "
                               f"z {total_len})")

        # dezaktywacja guzików jeśli trzeba
        self.prev_btn.state(["!disabled"] if page_num > 1 else ["disabled"])
        self.next_btn.state(["!disabled"] if page_num < total_pages else ["disabled"])

    def _save_result(self) -> None:
        """Zapis wyniku do CSV"""
        if self.df is None:
            messagebox.showwarning("Brak danych", "Najpierw wczytaj plik CSV!")
            return

        # Jeśli nie ma wyników przetwarzania, zapisz oryginalne dane
        df_to_save = self.current_result_df if self.current_result_df is not None else self.df

        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if file_path:
            try:
                df_to_save.to_csv(file_path, index=False)
                messagebox.showinfo("Sukces", f"Dane zapisano w: {file_path}")
            except Exception as e:
                messagebox.showerror("Błąd", f"Błąd podczas zapisu: {str(e)}")

    def _build_duplicates_tab(self, parent):
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill="both", expand=True, padx=10, pady=10)
        ttk.Label(control_frame, text="Kolumny do sprawdzania:").pack(anchor="w")
        self.duplicates_listbox = tk.Listbox(control_frame, selectmode="multiple", height=6, exportselection=False)
        self.duplicates_listbox.pack(fill="x", padx=5, pady=5)

        # Nowe przyciski
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(fill="x", pady=5)
        ttk.Button(btn_frame, text="Wybierz wszystkie", command=lambda: self._select_all(self.duplicates_listbox)).pack(
            side="left", padx=5)
        ttk.Button(btn_frame, text="Czyść zaznaczenie",
                   command=lambda: self._clear_selection(self.duplicates_listbox)).pack(side="left", padx=5)

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
        ttk.Button(control_frame, text="Usuń duplikaty", command=self._run_duplicate_removal).pack(side="right", padx=5,
                                                                                                   pady=5)

    def _run_duplicate_removal(self) -> None:
        if self.df is None:
            messagebox.showwarning("Brak danych", "Proszę najpierw wczytać plik CSV!")
            return

        self._set_busy("Usuwanie duplikatów…")
        try:
            selected_cols = [self.duplicates_listbox.get(i)
                             for i in self.duplicates_listbox.curselection()]
            result = usun_duplikaty(
                self.df, kolumny=selected_cols or None,
                tryb=self.duplicates_mode.get(), wyswietlaj_info=True
            )

            if result['liczba_duplikatow'] > 0:
                self._commit_df(result, dict_key='df_cleaned')
                messagebox.showinfo(
                    "Sukces",
                    f"Usunięto {result['liczba_duplikatow']} duplikatów!"
                )
            else:
                messagebox.showinfo("Informacja", "Nie znaleziono duplikatów do usunięcia")
        except Exception as e:
            messagebox.showerror("Błąd", str(e))
        finally:
            self._set_ready()

    def _build_missing_tab(self, parent):
        """Zakładka do obsługi brakujących wartości"""
        notebook = ttk.Notebook(parent)
        notebook.pack(fill='both', expand=True)

        # Podzakładka 1: Wypełnianie braków
        fill_frame = ttk.Frame(notebook)
        self._build_fill_missing_tab(fill_frame)
        notebook.add(fill_frame, text="Wypełnij")

        # Podzakładka 2: Usuwanie braków
        remove_frame = ttk.Frame(notebook)
        self._build_remove_missing_tab(remove_frame)
        notebook.add(remove_frame, text="Usuń")

    def _build_fill_missing_tab(self, parent):
        """Panel do wypełniania brakujących wartości"""
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Lista kolumn
        ttk.Label(control_frame, text="Wybierz kolumny:").pack(anchor="w")
        self.missing_listbox = tk.Listbox(control_frame, selectmode="multiple", height=5, exportselection=False)
        self.missing_listbox.pack(fill="x", padx=5, pady=5)

        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(fill="x", pady=5)
        ttk.Button(btn_frame, text="Wybierz wszystkie", command=lambda: self._select_all(self.missing_listbox)).pack(
            side="left", padx=5)
        ttk.Button(btn_frame, text="Czyść zaznaczenie",
                   command=lambda: self._clear_selection(self.missing_listbox)).pack(side="left", padx=5)

        # Metoda wypełnienia
        self.fill_method = tk.StringVar(value="srednia")
        methods = {
            'srednia': 'Średnia',
            'mediana': 'Mediana',
            'moda': 'Moda',
            'stała': 'Wartość stała'
        }
        method_frame = ttk.Frame(control_frame)
        method_frame.pack(fill="x", pady=5)
        for i, (key, val) in enumerate(methods.items()):
            ttk.Radiobutton(method_frame, text=val, variable=self.fill_method, value=key).grid(row=0, column=i, padx=5)

        # Pole dla wartości stałej
        self.const_value = ttk.Entry(control_frame)
        ttk.Label(control_frame, text="Wartość stała:").pack(anchor="w", pady=(10, 0))
        self.const_value.pack(fill="x", padx=5)

        # Przyciski wykonania
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(fill="x", pady=10)
        ttk.Button(btn_frame, text="Wypełnij braki", command=self._run_fill_missing) \
            .pack(side="right", padx=5)
        ttk.Button(btn_frame, text="Edytuj ręcznie", command=self._edit_selected_row) \
            .pack(side="right", padx=5)

    def _edit_selected_row(self):
        # Jeśli nie ma wyniku operacji, użyj oryginalnych danych
        df_to_edit = self.current_result_df if self.current_result_df is not None else self.df
        if df_to_edit is None:
            messagebox.showwarning("Brak danych", "Najpierw wczytaj plik CSV!")
            return

        selected_items = self.result_tree.selection()
        if not selected_items:
            messagebox.showwarning("Brak wyboru", "Wybierz wiersz do edycji.")
            return

        item = selected_items[0]
        item_index = self.result_tree.index(item)

        if self.page_size == -1:
            start = 0
        else:
            start = self.current_page * self.page_size
        actual_index = start + item_index

        if actual_index >= len(df_to_edit):
            messagebox.showerror("Błąd", "Nieprawidłowy indeks wiersza.")
            return

        self._open_edit_dialog(actual_index, df_to_edit)

    def _open_edit_dialog(self, index, df):
        dialog = tk.Toplevel(self)
        dialog.title("Edytuj wiersz")
        dialog.geometry("400x500")
        dialog.transient(self)
        dialog.grab_set()

        # Główny kontener z Canvas i Scrollbar
        main_frame = ttk.Frame(dialog)
        main_frame.pack(fill="both", expand=True)

        # Canvas z Scrollbar
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Ustawienie kolumny dla Canvas (0) i Scrollbara (1)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        # Konfiguracja siatki dla main_frame
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)

        # Wypełnianie scrollable_frame polami edycyjnymi
        row_data = df.iloc[index]
        entries = {}

        for col in df.columns:
            frame = ttk.Frame(scrollable_frame)
            frame.pack(fill="x", padx=10, pady=2)
            ttk.Label(frame, text=col).pack(anchor="w")
            entry = ttk.Entry(frame)
            entry.insert(0, str(row_data[col]))
            entry.pack(fill="x")
            entries[col] = entry

        # Ramka dla przycisków (poza obszarem przewijania)
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill="x", padx=10, pady=10)

        def save():
            try:
                for col, entry in entries.items():
                    value = entry.get()
                    dtype = df[col].dtype

                    if pd.api.types.is_integer_dtype(dtype):
                        df.at[index, col] = int(value)
                    elif pd.api.types.is_float_dtype(dtype):
                        df.at[index, col] = float(value)
                    else:
                        df.at[index, col] = value

                if self.current_result_df is None:
                    self.current_result_df = df.copy()

                self._display_dataframe(self.current_result_df if self.current_result_df is not None else df)
                dialog.destroy()
                messagebox.showinfo("Sukces", "Wiersz zaktualizowany.")
            except Exception as e:
                messagebox.showerror("Błąd", f"Nieprawidłowa wartość: {str(e)}")

        # Przyciski zawsze widoczne w dolnej części okna
        ttk.Button(button_frame, text="Zapisz", command=save).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Anuluj", command=dialog.destroy).pack(side="left", padx=5)

    def _build_remove_missing_tab(self, parent):
        """Panel do usuwania brakujących wartości"""
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Wybór osi
        self.missing_axis = tk.StringVar(value="wiersze")
        axis_frame = ttk.Frame(control_frame)
        axis_frame.pack(fill="x", pady=5)

        ttk.Radiobutton(axis_frame, text="Wiersze", variable=self.missing_axis, value="wiersze").pack(side="left")
        ttk.Radiobutton(axis_frame, text="Kolumny", variable=self.missing_axis, value="kolumny").pack(side="left")

        # Minimalna liczba wartości
        ttk.Label(control_frame, text="Min. liczba niepustych:").pack(anchor="w", pady=(10, 0))
        self.min_non_missing = ttk.Spinbox(control_frame, from_=1, to=100, width=5)
        self.min_non_missing.set(1)
        self.min_non_missing.pack(anchor="w", padx=5)

        # Przycisk wykonania
        ttk.Button(control_frame, text="Usuń braki", command=self._run_remove_missing) \
            .pack(side="right", padx=5, pady=10)

    def _clear_preprocessing_data(self, df: pd.DataFrame) -> None:
        """Resetuje stan po wczytaniu nowego pliku"""
        # Aktualizacja list kolumn
        for listbox in [getattr(self, 'duplicates_listbox', None),
                        getattr(self, 'missing_listbox', None)]:
            if listbox is not None:
                listbox.delete(0, tk.END)
                for col in df.columns:
                    listbox.insert(tk.END, col)

        if hasattr(self, 'encoding_listbox'):
            self.encoding_listbox.delete(0, tk.END)
            categorical_cols = [col for col in df.columns if
                                pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_object_dtype(df[col])]
            for col in categorical_cols:
                self.encoding_listbox.insert(tk.END, col)

        if hasattr(self, 'target_combobox'):
            numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
            self.target_combobox["values"] = numeric_cols
            if numeric_cols:
                self.target_combobox.set(numeric_cols[0])

        if hasattr(self, 'scaling_listbox'):
            self.scaling_listbox.delete(0, tk.END)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            for col in numeric_cols:
                self.scaling_listbox.insert(tk.END, col)

        if hasattr(self, 'replace_col_combobox'):
            self.replace_col_combobox["values"] = df.columns.tolist()
            if df.columns.any():
                self.replace_col_combobox.set(df.columns[0])

        # Czyszczenie reguł
        self._replacement_rules = {}
        self._update_rules_listbox()

        if hasattr(self, 'x_col'):
            self._update_columns(df)

        if self.path:
            self.file_info_var.set(f"Wczytano: {self.path.split('/')[-1]} ({len(df)}×{len(df.columns)})")
        else:
            self.file_info_var.set("Brak wczytanego pliku")
    def _run_fill_missing(self) -> None:
        if self.df is None:
            messagebox.showwarning("Brak danych", "Proszę najpierw wczytać plik CSV!")
            return

        selected_cols = [self.missing_listbox.get(i)
                         for i in self.missing_listbox.curselection()]
        method = self.fill_method.get()
        const_val = self.const_value.get() if method == 'stała' else None
        reguly = {col: method for col in selected_cols} if selected_cols else None

        self._set_busy("Wypełnianie braków…")
        try:
            result = uzupelnij_braki(
                self.df, metoda=method, wartosc_stala=const_val,
                reguly=reguly, wyswietlaj_info=True
            )
            # <---  commit  --->
            self._commit_df(result)
            messagebox.showinfo("Sukces", "Braki wypełnione!")
        except Exception as e:
            messagebox.showerror("Błąd", str(e))
        finally:
            self._set_ready()

    def _run_remove_missing(self) -> None:
        if self.df is None:
            messagebox.showwarning("Brak danych", "Proszę najpierw wczytać plik CSV!")
            return

        self._set_busy("Usuwanie braków…")
        try:
            result = usun_braki(
                self.df,
                os_wiersze_kolumny=self.missing_axis.get(),
                liczba_min_niepustych=int(self.min_non_missing.get()),
                wyswietlaj_info=True
            )
            # <---  commit  --->
            self._commit_df(result)
            messagebox.showinfo("Sukces", "Brakujące wiersze/kolumny usunięte!")
        except Exception as e:
            messagebox.showerror("Błąd", str(e))
        finally:
            self._set_ready()

    def _build_encoding_tab(self, parent):
        """Zakładka do transformacji kategorycznych"""
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Wybór metody kodowania
        ttk.Label(control_frame, text="Metoda kodowania:").pack(anchor="w")
        self.encoding_method = tk.StringVar(value="one_hot")
        methods = [("One-Hot", "one_hot"), ("Binarne", "binary"), ("Target", "target")]
        method_frame = ttk.Frame(control_frame)
        method_frame.pack(fill="x", pady=5)

        for text, value in methods:
            ttk.Radiobutton(method_frame, text=text, variable=self.encoding_method, value=value).pack(side="left")

        # Lista kolumn do kodowania
        ttk.Label(control_frame, text="Kolumny do zakodowania:").pack(anchor="w", pady=(10, 0))
        self.encoding_listbox = tk.Listbox(control_frame, selectmode="multiple", height=5, exportselection=False)
        self.encoding_listbox.pack(fill="x", padx=5, pady=5)

        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(fill="x", pady=5)
        ttk.Button(btn_frame, text="Wybierz wszystkie", command=lambda: self._select_all(self.encoding_listbox)).pack(
            side="left", padx=5)
        ttk.Button(btn_frame, text="Czyść zaznaczenie",
                   command=lambda: self._clear_selection(self.encoding_listbox)).pack(side="left", padx=5)

        # Dodatkowe opcje dla poszczególnych metod
        self.encoding_options_frame = ttk.Frame(control_frame)
        self.encoding_options_frame.pack(fill="x", pady=5)

        # One-Hot options
        self.oh_drop_first = tk.BooleanVar()
        ttk.Checkbutton(self.encoding_options_frame, text="Usuń pierwszą kolumnę", variable=self.oh_drop_first).pack(
            side="left")

        # Target encoding options
        ttk.Label(self.encoding_options_frame, text="Kolumna docelowa:").pack(side="left", padx=5)
        self.target_combobox = ttk.Combobox(self.encoding_options_frame, state="readonly")
        self.target_combobox.pack(side="left")

        # Przycisk wykonania
        ttk.Button(control_frame, text="Zastosuj kodowanie", command=self._run_encoding).pack(side="right", padx=5,
                                                                                              pady=10)

    def _run_encoding(self) -> None:
        if self.df is None:
            messagebox.showwarning("Brak danych", "Proszę najpierw wczytać plik CSV!")
            return
        selected_cols = [self.encoding_listbox.get(i)
                         for i in self.encoding_listbox.curselection()]
        if not selected_cols:
            messagebox.showerror("Błąd", "Proszę wybrać kolumny do kodowania")
            return
        method = self.encoding_method.get()
        self._set_busy("Kodowanie kategorii…")
        try:
            if method == "one_hot":
                result = jedno_gorace_kodowanie(
                    self.df, kolumny=selected_cols,
                    usun_pierwsza=self.oh_drop_first.get(), wyswietl_informacje=True
                )
                self.current_result_df = result['df_zakodowany']
            elif method == "binary":
                result = binarne_kodowanie(
                    self.df, kolumny=selected_cols, wyswietlaj_informacje=True
                )
                self.current_result_df = result['df_zakodowany']
            else:  # target
                target_col = self.target_combobox.get()
                if not target_col:
                    raise ValueError("Proszę wybrać kolumnę docelową")
                result = kodowanie_docelowe(
                    self.df, kolumny=selected_cols, target=target_col,
                    wyswietlaj_informacje=True
                )
                self.current_result_df = result['df_encoded']

            self._display_dataframe(self.current_result_df)
            messagebox.showinfo("Sukces", "Pomyślnie zastosowano kodowanie!")
            # <---  commit  --->
            self._commit_df(result, dict_key='df_zakodowany')
            messagebox.showinfo("Sukces", "Kodowanie zakończone!")
        except Exception as e:
            messagebox.showerror("Błąd", str(e))
        finally:
            self._set_ready()

    def _build_scaling_tab(self, parent):
        """Zakładka do skalowania numerycznego"""
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Wybór metody skalowania
        ttk.Label(control_frame, text="Metoda skalowania:").pack(anchor="w")
        self.scaling_method = tk.StringVar(value="minmax")
        methods = [("Min-Max [0-1]", "minmax"), ("Standard (Z-score)", "standard")]
        method_frame = ttk.Frame(control_frame)
        method_frame.pack(fill="x", pady=5)

        for text, value in methods:
            ttk.Radiobutton(method_frame, text=text, variable=self.scaling_method, value=value).pack(side="left")

        # Lista kolumn do skalowania
        ttk.Label(control_frame, text="Kolumny do skalowania:").pack(anchor="w", pady=(10, 0))
        self.scaling_listbox = tk.Listbox(control_frame, selectmode="multiple", height=5, exportselection=False)
        self.scaling_listbox.pack(fill="x", padx=5, pady=5)

        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(fill="x", pady=5)
        ttk.Button(btn_frame, text="Wybierz wszystkie", command=lambda: self._select_all(self.scaling_listbox)).pack(
            side="left", padx=5)
        ttk.Button(btn_frame, text="Czyść zaznaczenie",
                   command=lambda: self._clear_selection(self.scaling_listbox)).pack(side="left", padx=5)

        # Przycisk wykonania
        ttk.Button(control_frame, text="Zastosuj skalowanie", command=self._run_scaling).pack(side="right", padx=5,
                                                                                              pady=10)

    def _run_scaling(self) -> None:
        if self.df is None:
            messagebox.showwarning("Brak danych", "Proszę najpierw wczytać plik CSV!")
            return

        selected_cols = [self.scaling_listbox.get(i)
                         for i in self.scaling_listbox.curselection()]
        if not selected_cols:
            messagebox.showerror("Błąd", "Proszę wybrać kolumny do skalowania")
            return

        method = self.scaling_method.get()
        self._set_busy("Skalowanie…")
        try:
            if method == "minmax":
                result = minmax_scaler(
                    self.df, kolumny=selected_cols,
                    wyswietlaj_informacje=True, zwroc_tylko_dane=True
                )
            else:
                result = standard_scaler(
                    self.df, kolumny=selected_cols,
                    wyswietlaj_informacje=True, zwroc_tylko_dane=True
                )
            # <---  commit  --->
            self._commit_df(result)
            messagebox.showinfo("Sukces", "Skalowanie zakończone!")
        except Exception as e:
            messagebox.showerror("Błąd", str(e))
        finally:
            self._set_ready()

    def _build_value_replacement_tab(self, parent):
        """Zakładka do zamiany wartości w DataFrame"""
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Sekcja ręcznej zamiany
        ttk.Label(control_frame, text="Ręczna zamiana:", font=('Helvetica', 10, 'bold')).pack(anchor="w", pady=(0, 5))

        # Wybór kolumny
        col_frame = ttk.Frame(control_frame)
        col_frame.pack(fill="x", pady=2)
        ttk.Label(col_frame, text="Kolumna:").pack(side="left")
        self.replace_col_combobox = ttk.Combobox(col_frame, state="readonly", width=20)
        self.replace_col_combobox.pack(side="left", padx=5)

        # Wartości do zamiany
        val_frame = ttk.Frame(control_frame)
        val_frame.pack(fill="x", pady=2)
        ttk.Label(val_frame, text="Stara wartość:").pack(side="left")
        self.old_val_entry = ttk.Entry(val_frame)
        self.old_val_entry.pack(side="left", padx=5)

        ttk.Label(val_frame, text="Nowa wartość:").pack(side="left", padx=(10, 0))
        self.new_val_entry = ttk.Entry(val_frame)
        self.new_val_entry.pack(side="left", padx=5)

        # Przyciski ręcznej zamiany
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(fill="x", pady=5)
        ttk.Button(btn_frame, text="Dodaj regułę", command=self._add_replacement_rule).pack(side="left")
        ttk.Button(btn_frame, text="Zastosuj zamianę", command=self._run_value_replacement).pack(side="right")

        # Lista aktywnych reguł
        ttk.Label(control_frame, text="Aktywne reguły:", font=('Helvetica', 10, 'bold')).pack(anchor="w", pady=(10, 0))
        self.rules_listbox = tk.Listbox(control_frame, height=6, selectmode="single")
        self.rules_listbox.pack(fill="both", expand=True, pady=5)

        # Przyciski zarządzania regułami
        manage_frame = ttk.Frame(control_frame)
        manage_frame.pack(fill="x")
        ttk.Button(manage_frame, text="Usuń regułę", command=self._remove_rule).pack(side="left")
        ttk.Button(manage_frame, text="Wyczyść listę", command=self._clear_rules).pack(side="right")

    def _add_replacement_rule(self):
        """Dodaje nową regułę zamiany"""
        col = self.replace_col_combobox.get()
        old_val = self.old_val_entry.get()
        new_val = self.new_val_entry.get()

        if not col or not old_val or not new_val:
            messagebox.showwarning("Brak danych", "Wypełnij wszystkie pola!")
            return

        # Konwersja typów dla wartości numerycznych
        try:
            old_val = float(old_val) if '.' in old_val else int(old_val)
        except ValueError:
            pass

        try:
            new_val = float(new_val) if '.' in new_val else int(new_val)
        except ValueError:
            pass

        # Dodanie reguły do słownika
        if col not in self._replacement_rules:
            self._replacement_rules[col] = {}

        self._replacement_rules[col][old_val] = new_val
        self._update_rules_listbox()

    def _update_rules_listbox(self):
        """Aktualizuje listę reguł w UI"""
        self.rules_listbox.delete(0, tk.END)
        for col, rules in self._replacement_rules.items():
            for old_val, new_val in rules.items():
                self.rules_listbox.insert(tk.END, f"{col}: {old_val} → {new_val}")

    def _remove_rule(self):
        """Usuwa zaznaczoną regułę"""
        selection = self.rules_listbox.curselection()
        if not selection:
            return

        index = selection[0]
        items = [item for item in self.rules_listbox.get(0, tk.END)]
        selected_item = items[index]

        # Parsowanie usuwanej reguły
        col_part, vals_part = selected_item.split(":")
        col = col_part.strip()
        old_val = vals_part.split("→")[0].strip()

        # Konwersja wartości jeśli potrzeba
        try:
            old_val = float(old_val) if '.' in old_val else int(old_val)
        except ValueError:
            pass

        # Usuwanie z słownika reguł
        if col in self._replacement_rules and old_val in self._replacement_rules[col]:
            del self._replacement_rules[col][old_val]
            if not self._replacement_rules[col]:
                del self._replacement_rules[col]

        self._update_rules_listbox()

    def _clear_rules(self):
        """Czyści wszystkie reguły"""
        self._replacement_rules = {}
        self._update_rules_listbox()

    def _run_value_replacement(self) -> None:
        if self.df is None:
            messagebox.showwarning("Brak danych", "Najpierw wczytaj plik CSV!")
            return
        self._set_busy("Zamiana wartości…")
        try:
            result = zamien_wartosci(
                self.df, reguly=self._replacement_rules,
                wyswietlaj_informacje=True
            )

            self._commit_df(result)
            messagebox.showinfo("Sukces", "Wartości zamienione!")
        except Exception as e:
            messagebox.showerror("Błąd", str(e))
        finally:
            self._set_ready()

    # ──────────────────────────────────────────────────────────────
    #  STATYSTYKA
    # ──────────────────────────────────────────────────────────────
    def _load_stats(self) -> None:
        self._clear_tabs()
        self._build_numeric_stats_tab()
        self._build_corr_tab()
        self._build_visualization_tab()
        self._build_non_numeric_stats_tab()

        # Odśwież dane jeśli są dostępne
        if self.df is not None:
            self._update_all_columns(self.df)
            if hasattr(self, 'x_col'):
                self._update_visualization_columns(self.df)

    # ---------- Statystyki liczbowe ---------------------------------------
    def _build_numeric_stats_tab(self) -> None:
        """
        Zakładka "Statystyki liczbowe".  Użytkownik wybiera kolumny numeryczne,
        naciska 'Oblicz statystyki'.  Lista kolumn powinna być odświeżona po wczytaniu CSV.
        """
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="Statystyki liczbowe")

        # Lista dostępnych kolumn numerycznych
        lbl_cols = ttk.Label(tab, text="Wybierz kolumny (Ctrl+klik):")
        lbl_cols.pack(anchor="w", padx=10)

        listbox = tk.Listbox(tab, selectmode="multiple", height=6, exportselection=False)
        listbox.pack(fill="x", padx=10, pady=(0, 10))

        # Tabela wyników
        cols = ("kolumna", "średnia", "mediana", "min", "max", "std", "liczba")
        tree, ybar, xbar = self._make_treeview(tab, cols)
        tree.pack(fill="both", expand=True, padx=10, pady=5)

        def update_selector(df: pd.DataFrame) -> None:
            """
            Callback: uzupełnia listbox nazwami kolumn numerycznych.
            Wywoływane zawsze po wczytaniu CSV (przez _update_all_columns).
            """
            listbox.delete(0, tk.END)
            numeric = df.select_dtypes(include=[np.number]).columns
            for col in numeric:
                listbox.insert(tk.END, col)

        def run() -> None:
            """Funkcja wywoływana po kliknięciu 'Oblicz statystyki'."""
            if not self.path:
                messagebox.showwarning("Uwaga", "Najpierw wczytaj plik.")
                return
            selection = [listbox.get(i) for i in listbox.curselection()]
            _, wyniki = analizuj_dane_numeryczne(self.path,
                                                 wybrane_kolumny=selection or None)

            # Odśwież Treeview
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

        ttk.Button(tab, text="Oblicz statystyki", command=run) \
            .pack(anchor="w", padx=10, pady=(0, 6))

        # ========== ARMATURA: loader CSV (tylko do odświeżenia listy) ==========
        self._add_loader(tab, on_success=update_selector)

    def _build_non_numeric_stats_tab(self) -> None:
        """
        Zakładka "Statystyki nieliczone" – pokazuje informacje o kolumnach kategorycznych.
        Po wczytaniu CSV musi odświeżyć tabelę kolumn nieliczących.
        """
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="Statystyki nieliczone")

        # ARMATURA: loader CSV do odświeżenia (wywoła self._update_all_columns)
        self._add_loader(tab, on_success=self._update_all_columns)

        # Tytuł
        ttk.Label(tab, text="Statystyki dla kolumn nielicznych:",
                  font=('Helvetica', 10, 'bold')).pack(anchor="w", padx=10, pady=(10, 5))

        # Tabela wyników
        cols = ("kolumna", "liczba wystąpień", "unikalne wartości",
                "najczęstsza wartość", "częstotliwość [%]", "wypełnienie [%]")
        tree, ybar, xbar = self._make_treeview(tab, cols)
        tree.pack(fill="both", expand=True, padx=10, pady=5)

        # Przycisk analizy
        ttk.Button(tab, text="Oblicz statystyki nieliczone",
                   command=lambda: self._run_non_numeric_stats(tree)) \
            .pack(anchor="w", padx=10, pady=(0, 6))

    def _run_non_numeric_stats(self, tree: ttk.Treeview) -> None:
        if self.df is None:
            messagebox.showwarning("Brak danych", "Najpierw wczytaj plik CSV!")
            return
        self._set_busy("Analiza nieliczbowa…")
        try:
            from Backend.Statystyka import oblicz_statystyki_nie_numeryczne
            stats = oblicz_statystyki_nie_numeryczne(self.df)
            tree.delete(*tree.get_children())
            for col_name, col_stats in stats.items():
                tree.insert("", "end", values=(
                    col_name,
                    col_stats["liczba_wystapien"],
                    col_stats["wartosci_unikalne"],
                    col_stats["najczestsza_wartosc"],
                    f"{col_stats['czestotliwosc_najczestszej'] * 100:.2f}",
                    f"{col_stats['procent_wypelnienia']:.2f}"
                ))
        except Exception as e:
            messagebox.showerror("Błąd", str(e))
        finally:
            self._set_ready()

    # ---------- Korelacje -------------------------------------------------
    # wewnątrz klasy MainApp
    def _build_corr_tab(self) -> None:
        """
        Zakładka "Korelacje": użytkownik wybiera metodę (Pearson/Spearman),
        naciska 'Oblicz korelacje', a następnie widzi macierz korelacji.
        """
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="Korelacje")

        # ARMATURA: loader CSV (odświeży listy kolumn, jeśli potrzebne)
        self._add_loader(tab, on_success=self._update_all_columns)

        # Wybór metody
        method_var = tk.StringVar(value="pearson")
        frm_select = ttk.Frame(tab);
        frm_select.pack(anchor="w", padx=10, pady=(0, 8))
        ttk.Radiobutton(frm_select, text="Pearson", variable=method_var, value="pearson").pack(side="left")
        ttk.Radiobutton(frm_select, text="Spearman", variable=method_var, value="spearman").pack(side="left")

        # Pusta tabela (Treeview)
        tree, ybar, xbar = self._make_treeview(tab, cols=())
        tree.pack(fill="both", expand=True, padx=10, pady=5)

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

            # Konfiguracja kolumn w Treeview
            cols = ("Variable", *df_corr.columns)
            tree.config(columns=cols, show="headings")
            tree.heading("Variable", text="Variable")
            tree.column("Variable", width=130, anchor="w")

            for col in df_corr.columns:
                tree.heading(col, text=col)
                tree.column(col, width=90, anchor="center")

            # Wypełniamy wiersze
            tree.delete(*tree.get_children())
            for row_name, values in df_corr.iterrows():
                tree.insert("", "end", values=(row_name, *np.round(values.values, 4)))

        ttk.Button(tab, text="Oblicz korelacje", command=run_corr) \
            .pack(anchor="w", padx=10, pady=(0, 6))

    def _build_visualization_tab(self) -> None:
        """
        Zakładka "Wizualizacje" - ZMODYFIKOWANA WERSJA bez sekcji historii
        """
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="Wizualizacje")

        # ARMATURA: loader CSV
        self._add_loader(tab, on_success=self._update_visualization_columns)

        # Główny układ: lewa część = panel kontroli, prawa część = kanwa wykresu
        main_frame = ttk.Frame(tab)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # ────── PANEL KONTROLNY Z PRZEWIJANIEM ──────
        # Kontener dla panelu kontrolnego
        control_container = ttk.Frame(main_frame)
        control_container.pack(side="left", fill="y", padx=(0, 10))

        # Canvas z scrollbarem dla panelu kontrolnego
        control_canvas = tk.Canvas(control_container, width=280, highlightthickness=0)
        control_scrollbar = ttk.Scrollbar(control_container, orient="vertical", command=control_canvas.yview)
        control_frame = ttk.Frame(control_canvas)

        # Konfiguracja przewijania
        control_frame.bind(
            "<Configure>",
            lambda e: control_canvas.configure(scrollregion=control_canvas.bbox("all"))
        )

        control_canvas.create_window((0, 0), window=control_frame, anchor="nw")
        control_canvas.configure(yscrollcommand=control_scrollbar.set)

        # Pack canvas i scrollbar
        control_canvas.pack(side="left", fill="both", expand=True)
        control_scrollbar.pack(side="right", fill="y")

        # Bind mousewheel to canvas
        def _on_mousewheel(event):
            control_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        control_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # ────── ZAWARTOŚĆ PANELU KONTROLNEGO ──────

        # Nagłówek z przyciskami akcji
        header_frame = ttk.Frame(control_frame)
        header_frame.pack(fill="x", pady=(0, 10), padx=5)

        ttk.Label(header_frame, text="Interaktywne Wykresy",
                  font=("Arial", 10, "bold")).pack(anchor="w")

        # Przyciski akcji
        action_frame = ttk.Frame(header_frame)
        action_frame.pack(fill="x", pady=5)

        ttk.Button(action_frame, text="💾", width=4,
                   command=self._save_current_chart).pack(side="left", padx=1)
        ttk.Button(action_frame, text="⟲", width=4,
                   command=self._refresh_current_chart).pack(side="left", padx=1)

        # Separator
        ttk.Separator(control_frame, orient="horizontal").pack(fill="x", pady=5, padx=5)

        # ═══ SEKCJA PODSTAWOWA - KOMPAKTOWA ═══
        basic_frame = ttk.LabelFrame(control_frame, text="Podstawowe", padding=8)
        basic_frame.pack(fill="x", pady=3, padx=5)

        # Typ wykresu
        ttk.Label(basic_frame, text="Typ:", font=("Arial", 8)).pack(anchor="w")
        self.chart_type = ttk.Combobox(basic_frame, state="readonly", height=4,
                                       values=["scatter", "line", "bar", "pie"])
        self.chart_type.set("scatter")
        self.chart_type.pack(fill="x", pady=1)
        self.chart_type.bind("<<ComboboxSelected>>", self._update_chart_options)

        # Kolumny - kompaktowo
        for label, attr in [("X:", "x_col"), ("Y:", "y_col"), ("Kolor:", "hue_col")]:
            ttk.Label(basic_frame, text=label, font=("Arial", 8)).pack(anchor="w", pady=(3, 0))
            combo = ttk.Combobox(basic_frame, state="readonly", height=4)
            combo.pack(fill="x", pady=1)
            setattr(self, attr, combo)

        # ═══ OPCJE ZAAWANSOWANE - KOMPAKTOWE ═══
        self.options_frame = ttk.LabelFrame(control_frame, text="Opcje", padding=6)
        self.options_frame.pack(fill="x", pady=3, padx=5)

        # ═══ STYL - KOMPAKTOWY ═══
        style_frame = ttk.LabelFrame(control_frame, text="Styl", padding=6)
        style_frame.pack(fill="x", pady=3, padx=5)

        # Paleta w jednej linii
        palette_frame = ttk.Frame(style_frame)
        palette_frame.pack(fill="x", pady=1)
        ttk.Label(palette_frame, text="Paleta:", font=("Arial", 8), width=8).pack(side="left")
        self.palette_var = tk.StringVar(value="pastel")
        palette_combo = ttk.Combobox(palette_frame, textvariable=self.palette_var,
                                     values=["pastel", "Set1", "Set2", "deep", "bright"],
                                     state="readonly", width=10, height=4)
        palette_combo.pack(side="right", fill="x", expand=True)

        # ═══ ETYKIETY - KOMPAKTOWE ═══
        labels_frame = ttk.LabelFrame(control_frame, text="Etykiety", padding=6)
        labels_frame.pack(fill="x", pady=3, padx=5)

        for lbl, attr in [("Tytuł:", "chart_title"), ("X:", "x_label"), ("Y:", "y_label")]:
            frame = ttk.Frame(labels_frame)
            frame.pack(fill="x", pady=1)
            ttk.Label(frame, text=lbl, font=("Arial", 8), width=6).pack(side="left")
            entry = ttk.Entry(frame, font=("Arial", 8))
            entry.pack(side="right", fill="x", expand=True)
            setattr(self, attr, entry)

        # ═══ GŁÓWNY PRZYCISK - ZAWSZE WIDOCZNY ═══
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill="x", pady=10, padx=5)

        self.generate_btn = ttk.Button(button_frame, text="🎯 GENERUJ WYKRES",
                                       command=self._generate_plot_enhanced,
                                       style="Accent.TButton")
        self.generate_btn.pack(fill="x", ipady=5)  # Większy przycisk

        # Dodaj trochę miejsca na dole
        ttk.Frame(control_frame, height=20).pack()

        # ────── OBSZAR WYKRESU ──────
        chart_frame = ttk.Frame(main_frame)
        chart_frame.pack(side="right", fill="both", expand=True)

        # Toolbar matplotlib
        toolbar_frame = ttk.Frame(chart_frame)
        toolbar_frame.pack(fill="x")

        self.figure = plt.figure(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, master=chart_frame)

        # Dodaj toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Info o wykresie
        self.chart_info_label_var = tk.StringVar(value="Gotowy do generowania")
        info_label = ttk.Label(chart_frame, textvariable=self.chart_info_label_var,
                               font=("Arial", 8), foreground="blue")
        info_label.pack(anchor="w", padx=5, pady=2)

        # Inicjalizacja
        if self.df is not None:
            self._update_visualization_columns(self.df)
        self._update_chart_options()

    def _generate_plot_enhanced(self) -> None:
        """Rozszerzona wersja generowania wykresów z historią"""
        if self.df is None:
            messagebox.showwarning("Błąd", "Najpierw wczytaj plik CSV!")
            return

        self._set_busy("Generowanie wykresu…")
        try:
            self.figure.clear()
            ax = self.figure.add_subplot(111)

            # Podstawowe parametry
            params = {
                "df": self.df,
                "typ_wykresu": self.chart_type.get(),
                "kolumna_x": self.x_col.get() or None,
                "nazwa_wykresu": self.chart_title.get() or None,
                "etykieta_x": self.x_label.get() or None,
                "etykieta_y": self.y_label.get() or None,
                "palette": getattr(self, 'palette_var', tk.StringVar(value="pastel")).get(),
                "fig": self.figure,
                "ax": ax
            }

            # Kolumna Y tylko jeśli została wybrana lub jest wymagana
            chart_type = self.chart_type.get()
            if chart_type != "pie":  # Pie nigdy nie używa Y
                y_value = self.y_col.get()
                if y_value:  # Dodaj tylko jeśli Y jest wybrane
                    params["kolumna_y"] = y_value

            # Kolumna hue - tylko dla obsługiwanych typów i gdy wybrano wartość
            if chart_type in ("scatter", "line", "bar"):
                hue_value = self.hue_col.get()
                if hue_value and hue_value != "brak":
                    params["kolumna_hue"] = hue_value

            # Dodatkowe parametry z _chart_vars
            if hasattr(self, '_chart_vars'):
                known_params = {'regline', 'sort_values', 'maks_kategorie', 'min_procent',
                                'show_percentages', 'pie_style'}  # Znane parametry

                for key, var in self._chart_vars.items():
                    if key in known_params and isinstance(var, (tk.BooleanVar, tk.IntVar, tk.DoubleVar, tk.StringVar)):
                        params[key] = var.get()

            rysuj_wykres(**params)

            # Dodaj do historii
            self._add_chart_to_history(params.copy())

            self.canvas.draw()

            # Aktualizuj info
            chart_type = params["typ_wykresu"]
            info = f"Typ: {chart_type} | Rekordów: {len(self.df)}"
            self.chart_info_label_var.set(info)

        except Exception as e:
            messagebox.showerror("Błąd", str(e))
        finally:
            self._set_ready()

    def _add_chart_to_history(self, params):
        """Dodaje wykres do historii"""
        # Usuń przyszłe wykresy jeśli jesteśmy w środku historii
        if self.current_chart_index < len(self.chart_history) - 1:
            self.chart_history = self.chart_history[:self.current_chart_index + 1]

        # Dodaj nowy wykres
        self.chart_history.append(params)
        self.current_chart_index = len(self.chart_history) - 1

        # Ograniczenie do 5 wykresów
        if len(self.chart_history) > 5:
            self.chart_history.pop(0)
            self.current_chart_index -= 1

        self._update_history_buttons()

    def _update_history_buttons(self):
        """Aktualizuje przyciski historii"""
        if not hasattr(self, 'prev_chart_btn'):
            return

        can_go_back = self.current_chart_index > 0
        can_go_forward = self.current_chart_index < len(self.chart_history) - 1

        self.prev_chart_btn.config(state="normal" if can_go_back else "disabled")
        self.next_chart_btn.config(state="normal" if can_go_forward else "disabled")

        total = len(self.chart_history)
        current = self.current_chart_index + 1 if total > 0 else 0
        self.chart_info_var.set(f"{current}/{total}")

    def _prev_chart(self):
        """Poprzedni wykres"""
        if self.current_chart_index > 0:
            self.current_chart_index -= 1
            self._load_chart_from_history()

    def _next_chart(self):
        """Następny wykres"""
        if self.current_chart_index < len(self.chart_history) - 1:
            self.current_chart_index += 1
            self._load_chart_from_history()

    def _load_chart_from_history(self):
        """Ładuje wykres z historii"""
        if 0 <= self.current_chart_index < len(self.chart_history):
            params = self.chart_history[self.current_chart_index]
            try:
                self.figure.clear()
                rysuj_wykres(**params)
                self.canvas.draw()
                self._update_history_buttons()
            except Exception as e:
                messagebox.showerror("Błąd", f"Nie można załadować z historii:\n{str(e)}")

    def _save_current_chart(self):
        """Zapisuje obecny wykres"""
        if not hasattr(self, 'chart_history') or not self.chart_history:
            messagebox.showwarning("Brak wykresu", "Najpierw wygeneruj wykres!")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("All files", "*.*")]
        )

        if file_path:
            try:
                self.figure.savefig(file_path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Sukces", f"Wykres zapisany: {file_path}")
            except Exception as e:
                messagebox.showerror("Błąd", f"Nie można zapisać:\n{str(e)}")

    def _refresh_current_chart(self):
        """Odświeża obecny wykres"""
        if hasattr(self, 'chart_history') and self.chart_history:
            self._load_chart_from_history()

    def _update_chart_options(self, event=None) -> None:
        """Pokazuje tylko te opcje, które pasują do wybranego typu wykresu."""
        # Wyczyść panel opcji
        for w in self.options_frame.winfo_children():
            w.destroy()

        chart_type = self.chart_type.get()

        # Reset / aktywizacja głównych comboboxów zależnie od typu wykresu
        needs_x = chart_type in ("scatter", "line", "bar", "pie")  # Wszystkie potrzebują X
        needs_y = chart_type in ("scatter", "line")  # Tylko scatter i line zawsze potrzebują Y
        optional_y = chart_type in ("bar")  # Bar może, ale nie musi mieć Y

        # Pole Kolor tylko dla scatter i line, a nie dla bar (usunięte dla wykresu słupkowego)
        needs_hue = chart_type in ("scatter", "line")

        self._set_combo_state(self.x_col, needs_x)

        # Dla kolumny Y: albo obowiązkowa, albo opcjonalna
        if needs_y:
            self._set_combo_state(self.y_col, True)
        elif optional_y:
            # Bar może, ale nie musi mieć Y - ustaw stan na "readonly" ale bez czyszczenia
            self.y_col.config(state="readonly")
        else:
            # Pie nie używa Y wcale - wyczyść i deaktywuj
            self.y_col.set("")
            self.y_col.config(state="disabled")

        # Pole kolor - tylko dla scatter i line
        self._set_combo_state(self.hue_col, needs_hue)

        # Słownik zmiennych pomocniczych (tworzony raz)
        if not hasattr(self, "_chart_vars"):
            self._chart_vars = {}

        # Dodatkowe opcje specyficzne dla typów wykresów
        if chart_type in ("scatter", "line"):
            self._chart_vars["regline"] = tk.BooleanVar(value=False)
            ttk.Checkbutton(self.options_frame, text="Linia regresji",
                            variable=self._chart_vars["regline"]).pack(anchor="w")

        if chart_type == "bar":
            self._chart_vars["sort_values"] = tk.BooleanVar(value=True)
            ttk.Checkbutton(self.options_frame, text="Sortuj wartości",
                            variable=self._chart_vars["sort_values"]).pack(anchor="w")

            # Dodaj informację o dwóch trybach dla Bar
            ttk.Label(self.options_frame, text="Bar - tryby pracy:",
                      font=("Arial", 9, "bold")).pack(anchor="w", pady=(5, 0))
            ttk.Label(self.options_frame,
                      text="- Z kolumną Y: tradycyjny wykres słupkowy",
                      font=("Arial", 8)).pack(anchor="w")
            ttk.Label(self.options_frame,
                      text="- Bez kolumny Y: wykres liczebności",
                      font=("Arial", 8)).pack(anchor="w")

        if chart_type == "pie":
            ttk.Label(self.options_frame, text="Maks kategorii:").pack(anchor="w")
            self._chart_vars["maks_kategorie"] = tk.IntVar(value=8)
            ttk.Spinbox(self.options_frame, from_=3, to=15,
                        textvariable=self._chart_vars["maks_kategorie"]).pack(anchor="w")

            ttk.Label(self.options_frame, text="Min. procent:").pack(anchor="w")
            self._chart_vars["min_procent"] = tk.DoubleVar(value=1.0)
            ttk.Spinbox(self.options_frame, from_=0.1, to=100.0, increment=0.1,
                        textvariable=self._chart_vars["min_procent"]).pack(anchor="w")
            style_frame = ttk.Frame(self.options_frame)
            style_frame.pack(fill="x", pady=2)

            self._chart_vars["pie_style"] = tk.StringVar(value="full")
            ttk.Radiobutton(style_frame, text="Pełny krąg",
                            variable=self._chart_vars["pie_style"], value="full").pack(side="left")
            ttk.Radiobutton(style_frame, text="Pączek",
                            variable=self._chart_vars["pie_style"], value="donut").pack(side="left")

            # NOWA OPCJA: Pokazuj procenty
            self._chart_vars["show_percentages"] = tk.BooleanVar(value=True)
            ttk.Checkbutton(self.options_frame, text="Pokaż procenty",
                            variable=self._chart_vars["show_percentages"]).pack(anchor="w")
    def _update_columns(self, df: pd.DataFrame) -> None:
        """Aktualizuje listy kolumn po wczytaniu danych"""
        cols = df.columns.tolist()
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include=["category", "object"]).columns.tolist()

        self.x_col["values"] = cols
        self.y_col["values"] = numeric_cols
        self.hue_col["values"] = cat_cols

        if cols:
            self.x_col.set(cols[0])
            self.y_col.set(numeric_cols[0] if numeric_cols else "")
            self.hue_col.set(cat_cols[0] if cat_cols else "")

    def _generate_plot(self) -> None:
        if self.df is None:
            messagebox.showwarning("Błąd", "Najpierw wczytaj plik CSV!")
            return

        self._set_busy("Generowanie wykresu…")
        try:
            self.figure.clear()
            ax = self.figure.add_subplot(111)

            params = {
                "df": self.df,
                "typ_wykresu": self.chart_type.get(),
                "kolumna_x": self.x_col.get() or None,
                "kolumna_y": self.y_col.get() if self.chart_type.get() != "pie" else None,
                "kolumna_hue": self.hue_col.get() if self.hue_col.get() != "brak" else None,
                "nazwa_wykresu": self.chart_title.get() or None,
                "etykieta_x": self.x_label.get() or None,
                "etykieta_y": self.y_label.get() or None,
                "regline": False,
                "sort_values": False,
                "maks_kategorie": 8,
                "min_procent": 1.0,
                "fig": self.figure,
                "ax": ax
            }
            if hasattr(self, '_chart_vars'):
                for key, var in self._chart_vars.items():
                    if isinstance(var, (tk.BooleanVar, tk.IntVar, tk.DoubleVar)):
                        params[key] = var.get()

            rysuj_wykres(**params)
            self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Błąd", str(e))
        finally:
            self._set_ready()

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

# --------------------------------------AI--------------------------------------

    def _build_ai_window(self) -> None:
        """Tworzy zakładki AI w głównym notebooku"""
        self._clear_tabs()

        # Zakładka klasyfikacji
        classification_tab = ttk.Frame(self.nb)
        self.nb.add(classification_tab, text="Klasyfikacja")
        self._build_classification_tab(classification_tab)

        # Zakładka klasteryzacji
        clustering_tab = ttk.Frame(self.nb)
        self.nb.add(clustering_tab, text="Klasteryzacja")
        self._build_clustering_tab(clustering_tab)

        # Odśwież dane jeśli są dostępne
        if self.df is not None:
            self._update_all_columns(self.df)
            if hasattr(self, 'feature_listbox'):
                self._update_classification_columns(self.df)
            if hasattr(self, 'clustering_listbox'):
                self._update_clustering_columns(self.df)

    # ─────────────────────────────────────────────
    def _update_clustering_columns(self, df: pd.DataFrame) -> None:
        """Aktualizuje listy kolumn w zakładce klasteryzacji"""
        if hasattr(self, 'clustering_listbox'):
            self.clustering_listbox.delete(0, tk.END)
            for col in df.columns:
                self.clustering_listbox.insert(tk.END, col)
    def _update_classification_columns(self, df: pd.DataFrame) -> None:
        """Aktualizuje listy kolumn w zakładce klasyfikacji"""
        if hasattr(self, 'feature_listbox'):
            self.feature_listbox.delete(0, tk.END)
            for col in df.columns:
                self.feature_listbox.insert(tk.END, col)

        if hasattr(self, 'target_combobox'):
            self.target_combobox['values'] = df.columns.tolist()
            if df.columns.any():
                self.target_combobox.set(df.columns[0])
    # ─────────────────────────────────────────────
    #  >>> 2. ZAKŁADKA - KLASYFIKACJA  <<<
    # ─────────────────────────────────────────────
    def _build_classification_tab(self, parent: ttk.Frame) -> None:
        """
        Buduje interfejs do klasyfikacji: wybór cech, kolumna docelowa, itp.
        Lista cech i lista kolumn docelowych będą odświeżone przez _update_all_columns.
        """
        main_frame = ttk.Frame(parent);
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # ARMATURA: loader CSV, aby po wczytaniu globalnie załadować kolumny do listboxów
        self._add_loader(parent, on_success=self._update_all_columns)

        control = ttk.Frame(main_frame);
        control.pack(fill="x", padx=10, pady=5)
        ttk.Label(control, text="Wybierz cechy (Ctrl+klik):").pack(anchor="w")
        self.feature_listbox = tk.Listbox(control, selectmode="multiple", height=7, exportselection=False)
        self.feature_listbox.pack(fill="x", padx=5, pady=5)
        self.feature_listbox.bind("<<ListboxSelect>>", self._check_classif_ready)

        btn_frame = ttk.Frame(control)
        btn_frame.pack(fill="x", pady=5)
        ttk.Button(btn_frame, text="Wybierz wszystkie", command=lambda: self._select_all(self.feature_listbox)).pack(
            side="left", padx=5)
        ttk.Button(btn_frame, text="Czyść zaznaczenie",
                   command=lambda: self._clear_selection(self.feature_listbox)).pack(side="left", padx=5)

        ttk.Label(control, text="Kolumna docelowa:").pack(anchor="w")
        self.target_combobox = ttk.Combobox(control, state="readonly")
        self.target_combobox.pack(fill="x", padx=5, pady=5)
        self.target_combobox.bind("<<ComboboxSelected>>", self._check_classif_ready)

        pframe = ttk.Frame(control);
        pframe.pack(fill="x", pady=5)
        ttk.Label(pframe, text="Test size:").pack(side="left")
        self.test_size = ttk.Spinbox(pframe, from_=0.1, to=0.5, increment=0.05, width=5)
        self.test_size.set(0.3);
        self.test_size.pack(side="left", padx=6)
        ttk.Label(pframe, text="Random seed:").pack(side="left")
        self.seed_entry = ttk.Entry(pframe, width=8)
        self.seed_entry.insert(0, str(RANDOM_SEED));
        self.seed_entry.pack(side="left", padx=6)

        self.run_classif_btn = ttk.Button(control, text="Uruchom klasyfikację",
                                          command=self._run_classification, state="disabled")
        self.run_classif_btn.pack(pady=8)

        # ---- tabela wyników + paginacja + metryki ----
        self.classification_tree, _, _ = self._make_treeview(main_frame, [])
        self.classification_tree.pack(fill="both", expand=True, padx=10, pady=5)

        # paginacja
        self._init_paginator("cls")
        self._build_paginator_ui(
            parent=main_frame,
            name="cls",
            prev_cmd=lambda: self._prev_generic("cls", self.classification_tree),
            next_cmd=lambda: self._next_generic("cls", self.classification_tree),
            size_cmd=lambda e: self._change_page_size_generic(e, "cls", self.classification_tree)
        )

        self.classif_metrics = ttk.Label(main_frame, text="", font=("Consolas", 10))
        self.classif_metrics.pack(anchor="w", padx=12)

    def _run_classification(self) -> None:
        if self.df is None:
            messagebox.showwarning("Brak danych", "Najpierw wczytaj plik CSV!")
            return

        feats_idx = self.feature_listbox.curselection()
        features = [self.feature_listbox.get(i) for i in feats_idx]
        target = self.target_combobox.get()
        if not features or not target:
            messagebox.showerror("Błąd", "Wybierz cechy *i* kolumnę docelową.")
            return

        try:
            test_size = float(self.test_size.get())
            seed = int(self.seed_entry.get())
        except ValueError:
            messagebox.showerror("Błąd", "Nieprawidłowy test_size lub seed.")
            return

        self._set_busy("Klasyfikacja…")
        try:
            X = self.df[features]
            y = self.df[target]
            if y.dtype == object or str(y.dtype).startswith("category"):
                y = y.astype("category").cat.codes

            preds = classify_and_return_predictions(X, y, test_size=test_size, random_state=seed)

            # zapamiętaj DF i pokaż 1. stronę
            self.cls_df = preds
            self._show_page("cls", self.classification_tree)

            # metryki
            _, X_test, _, y_test = train_test_split(
                pd.get_dummies(X, drop_first=True), y,
                test_size=test_size, random_state=seed, stratify=y
            )
            log_acc = accuracy_score(y_test, preds.loc[X_test.index, "logreg_pred"])
            dt_acc = accuracy_score(y_test, preds.loc[X_test.index, "dtree_pred"])
            self.classif_metrics.config(text=f"LogReg acc={log_acc:.3f} | DTree acc={dt_acc:.3f}")

        except Exception as e:
            messagebox.showerror("Błąd klasyfikacji", str(e))
        finally:
            self._set_ready()

    # ─────────────────────────────────────────────
    #  >>> 3. ZAKŁADKA - KLASTERYZACJA  <<<
    # ─────────────────────────────────────────────
    def _build_clustering_tab(self, parent: ttk.Frame) -> None:
        """
        Zakładka Klasteryzacja: wybór kolumn do klasteryzacji, liczbę klastrów, itp.
        Lista kolumn jest odświeżana przez _update_all_columns.
        """
        main_frame = ttk.Frame(parent);
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # ARMATURA: loader CSV (odświeży listę cech)
        self._add_loader(parent, on_success=self._update_all_columns)

        control_frame = ttk.Frame(main_frame);
        control_frame.pack(fill="x", padx=10, pady=5)
        ttk.Label(control_frame, text="Wybierz cechy do klasteryzacji (Ctrl+klik):").pack(anchor="w")

        self.clustering_listbox = tk.Listbox(control_frame, selectmode="multiple", height=7, exportselection=False)
        self.clustering_listbox.pack(fill="x", padx=5, pady=5)

        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(fill="x", pady=5)
        ttk.Button(btn_frame, text="Wybierz wszystkie", command=lambda: self._select_all(self.clustering_listbox)).pack(
            side="left", padx=5)
        ttk.Button(btn_frame, text="Czyść zaznaczenie",
                   command=lambda: self._clear_selection(self.clustering_listbox)).pack(side="left", padx=5)

        param_frame = ttk.Frame(control_frame);
        param_frame.pack(fill="x", pady=5)
        ttk.Label(param_frame, text="Liczba klastrów:").pack(side="left")
        self.n_clusters = ttk.Spinbox(param_frame, from_=2, to=10, width=5)
        self.n_clusters.set(4);
        self.n_clusters.pack(side="left", padx=5)
        ttk.Label(param_frame, text="Random seed:").pack(side="left")
        self.clust_seed_entry = ttk.Entry(param_frame, width=8)
        self.clust_seed_entry.insert(0, str(RANDOM_SEED));
        self.clust_seed_entry.pack(side="left", padx=5)

        ttk.Button(control_frame, text="Uruchom klasteryzację",
                   command=self._run_clustering).pack(pady=8)

        self.clustering_tree, _, _ = self._make_treeview(main_frame, [])
        self.clustering_tree.pack(fill="both", expand=True, padx=10, pady=5)

        # paginacja
        self._init_paginator("clu")
        self._build_paginator_ui(
            parent=main_frame,
            name="clu",
            prev_cmd=lambda: self._prev_generic("clu", self.clustering_tree),
            next_cmd=lambda: self._next_generic("clu", self.clustering_tree),
            size_cmd=lambda e: self._change_page_size_generic(e, "clu", self.clustering_tree)
        )

        self.metrics_label = ttk.Label(main_frame, text="", font=("Consolas", 10))
        self.metrics_label.pack(anchor="w", padx=12)

    def _check_classif_ready(self, evt=None):
        feats = self.feature_listbox.curselection()
        tgt = self.target_combobox.get()
        state = "normal" if feats and tgt else "disabled"
        self.run_classif_btn.config(state=state)

    def _run_clustering(self) -> None:
        if self.df is None:
            messagebox.showwarning("Brak danych", "Najpierw wczytaj plik CSV!")
            return

        feat_idx = self.clustering_listbox.curselection()
        features = [self.clustering_listbox.get(i) for i in feat_idx]
        if not features:
            messagebox.showerror("Błąd", "Wybierz cechy do klasteryzacji!")
            return

        n_clusters = int(self.n_clusters.get())
        seed = int(self.clust_seed_entry.get())

        self._set_busy("Klasteryzacja…")
        try:
            X = self.df[features]
            df_with_clusters, metrics = cluster_kmeans(X, n_clusters=n_clusters, seed=seed)

            self.clu_df = df_with_clusters
            self._show_page("clu", self.clustering_tree)

            txt = f"Inertia: {metrics['inertia']:.2f}"
            txt += f"   |   Silhouette: {metrics['silhouette']:.4f}" if metrics["silhouette"] else "   |   Silhouette: brak"
            self.metrics_label.config(text=txt)

        except Exception as e:
            messagebox.showerror("Błąd", str(e))
        finally:
            self._set_ready()

    def on_close(self):
        self.destroy()
        plt.close('all')  # zamyka wszystkie figury matplotlib


if __name__ == "__main__":
    app = MainApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
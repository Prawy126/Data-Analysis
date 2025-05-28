import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy  as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

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
        self.minsize(820, 480)

        # Zmienne aplikacji
        self.current_result_df: pd.DataFrame | None = None
        self.df: pd.DataFrame | None = None
        self.path: str | None = None

        # Zmienne kontrolne wykresów / AI
        self.regline_var = tk.BooleanVar()
        self.sort_values_var = tk.BooleanVar()
        self.selected_features = []
        self.target_var = tk.StringVar()
        self._chart_vars = {}
        self._replacement_rules = {}

        # Pasek menu
        self.menubar = tk.Menu(self)
        self.config(menu=self.menubar)
        akcje_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Akcje", menu=akcje_menu)
        akcje_menu.add_command(label="Pre-processing", command=self._load_pre)
        akcje_menu.add_command(label="Statystyka", command=self._load_stats)
        akcje_menu.add_command(label="AI", command=self._build_ai_window)
        akcje_menu.add_separator()
        akcje_menu.add_command(label="Zamknij", command=self.quit)

        # Notebook (główne zakładki)
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=10, pady=10)

        # Pasek stanu + flaga busy
        self.status_var = tk.StringVar(value="Gotowe")
        self.status_bar = ttk.Label(self, textvariable=self.status_var,
                                    relief="sunken", anchor="w", padding=(6, 2))
        self.status_bar.pack(side="bottom", fill="x")

        #  POD SYSTEMEM STATUSU …
        self.page_size = 100  # domyślny rozmiar strony
        self.current_page = 0  # numer strony (0-indeks)
        self._pagination_df_id = None  # identyfikacja DataFrame dla paginacji

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
        """Przycisk + etykieta. on_success(df) wywoływane po udanym wczytaniu."""
        frm = ttk.Frame(parent);
        frm.pack(fill="x", padx=10, pady=(5, 12))
        info = tk.StringVar(value="(brak pliku)")

        def choose() -> None:
            fp = filedialog.askopenfilename(
                title="Wybierz plik CSV",
                filetypes=[("CSV", "*.csv"), ("Wszystkie", "*.*")]
            )
            if not fp: return
            self._set_busy("Wczytywanie pliku…")
            df = wczytaj_csv(fp, separator=None, wyswietlaj_informacje=True)
            if df is None:
                self._set_ready()
                messagebox.showerror("Błąd", "Nie udało się wczytać pliku.")
                return
            self.df, self.path = df, fp
            info.set(f"{fp.split('/')[-1]}  ({len(df)}×{len(df.columns)})")
            messagebox.showinfo("OK", "Plik wczytany!")
            if on_success:
                on_success(df)
            else:
                self._update_all_columns(df)
            self._set_ready()

        ttk.Button(frm, text="Wczytaj plik CSV", command=choose).pack(side="left")
        ttk.Label(frm, textvariable=info).pack(side="left", padx=10)

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

    def _change_page_size(self, evt=None):
        val = self.page_size_cmb.get()
        self.page_size = -1 if val == "Wszystkie" else int(val)
        self.current_page = 0
        if self.current_result_df is not None:
            self._display_dataframe(self.current_result_df)

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

    # W metodzie _update_all_columns dodajemy aktualizację dla wizualizacji:
    def _update_all_columns(self, df: pd.DataFrame) -> None:
        """Aktualizuje wszystkie listy kolumn w aplikacji"""
        # Aktualizacja zakładek pre-processingu
        self._clear_preprocessing_data(df)

        # Aktualizacja zakładki wizualizacji
        if hasattr(self, 'x_col'):
            self._update_visualization_columns(df)

        # Aktualizacja AI
        if hasattr(self, 'feature_listbox'):
            self.feature_listbox.delete(0, tk.END)
            for col in df.columns:
                self.feature_listbox.insert(tk.END, col)

        if hasattr(self, 'target_combobox'):
            self.target_combobox["values"] = df.columns.tolist()

        if hasattr(self, 'clustering_listbox'):
            self.clustering_listbox.delete(0, tk.END)
            for col in df.columns:
                self.clustering_listbox.insert(tk.END, col)

    def _update_visualization_columns(self, df: pd.DataFrame) -> None:
        """Aktualizuje listy kolumn w zakładce wizualizacji"""
        # Pobierz listy kolumn
        cols = df.columns.tolist()
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include=["category", "object"]).columns.tolist()

        # Aktualizuj comboboxy
        self.x_col["values"] = cols
        self.y_col["values"] = numeric_cols
        self.hue_col["values"] = cat_cols

        # Ustaw domyślne wartości jeśli istnieją
        self.x_col.set(cols[0] if cols else "")
        self.y_col.set(numeric_cols[0] if numeric_cols else "")
        self.hue_col.set(cat_cols[0] if cat_cols else "")


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

        # Nowa zakładka 3 - Braki danych
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

        # Tabela wyników
        self.result_tree, ybar, xbar = self._make_treeview(tab, [])
        self.result_tree.pack(fill="both", expand=True, padx=10, pady=10)
        self._setup_pagination_controls(tab)

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
            if result is not None:
                self.current_result_df = result
                self._display_dataframe(result)
                self.save_btn.config(state="normal")
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
                self.current_result_df = result['df_cleaned']
                self._display_dataframe(self.current_result_df)
                self.save_btn.config(state="normal")
                messagebox.showinfo(
                    "Sukces",
                    f"Usunięto {result['liczba_duplikatow']} duplikatów!\n"
                    f"Nowa liczba wierszy: {len(self.current_result_df)}"
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

        # Przycisk wykonania
        ttk.Button(control_frame, text="Wypełnij braki", command=self._run_fill_missing) \
            .pack(side="right", padx=5, pady=10)

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
        self.current_result_df = None
        self.save_btn.config(state="disabled")
        self.result_tree.delete(*self.result_tree.get_children())

        # Aktualizacja list kolumn
        for listbox in [self.duplicates_listbox, self.missing_listbox]:
            listbox.delete(0, tk.END)
            for col in df.columns:
                listbox.insert(tk.END, col)

        self.encoding_listbox.delete(0, tk.END)
        categorical_cols = [col for col in df.columns if
                            pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_object_dtype(df[col])]
        for col in categorical_cols:
            self.encoding_listbox.insert(tk.END, col)

        # Aktualizacja target combobox
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        self.target_combobox["values"] = numeric_cols
        if numeric_cols:
            self.target_combobox.set(numeric_cols[0])

        self.scaling_listbox.delete(0, tk.END)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            self.scaling_listbox.insert(tk.END, col)

        self.replace_col_combobox["values"] = df.columns.tolist()
        if df.columns.any():
            self.replace_col_combobox.set(df.columns[0])

        # Czyszczenie reguł
        self._replacement_rules = {}
        self._update_rules_listbox()

        if hasattr(self, 'x_col'):
            self._update_columns(df)

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
            self.current_result_df = result
            self._display_dataframe(result)
            self.save_btn.config(state="normal")
            messagebox.showinfo("Sukces", "Pomyślnie wypełniono brakujące wartości!")
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
            self.current_result_df = result
            self._display_dataframe(result)
            self.save_btn.config(state="normal")
            messagebox.showinfo("Sukces", "Pomyślnie usunięto brakujące wartości!")
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
            messagebox.showerror("Błąd", "Proszę wybrać przynajmniej jedną kolumnę")
            return

        method = self.encoding_method.get()
        self._set_busy("Kodowanie kategorii…")
        try:
            if method == "one_hot":
                result = jedno_gorace_kodowanie(
                    self.df, kolumny=selected_cols,
                    usun_pierwsza=self.oh_drop_first.get(), wyswietl_informacje=True
                )
            elif method == "binary":
                result = binarne_kodowanie(
                    self.df, kolumny=selected_cols, wyswietlaj_informacje=True
                )
            else:  # target
                target_col = self.target_combobox.get()
                if not target_col:
                    raise ValueError("Proszę wybrać kolumnę docelową")
                result = kodowanie_docelowe(
                    self.df, kolumny=selected_cols, target=target_col,
                    wyswietlaj_informacje=True
                )

            if result and 'df_zakodowany' in result:
                self.current_result_df = result['df_zakodowany']
                self._display_dataframe(self.current_result_df)
                self.save_btn.config(state="normal")
                messagebox.showinfo("Sukces", "Pomyślnie zastosowano kodowanie!")
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
            self.current_result_df = result
            self._display_dataframe(result)
            self.save_btn.config(state="normal")
            messagebox.showinfo("Sukces", "Pomyślnie zastosowano skalowanie!")
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
            self.current_result_df = result
            self._display_dataframe(result)
            self.save_btn.config(state="normal")
            messagebox.showinfo("Sukces", "Pomyślnie zastosowano zmiany!")
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

    def _build_non_numeric_stats_tab(self) -> None:
        """Zakładka do analizy statystyk nielicyznych"""
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="Statystyki nieliczone")

        # Tytuł sekcji
        ttk.Label(tab, text="Statystyki dla kolumn nielicyznych:", font=('Helvetica', 10, 'bold')).pack(anchor="w",
                                                                                                        padx=10,
                                                                                                        pady=(10, 5))

        # Tabela wyników
        cols = ("kolumna", "liczba wystąpień", "unikalne wartości", "najczęstsza wartość", "częstotliwość [%]",
                "wypełnienie [%]")
        tree, ybar, xbar = self._make_treeview(tab, cols)
        tree.pack(fill="both", expand=True, padx=10, pady=5)

        # Przycisk analizy
        ttk.Button(tab, text="Oblicz statystyki nieliczone", command=lambda: self._run_non_numeric_stats(tree)).pack(
            anchor="w", padx=10, pady=(0, 6))

        # Dodanie loadera CSV
        self._add_loader(tab, on_success=lambda df: None)  # Loader tylko do wczytania danych

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

    def _build_visualization_tab(self) -> None:
        """Zakładka do tworzenia wizualizacji (z kontrolkami zależnymi od typu)."""
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="Wizualizacje")

        # Loader CSV
        self._add_loader(tab, on_success=lambda df: self._update_visualization_columns(df))

        # Główny układ
        main_frame = ttk.Frame(tab);
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # ───────── panel kontrolny ─────────
        control_frame = ttk.Frame(main_frame);
        control_frame.pack(side="left", fill="y", padx=10)

        ttk.Label(control_frame, text="Typ wykresu:").pack(anchor="w")
        self.chart_type = ttk.Combobox(
            control_frame, state="readonly",
            values=["scatter", "line", "bar", "heatmap", "pie"]
        )
        self.chart_type.set("scatter")
        self.chart_type.pack(fill="x", pady=5)
        self.chart_type.bind("<<ComboboxSelected>>", self._update_chart_options)

        ttk.Label(control_frame, text="Kolumna X:").pack(anchor="w")
        self.x_col = ttk.Combobox(control_frame, state="readonly")
        self.x_col.pack(fill="x", pady=2)

        ttk.Label(control_frame, text="Kolumna Y:").pack(anchor="w")
        self.y_col = ttk.Combobox(control_frame, state="readonly")
        self.y_col.pack(fill="x", pady=2)

        ttk.Label(control_frame, text="Kolor (hue):").pack(anchor="w")
        self.hue_col = ttk.Combobox(control_frame, state="readonly")
        self.hue_col.pack(fill="x", pady=2)

        self.options_frame = ttk.Frame(control_frame);
        self.options_frame.pack(fill="x", pady=6)

        # Tytuły / etykiety
        for lbl, attr in [("Tytuł wykresu:", "chart_title"),
                          ("Etykieta X:", "x_label"),
                          ("Etykieta Y:", "y_label")]:
            ttk.Label(control_frame, text=lbl).pack(anchor="w")
            setattr(self, attr, ttk.Entry(control_frame))
            getattr(self, attr).pack(fill="x", pady=2)

        ttk.Button(control_frame, text="Generuj wykres", command=self._generate_plot) \
            .pack(side="bottom", pady=10)

        # ───────── obszar rysunku ─────────
        self.figure = plt.figure(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, master=main_frame)
        self.canvas.get_tk_widget().pack(side="right", fill="both", expand=True)

        # Po pierwszym uruchomieniu
        if self.df is not None:
            self._update_visualization_columns(self.df)
        self._update_chart_options()  # ustawia stany kontrolek

    def _update_chart_options(self, event=None) -> None:
        """Pokazuje tylko te opcje, które pasują do wybranego typu wykresu."""
        # Wyczyść panel opcji
        for w in self.options_frame.winfo_children():
            w.destroy()

        chart_type = self.chart_type.get()

        # Reset / aktywizacja głównych comboboxów
        needs_x = chart_type in ("scatter", "line", "bar", "pie")
        needs_y = chart_type in ("scatter", "line", "bar")
        needs_hue = chart_type in ("scatter", "line")

        self._set_combo_state(self.x_col, needs_x)
        self._set_combo_state(self.y_col, needs_y)
        self._set_combo_state(self.hue_col, needs_hue)

        # Słownik zmiennych pomocniczych (tworzony raz)
        if not hasattr(self, "_chart_vars"):
            self._chart_vars = {}

        # Dodatkowe opcje specyficzne
        if chart_type in ("scatter", "line"):
            self._chart_vars["regline"] = tk.BooleanVar(value=False)
            ttk.Checkbutton(self.options_frame, text="Linia regresji",
                            variable=self._chart_vars["regline"]).pack(anchor="w")

        if chart_type == "bar":
            self._chart_vars["sort_values"] = tk.BooleanVar(value=True)
            ttk.Checkbutton(self.options_frame, text="Sortuj wartości",
                            variable=self._chart_vars["sort_values"]).pack(anchor="w")

        if chart_type == "pie":
            ttk.Label(self.options_frame, text="Maks kategorii:").pack(anchor="w")
            self._chart_vars["maks_kategorie"] = tk.IntVar(value=8)
            ttk.Spinbox(self.options_frame, from_=3, to=15,
                        textvariable=self._chart_vars["maks_kategorie"]).pack(anchor="w")

            ttk.Label(self.options_frame, text="Min. procent:").pack(anchor="w")
            self._chart_vars["min_procent"] = tk.DoubleVar(value=1.0)
            ttk.Spinbox(self.options_frame, from_=0.1, to=100.0, increment=0.1,
                        textvariable=self._chart_vars["min_procent"]).pack(anchor="w")

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
                "kolumna_hue": self.hue_col.get() or None,
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

        # Dodaj loader CSV do zakładki klasyfikacji
        self._add_loader(classification_tab, on_success=lambda df: self._update_all_columns(df))

        # Buduj resztę interfejsu klasyfikacji
        self._build_classification_tab(classification_tab)

        # Zakładka klasteryzacji
        clustering_tab = ttk.Frame(self.nb)
        self.nb.add(clustering_tab, text="Klasteryzacja")

        # Dodaj loader CSV do zakładki klasteryzacji
        self._add_loader(clustering_tab, on_success=lambda df: self._update_all_columns(df))

        # Buduj resztę interfejsu klasteryzacji
        self._build_clustering_tab(clustering_tab)

    def _build_classification_tab(self, parent: ttk.Frame) -> None:
        main_frame = ttk.Frame(parent);
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        control = ttk.Frame(main_frame);
        control.pack(fill="x", padx=10, pady=5)

        ttk.Label(control, text="Wybierz cechy (Ctrl+klik):").pack(anchor="w")
        self.feature_listbox = tk.Listbox(control, selectmode="multiple", height=7, exportselection=False)
        self.feature_listbox.pack(fill="x", padx=5, pady=5)
        self.feature_listbox.bind("<<ListboxSelect>>", self._check_classif_ready)

        ttk.Label(control, text="Kolumna docelowa:").pack(anchor="w")
        self.target_combobox = ttk.Combobox(control, state="readonly")
        self.target_combobox.pack(fill="x", padx=5, pady=5)
        self.target_combobox.bind("<<ComboboxSelected>>", self._check_classif_ready)

        pframe = ttk.Frame(control);
        pframe.pack(fill="x", pady=5)
        ttk.Label(pframe, text="Test size:").pack(side="left")
        self.test_size = ttk.Spinbox(pframe, from_=0.1, to=0.5, increment=0.05, width=5);
        self.test_size.set(0.3)
        self.test_size.pack(side="left", padx=6)
        ttk.Label(pframe, text="Random seed:").pack(side="left")
        self.seed_entry = ttk.Entry(pframe, width=8);
        self.seed_entry.insert(0, str(RANDOM_SEED))
        self.seed_entry.pack(side="left", padx=6)

        self.run_classif_btn = ttk.Button(control, text="Uruchom klasyfikację",
                                          command=self._run_classification, state="disabled")
        self.run_classif_btn.pack(pady=8)

        # tabela + metryki
        self.classification_tree, _, _ = self._make_treeview(main_frame, [])
        self.classification_tree.pack(fill="both", expand=True, padx=10, pady=5)
        self.classif_metrics = ttk.Label(main_frame, text="", font=("Consolas", 10))
        self.classif_metrics.pack(anchor="w", padx=12)

        if self.df is not None:
            self._update_all_columns(self.df)

    def _run_classification(self) -> None:
        if self.df is None:
            messagebox.showwarning("Brak danych", "Najpierw wczytaj plik CSV!")
            return

        feats_idx = self.feature_listbox.curselection()
        features = [self.feature_listbox.get(i) for i in feats_idx]
        target = self.target_combobox.get()

        # sanity-check – powinno być już sprawdzone przez _check_classif_ready
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

            # tabela
            self.classification_tree.delete(*self.classification_tree.get_children())
            self.classification_tree["columns"] = list(preds.columns)
            for col in preds.columns:
                self.classification_tree.heading(col, text=col)
                self.classification_tree.column(col, width=90, anchor="center")
            for _, row in preds.head(200).iterrows():  # limit 200 wierszy
                self.classification_tree.insert("", "end", values=list(row))

            # metryki (obliczone już w funkcji backendowej → print; policzmy ponownie)
            _, X_test, _, y_test = train_test_split(
                pd.get_dummies(X, drop_first=True),
                y, test_size=test_size, random_state=seed, stratify=y
            )
            log_acc = accuracy_score(y_test, preds.loc[X_test.index, "logreg_pred"])
            dt_acc = accuracy_score(y_test, preds.loc[X_test.index, "dtree_pred"])
            self.classif_metrics.config(
                text=f"LogReg  acc={log_acc:.3f}   |   DTree  acc={dt_acc:.3f}"
            )

        except Exception as e:
            messagebox.showerror("Błąd klasyfikacji", str(e))
        finally:
            self._set_ready()

    def _build_clustering_tab(self, parent: ttk.Frame) -> None:
        """Zakładka do klasteryzacji danych"""
        # Główny kontener na elementy klasteryzacji
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill="x", padx=10, pady=5)

        # Wybór cech
        ttk.Label(control_frame, text="Wybierz cechy do klasteryzacji (Ctrl+klik):").pack(anchor="w")
        self.clustering_listbox = tk.Listbox(
            control_frame, selectmode="multiple", height=6, exportselection=False
        )
        self.clustering_listbox.pack(fill="x", padx=5, pady=5)

        # Parametry klastera
        param_frame = ttk.Frame(control_frame)
        param_frame.pack(fill="x", pady=5)

        ttk.Label(param_frame, text="Liczba klastrów:").pack(side="left")
        self.n_clusters = ttk.Spinbox(param_frame, from_=2, to=10, width=5)
        self.n_clusters.set(4)
        self.n_clusters.pack(side="left", padx=5)

        ttk.Label(param_frame, text="Random seed:").pack(side="left")
        self.clust_seed_entry = ttk.Entry(param_frame, width=10)
        self.clust_seed_entry.insert(0, str(RANDOM_SEED))
        self.clust_seed_entry.pack(side="left", padx=5)

        # Przycisk uruchamiający
        ttk.Button(
            control_frame,
            text="Uruchom klasteryzację",
            command=self._run_clustering
        ).pack(pady=5)

        # Tabela wyników
        self.clustering_tree, _, _ = self._make_treeview(main_frame, [])
        self.clustering_tree.pack(fill="both", expand=True, padx=10, pady=5)

        # Pole tekstowe na metryki
        self.metrics_label = ttk.Label(main_frame, text="", font=("Consolas", 10))
        self.metrics_label.pack(padx=10, pady=5)

        # Jeśli dane są już załadowane, wypełnij listę cech
        if self.df is not None:
            self._update_all_columns(self.df)

    def _check_classif_ready(self, evt=None):
        feats = self.feature_listbox.curselection()
        tgt = self.target_combobox.get()
        state = "normal" if feats and tgt else "disabled"
        self.run_classif_btn.config(state=state)

    def _run_clustering(self) -> None:
        if self.df is None:
            messagebox.showwarning("Brak danych", "Najpierw wczytaj plik CSV!")
            return

        feature_indices = self.clustering_listbox.curselection()
        features = [self.clustering_listbox.get(i) for i in feature_indices]
        if not features:
            messagebox.showerror("Błąd", "Wybierz cechy do klasteryzacji!")
            return

        n_clusters = int(self.n_clusters.get())
        seed = int(self.clust_seed_entry.get())

        self._set_busy("Klasteryzacja…")
        try:
            X = self.df[features]
            df_with_clusters, metrics = cluster_kmeans(X, n_clusters=n_clusters, seed=seed)

            self.clustering_tree.delete(*self.clustering_tree.get_children())
            self.clustering_tree["columns"] = list(df_with_clusters.columns)
            for col in df_with_clusters.columns:
                self.clustering_tree.heading(col, text=col)
                self.clustering_tree.column(col, width=100, anchor="center")
            for _, row in df_with_clusters.head(50).iterrows():
                self.clustering_tree.insert("", "end", values=list(row))

            self.metrics_label.config(
                text=(f"Inertia: {metrics['inertia']:.2f}\n"
                      f"Silhouette: {metrics['silhouette']:.4f}"
                      if metrics["silhouette"] else "Silhouette: brak")
            )
        except Exception as e:
            messagebox.showerror("Błąd", str(e))
        finally:
            self._set_ready()





if __name__ == "__main__":
    MainApp().mainloop()

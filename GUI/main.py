import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import csv


class AnalysisPanel:
    """Bazowa klasa dla wszystkich paneli analizy danych"""

    def __init__(self, name):
        self.name = name
        self.frame = None

    def create_ui(self, parent):
        """Tworzy interfejs użytkownika dla panelu"""
        self.frame = ttk.Frame(parent)
        return self.frame

    def analyze_data(self, df, selected_columns):
        """Analizuje dane i zwraca wyniki"""
        raise NotImplementedError

    def update_ui(self, results):
        """Aktualizuje interfejs użytkownika wynikami analizy"""
        raise NotImplementedError


class NumericalPanel(AnalysisPanel):
    def __init__(self):
        super().__init__("Dane numeryczne")
        self.treeview = None

    def create_ui(self, parent):
        frame = super().create_ui(parent)

        # Tabela dla danych numerycznych
        self.treeview = ttk.Treeview(frame,
                                     columns=("Średnia", "Mediana", "Min", "Max", "Odchylenie", "Liczba wartości"),
                                     show="headings")

        self.treeview.pack(padx=10, pady=10, fill="both", expand=True)

        # Nagłówki kolumn
        self.treeview.heading("Średnia", text="Średnia")
        self.treeview.heading("Mediana", text="Mediana")
        self.treeview.heading("Min", text="Min")
        self.treeview.heading("Max", text="Max")
        self.treeview.heading("Odchylenie", text="Odchylenie")
        self.treeview.heading("Liczba wartości", text="Liczba wartości")

        return frame

    def analyze_data(self, df, selected_columns):
        """Analizuje dane numeryczne"""
        results = {}
        for col in selected_columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                results[col] = {
                    'średnia': df[col].mean(),
                    'mediana': df[col].median(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'odchylenie_std': df[col].std(),
                    'liczba_wartości': df[col].count()
                }
        return results

    def update_ui(self, results):
        """Aktualizuje tabelę danymi numerycznymi"""
        # Czyszczenie tabeli
        for item in self.treeview.get_children():
            self.treeview.delete(item)

        # Wypełnianie tabeli
        for kolumna, stats in results.items():
            self.treeview.insert("", "end", text=kolumna, values=(
                round(stats['średnia'], 2),
                round(stats['mediana'], 2),
                round(stats['min'], 2),
                round(stats['max'], 2),
                round(stats['odchylenie_std'], 2),
                stats['liczba_wartości']
            ))


class NonNumericalPanel(AnalysisPanel):
    def __init__(self):
        super().__init__("Dane nie-numeryczne")
        self.treeview = None

    def create_ui(self, parent):
        frame = super().create_ui(parent)

        # Tabela dla danych nie-numerycznych
        self.treeview = ttk.Treeview(frame,
                                     columns=("Wystąpienia", "Unikalne", "Najczęstsza", "Częstość",
                                              "Wypełnienie", "Min długość", "Max długość", "Średnia długość"),
                                     show="headings")

        self.treeview.pack(padx=10, pady=10, fill="both", expand=True)

        # Nagłówki kolumn
        self.treeview.heading("Wystąpienia", text="Wystąpienia")
        self.treeview.heading("Unikalne", text="Unikalne")
        self.treeview.heading("Najczęstsza", text="Najczęstsza")
        self.treeview.heading("Częstość", text="Częstość")
        self.treeview.heading("Wypełnienie", text="Wypełnienie")
        self.treeview.heading("Min długość", text="Min długość")
        self.treeview.heading("Max długość", text="Max długość")
        self.treeview.heading("Średnia długość", text="Średnia długość")

        return frame

    def analyze_data(self, df, selected_columns):
        """Analizuje dane nie-numeryczne"""
        results = {}
        for col in selected_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                non_null_values = df[col].dropna()
                value_counts = non_null_values.value_counts()

                results[col] = {
                    'liczba_wystapien': len(non_null_values),
                    'wartosci_unikalne': len(value_counts),
                    'najczestsza_wartosc': value_counts.index[0] if not value_counts.empty else None,
                    'czestotliwosc_najczestszej': value_counts.iloc[0] / len(
                        non_null_values) if not value_counts.empty else 0,
                    'procent_wypelnienia': (len(non_null_values) / len(df[col])) * 100,
                    'dlugosc_min': non_null_values.astype(str).str.len().min() if not non_null_values.empty else 0,
                    'dlugosc_max': non_null_values.astype(str).str.len().max() if not non_null_values.empty else 0,
                    'dlugosc_srednia': non_null_values.astype(str).str.len().mean() if not non_null_values.empty else 0
                }
        return results

    def update_ui(self, results):
        """Aktualizuje tabelę danymi nie-numerycznymi"""
        # Czyszczenie tabeli
        for item in self.treeview.get_children():
            self.treeview.delete(item)

        # Wypełnianie tabeli
        for kolumna, stats in results.items():
            self.treeview.insert("", "end", text=kolumna, values=(
                stats['liczba_wystapien'],
                stats['wartosci_unikalne'],
                stats['najczestsza_wartosc'],
                f"{round(stats['czestotliwosc_najczestszej'] * 100, 2)}%",
                f"{round(stats['procent_wypelnienia'], 2)}%",
                round(stats['dlugosc_min'], 2),
                round(stats['dlugosc_max'], 2),
                round(stats['dlugosc_srednia'], 2)
            ))


class KorelacjaPanel(AnalysisPanel):
    def __init__(self):
        super().__init__("Korelacja")
        self.treeview = None
        self.method_var = tk.StringVar(value="pearson")

    def create_ui(self, parent):
        frame = super().create_ui(parent)

        # Wybór metody
        method_frame = ttk.Frame(frame)
        method_frame.pack(pady=5)

        tk.Label(method_frame, text="Metoda korelacji:").pack(side="left", padx=5)
        tk.Radiobutton(method_frame, text="Pearson", variable=self.method_var, value="pearson").pack(side="left",
                                                                                                     padx=5)
        tk.Radiobutton(method_frame, text="Spearman", variable=self.method_var, value="spearman").pack(side="left",
                                                                                                       padx=5)

        # Tabela dla macierzy korelacji
        self.treeview = ttk.Treeview(frame, show="tree")
        self.treeview.pack(padx=10, pady=10, fill="both", expand=True)

        # Pionowy scrollbar
        yscroll = ttk.Scrollbar(frame, orient="vertical", command=self.treeview.yview)
        yscroll.pack(side="right", fill="y")
        self.treeview.configure(yscrollcommand=yscroll.set)

        # Poziomy scrollbar
        xscroll = ttk.Scrollbar(frame, orient="horizontal", command=self.treeview.xview)
        xscroll.pack(side="bottom", fill="x")
        self.treeview.configure(xscrollcommand=xscroll.set)

        return frame

    def analyze_data(self, df, selected_columns):
        """Analizuje korelację wybranych kolumn"""
        if not selected_columns:
            return None

        # Filtruj tylko rzeczywiście istniejące i numeryczne kolumny
        numeric_cols = [col for col in selected_columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]

        if len(numeric_cols) < 2:
            return None

        try:
            if self.method_var.get() == "pearson":
                return df[numeric_cols].corr(method='pearson')
            else:
                return df[numeric_cols].corr(method='spearman')
        except Exception as e:
            print(f"Błąd podczas obliczania korelacji: {str(e)}")
            return None

    def update_ui(self, results):
        """Aktualizuje tabelę danymi korelacji"""
        # Czyszczenie tabeli
        self.treeview.delete(*self.treeview.get_children())

        # Czyszczenie kolumn
        for col in self.treeview["columns"]:
            self.treeview.heading(col, text="")

        if results is None or results.empty:
            self.treeview.insert("", "end", text="Brak danych numerycznych do analizy korelacji")
            return

        # Ustawienie nowych kolumn
        columns = list(results.columns)
        self.treeview["columns"] = columns

        # Nagłówki kolumn
        for col in columns:
            self.treeview.heading(col, text=col)
            self.treeview.column(col, width=100, anchor="center")

        # Wypełnianie tabeli
        for row_name, row_data in results.iterrows():
            values = [round(row_data[col], 4) for col in columns]
            self.treeview.insert("", "end", text=row_name, values=tuple(values))


class MainWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Analizator Danych")
        self.root.geometry("1000x700")

        # Zmienne
        self.separator_var = tk.StringVar(value=",")
        self.file_path = tk.StringVar()
        self.panels = []
        self.df = None

        # Inicjalizacja UI
        self.create_ui()

        # Rejestracja paneli
        self.register_panel(NumericalPanel())
        self.register_panel(NonNumericalPanel())
        self.register_panel(KorelacjaPanel())

    def create_ui(self):
        """Tworzy interfejs użytkownika"""
        # Panel zakładek
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(pady=10, expand=True, fill="both")

        # Panel wejścia
        wejscie_frame = ttk.Frame(self.root)
        wejscie_frame.pack(pady=10)

        # Ścieżka do pliku
        label_sciezka = tk.Label(wejscie_frame, text="Ścieżka do pliku CSV:")
        label_sciezka.grid(row=0, column=0, padx=5)

        self.entry_sciezka = tk.Entry(wejscie_frame, width=50, textvariable=self.file_path)
        self.entry_sciezka.grid(row=0, column=1, padx=5)

        button_otworz_plik = tk.Button(wejscie_frame, text="Wybierz plik", command=self.otworz_plik)
        button_otworz_plik.grid(row=0, column=2, padx=5)

        # Radio buttons do wyboru separatora
        radio_frame = ttk.Frame(wejscie_frame)
        radio_frame.grid(row=1, column=0, columnspan=3, pady=5)

        tk.Label(radio_frame, text="Separator:").pack(side="left", padx=5)
        tk.Radiobutton(radio_frame, text="Auto", variable=self.separator_var, value="auto").pack(side="left", padx=5)
        tk.Radiobutton(radio_frame, text="Przecinek (,)", variable=self.separator_var, value=",").pack(side="left",
                                                                                                       padx=5)
        tk.Radiobutton(radio_frame, text="Średnik (;)", variable=self.separator_var, value=";").pack(side="left",
                                                                                                     padx=5)
        tk.Radiobutton(radio_frame, text="Tabulator", variable=self.separator_var, value="\t").pack(side="left", padx=5)

        # Lista kolumn do wyboru
        label_wybierz_kolumny = tk.Label(self.root, text="Wybierz kolumny do analizy:")
        label_wybierz_kolumny.pack(pady=5)

        self.listbox_kolumny = tk.Listbox(self.root, selectmode=tk.MULTIPLE, height=5)
        self.listbox_kolumny.pack()

        # Przycisk analizy
        button_analizuj = tk.Button(self.root, text="Analizuj dane", command=self.wykonaj_analize)
        button_analizuj.pack(pady=10)

    def register_panel(self, panel):
        """Rejestruje nowy panel analizy"""
        self.panels.append(panel)
        panel_frame = panel.create_ui(self.notebook)
        self.notebook.add(panel_frame, text=panel.name)

    def wykryj_separator(self, sciezka_pliku):
        """Wykrywa separator w pliku CSV z dodatkową walidacją"""
        try:
            with open(sciezka_pliku, "r") as f:
                sample = f.read(1024)
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample).delimiter

            # Dodatkowa walidacja: jeśli zawiera średniki i nie został wykryty separator
            if ';' in sample.split('\n')[0] and delimiter != ';':
                return ';'
            return delimiter
        except:
            # Domyślny separator, jeśli auto-wykrywanie nie działa
            return ";"

    def otworz_plik(self):
        sciezka_pliku = filedialog.askopenfilename(
            title="Wybierz plik CSV",
            filetypes=[("Pliki CSV", "*.csv"), ("Wszystkie pliki", "*.*")]
        )
        if sciezka_pliku:
            self.file_path.set(sciezka_pliku)

            separator = self.separator_var.get()
            if separator == "auto":
                try:
                    separator = self.wykryj_separator(sciezka_pliku)
                except Exception as e:
                    messagebox.showerror("Błąd", f"Nie można wykryć separatora: {str(e)}")
                    return

            try:
                self.df = pd.read_csv(sciezka_pliku, sep=separator, encoding='utf-8-sig')
                self.listbox_kolumny.delete(0, tk.END)
                for kolumna in self.df.columns:
                    self.listbox_kolumny.insert(tk.END, kolumna)
            except Exception as e:
                messagebox.showerror("Błąd", f"Błąd podczas wczytywania pliku: {str(e)}")

    def wykonaj_analize(self):
        """Uruchamia analizę danych"""
        sciezka_pliku = self.file_path.get()
        if not sciezka_pliku:
            messagebox.showerror("Błąd", "Nie wybrano pliku CSV.")
            return

        separator = self.separator_var.get()
        if separator == "auto":
            separator = self.wykryj_separator(sciezka_pliku)

        # Pobierz wybrane kolumny
        selected_indices = self.listbox_kolumny.curselection()
        selected_columns = [self.listbox_kolumny.get(i) for i in selected_indices]

        if not selected_columns:
            messagebox.showwarning("Ostrzeżenie", "Nie wybrano żadnych kolumn do analizy.")
            return

        # Załaduj dane
        try:
            self.df = pd.read_csv(sciezka_pliku, sep=separator)
        except Exception as e:
            messagebox.showerror("Błąd", f"Błąd podczas wczytywania pliku: {str(e)}")
            return

        # Analizuj dane dla każdego panelu
        for panel in self.panels:
            results = panel.analyze_data(self.df, selected_columns)
            panel.update_ui(results)


if __name__ == "__main__":
    root = tk.Tk()
    app = MainWindow(root)
    root.mainloop()
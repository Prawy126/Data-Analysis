import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from Backend.Statystyka import analizuj_dane_numeryczne, oblicz_statystyki_nie_numeryczne
import pandas as pd
import csv

# Funkcja do wykrywania separatora
def wykryj_separator(sciezka_pliku):
    with open(sciezka_pliku, "r") as f:
        sample = f.read(1024)
    sniffer = csv.Sniffer()
    delimiter = sniffer.sniff(sample).delimiter
    return delimiter

# Funkcja do otwierania pliku CSV
def otworz_plik():
    sciezka_pliku = filedialog.askopenfilename(
        title="Wybierz plik CSV",
        filetypes=[("Pliki CSV", "*.csv"), ("Wszystkie pliki", "*.*")]
    )
    if sciezka_pliku:
        entry_sciezka.delete(0, tk.END)
        entry_sciezka.insert(0, sciezka_pliku)

        separator = separator_var.get()
        if separator == "auto":
            separator = wykryj_separator(sciezka_pliku)

        try:
            df = pd.read_csv(sciezka_pliku, sep=separator)
            listbox_wybierz_kolumny.delete(0, tk.END)
            for kolumna in df.columns:
                listbox_wybierz_kolumny.insert(tk.END, kolumna)
        except Exception as e:
            messagebox.showerror("Błąd", f"Błąd podczas wczytywania pliku: {str(e)}")

# Funkcja do analizy danych
def wykonaj_analize():
    sciezka_pliku = entry_sciezka.get()
    if not sciezka_pliku:
        messagebox.showerror("Błąd", "Nie wybrano pliku CSV.")
        return

    separator = separator_var.get()
    if separator == "auto":
        separator = wykryj_separator(sciezka_pliku)

    wybrane_kolumny = listbox_wybierz_kolumny.curselection()
    wybrane_kolumny = [listbox_wybierz_kolumny.get(i) for i in wybrane_kolumny]

    try:
        df = pd.read_csv(sciezka_pliku, sep=separator)
    except Exception as e:
        messagebox.showerror("Błąd", f"Błąd podczas wczytywania pliku: {str(e)}")
        return

    # Analiza danych numerycznych
    wartosci_numeryczne, statystyki_numeryczne = analizuj_dane_numeryczne(sciezka_pliku, separator=separator, wybrane_kolumny=wybrane_kolumny)

    # Analiza danych nie-numerycznych
    statystyki_nie_numeryczne = oblicz_statystyki_nie_numeryczne(df)

    # Czyszczenie tabel
    for i in tabela_numeryczne.get_children():
        tabela_numeryczne.delete(i)
    for i in tabela_nie_numeryczne.get_children():
        tabela_nie_numeryczne.delete(i)

    # Wypełnianie tabeli danymi numerycznymi
    for kolumna, stats in statystyki_numeryczne.items():
        tabela_numeryczne.insert("", "end", text=kolumna, values=(
            round(stats['średnia'], 2),
            round(stats['mediana'], 2),
            round(stats['min'], 2),
            round(stats['max'], 2),
            round(stats['odchylenie_std'], 2),
            stats['liczba_wartości']
        ))

    # Wypełnianie tabeli danymi nie-numerycznymi
    for kolumna, stats in statystyki_nie_numeryczne.items():
        tabela_nie_numeryczne.insert("", "end", text=kolumna, values=(
            stats['liczba_wystapien'],
            stats['wartosci_unikalne'],
            stats['najczestsza_wartosc'],
            f"{round(stats['czestotliwosc_najczestszej'] * 100, 2)}%",
            f"{round(stats['procent_wypelnienia'], 2)}%",
            round(stats['dlugosc_min'], 2),
            round(stats['dlugosc_max'], 2),
            round(stats['dlugosc_srednia'], 2)
        ))

# Utworzenie głównego okna
root = tk.Tk()
root.title("Analizator Danych")
root.geometry("1000x700")

# Zmienne interfejsu
separator_var = tk.StringVar(value=",")

# Panel zakładek
notebook = ttk.Notebook(root)
notebook.pack(pady=10, expand=True, fill="both")

# Zakładka 1 - Dane numeryczne
frame_numeryczne = ttk.Frame(notebook)
notebook.add(frame_numeryczne, text="Dane numeryczne")

# Zakładka 2 - Dane nie-numeryczne
frame_nie_numeryczne = ttk.Frame(notebook)
notebook.add(frame_nie_numeryczne, text="Dane nie-numeryczne")

# Panel wejścia
wejscie_frame = ttk.Frame(root)
wejscie_frame.pack(pady=10)

label_sciezka = tk.Label(wejscie_frame, text="Ścieżka do pliku CSV:")
label_sciezka.grid(row=0, column=0, padx=5)

entry_sciezka = tk.Entry(wejscie_frame, width=50)
entry_sciezka.grid(row=0, column=1, padx=5)

button_otworz_plik = tk.Button(wejscie_frame, text="Wybierz plik", command=otworz_plik)
button_otworz_plik.grid(row=0, column=2, padx=5)

# Radio buttons do wyboru separatora
radio_frame = ttk.Frame(wejscie_frame)
radio_frame.grid(row=1, column=0, columnspan=3, pady=5)

radio_auto = tk.Radiobutton(radio_frame, text="Auto", variable=separator_var, value="auto")
radio_comma = tk.Radiobutton(radio_frame, text="Przecinek (,)", variable=separator_var, value=",")
radio_semicolon = tk.Radiobutton(radio_frame, text="Średnik (;)", variable=separator_var, value=";")
radio_tab = tk.Radiobutton(radio_frame, text="Tabulator", variable=separator_var, value="\t")

radio_auto.pack(side="left", padx=5)
radio_comma.pack(side="left", padx=5)
radio_semicolon.pack(side="left", padx=5)
radio_tab.pack(side="left", padx=5)

# Lista kolumn do wyboru
label_wybierz_kolumny = tk.Label(root, text="Wybierz kolumny do analizy:")
label_wybierz_kolumny.pack(pady=5)

listbox_wybierz_kolumny = tk.Listbox(root, selectmode=tk.MULTIPLE, height=5)
listbox_wybierz_kolumny.pack()

# Przycisk do analizy
button_analizuj = tk.Button(root, text="Analizuj dane", command=wykonaj_analize)
button_analizuj.pack(pady=10)

# Tabela - Dane numeryczne
tabela_numeryczne = ttk.Treeview(frame_numeryczne, columns=("Średnia", "Mediana", "Min", "Max", "Odchylenie", "Liczba wartości"), show="headings")
tabela_numeryczne.pack(padx=10, pady=10, fill="both", expand=True)

tabela_numeryczne.heading("Średnia", text="Średnia")
tabela_numeryczne.heading("Mediana", text="Mediana")
tabela_numeryczne.heading("Min", text="Min")
tabela_numeryczne.heading("Max", text="Max")
tabela_numeryczne.heading("Odchylenie", text="Odchylenie")
tabela_numeryczne.heading("Liczba wartości", text="Liczba wartości")

# Tabela - Dane nie-numeryczne
tabela_nie_numeryczne = ttk.Treeview(frame_nie_numeryczne, columns=("Wystąpienia", "Unikalne", "Najczęstsza", "Częstość", "Wypełnienie", "Min długość", "Max długość", "Średnia długość"), show="headings")
tabela_nie_numeryczne.pack(padx=10, pady=10, fill="both", expand=True)

tabela_nie_numeryczne.heading("Wystąpienia", text="Wystąpienia")
tabela_nie_numeryczne.heading("Unikalne", text="Unikalne")
tabela_nie_numeryczne.heading("Najczęstsza", text="Najczęstsza")
tabela_nie_numeryczne.heading("Częstość", text="Częstość")
tabela_nie_numeryczne.heading("Wypełnienie", text="Wypełnienie")
tabela_nie_numeryczne.heading("Min długość", text="Min długość")
tabela_nie_numeryczne.heading("Max długość", text="Max długość")
tabela_nie_numeryczne.heading("Średnia długość", text="Średnia długość")

# Uruchomienie aplikacji
root.mainloop()
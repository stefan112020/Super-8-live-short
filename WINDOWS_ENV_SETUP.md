# Ghid de configurare Windows pentru Super 8

Acest ghid descrie pașii necesari pentru a rula backtest-ul din `Suber_8_Backtest_Simboluri.ipynb` direct din Anaconda Prompt. Instrucțiunile sunt împărțite pentru **PowerShell** (implicit în Anaconda Prompt) și **Command Prompt (cmd.exe)**.

> **Cum îți dai seama în ce shell ești?**
>
> * Promptul care arată `(base) PS C:\Users\Stefan>` sau `(super8) PS ...` înseamnă **PowerShell** – folosește comenzile cu `$env:`.
> * Promptul care arată `(base) C:\Users\Stefan>` sau `(super8) C:\Users\Stefan>` (fără `PS`) este **Command Prompt** – folosește comenzile cu `set "CHEIE=valoare"`.
>
> Nu amesteca sintaxele: comenzile `$env:` nu funcționează în Command Prompt, iar `set` cu ghilimele duble nu funcționează în PowerShell.

## 1. Activarea mediului conda

1. Deschide Anaconda Prompt.
2. Creează un mediu dedicat (o singură dată):
   ```powershell
   conda create -n super8 python=3.10
   ```
3. Activează mediul înainte de fiecare sesiune:
   ```powershell
   conda activate super8
   ```

## 2. Instalarea dependențelor minime

Cu mediul activ:
```powershell
conda install pandas numpy scipy
```
Confirmă cu `y` când este necesar.

Instalează și Jupyter Notebook (este inclus în majoritatea distribuțiilor, dar dacă primești mesajul „`'jupyter' is not recognized`”, rulează):
```powershell
conda install notebook
```
După instalare poți verifica disponibilitatea cu `where jupyter` (Command Prompt) sau `Get-Command jupyter` (PowerShell).

## 3. Setarea variabilelor de mediu

### 3.1 Varianta PowerShell (Anaconda Prompt implicit)
Copiază și rulează **fiecare linie** separat:
```powershell
$env:SUPER8_DATA_DIR = 'C:\Users\Stefan\Desktop\Curs Trading\Strategii\Simboluri_Binance'
$env:BINANCE_CFG_PATH = 'C:\Users\Stefan\Desktop\Curs Trading\Strategii\binance.cfg'
$env:SUPER8_RESULTS_PATH = 'C:\Users\Stefan\Desktop\Curs Trading\Strategii\rezultate\rezultate_optimizare.csv'
$env:SUPER8_SYMBOLS_FILE = 'C:\Users\Stefan\Desktop\Curs Trading\Strategii\simboluri.csv'
```
> **Observație:** Folosește ghilimele simple `'...'` numai în PowerShell. După lipire, apasă `Enter` pentru a executa linia curentă.

### 3.2 Varianta Command Prompt (cmd.exe)
Dacă preferi Command Prompt, folosește `set` și ghilimele duble:
```cmd
set "SUPER8_DATA_DIR=C:\Users\Stefan\Desktop\Curs Trading\Strategii\Simboluri_Binance"
set "BINANCE_CFG_PATH=C:\Users\Stefan\Desktop\Curs Trading\Strategii\binance.cfg"
set "SUPER8_RESULTS_PATH=C:\Users\Stefan\Desktop\Curs Trading\Strategii\rezultate\rezultate_optimizare.csv"
set "SUPER8_SYMBOLS_FILE=C:\Users\Stefan\Desktop\Curs Trading\Strategii\simboluri.csv"
```
> **Important:** Nu folosi apostrofi `'` în Command Prompt; rămâi la `set "CHEIE=valoare"` pentru a evita erorile „filename syntax is incorrect”.

> **Observație despre lipire:** dacă lipești mai multe linii deodată (de exemplu, dintr-un bloc marcat cu ```powershell```), Command Prompt va interpreta și rândul `powershell` ca și cum ar fi o comandă și va eșua cu „The system cannot find the path specified.” Copiază și rulează **o singură linie pe rând**.

### 3.3 Verificarea valorilor
După setare, verifică valorile:
- PowerShell:
  ```powershell
  $env:SUPER8_DATA_DIR
  $env:BINANCE_CFG_PATH
  $env:SUPER8_RESULTS_PATH
  $env:SUPER8_SYMBOLS_FILE
  ```
- Command Prompt:
  ```cmd
  echo %SUPER8_DATA_DIR%
  echo %BINANCE_CFG_PATH%
  echo %SUPER8_RESULTS_PATH%
  echo %SUPER8_SYMBOLS_FILE%
  ```

## 4. Rulare notebook în Jupyter (opțional)
Din aceeași sesiune Anaconda Prompt:
```powershell
jupyter notebook
```
În interfață, rulează celula de verificare a variabilelor de mediu și apoi backtest-ul.

## 5. Rulare directă din linia de comandă
Dacă vrei să rulezi fără interfață grafică:
1. Exportă notebook-ul ca script (o singură dată sau când îl actualizezi):
   ```powershell
   jupyter nbconvert --to script Suber_8_Backtest_Simboluri.ipynb
   ```
2. Rulează scriptul generat:
   ```powershell
   python Suber_8_Backtest_Simboluri.py
   ```
Rezultatele vor fi scrise în fișierul indicat prin `SUPER8_RESULTS_PATH`.

> **Dacă apare eroarea `jupyter is not recognized`** după instalarea pachetului `notebook`, închide fereastra Anaconda Prompt și deschide alta nouă, apoi rulează `conda activate super8` înainte de `jupyter notebook`. Astfel se reîncarcă variabilele de mediu și se pune la dispoziție comanda `jupyter`.

## 6. Sfaturi suplimentare
- Dacă închizi Anaconda Prompt, la următoarea sesiune trebuie să rulezi din nou pașii de activare a mediului și de setare a variabilelor de mediu.
- Pentru setări permanente, folosește `setx` din Command Prompt pentru fiecare variabilă (nu este necesar pentru test rapid).

Prin urmarea acestor pași, erorile din capturile trimise (legate de utilizarea apostrofilor `'` în Command Prompt) sunt eliminate, iar backtest-ul poate rula fie din notebook, fie direct cu `python`.

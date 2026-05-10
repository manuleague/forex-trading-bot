# Ghid de finalizare a documentatiei

Acest folder contine documentatia tehnica in format LaTeX pentru proiectul de diploma:

- `main.tex` este fisierul principal care trebuie compilat.
- `continut/rezumat.tex` contine rezumatul lucrarii.
- `continut/capitol1` - introducere, obiective, metodologie.
- `continut/capitol2` - fundamentare teoretica si bibliografie.
- `continut/capitol3` - arhitectura, cerinte, strategii, risc, flux live.
- `continut/capitol4` - testare, rezultate experimentale si limitari.
- `continut/capitol5` - concluzii si directii viitoare.
- `anexe/anexe.tex` - comenzi, artefacte, parametri si fragmente de cod.

## Figuri incluse

Am inclus deja urmatoarele figuri in lucrare:

- Capitolul 3: `coperti/arhitectura_sistem.png`, arhitectura generala.
- Capitolul 3: `continut/capitol3/figuri/pipeline_strategii.png`, pipeline-ul decizional.
- Capitolul 3: `continut/capitol3/figuri/flux_backtesting.png`, fluxul de backtesting.
- Capitolul 3: `continut/capitol3/figuri/flux_live_paper.png`, fluxul live paper.
- Capitolul 4: `continut/capitol4/figuri/comparatie_rezultate_backtest.png`, comparatia rezultatelor EURUSD/GBPUSD/USDJPY.

## Figuri recomandate daca vrei sa intaresti lucrarea

Acestea nu sunt obligatorii, dar ar face documentatia mai vizuala:

1. In Capitolul 4, dupa tabelul cu rezultate, poti adauga o captura din `output/backtest_USDJPY_D_20180101_to_20241231_20260505_142456/dashboard.html`.
2. In Capitolul 4, in sectiunea de analiza a riscului, poti adauga imaginea exportata din `output/backtest_USDJPY_D_20180101_to_20241231_20260505_142456/performance_dashboard.svg`, convertita in PNG sau PDF.
3. In Anexe, poti adauga o captura cu structura directorului `output` dupa o rulare completa.
4. In Capitolul 3, daca profesorul cere o diagrama UML, poti transforma tabelul de arhitectura intr-o diagrama de componente cu modulele `data`, `strategies`, `core`, `backtest`, `live` si `logs`.

## Compilare

Template-ul foloseste `minted`, deci compilarea LaTeX trebuie facuta cu `-shell-escape`.

Exemplu:

```bash
pdflatex -shell-escape main.tex
bibtex main
pdflatex -shell-escape main.tex
pdflatex -shell-escape main.tex
```

Pe aceasta masina nu am gasit `pdflatex` instalat in PATH, deci nu am putut compila local PDF-ul. Documentatia este pregatita pentru Overleaf sau pentru o distributie locala TeX Live/MiKTeX cu Pygments instalat.

## Verificari finale inainte de predare

- Completeaza/valideaza datele din `macros/student.tex`: nume, titlu, coordonator, program de studii, promotie.
- Inlocuieste PDF-ul declaratiei de autenticitate din `pdf/declaratie_autenticitate_diploma.pdf` cu varianta semnata, daca este cazul.
- Verifica daca profesorul cere diacritice integral in text; unele fisiere mostenite din template pot aparea cu encoding gresit in anumite terminale, dar LaTeX/Overleaf ar trebui sa le citeasca drept UTF-8.
- Verifica bibliografia si elimina sursele demo care nu sunt citate in lucrare, daca vrei o lista bibliografica mai curata.
- Pastreaza rapoartele experimentale din `output`, pentru ca valorile din Capitolul 4 sunt extrase din acele fisiere.

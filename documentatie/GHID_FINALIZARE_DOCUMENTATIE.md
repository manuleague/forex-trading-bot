# Ghid finalizare documentatie

Documentatia este organizata conform structurii din `Structura proiectului de diploma v1.0.pdf`.

## Fisier principal

Compileaza `documentatie/main.tex`.

Template-ul foloseste o mapare compatibila `minted` -> `listings`, deci nu este necesar `-shell-escape`:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Continut inclus

- Rezumat cu termeni cheie.
- Capitolul 1: introducere, context, obiective, metodologie, structura.
- Capitolul 2: fundamentare teoretica, solutii similare, lacune si cerinte derivate.
- Capitolul 3: cerinte utilizator, cerinte functionale/nefunctionale, tehnologii, arhitectura, module, strategii, risc, live paper, algoritm.
- Capitolul 4: instalare, configurare, scenarii, metrici, rezultate, risc, resurse, fiabilitate, securitate, scalabilitate, observabilitate Prometheus/Grafana, capturi recomandate.
- Capitolul 5: concluzii, contributii personale, comparatie, directii viitoare.
- Anexe: comenzi, artefacte, parametri, checklist, fragmente de cod.

## Figuri deja incluse

- `coperti/arhitectura_sistem.png`
- `continut/capitol3/figuri/pipeline_strategii.png`
- `continut/capitol3/figuri/flux_backtesting.png`
- `continut/capitol3/figuri/flux_live_paper.png`
- `continut/capitol4/figuri/comparatie_rezultate_backtest.png`

## Ce mai trebuie doar administrativ

- Verifica datele din `macros/student.tex`.
- Inlocuieste declaratia de autenticitate cu varianta semnata.
- Compileaza PDF-ul final in Overleaf sau local cu MiKTeX/TeX Live.
- Optional, adauga o captura reala din `dashboard.html` pentru rularea USDJPY.
- Optional, adauga capturi Grafana pentru dashboard-urile EURUSD/GBPUSD/USDJPY si Prometheus Targets (`localhost:39090/targets`) daca profesorul cere dovada vizuala a monitorizarii live.

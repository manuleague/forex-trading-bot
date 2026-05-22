# Checklist cerinte licenta

Verificare dupa documentul `Structura proiectului de diploma v1.0.pdf`.

| Cerinta | Status | Unde este acoperita |
|---|---:|---|
| Format LaTeX / Word | Inclus | `documentatie/main.tex` |
| Font 12, margini 2 cm, spatiere 1 | Inclus | `macros/dimensiuni.tex`, `main.tex` |
| Rezumat max. 1 pagina, 150-250 cuvinte, keywords | Inclus | `continut/rezumat.tex` |
| Introducere: context, obiective, metodologie, structura | Inclus | Capitolul 1 |
| Fundamentare teoretica | Inclus | Capitolul 2 |
| Solutii similare si comparatie | Inclus | Capitolul 2 |
| Lacune/stadiu actual si cerinte derivate | Inclus | Capitolul 2 |
| Cerinte ale utilizatorului | Inclus | Capitolul 3 |
| Cerinte functionale si nefunctionale | Inclus | Capitolul 3 |
| Arhitectura/modelarea sistemului | Inclus | Capitolul 3 + figuri |
| Descriere module software | Inclus | Capitolul 3 |
| Alegerea tehnologiilor si justificare | Inclus | Capitolul 3 |
| Algoritmi, pseudocod, formule | Inclus | Capitolul 3 + Anexe |
| Testare si punere in functiune | Inclus | Capitolul 4 |
| Parametri de configurare si instalare | Inclus | Capitolul 4 + Anexe |
| Metrici, benchmarks, rezultate experimentale | Inclus | Capitolul 4 |
| Observabilitate live Prometheus/Grafana | Inclus | Capitolul 3, Capitolul 4, Anexe |
| Grafice/tabele cu rezultate | Inclus | Capitolul 4 |
| Fiabilitate, securitate, scalabilitate | Inclus | Capitolul 4 |
| Capturi/artefacte vizuale relevante | Inclus | Capitolul 4 |
| Contributii personale | Inclus | Capitolul 5 |
| Concluzii si directii viitoare | Inclus | Capitolul 5 |
| Bibliografie IEEE | Inclus | `bibliografie/bibliografie.bib` |
| Anexe cu comenzi, parametri si cod relevant | Inclus | `anexe/anexe.tex` |

## Observatii finale

- Pentru predare trebuie compilat PDF-ul final.
- Daca profesorul cere dovada vizuala din aplicatie, adauga o captura reala din `output/.../dashboard.html`.
- Pentru observabilitate live, capturile recomandate sunt Grafana pe `localhost:33000` si Prometheus Targets pe `localhost:39090/targets`.
- Inlocuieste declaratia de autenticitate cu varianta semnata, daca este ceruta.

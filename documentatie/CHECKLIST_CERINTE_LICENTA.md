# Checklist cerinte documentatie licenta

Aceasta lista poate fi folosita inainte de predare pentru a verifica rapid daca documentatia acopera elementele asteptate intr-un proiect tehnic.

| Zona | Status | Unde este acoperita |
|---|---:|---|
| Tema, motivatie, context | Inclus | Capitolul 1 |
| Obiective generale si specifice | Inclus | Capitolul 1 |
| Metodologie de lucru | Inclus | Capitolul 1 |
| Studiu teoretic si bibliografic | Inclus | Capitolul 2 |
| Cerinte functionale | Inclus | Capitolul 3 |
| Cerinte nefunctionale | Inclus | Capitolul 3 |
| Arhitectura sistemului | Inclus | Capitolul 3 + figura arhitectura |
| Flux de date si flux decizional | Inclus | Capitolul 3 + figuri pipeline/backtesting/live |
| Descriere module software | Inclus | Capitolul 3 |
| Algoritmi si formule | Inclus | Capitolul 3 |
| Managementul riscului | Inclus | Capitolele 2, 3 si 4 |
| Integrare cu broker/API extern | Inclus | Capitolul 3 si Capitolul 4 |
| Instalare si rulare | Inclus | Capitolul 4 si Anexe |
| Scenarii de test | Inclus | Capitolul 4 |
| Rezultate experimentale | Inclus | Capitolul 4 |
| Interpretarea rezultatelor | Inclus | Capitolul 4 |
| Limitari | Inclus | Capitolul 4 si Capitolul 5 |
| Concluzii si dezvoltari viitoare | Inclus | Capitolul 5 |
| Anexe cu comenzi, fisiere si cod | Inclus | Anexe |
| Bibliografie IEEE | Inclus | `bibliografie/bibliografie.bib` |

## Observatii importante

- Valorile experimentale din Capitolul 4 provin din fisierele `summary.txt` din `output`.
- Modul live este documentat explicit ca paper trading, nu ca tranzactionare reala.
- Pentru o forma finala foarte curata, recomand eliminarea imaginilor demo ramase din template daca nu mai sunt folosite.
- Pentru compilare locala este necesar `pdflatex` si rularea cu `-shell-escape`, deoarece template-ul foloseste `minted`.

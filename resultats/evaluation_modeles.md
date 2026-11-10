# Evaluation du modele de prediction des loyers - appartements

Jeu de donnees : 1409 annonces, 74 quartiers, loyer median 700 000 F CFA.
Protocole : validation croisee 5 plis x 3 repetitions ; chaque valeur est une moyenne sur les 15 plis, l'ecart-type du R2 est entre parentheses.

## 1. Comparaison des modeles

| Modele | MAE (F CFA) | RMSE (F CFA) | R2 | R2 (log) | Erreur relative mediane | Predictions a +/- 20 % |
|---|---:|---:|---:|---:|---:|---:|
| Reference : loyer median global | 532 181 | 790 774 | -0.088 (+/- 0.026) | -0.004 | 54.2 % | 18.9 % |
| Reference : loyer median du quartier | 387 141 | 612 855 | 0.346 (+/- 0.069) | 0.471 | 33.4 % | 30.8 % |
| Regression lineaire (Ridge, log du loyer) | 318 975 | 521 709 | 0.525 (+/- 0.067) | 0.696 | 27.4 % | 37.0 % |
| Gradient Boosting | 302 942 | 494 379 | 0.573 (+/- 0.067) | 0.713 | 27.2 % | 37.4 % |
| Random Forest (modele retenu) | 305 158 | 499 518 | 0.564 (+/- 0.074) | 0.698 | 26.0 % | 38.8 % |

## 2. Contribution des groupes de variables (Random Forest)

| Variables utilisees | MAE (F CFA) | RMSE (F CFA) | R2 | R2 (log) | Erreur relative mediane | Predictions a +/- 20 % |
|---|---:|---:|---:|---:|---:|---:|
| Modele retenu (chambres, type, standing, quartier, coordonnees) | 305 158 | 499 518 | 0.564 (+/- 0.074) | 0.698 | 26.0 % | 38.8 % |
| + dates de publication (configuration d'origine) | 319 123 | 526 236 | 0.517 (+/- 0.070) | 0.674 | 26.6 % | 38.4 % |
| Sans coordonnees GPS (quartier seul) | 311 994 | 513 029 | 0.541 (+/- 0.062) | 0.670 | 27.1 % | 37.6 % |
| Sans nom de quartier (coordonnees seules) | 305 202 | 499 187 | 0.565 (+/- 0.073) | 0.697 | 26.3 % | 38.9 % |
| Sans aucune localisation | 417 639 | 642 341 | 0.281 (+/- 0.057) | 0.428 | 38.0 % | 27.3 % |
| Localisation seule | 383 400 | 611 314 | 0.350 (+/- 0.066) | 0.505 | 33.6 % | 29.5 % |
| Sans standing | 307 191 | 500 764 | 0.562 (+/- 0.070) | 0.691 | 26.4 % | 37.9 % |
| Sans type de logement | 308 703 | 503 747 | 0.557 (+/- 0.073) | 0.696 | 27.1 % | 38.0 % |
| Sans nombre de chambres | 342 867 | 551 641 | 0.470 (+/- 0.054) | 0.619 | 29.2 % | 34.6 % |

## 3. Importance des variables par permutation

Baisse moyenne du R2 hors echantillon quand on melange la variable.

| Variable | Baisse du R2 | Ecart-type |
|---|---:|---:|
| Latitude | 0.546 | 0.091 |
| Chambres numeriques | 0.415 | 0.052 |
| Longitude | 0.198 | 0.027 |
| Type logement | 0.040 | 0.019 |
| Quartier nettoye | 0.022 | 0.011 |
| Standing | 0.014 | 0.010 |

## 4. Erreur selon le niveau de loyer

Biais moyen positif = le modele surestime.

| Tranche de loyer (F CFA) | Annonces | Loyer median | MAE | Erreur relative mediane | Biais moyen |
|---|---:|---:|---:|---:|---:|
| < 300 000 | 257 | 250 000 | 85 084 | 24.0 % | 70 414 |
| 300 000 - 600 000 | 393 | 450 000 | 184 895 | 26.2 % | 124 098 |
| 600 000 - 1 000 000 | 333 | 800 000 | 225 409 | 23.5 % | 47 471 |
| 1 000 000 - 2 000 000 | 320 | 1 500 000 | 453 971 | 27.5 % | -290 826 |
| > 2 000 000 | 106 | 2 700 000 | 1 053 053 | 29.7 % | -997 906 |

## 5. Erreur dans les 12 quartiers les plus representes

| Quartier | Annonces | Loyer median | MAE | Erreur relative mediane | Biais moyen |
|---|---:|---:|---:|---:|---:|
| Almadies | 246 | 1 000 000 | 393 791 | 29.5 % | -128 203 |
| Mermoz | 134 | 950 000 | 473 939 | 38.3 % | -126 420 |
| Mamelles | 120 | 500 000 | 192 474 | 22.6 % | -58 934 |
| Point E | 107 | 1 000 000 | 312 899 | 23.0 % | -96 619 |
| Ngor | 69 | 750 000 | 358 203 | 33.8 % | -99 542 |
| Fann | 65 | 2 000 000 | 555 471 | 21.7 % | -127 797 |
| Plateau | 62 | 1 700 000 | 523 887 | 28.5 % | -142 814 |
| Virage | 55 | 1 000 000 | 301 270 | 24.4 % | -60 567 |
| Sacré-Cœur | 51 | 500 000 | 150 272 | 21.0 % | -32 606 |
| Ouakam | 46 | 450 000 | 486 789 | 43.1 % | -135 740 |
| Cite Keur Gorgui | 36 | 650 000 | 251 946 | 21.1 % | -103 564 |
| Yoff | 31 | 400 000 | 132 479 | 25.7 % | 2 620 |

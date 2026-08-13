---
module: methodologie
titre: Méthodologie et évaluation
ordre: 3
resume: Le module le plus rentable et le plus négligé. C'est lui qui sépare le débutant du praticien.
---

## Découpage des données
<!-- slug: decoupage-donnees | difficulte: 1 | prereqs:  -->

**En une phrase** — Séparer les données en jeux distincts pour que le score annoncé reflète la performance sur des données jamais vues.

**Explication** — Trois rôles, à ne jamais confondre. Le jeu d'**entraînement** sert à ajuster les paramètres. Le jeu de **validation** sert à choisir : hyperparamètres, features, algorithme, seuil. Le jeu de **test** sert une seule fois, à la fin, pour annoncer un chiffre. Dès qu'on regarde le test pour prendre une décision, il cesse d'être un test et devient un jeu de validation — et le score qu'il produit devient optimiste.

La proportion habituelle est 60/20/20, ou 80/20 avec validation croisée sur les 80. Ce qui importe davantage est la **manière** de découper. Un découpage aléatoire suppose que les lignes sont indépendantes et identiquement distribuées. Cette hypothèse est fausse dès qu'il y a une dimension temporelle (il faut couper par date), des groupes (plusieurs lignes par patient : couper par patient), ou un déséquilibre de classes (il faut stratifier pour garder les mêmes proportions).

Le cas temporel mérite une insistance particulière : prédire le passé à partir du futur donne des scores magnifiques et un modèle inutilisable. Si les données ont une date, le test doit être **postérieur** à l'entraînement, comme en production.

**Cas d'utilisation**
- Toujours, sans exception, avant la première ligne de modélisation.
- `stratify=y` dès qu'une classe représente moins de 20 % des données.
- `GroupShuffleSplit` quand plusieurs lignes proviennent d'une même entité.
- Découpage temporel dès qu'une date existe, même si le problème ne semble pas temporel.

**Algorithme**
```text
1. Identifier l'unité d'indépendance : ligne ? patient ? session ? jour ?
2. Identifier s'il existe un axe temporel. Si oui, découper par date.
3. Sinon, découpage aléatoire stratifié sur la cible.
4. Mettre le jeu de test de côté et NE PLUS Y TOUCHER.
5. Tout le prétraitement (imputation, encodage, échelle) s'apprend sur
   l'entraînement seul et s'applique aux autres jeux. Un Pipeline le garantit.
6. Noter la date et la logique du découpage dans le code, pas dans sa tête.
```

**Implémentation**
```python
from sklearn.model_selection import train_test_split, GroupShuffleSplit
import numpy as np

# Cas standard : stratifié
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                          stratify=y, random_state=0)

# Cas groupé : aucun patient ne doit apparaître dans deux jeux
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
i_tr, i_te = next(gss.split(X, y, groups=df['patient_id']))

# Cas temporel : le test est postérieur, comme en production
coupe = df['date'].quantile(0.8)
tr, te = df[df['date'] <= coupe], df[df['date'] > coupe]

# Vérification systématique après tout découpage
print("proportions :", y_tr.mean().round(4), y_te.mean().round(4))
print("chevauchement d'identifiants :",
      len(set(tr['client_id']) & set(te['client_id'])))     # doit valoir 0
```

**Outils** — `scikit-learn.model_selection`.

**Alternatives open-source**
- *Bibliothèques* : **scikit-learn** couvre tous les cas (`StratifiedShuffleSplit`, `GroupKFold`, `TimeSeriesSplit`, `StratifiedGroupKFold`) ; **sktime** pour des découpages temporels avancés avec horizons multiples ; **DeepChecks** vérifie automatiquement qu'un découpage ne fuit pas et compare les distributions des jeux.
- *Outils graphiques* : **DeepChecks** produit un rapport HTML de validation de découpage ; **Evidently AI** compare visuellement les distributions entraînement/test et alerte sur les écarts ; **Orange** propose le bloc Data Sampler avec stratification.

**Astuces**
- Fixer `random_state` partout. Un score qui varie de 3 points entre deux exécutions n'est pas comparable, et cette variabilité elle-même est une information à mesurer.
- Comparer les distributions des features entre entraînement et test. Un écart net signale un découpage cassé ou une dérive réelle.
- Vérifier l'absence d'identifiants communs entre les jeux. Une ligne dupliquée dans le dataset original suffit à créer une fuite.
- Si le dataset fait moins de 1 000 lignes, un jeu de test unique est trop bruité : passer à une validation croisée imbriquée.
- Le jeu de test doit ressembler à la production, pas aux données d'entraînement. Si le modèle tournera sur des clients nouveaux, le test doit contenir des clients nouveaux.

## Validation croisée
<!-- slug: validation-croisee | difficulte: 2 | prereqs: decoupage-donnees -->

**En une phrase** — Découper l'entraînement en $k$ blocs, entraîner $k$ fois en laissant chaque bloc de côté à son tour, et moyenner les scores.

**Explication** — Un jeu de validation unique donne une estimation bruitée : le score dépend des lignes tombées dedans. La validation croisée en $k$ blocs utilise toutes les données pour valider, chacune une fois, et fournit à la fois une **moyenne** et un **écart type**. Cet écart type est aussi important que la moyenne : deux modèles à 0,84 avec des écarts de 0,01 et de 0,06 ne se valent pas du tout.

$k=5$ est le standard, $k=10$ quand les données sont rares, $k=n$ (*leave-one-out*) presque jamais — très coûteux, et avec une variance élevée contrairement à l'intuition. La variante **stratifiée** conserve les proportions de classes dans chaque bloc et devrait être le défaut en classification (scikit-learn l'applique automatiquement).

Le point crucial concerne le réglage d'hyperparamètres. Si on choisit les hyperparamètres avec la validation croisée puis qu'on annonce le meilleur score de cette même validation croisée, ce chiffre est optimiste : on a sélectionné le maximum d'un ensemble de valeurs bruitées. La solution correcte est la **validation croisée imbriquée** — une boucle interne pour choisir, une boucle externe pour estimer. Coûteuse, mais c'est la seule estimation honnête.

**Cas d'utilisation**
- Comparer des modèles ou des hyperparamètres sur moins de 100 000 lignes.
- Estimer la variabilité d'un score, donc savoir si un écart est réel.
- Générer des prédictions hors-échantillon pour tout le dataset (`cross_val_predict`), base de l'empilement de modèles.
- Inutile sur des millions de lignes : un jeu de validation unique y est déjà stable, et $k$ entraînements coûtent cher.

**Algorithme**
```text
Validation croisée simple :
1. Partitionner l'entraînement en k blocs (stratifiés si classification).
2. Pour i = 1..k :
     entraîner sur tous les blocs sauf i ; scorer sur le bloc i.
3. Rapporter moyenne ± écart type.

Validation croisée imbriquée (la seule estimation non biaisée) :
Pour chaque bloc externe i :
   a. Sur les k-1 autres blocs, faire une validation croisée interne
      pour choisir les hyperparamètres.
   b. Réentraîner avec ces hyperparamètres sur ces k-1 blocs.
   c. Scorer sur le bloc i.
Rapporter la moyenne des scores externes.
```

**Implémentation**
```python
import numpy as np
from sklearn.model_selection import (cross_validate, StratifiedKFold,
                                     TimeSeriesSplit, GridSearchCV, cross_val_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

res = cross_validate(pipe, X, y, cv=cv,
                     scoring=['roc_auc', 'average_precision', 'f1'],
                     return_train_score=True, n_jobs=-1)
for m in ['roc_auc', 'average_precision', 'f1']:
    t, v = res[f'train_{m}'], res[f'test_{m}']
    print(f"{m:20s} val {v.mean():.3f} ± {v.std():.3f}   (train {t.mean():.3f})")
# Un grand écart train/val signale du surapprentissage.

# Séries temporelles : jamais de mélange, entraînement toujours dans le passé
tscv = TimeSeriesSplit(n_splits=5, gap=7)     # gap évite la fuite par proximité

# Validation croisée imbriquée
interne = GridSearchCV(pipe, {'logisticregression__C': np.logspace(-3, 2, 6)},
                       cv=StratifiedKFold(4), scoring='roc_auc', n_jobs=-1)
scores = cross_val_score(interne, X, y, cv=cv, scoring='roc_auc')
print(f"estimation honnête : {scores.mean():.3f} ± {scores.std():.3f}")
```

**Outils** — `scikit-learn.model_selection` : `cross_validate`, `GridSearchCV`, `HalvingGridSearchCV`.

**Alternatives open-source**
- *Bibliothèques* : **Optuna** remplace la recherche sur grille par une recherche bayésienne, avec élagage des essais peu prometteurs — 10 fois moins d'entraînements pour un meilleur résultat ; **scikit-learn** `HalvingRandomSearchCV` applique le *successive halving* sans dépendance externe ; **Ray Tune** pour distribuer la recherche sur plusieurs machines ; **sktime** et **mlforecast** pour la validation temporelle glissante ; **MAPIE** pour des intervalles de prédiction garantis par validation croisée conforme.
- *Outils graphiques* : **Optuna Dashboard** montre l'importance des hyperparamètres et l'historique de recherche ; **MLflow UI** compare les exécutions et leurs métriques ; **H2O AutoML** produit un tableau de comparaison de modèles validés croisés ; **Orange** propose Test & Score avec plusieurs modèles côte à côte.

**Astuces**
- Toujours mettre le prétraitement dans un `Pipeline`. Standardiser avant la validation croisée fait fuiter les statistiques des blocs de validation vers l'entraînement, et gonfle le score de façon invisible.
- Rapporter l'écart type. Un modèle à 0,84 ± 0,06 n'est pas meilleur qu'un modèle à 0,82 ± 0,01 ; il est plus fragile.
- `shuffle=True` est nécessaire si le dataset est trié (par classe, par date, par identifiant). Sans lui, les blocs sont systématiquement biaisés.
- `cross_val_predict` donne une prédiction hors-échantillon pour chaque ligne : parfait pour tracer une matrice de confusion honnête ou pour construire un méta-modèle.
- Une différence énorme entre score d'entraînement et score de validation est du surapprentissage. Une différence énorme entre validation et test est une fuite dans la sélection.
- La recherche sur grille explose combinatoirement. Au-delà de trois hyperparamètres, passer à Optuna ou à une recherche aléatoire : sur un même budget, elle trouve mieux.

## Fuite de données
<!-- slug: fuite-de-donnees | difficulte: 3 | prereqs: validation-croisee -->

**En une phrase** — Toute information présente à l'entraînement mais indisponible au moment réel de la prédiction, qui produit des scores excellents et un modèle inutile.

**Explication** — La fuite est l'erreur la plus coûteuse du ML appliqué, parce qu'elle ne provoque aucune alerte : elle *améliore* les métriques. Un projet peut passer en production avec une AUC de 0,97 en validation et s'effondrer à 0,61, sans qu'aucune ligne de code soit fausse.

Quatre formes principales. La **fuite de prétraitement** : standardiser, imputer ou sélectionner des features sur l'ensemble complet avant le découpage — les statistiques du test ont alors influencé l'entraînement. La **fuite temporelle** : une feature calculée avec des données postérieures à la date de prédiction (moyenne sur l'ensemble de l'historique, y compris le futur). La **fuite de cible** : une feature qui est une conséquence de la cible plutôt qu'une cause — le montant remboursé pour prédire un défaut de paiement, le fait qu'un traitement ait été administré pour prédire un diagnostic. Et la **fuite par duplication** : des lignes quasi identiques réparties entre entraînement et test.

Le signal d'alarme le plus fiable est une performance **trop bonne**. Une AUC supérieure à 0,95 sur un problème métier réel doit déclencher une enquête, pas une célébration. La question à se poser pour chaque feature est unique et suffisante : *cette valeur était-elle connue, sous cette forme, à l'instant où la prédiction doit être faite ?*

**Cas d'utilisation**
- Audit obligatoire de tout modèle avant mise en production.
- Explication d'un écart entre performance de validation et performance réelle.
- Revue de code de features : la liste ci-dessous est une checklist opérationnelle.

**Algorithme**
```text
Audit anti-fuite, dans l'ordre :
1. Trier les features par importance. Si une seule domine massivement,
   la suspecter en premier : l'enlever et voir si le score s'effondre.
2. Pour chaque feature du top 10, répondre par écrit : à quel instant
   cette valeur est-elle disponible en production ?
3. Vérifier que tout prétraitement est dans un Pipeline (donc appris
   dans chaque bloc de validation croisée).
4. Chercher les doublons : df.duplicated(subset=features).sum().
5. Vérifier les identifiants : aucun ID ne doit être une feature, et aucun
   ID ne doit apparaître dans deux jeux.
6. Si un axe temporel existe, revalider avec un découpage temporel strict.
   Un écart important entre validation aléatoire et validation temporelle
   EST la preuve d'une fuite.
7. Comparer le score au meilleur résultat publié sur un problème similaire.
   Un écart de 20 points en votre faveur n'arrive jamais.
```

**Implémentation**
```python
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# FAUX : les statistiques sont calculées sur tout le dataset, test inclus
X_std = StandardScaler().fit_transform(X)            # <-- fuite
X_tr, X_te = train_test_split(X_std, ...)

# JUSTE : le pipeline apprend uniquement sur les blocs d'entraînement
prep = ColumnTransformer([
    ('num', Pipeline([('imp', SimpleImputer(strategy='median')),
                      ('sc', StandardScaler())]), colonnes_num),
    ('cat', OneHotEncoder(handle_unknown='ignore', min_frequency=10), colonnes_cat),
])
modele = Pipeline([('prep', prep), ('clf', LGBMClassifier())])
scores = cross_val_score(modele, X, y, cv=5)          # aucune fuite possible

# Fuite temporelle : une moyenne glissante doit EXCLURE la ligne courante
df = df.sort_values('date')
df['moy_client_7j'] = (df.groupby('client_id')['montant']
                         .transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean()))
#                                              ^^^^^^^^ shift(1) : sans lui, fuite

# Détecteur pragmatique : comparer validation aléatoire et validation temporelle
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
alea = cross_val_score(modele, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0))
temp = cross_val_score(modele, X, y, cv=TimeSeriesSplit(5))
print(f"aléatoire {alea.mean():.3f}  temporel {temp.mean():.3f}")
# Écart > 0.05 : enquêter avant toute autre chose.

# Doublons quasi identiques
print("doublons exacts :", df.duplicated(subset=features).sum())
```

**Outils** — `scikit-learn` (`Pipeline`, `ColumnTransformer`), `pandas` pour l'audit.

**Alternatives open-source**
- *Bibliothèques* : **DeepChecks** exécute une suite de tests dédiés à la fuite (fuite d'identifiants, chevauchement entraînement/test, features suspectes) et produit un rapport ; **SHAP** révèle les features dominantes, premier indice d'une fuite ; **feature-engine** propose des transformations sûres et compatibles Pipeline ; **Evidently AI** compare les distributions et détecte les incohérences temporelles.
- *Outils graphiques* : **DeepChecks** en rapport HTML autonome est le plus direct ; **Evidently** pour le suivi visuel entraînement/production ; **pandas-profiling / ydata-profiling** met en évidence les corrélations extrêmes avec la cible, souvent le premier signe.

**Astuces**
- Une AUC supérieure à 0,95 sur un problème réel est une fuite jusqu'à preuve du contraire. Les problèmes métier authentiques plafonnent autour de 0,75-0,90.
- Ne jamais garder d'identifiant comme feature. Un numéro de client corrélé à la date d'inscription encode l'ancienneté, donc parfois la cible.
- Tout `fit` doit être précédé mentalement de la question « sur quelles données ? ». Un `fit_transform` sur `X` complet est presque toujours un bug.
- Les moyennes par groupe (encodage par cible) fuient si elles incluent la ligne courante. Utiliser un encodage validé croisé ou `shift(1)`.
- La fuite peut venir de l'amont : une base de données mise à jour rétroactivement fait apparaître dans l'historique des valeurs qui n'existaient pas à l'époque. C'est le problème du *point-in-time correctness*, et il exige des données horodatées.
- Écrire, pour chaque feature, une phrase justifiant sa disponibilité au moment de la prédiction. Cette page de documentation coûte une heure et sauve des semaines.

## Compromis biais-variance
<!-- slug: biais-variance | difficulte: 3 | prereqs: validation-croisee -->

**En une phrase** — L'erreur d'un modèle se décompose en erreur de simplification (biais), en sensibilité aux données d'entraînement (variance), et en bruit irréductible.

**Explication** — Formellement, l'erreur quadratique attendue en un point se décompose en $\text{Biais}^2 + \text{Variance} + \sigma^2$. Le **biais** est l'écart entre la prédiction moyenne du modèle (sur tous les échantillons possibles) et la vérité : c'est l'erreur due à des hypothèses trop rigides. La **variance** est la dispersion des prédictions d'un échantillon à l'autre : c'est l'erreur due à une sensibilité excessive. Le **bruit** ne peut être réduit par aucun modèle.

Le diagnostic pratique se lit sur deux courbes. Le **sous-apprentissage** (biais élevé) donne une erreur d'entraînement élevée *et* une erreur de validation élevée, proches l'une de l'autre — le modèle n'arrive même pas à apprendre ses propres données. Le **surapprentissage** (variance élevée) donne une erreur d'entraînement faible et une erreur de validation nettement supérieure. La distinction est essentielle car les remèdes sont **opposés** : contre le biais il faut plus de complexité, plus de features, moins de régularisation ; contre la variance il faut plus de données, plus de régularisation, moins de complexité.

Une nuance moderne mérite d'être connue : le phénomène de **double descente**. Sur les modèles très surparamétrés (réseaux profonds), l'erreur de test remonte à l'approche du point d'interpolation, puis **redescend** quand on continue d'augmenter la capacité. La courbe en U classique reste juste pour les modèles classiques, mais elle n'est pas toute l'histoire.

**Cas d'utilisation**
- Décider quoi faire quand un modèle est décevant : ajouter des données ou changer de modèle ? La réponse dépend entièrement du diagnostic.
- Interpréter une courbe d'apprentissage.
- Justifier le choix d'une méthode d'ensemble : le bagging attaque la variance, le boosting attaque le biais.

**Algorithme**
```text
Diagnostic en trois mesures :
1. Erreur sur l'entraînement (E_tr) et sur la validation (E_val).
2. Estimer l'erreur irréductible : performance humaine, ou meilleur
   résultat publié sur ce type de problème.

   E_tr élevée, E_val ~ E_tr        -> BIAIS. Le modèle est trop simple.
      Remèdes : modèle plus expressif, features d'interaction, moins de
      régularisation, entraîner plus longtemps.

   E_tr faible, E_val >> E_tr       -> VARIANCE. Le modèle mémorise.
      Remèdes : plus de données, régularisation, réduire la complexité,
      ensembles, augmentation de données, arrêt précoce.

   E_tr faible, E_val ~ E_tr, mais insuffisant pour le métier
      -> le problème est ailleurs : données, définition de la cible, bruit.

3. Tracer la courbe d'apprentissage (score vs taille d'échantillon) :
   si les deux courbes se rejoignent haut, plus de données n'aidera pas.
   Si l'écart persiste, plus de données aidera.
```

**Implémentation**
```python
import numpy as np, matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, validation_curve

# Courbe d'apprentissage : plus de données servirait-il à quelque chose ?
tailles, s_tr, s_val = learning_curve(
    modele, X, y, cv=5, scoring='roc_auc', n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 8))

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
a1.plot(tailles, s_tr.mean(1), 'o-', label='entraînement')
a1.plot(tailles, s_val.mean(1), 'o-', label='validation')
a1.fill_between(tailles, s_val.mean(1) - s_val.std(1), s_val.mean(1) + s_val.std(1), alpha=.2)
a1.set(xlabel="taille d'entraînement", ylabel='AUC', title="Courbe d'apprentissage")
a1.legend()

# Courbe de validation : où est l'optimum de complexité ?
profondeurs = [1, 2, 3, 4, 6, 8, 12, 20, None]
c_tr, c_val = validation_curve(DecisionTreeClassifier(random_state=0), X, y,
                               param_name='max_depth', param_range=profondeurs,
                               cv=5, scoring='roc_auc', n_jobs=-1)
x = [p if p else 25 for p in profondeurs]
a2.plot(x, c_tr.mean(1), 'o-', label='entraînement')
a2.plot(x, c_val.mean(1), 'o-', label='validation')
a2.set(xlabel='profondeur max', title='Courbe de validation')
a2.legend()

# Décomposition empirique biais / variance par bootstrap
def decomposer(faire_modele, X_tr, y_tr, X_te, y_te, B=50, graine=0):
    rng = np.random.default_rng(graine)
    P = np.stack([faire_modele().fit(*_ech(X_tr, y_tr, rng)).predict(X_te) for _ in range(B)])
    biais2 = ((P.mean(0) - y_te) ** 2).mean()
    variance = P.var(0).mean()
    return biais2, variance

def _ech(X, y, rng):
    i = rng.integers(0, len(y), len(y))
    return X[i], y[i]
```

**Outils** — `scikit-learn` (`learning_curve`, `validation_curve`).

**Alternatives open-source**
- *Bibliothèques* : **Yellowbrick** produit directement les visualiseurs `LearningCurve`, `ValidationCurve` et `ResidualsPlot`, prêts à mettre dans un rapport ; **mlxtend** propose `bias_variance_decomp`, une décomposition empirique clé en main ; **scikit-learn** `LearningCurveDisplay` depuis la version 1.2.
- *Outils graphiques* : **TensorBoard** pour suivre l'écart entraînement/validation en direct pendant un entraînement long ; **Weights & Biases** (cœur libre) pour comparer visuellement plusieurs configurations ; **Orange** avec le bloc Test & Score et sa courbe d'apprentissage intégrée.

**Astuces**
- Toujours tracer les deux courbes ensemble. Un score de validation seul ne permet aucun diagnostic.
- Avant d'accuser le modèle, faire le test du sur-apprentissage volontaire : sur 20 exemples, un modèle correct doit atteindre une erreur quasi nulle. S'il n'y arrive pas, le bug est dans le code ou les données.
- « Ajouter des données » ne résout jamais un problème de biais. Cette dépense est le gaspillage le plus fréquent en entreprise.
- Le bagging (forêts) réduit la variance sans toucher au biais ; le boosting réduit le biais et peut augmenter la variance. Choisir selon le diagnostic, pas selon la mode.
- Un écart entraînement/validation important n'est pas toujours grave : sur du boosting bien réglé avec arrêt précoce, un écart de 5 points est normal et le modèle généralise bien.
- L'erreur irréductible est souvent sous-estimée. Sur un problème où deux experts humains ne s'accordent que dans 85 % des cas, aucun modèle ne dépassera durablement 85 %.

## Métriques de classification
<!-- slug: metriques-classification | difficulte: 2 | prereqs: probabilites-bayes -->

**En une phrase** — Choisir l'indicateur qui reflète le coût réel des erreurs, ce que l'exactitude ne fait presque jamais.

**Explication** — Tout part de la matrice de confusion : vrais positifs (VP), faux positifs (FP), vrais négatifs (VN), faux négatifs (FN). De là se déduisent la **précision** $VP/(VP+FP)$ — « quand je dis oui, ai-je raison ? » — et le **rappel** $VP/(VP+FN)$ — « quelle proportion des positifs ai-je trouvée ? ». Les deux s'opposent : baisser le seuil augmente le rappel et dégrade la précision. Le **F1** est leur moyenne harmonique, pratique mais qui masque l'arbitrage ; le $F_\beta$ permet de pondérer ($\beta = 2$ favorise le rappel).

L'**exactitude** (*accuracy*) est trompeuse dès qu'il y a déséquilibre : sur 1 % de fraudes, prédire « jamais de fraude » donne 99 % d'exactitude et une valeur métier nulle. C'est l'erreur la plus commune dans les projets débutants.

Deux métriques s'affranchissent du seuil. L'**AUC-ROC** est la probabilité qu'un positif tiré au hasard reçoive un score supérieur à un négatif tiré au hasard : elle mesure la capacité de **classement**, indépendamment de la prévalence — d'où sa popularité, mais aussi son défaut, elle reste optimiste sur des classes très déséquilibrées. L'**AUC-PR** (précision-rappel moyenne) est bien plus informative dans ce cas, parce que sa valeur de référence est la prévalence elle-même, pas 0,5.

**Cas d'utilisation**
- AUC-PR ou rappel à précision fixée : fraude, maladies rares, modération — tout ce qui a une classe minoritaire.
- Rappel prioritaire quand un faux négatif est coûteux : dépistage médical, détection de panne.
- Précision prioritaire quand un faux positif est coûteux : blocage de compte, envoi commercial.
- Log-loss ou Brier quand la probabilité elle-même est utilisée (tarification, allocation de budget).
- Kappa de Cohen ou MCC pour un résumé unique robuste au déséquilibre.

**Algorithme**
```text
Choix de la métrique, dans l'ordre des questions :
1. La sortie utilisée est-elle une probabilité ou une décision ?
     probabilité -> log-loss, Brier, courbe de calibration
     décision    -> suite ci-dessous
2. Les classes sont-elles équilibrées (min > 20 %) ?
     oui -> accuracy et F1 sont acceptables ; AUC-ROC pour le classement
     non -> AUC-PR, rappel@précision, ou coût métier explicite
3. Le coût d'un FP et celui d'un FN sont-ils différents ?
     oui -> construire la métrique de coût, optimiser le seuil dessus
4. Multi-classe ?
     macro   : chaque classe compte autant (favorise les rares)
     weighted: pondéré par l'effectif (proche de l'accuracy)
     micro   : agrège tous les comptes (= accuracy en mono-étiquette)
```

**Implémentation**
```python
import numpy as np
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_auc_score, average_precision_score,
                             precision_recall_curve, log_loss, matthews_corrcoef,
                             ConfusionMatrixDisplay, PrecisionRecallDisplay)

p = modele.predict_proba(X_te)[:, 1]

print(f"AUC-ROC {roc_auc_score(y_te, p):.3f}")
print(f"AUC-PR  {average_precision_score(y_te, p):.3f}  (référence = {y_te.mean():.3f})")
print(f"log-loss {log_loss(y_te, p):.4f}")

# Le seuil se CHOISIT. Ici : le plus haut rappel avec au moins 80 % de précision.
prec, rap, seuils = precision_recall_curve(y_te, p)
ok = prec[:-1] >= 0.80
seuil = seuils[ok][np.argmax(rap[:-1][ok])]
print(f"seuil retenu {seuil:.3f} -> rappel {rap[:-1][ok].max():.3f}")

y_pred = (p >= seuil).astype(int)
print(classification_report(y_te, y_pred, digits=3))
print(confusion_matrix(y_te, y_pred))
print("MCC :", round(matthews_corrcoef(y_te, y_pred), 3))

# Seuil optimal par coût métier — la seule approche défendable en entreprise
COUT_FP, COUT_FN = 1, 12
grille = np.linspace(0.01, 0.99, 199)
couts = [((p >= s) & (y_te == 0)).sum() * COUT_FP +
         ((p < s) & (y_te == 1)).sum() * COUT_FN for s in grille]
print("seuil coût-optimal :", round(grille[int(np.argmin(couts))], 3))
```

**Outils** — `scikit-learn.metrics`, et ses classes `*Display` pour les graphiques.

**Alternatives open-source**
- *Bibliothèques* : **torchmetrics** pour les mêmes métriques calculées par lots sur GPU ; **imbalanced-learn** ajoute la moyenne géométrique et l'indice d'équilibre ; **Yellowbrick** produit les visualiseurs `ClassificationReport`, `ROCAUC`, `DiscriminationThreshold` ; **scikit-plot** pour des figures rapides ; **fairlearn** pour les métriques d'équité par sous-groupe, obligatoires dans certains contextes.
- *Outils graphiques* : **Evidently AI** génère un rapport de performance de classification complet en HTML ; **MLflow UI** stocke et compare les métriques entre exécutions ; **TensorBoard** pour le suivi pendant l'entraînement ; **Orange** affiche matrices de confusion et courbes ROC en blocs.

**Astuces**
- Ne jamais rapporter l'exactitude seule. Toujours la comparer à la fréquence de la classe majoritaire — le vrai point de référence.
- `predict()` applique un seuil de 0,5 arbitraire. Sur des classes déséquilibrées, il produit un modèle inutilisable alors que le classement est bon. Travailler avec `predict_proba` et choisir le seuil.
- Le seuil se choisit sur la **validation**, jamais sur le test : c'est un hyperparamètre comme un autre.
- Sur 1 % de positifs, une AUC-ROC de 0,90 peut correspondre à une AUC-PR de 0,15. Toujours donner les deux, et rappeler la prévalence à côté de l'AUC-PR.
- En multi-classe, `average='macro'` traite les classes rares à égalité — c'est généralement ce qu'on veut, et ce n'est pas le défaut.
- Une matrice de confusion se lit toujours, même quand toutes les métriques sont calculées. Elle révèle *quelles* confusions se produisent, information qu'aucun scalaire ne contient.

## Métriques de régression
<!-- slug: metriques-regression | difficulte: 1 | prereqs: regression-lineaire -->

**En une phrase** — Mesurer l'écart entre valeurs prédites et réelles, en choisissant la forme de pénalité qui correspond au coût métier.

**Explication** — La **RMSE** $\sqrt{\frac{1}{n}\sum(y-\hat{y})^2}$ pénalise quadratiquement : une erreur de 10 pèse cent fois une erreur de 1. Elle est donc dominée par les grosses erreurs, ce qui est souhaitable quand les grandes erreurs sont réellement catastrophiques, et néfaste quand la cible contient des valeurs aberrantes. La **MAE** $\frac{1}{n}\sum|y-\hat{y}|$ pénalise linéairement, s'interprète directement dans l'unité de la cible, et résiste aux aberrations. Fait souvent ignoré : minimiser la RMSE conduit à prédire la **moyenne** conditionnelle, minimiser la MAE conduit à prédire la **médiane** conditionnelle. Ce n'est pas le même modèle.

Le **$R^2$** exprime la part de variance expliquée : 0 correspond à prédire la moyenne, 1 à la perfection, et les valeurs négatives sont possibles sur un jeu de test. Son avantage est d'être sans unité, son inconvénient de dépendre de la variance de l'échantillon — un $R^2$ n'est comparable qu'à l'intérieur d'un même jeu de données.

Pour des cibles positives et multiplicatives (prix, ventes, trafic), les métriques relatives sont plus pertinentes : **MAPE** (erreur en pourcentage, mais qui explose près de zéro et pénalise asymétriquement), **RMSLE** (erreur sur les logarithmes, qui traite une prédiction doublée comme une prédiction divisée par deux), ou **MASE** en séries temporelles (rapport à une prédiction naïve).

**Cas d'utilisation**
- RMSE quand les grosses erreurs coûtent disproportionnellement cher (dimensionnement, stock critique).
- MAE quand toutes les erreurs coûtent proportionnellement, ou en présence d'aberrations.
- RMSLE / MAPE quand l'erreur relative est ce qui importe : une erreur de 10 € sur 50 € n'est pas la même que sur 5 000 €.
- Perte quantile quand il faut un intervalle plutôt qu'un point (prévision de demande, stocks de sécurité).

**Algorithme**
```text
1. Établir une baseline : prédire la moyenne (ou la valeur de la veille en
   séries temporelles). Toute métrique s'interprète relativement à elle.
2. Choisir la métrique selon le coût métier :
     erreurs symétriques et absolues        -> MAE
     grosses erreurs très coûteuses         -> RMSE
     erreur relative                        -> RMSLE ou MAPE
     besoin d'un intervalle                 -> perte quantile (0.1 et 0.9)
     sous-estimation plus grave que sur-estimation -> perte asymétrique sur mesure
3. Rapporter la métrique ET son intervalle de confiance par bootstrap.
4. Tracer les résidus. Toujours. Aucune métrique ne remplace ce graphique.
```

**Implémentation**
```python
import numpy as np
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             mean_absolute_percentage_error, median_absolute_error)
from sklearn.dummy import DummyRegressor

base = DummyRegressor(strategy='mean').fit(X_tr, y_tr)
for nom, m in [('baseline', base), ('modele', modele)]:
    p = m.predict(X_te)
    print(f"{nom:9s} MAE {mean_absolute_error(y_te, p):8.2f}  "
          f"RMSE {mean_squared_error(y_te, p) ** 0.5:8.2f}  "
          f"MedAE {median_absolute_error(y_te, p):8.2f}  "
          f"R² {r2_score(y_te, p):6.3f}")

# RMSLE : pour des cibles positives et multiplicatives
def rmsle(y, p):
    return np.sqrt(np.mean((np.log1p(np.clip(p, 0, None)) - np.log1p(y)) ** 2))

# Prédiction d'intervalle par régression quantile
import lightgbm as lgb
bas = lgb.LGBMRegressor(objective='quantile', alpha=0.1).fit(X_tr, y_tr)
haut = lgb.LGBMRegressor(objective='quantile', alpha=0.9).fit(X_tr, y_tr)
couverture = ((y_te >= bas.predict(X_te)) & (y_te <= haut.predict(X_te))).mean()
print(f"couverture de l'intervalle 80 % : {couverture:.1%}")   # doit être proche de 0.80

# Perte asymétrique : sous-estimer un stock coûte 3 fois plus que surestimer
def cout_asymetrique(y, p, sous=3.0, sur=1.0):
    e = y - p
    return np.mean(np.where(e > 0, sous * e, -sur * e))
```

**Outils** — `scikit-learn.metrics`, `DummyRegressor` pour la baseline.

**Alternatives open-source**
- *Bibliothèques* : **MAPIE** produit des intervalles de prédiction avec garantie de couverture (prédiction conforme) ; **statsmodels** `QuantReg` pour la régression quantile classique ; **NGBoost** prédit une distribution complète, pas un point ; **darts** et **sktime** fournissent les métriques spécialisées des séries temporelles (MASE, sMAPE) ; **properscoring** pour le CRPS, la bonne métrique des prévisions probabilistes.
- *Outils graphiques* : **Yellowbrick** `ResidualsPlot` et `PredictionError` en deux lignes ; **Evidently AI** pour un rapport de régression complet avec analyse des résidus par segment ; **Orange** avec ses blocs de scoring et graphiques de dispersion.

**Astuces**
- Toujours comparer à `DummyRegressor`. Un RMSE de 4,2 ne veut rien dire sans savoir que la baseline est à 4,5 — ou à 40.
- Tracer les résidus contre les prédictions et contre chaque feature importante. Une structure visible dans les résidus signifie qu'il reste du signal à capter.
- Un $R^2$ ne se compare pas entre deux jeux de données différents. Sur un jeu à faible variance, un excellent modèle peut afficher un $R^2$ médiocre.
- La MAPE est indéfinie en zéro et pénalise plus la surestimation que la sous-estimation. Sur des cibles pouvant approcher zéro, utiliser sMAPE ou passer en RMSLE.
- Si on optimise la RMSE mais qu'on rapporte la MAE, le modèle n'est pas optimal pour ce qu'on mesure. Aligner la perte d'entraînement et la métrique d'évaluation.
- Une seule valeur aberrante dans la cible du jeu de test peut faire varier la RMSE de 30 %. Rapporter aussi la médiane des erreurs absolues (`MedAE`), beaucoup plus stable.

## Classes déséquilibrées
<!-- slug: classes-desequilibrees | difficulte: 3 | prereqs: metriques-classification -->

**En une phrase** — Quand une classe est rare, la plupart des algorithmes l'ignorent ; il faut agir sur la métrique, sur les poids, sur le seuil, et seulement en dernier recours sur les données.

**Explication** — Un déséquilibre de 1:100 ou 1:10 000 est la norme en fraude, en défaut de paiement, en maladie rare, en panne industrielle. Le problème n'est pas que les algorithmes soient mauvais : c'est qu'ils optimisent correctement une fonction qui ne correspond pas à l'objectif. Minimiser le nombre d'erreurs, quand 99,9 % des cas sont négatifs, revient à prédire toujours négatif.

Il existe quatre leviers, à essayer dans cet ordre de rentabilité. **Un** : changer la métrique — AUC-PR, rappel à précision fixée, coût métier. C'est gratuit et souvent suffisant. **Deux** : ajuster le **seuil** de décision, également gratuit, et le levier le plus sous-utilisé. **Trois** : pondérer les classes dans la fonction de perte (`class_weight='balanced'`, `scale_pos_weight`) — un exemple rare compte alors autant que plusieurs exemples fréquents. **Quatre**, et seulement si les trois premiers ne suffisent pas : rééchantillonner. Le sous-échantillonnage de la majorité jette de l'information ; le sur-échantillonnage naïf duplique et favorise le surapprentissage ; **SMOTE** interpole entre voisins de la classe minoritaire pour créer des exemples synthétiques.

Un point d'attention majeur : le rééchantillonnage **doit** se faire à l'intérieur de la validation croisée, sur le seul bloc d'entraînement. Appliquer SMOTE avant le découpage crée des exemples synthétiques dérivés de points qui se retrouvent dans le jeu de test — une fuite classique qui gonfle spectaculairement les scores.

**Cas d'utilisation**
- Détection de fraude, de panne, de churn rare, de maladie.
- Modération de contenu, sécurité informatique, contrôle qualité industriel.
- Tout problème où l'événement d'intérêt est celui qu'on veut trouver *parce qu'il est rare*.
- Si la classe minoritaire dépasse 20 %, ce module est inutile : les méthodes standard suffisent.

**Algorithme**
```text
1. Mesurer la prévalence. Établir la baseline : que donne « toujours négatif » ?
2. Changer de métrique : AUC-PR, et rappel à un niveau de précision exigé
   par le métier. Abandonner l'exactitude.
3. Activer la pondération des classes (class_weight / scale_pos_weight).
4. Optimiser le seuil sur la validation, selon le coût métier des FP et FN.
5. Seulement si insuffisant : rééchantillonner DANS le pipeline de
   validation croisée (imblearn.Pipeline, jamais avant le découpage).
6. Vérifier la calibration : la pondération et le rééchantillonnage
   décalent les probabilités. Recalibrer si les valeurs sont utilisées.
7. Si la classe rare compte moins de ~50 exemples, changer de cadre :
   détection d'anomalies non supervisée plutôt que classification.
```

**Implémentation**
```python
import numpy as np
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedKFold, cross_val_score
import lightgbm as lgb

print("prévalence :", y.mean())

# Levier 3 : pondération — le meilleur rapport résultat / effort
ratio = (y == 0).sum() / (y == 1).sum()
clf = lgb.LGBMClassifier(scale_pos_weight=ratio, n_estimators=500, learning_rate=0.05)

# Levier 4 : rééchantillonnage DANS le pipeline, donc dans chaque bloc de CV
pipe = ImbPipeline([
    ('smote', SMOTE(k_neighbors=5, random_state=0)),      # appliqué au bloc d'entraînement seul
    ('sous', RandomUnderSampler(sampling_strategy=0.5, random_state=0)),
    ('clf', lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05)),
])
cv = StratifiedKFold(5, shuffle=True, random_state=0)
print("AUC-PR :", cross_val_score(pipe, X, y, cv=cv, scoring='average_precision').mean())

# Levier 2 : seuil par contrainte métier — « je peux traiter 500 alertes par jour »
p = clf.fit(X_tr, y_tr).predict_proba(X_val)[:, 1]
budget = 500
seuil = np.sort(p)[-budget]
rappel = ((p >= seuil) & (y_val == 1)).sum() / (y_val == 1).sum()
print(f"avec {budget} alertes : rappel = {rappel:.1%}")

# Cas extrême (moins de 50 positifs) : sortir du cadre supervisé
from sklearn.ensemble import IsolationForest
score_anomalie = -IsolationForest(contamination=0.01, random_state=0).fit(X_tr).score_samples(X_te)
```

**Outils** — `pip install imbalanced-learn`. `scikit-learn` fournit `class_weight`, LightGBM/XGBoost `scale_pos_weight`.

**Alternatives open-source**
- *Bibliothèques* : **imbalanced-learn** est la référence (SMOTE et ses 10 variantes, ADASYN, Tomek Links, EasyEnsemble, BalancedRandomForest) ; **PyOD** rassemble 40 algorithmes de détection d'anomalies pour les cas de rareté extrême ; **focal loss** (dans `torch` ou `focal-loss-pytorch`) pondère dynamiquement les exemples difficiles, très efficace en deep learning ; **CatBoost** avec `auto_class_weights='Balanced'` ; **SDV** pour générer des données synthétiques plus réalistes que SMOTE sur des données tabulaires structurées.
- *Outils graphiques* : **Evidently AI** pour suivre la prévalence et la performance par segment en production ; **Yellowbrick** `ClassBalance` et `DiscriminationThreshold` visualisent l'arbitraire du seuil ; **Orange** dispose de blocs de rééchantillonnage.

**Astuces**
- Essayer la pondération et l'ajustement du seuil **avant** SMOTE. Dans la majorité des cas, cela suffit, et cela ne fabrique aucune donnée.
- SMOTE interpole entre voisins : sur des features catégorielles encodées en one-hot, il produit des valeurs impossibles comme 0,4. Utiliser `SMOTENC` pour des données mixtes.
- `imblearn.pipeline.Pipeline` et non `sklearn.pipeline.Pipeline` : seul le premier applique le rééchantillonnage à l'entraînement et pas à la prédiction.
- Après pondération ou rééchantillonnage, les probabilités sortantes sont **surestimées** pour la classe rare. Si elles servent à une décision chiffrée (montant, priorité), les recalibrer sur des données non rééchantillonnées.
- Un rappel de 90 % avec une précision de 2 % signifie 50 fausses alertes par vraie détection. C'est parfois acceptable (dépistage), parfois absurde (blocage de compte). Chiffrer avec le métier avant d'optimiser.
- Avec moins de 50 exemples positifs, la classification supervisée n'est pas fiable, quelle que soit la technique. Reformuler en détection d'anomalies ou collecter plus d'étiquettes.

## Calibration des probabilités
<!-- slug: calibration | difficulte: 3 | prereqs: metriques-classification, classes-desequilibrees -->

**En une phrase** — Un modèle est calibré si, parmi les cas auxquels il attribue 30 %, environ 30 % sont effectivement positifs — ce qui est rarement le cas par défaut.

**Explication** — Un modèle peut **classer** parfaitement (AUC de 0,95) tout en produisant des probabilités totalement fausses. Or dès que la valeur numérique sert à un calcul — espérance de gain, montant de provision, priorisation avec seuils multiples, combinaison avec un autre modèle — c'est la calibration qui compte, pas le classement.

Chaque famille de modèles a son biais caractéristique. Les SVM et les forêts aléatoires sont **sous-confiants** : ils évitent les valeurs extrêmes, car la moyenne de nombreux arbres ne vaut jamais 0 ni 1. Naive Bayes est fortement **sur-confiant**, du fait de son hypothèse d'indépendance qui multiplie des évidences redondantes. Les réseaux profonds modernes sont sur-confiants, conséquence de l'entraînement prolongé sur une entropie croisée. La régression logistique et le gradient boosting bien réglé sont, eux, à peu près calibrés d'origine.

Deux méthodes de correction. La **régression de Platt** (sigmoïde) ajuste une sigmoïde sur les scores : deux paramètres, robuste avec peu de données, mais impose une forme. La **régression isotonique** ajuste une fonction croissante par morceaux : plus flexible, capable de corriger n'importe quelle déformation monotone, mais elle surapprend en dessous d'environ 1 000 exemples de calibration. Dans les deux cas, la calibration doit être ajustée sur des données **non vues** par le modèle, d'où l'usage de la validation croisée interne de `CalibratedClassifierCV`.

**Cas d'utilisation**
- La probabilité est utilisée dans un calcul : tarification, provision de risque, espérance de revenu, allocation de budget.
- Combinaison ou empilement de plusieurs modèles : des scores mal calibrés se combinent mal.
- Seuils multiples (« vert / orange / rouge » à 20 % et 60 % ») : ces bornes n'ont de sens que calibrées.
- Après pondération de classes ou rééchantillonnage, qui décalent systématiquement les probabilités.
- Inutile si seul le classement importe (choisir le top 100, ordonner des résultats).

**Algorithme**
```text
1. Diagnostiquer : tracer la courbe de fiabilité (probabilité prédite en
   abscisse, fréquence observée en ordonnée, par déciles de score).
   La diagonale = calibration parfaite. Au-dessus = sous-confiance.
2. Quantifier : Brier score, ou ECE (erreur de calibration attendue) =
   moyenne pondérée des |confiance - précision| par intervalle.
3. Corriger sur des données NON vues par le modèle :
     < 1000 exemples de calibration  -> sigmoïde (Platt)
     > 1000                          -> isotonique
4. Revérifier la courbe de fiabilité APRÈS correction.
5. Vérifier que l'AUC n'a pas bougé : la calibration est monotone, donc
   elle ne doit pas modifier le classement.
```

**Implémentation**
```python
import numpy as np, matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score

def ece(y, p, bins=10):
    """Erreur de calibration attendue : le résumé chiffré le plus utile."""
    bornes = np.linspace(0, 1, bins + 1)
    e, n = 0.0, len(y)
    for lo, hi in zip(bornes[:-1], bornes[1:]):
        m = (p > lo) & (p <= hi)
        if m.sum():
            e += m.sum() / n * abs(y[m].mean() - p[m].mean())
    return e

p_brut = modele.predict_proba(X_te)[:, 1]

# Correction : cv=5 ajuste le calibreur en validation croisée interne
cal = CalibratedClassifierCV(modele, method='isotonic', cv=5).fit(X_tr, y_tr)
p_cal = cal.predict_proba(X_te)[:, 1]

for nom, p in [('brut', p_brut), ('calibré', p_cal)]:
    print(f"{nom:8s} Brier {brier_score_loss(y_te, p):.4f}  "
          f"ECE {ece(y_te, p):.4f}  AUC {roc_auc_score(y_te, p):.4f}")
# L'AUC doit rester quasi identique ; Brier et ECE doivent baisser.

# Courbe de fiabilité
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot([0, 1], [0, 1], 'k--', lw=1, label='parfait')
for nom, p in [('brut', p_brut), ('calibré', p_cal)]:
    obs, pred = calibration_curve(y_te, p, n_bins=10, strategy='quantile')
    ax.plot(pred, obs, 'o-', label=nom)
ax.set(xlabel='probabilité prédite', ylabel='fréquence observée',
       title='Courbe de fiabilité')
ax.legend()

# Cas fréquent : rétablir la prévalence après un rééchantillonnage
def corriger_prevalence(p, prev_entrainement, prev_reelle):
    cote = p / (1 - p) * (prev_reelle / (1 - prev_reelle)) * \
           ((1 - prev_entrainement) / prev_entrainement)
    return cote / (1 + cote)
```

**Outils** — `scikit-learn.calibration` (`CalibratedClassifierCV`, `calibration_curve`).

**Alternatives open-source**
- *Bibliothèques* : **netcal** implémente une dizaine de méthodes de calibration dont le *temperature scaling*, standard pour les réseaux profonds (un seul paramètre, ne change pas les prédictions argmax) ; **MAPIE** pour des ensembles de prédiction à couverture garantie plutôt que des probabilités ponctuelles ; **torch-uncertainty** pour l'incertitude en deep learning (ensembles profonds, dropout au test) ; **venn-abers** pour des probabilités avec bornes.
- *Outils graphiques* : **Evidently AI** inclut un rapport de calibration ; **Yellowbrick** et **scikit-plot** produisent les courbes de fiabilité en une ligne ; **Uncertainty Toolbox** génère un ensemble de diagnostics visuels d'incertitude.

**Astuces**
- Le Brier score mesure à la fois le classement et la calibration ; l'ECE isole la calibration. Rapporter les deux, plus la courbe.
- Ne jamais calibrer sur les données d'entraînement du modèle : le calibreur apprendrait à corriger des scores déjà surappris et le résultat serait pire qu'avant.
- L'isotonique surapprend en dessous de 1 000 exemples de calibration et produit des paliers plats. En deçà, la sigmoïde est plus sûre.
- Le *temperature scaling* est le choix standard pour un réseau de neurones : un scalaire $T$ divise les logits, l'argmax est préservé, donc l'exactitude est inchangée et l'ECE baisse fortement.
- Après `class_weight='balanced'` ou SMOTE, la prévalence apprise n'est plus la prévalence réelle. La formule de correction des cotes ci-dessus rétablit l'échelle, ou bien on recalibre sur un échantillon non modifié.
- Vérifier que l'AUC ne change pas après calibration. Si elle change, le calibreur n'est pas monotone et quelque chose est cassé.

---
module: features-pipelines
titre: Features et pipelines
ordre: 4
resume: Le gain marginal d'une bonne feature dépasse presque toujours celui d'un meilleur algorithme.
---

## Encodage des variables catégorielles
<!-- slug: encodage-categoriel | difficulte: 2 | prereqs: pandas-manipulation -->

**En une phrase** — Transformer des modalités textuelles en nombres, sans inventer un ordre qui n'existe pas ni faire exploser la dimension.

**Explication** — Le **one-hot** crée une colonne binaire par modalité : aucun ordre supposé, mais $d$ devient énorme sur une variable à forte cardinalité (30 000 codes postaux). L'**encodage ordinal** attribue 0, 1, 2… : compact, mais il introduit un ordre arbitraire. Cet ordre est catastrophique pour un modèle linéaire (« Paris = 3 fois Lyon »), et parfaitement acceptable pour un arbre, qui peut isoler n'importe quelle valeur par des coupes successives — c'est pourquoi LightGBM et CatBoost préfèrent l'ordinal ou leur propre traitement natif.

Pour la forte cardinalité, l'**encodage par cible** remplace chaque modalité par la moyenne de la cible sur cette modalité. Très puissant, et très dangereux : calculé naïvement, il fait fuiter la cible. La modalité vue une seule fois reçoit exactement la valeur de sa propre ligne. Deux protections indispensables : le **lissage** vers la moyenne globale (proportionnel à l'effectif de la modalité), et le calcul **hors-échantillon** par validation croisée. CatBoost implémente une version ordonnée qui règle le problème structurellement.

Autres options utiles : l'encodage par **fréquence** (remplacer par le nombre d'occurrences), le regroupement des modalités rares dans un « autre », et le **hachage** quand la cardinalité est inconnue à l'avance.

**Cas d'utilisation**
- One-hot avec moins de 15 modalités, pour tout modèle linéaire ou à noyau.
- Ordinal ou natif pour les arbres et le boosting, quelle que soit la cardinalité.
- Encodage par cible au-delà de 50 modalités, avec lissage et validation croisée.
- Hachage pour du texte libre ou des identifiants ouverts (flux en ligne).

**Algorithme**
```text
1. Compter les modalités : df[col].nunique().
2. < 15 modalités  -> one-hot (drop='first' pour un modèle linéaire).
   15-50           -> one-hot avec regroupement des rares (min_frequency).
   > 50            -> encodage par cible lissé, OU natif si LightGBM/CatBoost.
3. Toujours gérer les modalités inconnues à la prédiction :
   handle_unknown='ignore' (one-hot) ou valeur par défaut (cible).
4. Encodage par cible, formule lissée pour la modalité m :
     enc(m) = (n_m * moyenne_m + k * moyenne_globale) / (n_m + k)
   avec k ~ 10 à 50. Calculé en validation croisée hors-échantillon.
```

**Implémentation**
```python
import numpy as np, pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder
from sklearn.compose import ColumnTransformer

# One-hot moderne : regroupe automatiquement les modalités rares
ohe = OneHotEncoder(handle_unknown='infrequent_if_exist',
                    min_frequency=20, sparse_output=False)

# Encodage par cible de scikit-learn (>= 1.3) : validation croisée interne intégrée
te = TargetEncoder(smooth='auto', cv=5, random_state=0)

prep = ColumnTransformer([
    ('faible_card', ohe, ['region', 'canal', 'segment']),
    ('forte_card', te, ['code_postal', 'produit_id']),
], remainder='passthrough')

# --- encodage par cible from scratch, avec lissage (palier 2) ---
def encodage_cible(serie, y, k=20):
    glob = y.mean()
    stats = pd.DataFrame({'x': serie, 'y': y}).groupby('x')['y'].agg(['mean', 'size'])
    lisse = (stats['size'] * stats['mean'] + k * glob) / (stats['size'] + k)
    return serie.map(lisse).fillna(glob), lisse, glob

# LightGBM : traitement natif, aucun encodage à faire
for c in colonnes_cat:
    df[c] = df[c].astype('category')
lgb.LGBMClassifier().fit(df[features], y)      # categorical_feature détecté

# Piège classique : appliquer un encodage appris sur tout le dataset
# -> toujours dans un ColumnTransformer, jamais avant train_test_split
```

**Outils** — `scikit-learn.preprocessing`, `pip install category_encoders`.

**Alternatives open-source**
- *Bibliothèques* : **category_encoders** offre une quinzaine d'encodeurs compatibles scikit-learn (CatBoost, James-Stein, WOE, binaire, base N, hachage) ; **CatBoost** gère la forte cardinalité mieux que tout encodage manuel ; **feature-engine** propose des encodeurs avec gestion explicite des modalités rares et inconnues ; **sklearn.feature_extraction.FeatureHasher** pour le hachage à dimension fixe.
- *Outils graphiques* : **OpenRefine** pour fusionner des modalités mal orthographiées avant tout encodage — étape souvent plus rentable que le choix de l'encodeur ; **KNIME** et **Orange** proposent des nœuds d'encodage visuels ; **ydata-profiling** liste cardinalités et modalités rares en un rapport.

**Astuces**
- `drop='first'` évite la colinéarité parfaite (piège de la variable muette) pour les modèles linéaires. Inutile et légèrement nuisible pour les arbres.
- Une modalité présente en test et absente en entraînement fait planter la prédiction si `handle_unknown` n'est pas réglé. Toujours anticiper.
- L'encodage par cible naïf est la deuxième source de fuite de données la plus fréquente, après le prétraitement hors pipeline.
- Le one-hot d'une variable à 30 000 modalités crée une matrice creuse : garder `sparse_output=True` et un modèle qui l'accepte, sinon la mémoire explose.
- Combiner deux catégorielles en une (`region_x_canal`) capture une interaction que les modèles linéaires ne verraient pas. Simple et souvent rentable.
- Regrouper les modalités à moins de 20 occurrences dans « autre » améliore presque toujours la généralisation.

## Valeurs manquantes
<!-- slug: valeurs-manquantes | difficulte: 2 | prereqs: pandas-manipulation -->

**En une phrase** — Décider quoi faire des trous, en sachant que le fait qu'une valeur manque est souvent lui-même une information.

**Explication** — Trois régimes, aux conséquences très différentes. **MCAR** (manquant complètement au hasard) : l'absence est indépendante de tout, l'imputation est sans biais. **MAR** : l'absence dépend d'autres variables observées, une imputation conditionnelle reste correcte. **MNAR** : l'absence dépend de la valeur elle-même — les hauts revenus ne répondent pas à la question sur le revenu. Là, aucune imputation n'est neutre, et il faut modéliser l'absence explicitement.

En pratique, la stratégie la plus rentable est presque toujours la même : **imputer par la médiane (numérique) ou le mode (catégoriel), et ajouter une colonne indicatrice** signalant que la valeur manquait. L'indicatrice permet au modèle d'apprendre que l'absence est prédictive, ce qui est fréquent : un champ non rempli dans un formulaire de crédit en dit long.

Les méthodes sophistiquées — kNN, imputation multiple par équations chaînées (`IterativeImputer`) — apportent un gain réel quand les features sont fortement corrélées, au prix d'un coût de calcul et d'une complexité de mise en production. Et LightGBM, XGBoost et CatBoost gèrent les manquants **nativement** : ils apprennent, à chaque nœud, de quel côté envoyer les valeurs absentes. C'est souvent le meilleur traitement, et il ne coûte rien.

**Cas d'utilisation**
- Médiane + indicatrice : le défaut raisonnable, à essayer d'abord.
- Manquants natifs du boosting : dès qu'on utilise LightGBM ou XGBoost.
- `IterativeImputer` sur des données médicales ou de capteurs fortement corrélées.
- Suppression de la colonne au-delà de 60 à 70 % de manquants — sauf si l'indicatrice seule est prédictive.
- Suppression de la ligne uniquement si les manquants sont rares (< 5 %) et MCAR.

**Algorithme**
```text
1. Cartographier : df.isna().mean().sort_values() — proportion par colonne.
2. Chercher la structure : les manquants sont-ils corrélés entre colonnes ?
   Un même incident de collecte crée des trous groupés.
3. Tester la prédictivité de l'absence : différence de moyenne de la cible
   entre lignes manquantes et lignes complètes. Si écart -> indicatrice obligatoire.
4. Choisir :
     > 60 % manquants        -> supprimer la colonne, garder l'indicatrice
     numérique               -> médiane + indicatrice
     catégoriel              -> modalité « INCONNU » (pas le mode : c'est une info)
     corrélations fortes     -> IterativeImputer
     modèle = boosting       -> ne rien faire, laisser les NaN
5. L'imputeur s'apprend sur l'entraînement seul -> dans le Pipeline.
```

**Implémentation**
```python
import numpy as np, pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer, MissingIndicator
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline, FeatureUnion

# Cartographie et test de prédictivité de l'absence
manque = df.isna().mean().sort_values(ascending=False)
for c in manque[manque > 0].index:
    m = df[c].isna()
    print(f"{c:25s} {manque[c]:5.1%}  cible: manquant {y[m].mean():.3f} "
          f"vs présent {y[~m].mean():.3f}")

# Défaut recommandé : médiane + indicatrice, en une étape
imp = SimpleImputer(strategy='median', add_indicator=True)

# Catégoriel : une modalité explicite, jamais le mode
imp_cat = SimpleImputer(strategy='constant', fill_value='INCONNU')

# Corrélations fortes : imputation itérative (MICE simplifié)
imp_iter = IterativeImputer(max_iter=10, random_state=0, sample_posterior=False)

# Boosting : ne rien imputer du tout
import lightgbm as lgb
lgb.LGBMClassifier().fit(X_avec_nan, y)     # les NaN sont un signal exploité

# Vérification après imputation : la distribution ne doit pas être déformée
avant, apres = df['revenu'].dropna(), pd.Series(imp.fit_transform(df[['revenu']])[:, 0])
print(avant.describe().round(2))
print(apres.describe().round(2))            # médiane identique, variance réduite : normal
```

**Outils** — `scikit-learn.impute`, `pandas` pour le diagnostic.

**Alternatives open-source**
- *Bibliothèques* : **miceforest** implémente l'imputation multiple par forêts aléatoires, la référence moderne et rapide ; **missingno** produit des visualisations de motifs de manquants (matrice, dendrogramme de corrélation d'absence) ; **feature-engine** offre des imputeurs avec traçabilité et compatibles Pipeline ; **fancyimpute** pour les méthodes matricielles (SoftImpute, complétion de matrice de faible rang).
- *Outils graphiques* : **missingno** est le premier réflexe visuel, trois lignes pour comprendre la structure des trous ; **OpenRefine** pour inspecter et corriger à la main sur des données sales ; **ydata-profiling** signale les colonnes problématiques dans son rapport ; **KNIME** propose des nœuds d'imputation configurables.

**Astuces**
- L'indicatrice de manquant est gratuite et gagne souvent plus que l'amélioration de la méthode d'imputation. La mettre par défaut.
- Ne jamais imputer par la moyenne sur une distribution asymétrique : la médiane est plus robuste et ne déplace pas le centre.
- Attention aux manquants déguisés : `-999`, `0`, `'N/A'`, `'inconnu'`, une date au 01/01/1900. Toujours regarder les valeurs extrêmes et les modalités avant d'imputer.
- Imputer avant le découpage fait fuiter la médiane du test dans l'entraînement. Erreur classique, effet réel mais faible — sauf sur petits jeux, où il devient significatif.
- L'imputation réduit mécaniquement la variance de la colonne et affaiblit ses corrélations. C'est acceptable pour la prédiction, pas pour une analyse statistique inférentielle.
- En production, une colonne subitement 100 % manquante (panne d'un capteur, changement d'API) est un scénario réel. L'imputeur doit pouvoir renvoyer une valeur par défaut sans planter.

## Mise à l'échelle et transformations
<!-- slug: mise-a-l-echelle | difficulte: 1 | prereqs: valeurs-manquantes -->

**En une phrase** — Ramener les features à des échelles comparables et corriger leur asymétrie, ce qui est obligatoire pour certains modèles et inutile pour d'autres.

**Explication** — `StandardScaler` centre et réduit : $(x - \mu)/\sigma$. C'est le défaut, il suppose une distribution à peu près symétrique et reste sensible aux valeurs aberrantes (qui gonflent $\sigma$). `MinMaxScaler` ramène dans $[0, 1]$ : utile quand une borne est requise (images, réseaux de neurones avec sigmoïde), très sensible aux extrêmes. `RobustScaler` utilise médiane et écart interquartile : le bon choix en présence d'aberrations. `QuantileTransformer` force la distribution vers une loi uniforme ou normale, ce qui écrase les aberrations mais détruit la forme.

Qui en a besoin ? Tout ce qui repose sur une **distance** (kNN, k-means, SVM à noyau RBF, ACP) et tout ce qui utilise une **descente de gradient** (régression logistique, réseaux de neurones) ou une **pénalité** (ridge, lasso). Qui n'en a pas besoin ? Les arbres et tous les ensembles d'arbres, invariants à toute transformation monotone. Standardiser avant un LightGBM ne fait aucun mal, mais aucun bien.

Distinct de la mise à l'échelle : la **correction d'asymétrie**. Une cible ou une feature très asymétrique à droite (revenus, prix, durées) bénéficie d'un `log1p`, ou d'une transformation de Yeo-Johnson qui gère aussi les valeurs négatives. Sur un modèle linéaire, l'effet peut être spectaculaire — bien supérieur à un changement d'algorithme.

**Cas d'utilisation**
- Obligatoire avant kNN, SVM, ACP, k-means, réseaux de neurones, régression régularisée.
- `RobustScaler` sur des données financières ou de capteurs comportant des extrêmes.
- `log1p` sur des cibles positives asymétriques : la transformation la plus rentable du ML tabulaire.
- Inutile pour arbres, forêts, boosting — sauf si l'on compare les features entre elles.

**Algorithme**
```text
1. Tracer l'histogramme de chaque feature numérique. Trois questions :
     l'échelle est-elle très différente des autres ?
     la distribution est-elle très asymétrique ?
     y a-t-il des valeurs extrêmes isolées ?
2. Choisir :
     symétrique, peu d'extrêmes      -> StandardScaler
     extrêmes présents               -> RobustScaler
     bornes nécessaires              -> MinMaxScaler
     très asymétrique positif        -> log1p puis StandardScaler
     asymétrie avec valeurs négatives-> PowerTransformer (Yeo-Johnson)
     distribution ingérable          -> QuantileTransformer
3. Apprendre sur l'entraînement, appliquer partout : dans un Pipeline.
4. Vérifier après transformation : moyenne ~0, écart ~1, histogramme lisible.
```

**Implémentation**
```python
import numpy as np
from sklearn.preprocessing import (StandardScaler, RobustScaler, MinMaxScaler,
                                  PowerTransformer, QuantileTransformer)
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline

# Choix par groupe de colonnes selon la forme observée
prep = ColumnTransformer([
    ('classique', StandardScaler(), cols_symetriques),
    ('robuste', RobustScaler(), cols_avec_extremes),
    ('asymetrique', Pipeline([('pt', PowerTransformer(method='yeo-johnson')),
                              ('sc', StandardScaler())]), cols_asymetriques),
])

# Transformer la CIBLE et inverser automatiquement à la prédiction
modele = TransformedTargetRegressor(
    regressor=Ridge(alpha=1.0),
    func=np.log1p, inverse_func=np.expm1,
)
modele.fit(X_tr, y_tr)
pred = modele.predict(X_te)      # déjà ramené à l'échelle d'origine

# --- from scratch : mesurer l'effet de la standardisation sur la convergence ---
def nb_pas_pour_converger(X, y, eta=0.01, tol=1e-4, max_pas=100_000):
    w = np.zeros(X.shape[1])
    for t in range(max_pas):
        g = 2 / len(y) * X.T @ (X @ w - y)
        w -= eta * g
        if np.linalg.norm(g) < tol:
            return t
    return max_pas

X_brut = np.c_[np.random.randn(500), 1000 * np.random.randn(500)]   # échelles 1 et 1000
y = X_brut @ [1.0, 0.002] + 0.1 * np.random.randn(500)
print("brut     :", nb_pas_pour_converger(X_brut, y))
print("standardisé :", nb_pas_pour_converger(StandardScaler().fit_transform(X_brut), y))
```

**Outils** — `scikit-learn.preprocessing`, `sklearn.compose.TransformedTargetRegressor`.

**Alternatives open-source**
- *Bibliothèques* : **feature-engine** propose des transformations avec détection automatique des colonnes concernées et gestion des cas limites (log sur valeur négative) ; **scipy.stats.boxcox** pour la transformation de Box-Cox avec estimation du paramètre ; **sklearn.preprocessing.SplineTransformer** pour capturer une non-linéarité tout en restant dans un modèle linéaire ; **KBinsDiscretizer** pour transformer une variable continue en catégories.
- *Outils graphiques* : **ydata-profiling** affiche l'asymétrie (*skewness*) et l'aplatissement de chaque colonne, ce qui indique directement quoi transformer ; **Orange** propose des blocs de normalisation ; **seaborn** `pairplot` avant/après pour juger visuellement.

**Astuces**
- La standardisation apprise sur tout le dataset est la fuite de données la plus fréquente du ML. Toujours dans un `Pipeline`.
- Vérifier l'asymétrie avec `df.skew()`. Au-delà de 1 en valeur absolue, essayer `log1p` et comparer les scores : c'est souvent le gain le plus facile du projet.
- `log1p` et `expm1` plutôt que `log` et `exp` : ils gèrent correctement les zéros et restent précis près de zéro.
- Attention au biais de rétro-transformation : $\exp(\text{moyenne des logs})$ n'est pas la moyenne. Sur une prédiction de montant, cela sous-estime systématiquement.
- Ne jamais standardiser les variables one-hot : cela détruit leur lisibilité pour un gain nul.
- `MinMaxScaler` sur l'entraînement ne garantit pas $[0, 1]$ en test : une valeur plus extrême sortira des bornes. Prévoir `clip=True` si le modèle en dépend.

## Features temporelles
<!-- slug: features-temporelles | difficulte: 3 | prereqs: mise-a-l-echelle, fuite-de-donnees -->

**En une phrase** — Extraire d'une date et d'un historique des variables prédictives, sans jamais utiliser d'information postérieure à l'instant de prédiction.

**Explication** — Une colonne date brute est inutilisable telle quelle. On en extrait des **composantes calendaires** : heure, jour de la semaine, mois, semaine ISO, jour férié, indicateur de week-end. Ces variables sont **cycliques** : le mois 12 est adjacent au mois 1, et un encodage entier ne le dit pas. La solution standard est l'encodage sinus/cosinus : $\sin(2\pi t/T)$ et $\cos(2\pi t/T)$, deux colonnes qui préservent la continuité circulaire. Pour un arbre, l'entier suffit ; pour un modèle linéaire ou un réseau, l'encodage cyclique change tout.

Viennent ensuite les features d'**historique**, généralement les plus prédictives : décalages (`lag`) — la valeur d'hier, de la semaine dernière, de l'an dernier à la même date — et **agrégats glissants** — moyenne, écart type, min, max sur les 7 ou 30 derniers jours. Ce sont elles qui portent le signal, et ce sont exactement elles qui produisent des fuites.

La règle est absolue : toute agrégation doit **exclure la ligne courante et tout ce qui la suit**. En pandas, cela signifie `shift(1)` avant `rolling()`, et un tri chronologique préalable. Une moyenne glissante centrée, ou un `expanding().mean()` sans décalage, contient la valeur à prédire. C'est la fuite la plus fréquente et la plus coûteuse de tout le ML appliqué.

**Cas d'utilisation**
- Prévision de demande, de trafic, de consommation.
- Détection de fraude : fréquence des transactions sur la dernière heure, écart au comportement habituel.
- Churn : évolution de l'activité sur les 3 derniers mois.
- Maintenance prédictive : tendance et variance des mesures de capteurs.
- Tout dataset contenant une colonne date, même sans objectif temporel explicite.

**Algorithme**
```text
1. Trier par entité puis par date. Vérifier qu'il n'y a pas de doublons de dates.
2. Composantes calendaires : heure, jour_semaine, mois, semaine, ferie, week-end.
3. Encoder les cycliques en sin/cos si le modèle est linéaire ou un réseau.
4. Décalages : shift(1), shift(7), shift(365) selon la saisonnalité observée.
5. Fenêtres glissantes, TOUJOURS après shift(1) :
     .shift(1).rolling(k).agg(['mean','std','min','max'])
   par entité : groupby('entite')[col].transform(...)
6. Différences et ratios : valeur / moyenne_30j, écart à la même période l'an passé.
7. Temps écoulé depuis le dernier événement (recence), compte d'événements
   sur la fenêtre (fréquence).
8. VALIDER par TimeSeriesSplit avec un gap. Comparer au score obtenu en
   validation aléatoire : un écart important révèle une fuite résiduelle.
```

**Implémentation**
```python
import numpy as np, pandas as pd

df = df.sort_values(['client_id', 'date']).reset_index(drop=True)

# 1. Composantes calendaires
d = df['date'].dt
df['heure'], df['jour_sem'], df['mois'] = d.hour, d.dayofweek, d.month
df['week_end'] = (df['jour_sem'] >= 5).astype(int)
df['fin_de_mois'] = d.is_month_end.astype(int)

# 2. Encodage cyclique : indispensable pour modèles linéaires et réseaux
for col, periode in [('heure', 24), ('jour_sem', 7), ('mois', 12)]:
    df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / periode)
    df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / periode)

# 3. Décalages et fenêtres — le shift(1) N'EST PAS OPTIONNEL
g = df.groupby('client_id')['montant']
for k in [1, 7, 30]:
    df[f'montant_lag{k}'] = g.shift(k)
for f in [7, 30]:
    df[f'moy_{f}j'] = g.transform(lambda s: s.shift(1).rolling(f, min_periods=1).mean())
    df[f'std_{f}j'] = g.transform(lambda s: s.shift(1).rolling(f, min_periods=2).std())

# 4. Features relatives : souvent plus prédictives que les valeurs absolues
df['ecart_habitude'] = df['montant_lag1'] / df['moy_30j'].replace(0, np.nan)

# 5. Récence : temps depuis la transaction précédente
df['jours_depuis'] = df.groupby('client_id')['date'].diff().dt.days

# 6. Validation temporelle stricte, avec gap pour éviter la fuite de proximité
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
tscv = TimeSeriesSplit(n_splits=5, gap=7, test_size=len(df) // 10)
print("temporel :", cross_val_score(modele, X, y, cv=tscv, scoring='neg_mean_absolute_error').mean())

# 7. Test anti-fuite : décaler la cible d'un jour ne doit pas améliorer le score
```

**Outils** — `pandas` (`shift`, `rolling`, `expanding`, `Grouper`), `numpy`.

**Alternatives open-source**
- *Bibliothèques* : **tsfresh** extrait automatiquement des centaines de features temporelles et les filtre par test statistique ; **sktime** et **darts** offrent des cadres complets de prévision avec validation temporelle correcte par construction ; **mlforecast** (Nixtla) construit lags et fenêtres pour un modèle de boosting, très rapide et sans fuite par conception ; **feature-engine** dispose de `LagFeatures` et `WindowFeatures` compatibles Pipeline ; **holidays** pour les jours fériés de n'importe quel pays ; **Prophet** pour une baseline de prévision immédiate.
- *Outils graphiques* : **Grafana** pour explorer visuellement des séries et repérer saisonnalités et ruptures ; **Kats** (Meta) pour la détection de changement ; **statsmodels** `plot_acf` / `plot_pacf` afin d'identifier les décalages pertinents avant de les créer ; **Orange** avec son extension Timeseries.

**Astuces**
- `shift(1)` avant tout `rolling` : à relire trois fois. Sans lui, le modèle voit la valeur qu'il doit prédire, l'AUC monte à 0,99 et le projet échoue en production.
- Toujours trier par entité **et** par date avant tout `groupby().shift()`. Sur un DataFrame non trié, les décalages sont silencieusement faux.
- L'autocorrélation (`plot_acf`) indique quels décalages créer. Les créer au hasard gaspille du temps de calcul.
- Le `gap` de `TimeSeriesSplit` protège contre la fuite de proximité : si les features utilisent une fenêtre de 7 jours, il faut au moins 7 jours d'écart entre entraînement et validation.
- Le jour de la semaine et l'indicateur de jour férié sont souvent les deux features les plus prédictives d'un problème de demande. Les créer avant toute chose.
- Vérifier que chaque feature est calculable en production **à l'instant de la prédiction**. Une moyenne sur 30 jours exige 30 jours d'historique disponible : que fait le modèle pour un nouveau client ?

## Sélection de features
<!-- slug: selection-features | difficulte: 2 | prereqs: forets-aleatoires, validation-croisee -->

**En une phrase** — Retirer les variables inutiles pour gagner en robustesse, en vitesse et en lisibilité, sans perdre de performance.

**Explication** — Trois familles de méthodes. Les méthodes **filtres** évaluent chaque feature indépendamment du modèle : variance nulle, corrélation avec la cible, information mutuelle, test du khi-deux. Rapides, elles ignorent les interactions et peuvent écarter une variable inutile seule mais décisive combinée à une autre. Les méthodes **enveloppes** entraînent un modèle sur des sous-ensembles : élimination récursive (RFE), sélection séquentielle. Bien plus efficaces, mais coûteuses et sujettes au surapprentissage de sélection si la validation n'est pas imbriquée. Les méthodes **intégrées** obtiennent la sélection comme produit de l'entraînement : lasso qui met des coefficients à zéro, importance des arbres.

La méthode la plus fiable en pratique est l'**importance par permutation** sur un jeu de validation : on mélange une colonne et on mesure la dégradation. Une feature dont la permutation ne dégrade rien n'apporte rien. Sa variante robuste, **Boruta**, compare l'importance de chaque feature à celle de copies aléatoires (features « ombres »), ce qui fournit un critère statistique de rejet au lieu d'un seuil arbitraire.

Attention toutefois : sur un boosting bien régularisé, la sélection de features apporte souvent **peu ou rien** en performance. Ses vrais bénéfices sont ailleurs — coût d'acquisition des données, latence, stabilité dans le temps, capacité à expliquer le modèle. Ce sont des raisons suffisantes, mais il faut les nommer.

**Cas d'utilisation**
- Des centaines ou milliers de features, dont beaucoup de bruit (capteurs, texte vectorisé, génomique).
- Réduction du coût d'acquisition : chaque variable collectée coûte de l'argent ou du temps.
- Contrainte de latence ou de taille de modèle en production.
- Besoin d'expliquer le modèle à un humain : dix variables se discutent, trois cents non.
- Inutile avec 20 features propres et un modèle régularisé.

**Algorithme**
```text
1. Éliminations gratuites et sans risque :
     variance nulle ou quasi nulle
     doublons de colonnes
     corrélation |r| > 0.95 entre deux features -> garder une seule
     identifiants et colonnes constantes
2. Importance par permutation sur un jeu de VALIDATION (pas d'entraînement),
   avec n_repeats >= 10 pour avoir un écart type.
3. Retirer les features dont l'importance moyenne est <= 0 à l'écart type près.
4. Réentraîner et comparer par validation croisée : le score doit être
   statistiquement inchangé. Sinon, remettre.
5. Répéter jusqu'à ce que le retrait dégrade réellement.
6. Rapporter le score par validation croisée IMBRIQUÉE : la sélection
   fait partie du modèle, donc du processus à valider.
```

**Implémentation**
```python
import numpy as np, pandas as pd
from sklearn.feature_selection import (VarianceThreshold, mutual_info_classif,
                                       RFECV, SelectFromModel)
from sklearn.inspection import permutation_importance

# 1. Éliminations gratuites
X = X.loc[:, X.std() > 0]
corr = X.corr().abs()
haut = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
a_jeter = [c for c in haut.columns if (haut[c] > 0.95).any()]
X = X.drop(columns=a_jeter)
print("retirées par corrélation :", a_jeter)

# 2. Importance par permutation : la mesure de référence
modele.fit(X_tr, y_tr)
imp = permutation_importance(modele, X_val, y_val, n_repeats=20,
                            scoring='roc_auc', random_state=0, n_jobs=-1)
res = pd.DataFrame({'moy': imp.importances_mean, 'std': imp.importances_std},
                   index=X.columns).sort_values('moy', ascending=False)
inutiles = res[res['moy'] <= res['std']].index.tolist()
print(f"{len(inutiles)} features candidates au retrait")

# 3. Élimination récursive avec validation croisée intégrée
rfe = RFECV(modele, step=0.1, cv=5, scoring='roc_auc', min_features_to_select=5, n_jobs=-1)
rfe.fit(X_tr, y_tr)
print("retenues :", rfe.n_features_, "/", X_tr.shape[1])

# 4. Boruta : critère statistique contre des features aléatoires
# pip install boruta
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
bor = BorutaPy(RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=6),
               n_estimators='auto', random_state=0).fit(X_tr.values, y_tr.values)
print("confirmées :", X_tr.columns[bor.support_].tolist())
```

**Outils** — `scikit-learn.feature_selection`, `sklearn.inspection.permutation_importance`.

**Alternatives open-source**
- *Bibliothèques* : **Boruta** (`boruta_py`) pour un critère de rejet fondé sur des features aléatoires, la méthode la plus défendable ; **SHAP** fournit des importances cohérentes et locales, utilisables pour sélectionner ; **featurewiz** et **mrmr-selection** implémentent la sélection par pertinence maximale et redondance minimale, très efficace en grande dimension ; **scikit-learn** `SequentialFeatureSelector` pour la sélection avant/arrière ; **feature-engine** propose des sélecteurs par performance de modèle univarié.
- *Outils graphiques* : **SHAP** `summary_plot` et `beeswarm` sont les meilleures visualisations d'importance existantes ; **Yellowbrick** `FeatureImportances` et `RFECV` en une ligne ; **Orange** avec son bloc Rank ; **H2O Flow** affiche les importances de variables pour tous ses modèles.

**Astuces**
- Ne jamais sélectionner sur l'ensemble complet avant le découpage : la sélection est un apprentissage, donc une source de fuite. Elle doit être dans le `Pipeline`.
- L'importance par impureté des arbres (`feature_importances_`) est biaisée vers les variables continues et à haute cardinalité. Elle ne doit pas servir à des décisions.
- Sur des features corrélées, l'importance par permutation les sous-évalue toutes : permuter l'une laisse l'information disponible dans l'autre. Grouper avant de permuter (permutation par groupe).
- Retirer une feature et constater un score identique n'est pas une preuve d'inutilité tant qu'on n'a pas regardé l'écart type de la validation croisée.
- Le gain principal n'est souvent pas la performance mais la **stabilité** : moins de features signifie moins de sources de dérive et de pannes en production.
- Documenter les features retirées et pourquoi. Sans cela, quelqu'un les remettra dans six mois.

## Pipelines scikit-learn
<!-- slug: pipelines | difficulte: 2 | prereqs: encodage-categoriel, mise-a-l-echelle, fuite-de-donnees -->

**En une phrase** — Enchaîner prétraitement et modèle dans un objet unique qui s'entraîne, se valide et se sérialise d'un bloc.

**Explication** — Un `Pipeline` est une liste d'étapes dont toutes sauf la dernière sont des transformateurs (`fit` + `transform`), la dernière étant un estimateur (`fit` + `predict`). Appeler `pipeline.fit(X, y)` propage : chaque étape apprend sur la sortie de la précédente. Appeler `predict` applique les `transform` appris, puis prédit.

L'intérêt n'est pas l'élégance mais la **correction**. Quand un pipeline entre dans une validation croisée, tout son prétraitement est réappris à l'intérieur de chaque bloc, sur les seules données d'entraînement de ce bloc. C'est mécaniquement impossible à faire fuiter. Sans pipeline, il faut réécrire cette discipline à la main dans chaque boucle, et personne ne le fait correctement longtemps.

Le `ColumnTransformer` complète le dispositif en appliquant des traitements différents à des groupes de colonnes — médiane et standardisation sur le numérique, imputation constante et one-hot sur le catégoriel — puis en concaténant les résultats. L'ensemble forme un objet unique, sérialisable avec `joblib`, qui prend un DataFrame brut en entrée et rend une prédiction. C'est exactement ce qu'il faut déployer : le modèle *et* son prétraitement, indissociables.

**Cas d'utilisation**
- Systématiquement, dès qu'il y a une seule étape de prétraitement.
- Recherche d'hyperparamètres portant à la fois sur le prétraitement et le modèle (`GridSearchCV` sur `imputer__strategy` et `clf__C` en même temps).
- Mise en production : un seul objet `joblib` contient tout.
- Données mixtes numériques/catégorielles/texte, traitées différemment.

**Algorithme**
```text
1. Lister les groupes de colonnes et leur traitement.
2. Construire un ColumnTransformer : (nom, transformateur, colonnes).
   remainder='drop' par défaut — le mettre explicitement pour éviter les surprises.
3. Pipeline([('prep', ct), ('clf', modele)]).
4. Régler les hyperparamètres avec la syntaxe étape__paramètre
   (double tiret bas), y compris sur les étapes de prétraitement.
5. cross_validate / GridSearchCV sur le pipeline entier : aucune fuite possible.
6. joblib.dump(pipeline) pour la production. Épingler les versions des
   bibliothèques : un pickle n'est pas portable entre versions majeures.
```

**Implémentation**
```python
import numpy as np, joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression

num = ['age', 'revenu', 'anciennete']
cat = ['region', 'segment', 'canal']

prep = ColumnTransformer([
    ('num', Pipeline([
        ('imp', SimpleImputer(strategy='median', add_indicator=True)),
        ('sc', StandardScaler()),
    ]), num),
    ('cat', Pipeline([
        ('imp', SimpleImputer(strategy='constant', fill_value='INCONNU')),
        ('ohe', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=20)),
    ]), cat),
], remainder='drop', verbose_feature_names_out=False)

pipe = Pipeline([('prep', prep), ('clf', LogisticRegression(max_iter=1000))])

# Régler prétraitement ET modèle ensemble, sans aucune fuite
grille = {
    'prep__num__imp__strategy': ['median', 'mean'],
    'prep__cat__ohe__min_frequency': [5, 20, 50],
    'clf__C': np.logspace(-3, 2, 6),
}
gs = GridSearchCV(pipe, grille, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                  scoring='roc_auc', n_jobs=-1).fit(X_tr, y_tr)
print(gs.best_params_, round(gs.best_score_, 4))

# Noms des features en sortie : indispensable pour interpréter les coefficients
noms = gs.best_estimator_['prep'].get_feature_names_out()

# Transformateur sur mesure, intégrable comme n'importe quelle étape
from sklearn.base import BaseEstimator, TransformerMixin

class RatiosMetier(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self                                    # rien à apprendre

    def transform(self, X):
        X = X.copy()
        X['charge_revenu'] = X['charges'] / X['revenu'].replace(0, np.nan)
        return X

# Production : un seul fichier contient prétraitement + modèle
joblib.dump(gs.best_estimator_, 'modele.joblib')
charge = joblib.load('modele.joblib')
charge.predict(df_brut_nouveau)          # DataFrame brut en entrée, prédiction en sortie
```

**Outils** — `scikit-learn` (`Pipeline`, `ColumnTransformer`, `FeatureUnion`), `joblib`.

**Alternatives open-source**
- *Bibliothèques* : **feature-engine** fournit des transformateurs qui acceptent et rendent des DataFrames avec noms de colonnes conservés, ce qui rend le débogage bien plus simple ; **skrub** (ex-dirty_cat) automatise le prétraitement de tables sales et les jointures floues ; **imbalanced-learn** `Pipeline` pour inclure du rééchantillonnage ; **sklearn-pandas** pour un mapping colonne par colonne ; **skops** remplace `joblib` par une sérialisation plus sûre, sans exécution de code arbitraire.
- *Outils graphiques* : `set_config(display='diagram')` de scikit-learn affiche le pipeline sous forme de diagramme cliquable dans un notebook — le meilleur moyen de vérifier sa structure ; **KNIME** et **Orange** construisent des pipelines par blocs visuels ; **MLflow** enregistre le pipeline complet comme artefact versionné.

**Astuces**
- `set_config(display='diagram')` puis afficher le pipeline : les erreurs de structure deviennent visibles immédiatement.
- `remainder='passthrough'` laisse passer les colonnes non listées **sans transformation**, y compris des identifiants ou du texte brut. Préférer `'drop'` et lister explicitement.
- `verbose_feature_names_out=False` évite les préfixes illisibles du type `num__sc__age`.
- Un pickle scikit-learn n'est pas portable entre versions majeures. Épingler la version dans le `requirements.txt` et enregistrer ce numéro à côté du modèle.
- Pour un transformateur sur mesure, hériter de `BaseEstimator` et `TransformerMixin` donne `fit_transform`, `get_params` et la compatibilité avec `GridSearchCV` gratuitement.
- Ne pas mettre la transformation de la **cible** dans le pipeline : elle relève de `TransformedTargetRegressor`, sinon la validation croisée se trompe d'échelle.
- Un pipeline entraîné est le livrable réel du projet, pas le notebook. Le sérialiser dès la première version, avec les données d'exemple qui permettent de le tester.

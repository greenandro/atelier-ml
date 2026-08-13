---
module: outillage
titre: Outillage données
ordre: 0
resume: Les gestes de base. Sans eux, chaque exercice de ML coûte trois fois plus de temps.
---

## Vectorisation NumPy
<!-- slug: numpy-vectorisation | difficulte: 1 | prereqs:  -->

**En une phrase** — Remplacer les boucles Python par des opérations sur des tableaux entiers, exécutées en C, ce qui va 10 à 100 fois plus vite.

**Explication** — Un tableau NumPy (`ndarray`) est un bloc de mémoire contigu, homogène en type, avec une forme (`shape`). Une liste Python contient des pointeurs vers des objets dispersés ; un `ndarray` contient les octets eux-mêmes. C'est cette différence qui permet à `a + b` de descendre dans une boucle compilée au lieu de repasser par l'interpréteur à chaque élément.

Le mécanisme qui fait tout le confort s'appelle le *broadcasting* : quand deux tableaux n'ont pas la même forme, NumPy étire virtuellement les dimensions de taille 1 pour les faire correspondre, sans copier la mémoire. Les formes sont alignées **par la droite** ; deux dimensions sont compatibles si elles sont égales, ou si l'une vaut 1. Ainsi `(100, 3) + (3,)` fonctionne — le vecteur est ajouté à chaque ligne — alors que `(100, 3) + (100,)` échoue. Il faut alors écrire `(100, 1)` avec `x[:, None]`.

En ML, tout se ramène à des produits matriciels : $\hat{y} = Xw + b$ où $X$ est de forme $(n, d)$, $w$ de forme $(d,)$ et le résultat de forme $(n,)$. Savoir lire ces formes mentalement est la compétence de débogage la plus rentable du domaine.

**Cas d'utilisation**
- Toute opération sur des données numériques : normalisation, distances, gradients.
- Calcul d'une distance euclidienne entre tous les points de deux ensembles, sans double boucle.
- Masquage conditionnel : `X[y == 1]` pour ne garder qu'une classe.
- Ce n'est **pas** l'outil pour des données hétérogènes avec des noms de colonnes : c'est le rôle de pandas.

**Algorithme**
```text
Réflexe à appliquer devant chaque boucle for :
1. Identifier ce qui varie dans la boucle -> c'est un axe du tableau.
2. Écrire la forme d'entrée et la forme de sortie souhaitée.
3. Choisir l'opération : élément par élément, produit matriciel (@),
   ou réduction (sum, mean, max) sur un axe.
4. Insérer les axes manquants avec [:, None] ou [None, :].
5. Vérifier avec .shape avant de vérifier les valeurs.
```

**Implémentation**
```python
import numpy as np

# Distances euclidiennes entre 1000 points et 5 centroïdes, sans boucle
X = np.random.randn(1000, 3)          # (n, d)
C = np.random.randn(5, 3)             # (k, d)

# X[:, None, :] -> (n, 1, d) ; C[None, :, :] -> (1, k, d)
# différence broadcastée -> (n, k, d), puis réduction sur d
D = np.sqrt(((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2))   # (n, k)
plus_proche = D.argmin(axis=1)        # (n,) index du centroïde le plus proche

# Standardisation colonne par colonne
X_std = (X - X.mean(axis=0)) / X.std(axis=0)

# Astuce mémoire : la version ci-dessus alloue n*k*d flottants.
# Version équivalente en O(n*k) via ||x-c||² = ||x||² - 2x·c + ||c||²
D2 = (X**2).sum(1)[:, None] - 2 * X @ C.T + (C**2).sum(1)[None, :]
```

**Outils** — `pip install numpy` (une seule dépendance, aucune configuration).

**Alternatives open-source**
- *Bibliothèques* : **PyTorch** (`torch.Tensor`) offre la même API avec l'autograd et le GPU en plus — de plus en plus le défaut, même sans réseau de neurones ; **JAX** ajoute la compilation JIT et la différentiation automatique fonctionnelle, excellent pour la recherche ; **CuPy** est un remplacement de NumPy sur GPU, changement d'import et rien d'autre.
- *Outils graphiques* : aucun n'a de sens ici — c'est un langage de calcul, pas un objet à manipuler à la souris.

**Astuces**
- `axis=0` agit *sur* les lignes donc produit un résultat par colonne. La confusion inverse est la source d'erreur numéro un : retenir « axis est l'axe qui disparaît ».
- Le *slicing* renvoie une **vue**, pas une copie : modifier `X[0:10]` modifie `X`. Utiliser `.copy()` dès qu'un doute existe. En revanche l'indexation par tableau (`X[[0, 5, 9]]`) copie.
- `float32` suffit presque toujours en ML et divise la mémoire par deux. NumPy crée du `float64` par défaut, PyTorch du `float32` — d'où des erreurs de type à la frontière.
- Ne jamais comparer des flottants avec `==` : utiliser `np.allclose(a, b)`.
- `np.random.seed(0)` est déprécié au profit de `rng = np.random.default_rng(0)`. La nouvelle API est meilleure et reproductible entre versions.

## Manipulation avec pandas
<!-- slug: pandas-manipulation | difficulte: 1 | prereqs: numpy-vectorisation -->

**En une phrase** — Le tableur programmable de Python : des colonnes nommées, typées, avec jointures, regroupements et gestion des valeurs manquantes.

**Explication** — Un `DataFrame` est un dictionnaire de `Series`, chaque `Series` étant un tableau NumPy plus un **index**. L'index est la particularité de pandas, et sa principale source de confusion : après un filtre, les positions et les étiquettes ne coïncident plus. D'où la règle absolue : `.loc[]` pour indexer par étiquette, `.iloc[]` pour indexer par position, jamais `df[...]` pour des lignes.

Trois opérations couvrent 90 % du travail réel. Le **filtrage** booléen (`df[df.age > 30]`), le **regroupement** (`groupby().agg()`) qui suit le schéma *split-apply-combine* — découper selon une clé, calculer sur chaque groupe, recoller les résultats — et la **jointure** (`merge`) qui reproduit les `JOIN` de SQL. Si tu maîtrises ces trois-là plus le pivot (`pivot_table`, `melt`), le reste s'improvise.

Le point à comprendre tôt : pandas est **lent et gourmand** dès quelques millions de lignes, parce qu'il matérialise tout en mémoire et copie souvent. Ce n'est pas un défaut à corriger, c'est une limite d'usage — au-delà, on passe à Polars ou à un moteur distribué.

**Cas d'utilisation**
- Chargement et nettoyage de CSV, Excel, JSON, SQL : le point d'entrée de tout projet.
- Analyse exploratoire : distributions, valeurs manquantes, corrélations, doublons.
- Agrégation temporelle : moyennes glissantes, resampling par mois.
- Au-delà de ~5 Go en mémoire, ou pour du calcul répété en production, ce n'est plus le bon outil.

**Algorithme**
```text
Rituel d'ouverture d'un dataset inconnu (10 minutes, jamais sauté) :
1. df.shape                 -> combien de lignes, combien de colonnes
2. df.head(20)              -> à quoi ça ressemble vraiment
3. df.info()                -> types réels et taille mémoire
4. df.isna().mean()         -> proportion de manquants par colonne
5. df.describe(include='all')-> ordres de grandeur et valeurs aberrantes
6. df.duplicated().sum()    -> doublons
7. df[cible].value_counts(normalize=True) -> équilibre des classes
8. df.nunique()             -> repérer les identifiants et les constantes
```

**Implémentation**
```python
import pandas as pd

df = pd.read_csv('ventes.csv', parse_dates=['date'])

# Filtrage + sélection : toujours .loc pour lignes ET colonnes
recents = df.loc[df['date'] >= '2024-01-01', ['produit', 'montant', 'region']]

# Split-apply-combine
resume = (df
    .groupby(['region', pd.Grouper(key='date', freq='ME')])
    .agg(total=('montant', 'sum'),
         panier_moyen=('montant', 'mean'),
         n=('montant', 'size'))
    .reset_index())

# Jointure avec contrôle : indicator révèle les clés orphelines
clients = pd.read_csv('clients.csv')
joint = df.merge(clients, on='client_id', how='left', indicator=True)
print(joint['_merge'].value_counts())   # left_only > 0 = problème de clés

# Chaînage lisible avec assign, sans variable intermédiaire
propre = (df
    .dropna(subset=['montant'])
    .assign(mois=lambda d: d['date'].dt.to_period('M'),
            marge=lambda d: d['montant'] - d['cout'])
    .query('marge > 0'))
```

**Outils** — `pip install pandas pyarrow` — `pyarrow` accélère la lecture et permet les types nullables et le format Parquet.

**Alternatives open-source**
- *Bibliothèques* : **Polars** (Rust, API en expressions, 5 à 30× plus rapide, exécution différée) est aujourd'hui le meilleur choix pour un nouveau projet ; **DuckDB** permet d'écrire du SQL directement sur des DataFrames et des fichiers Parquet, imbattable pour les agrégations ; **Dask** et **PySpark** pour ce qui ne tient pas en mémoire.
- *Outils graphiques* : **JupyterLab** avec le module de tableau interactif ; **ydata-profiling** génère un rapport HTML complet en une ligne (`ProfileReport(df)`) ; **Visidata** explore un CSV de plusieurs Go dans le terminal ; **OpenRefine** pour le nettoyage manuel de données sales et le rapprochement de valeurs mal orthographiées.

**Astuces**
- Le `SettingWithCopyWarning` n'est pas un détail : il signale que ta modification part peut-être dans une copie temporaire. La cause est toujours un double indexage (`df[df.a > 1]['b'] = 0`). Solution : `df.loc[df.a > 1, 'b'] = 0`.
- `parse_dates` au chargement évite des heures de conversions ratées plus tard.
- `df.memory_usage(deep=True)` révèle que les colonnes texte coûtent énormément. Convertir les colonnes à faible cardinalité en `category` divise souvent la mémoire par dix.
- `inplace=True` n'économise pas de mémoire (c'est un mythe) et casse le chaînage. Ne pas l'utiliser.
- `pd.read_csv(..., nrows=1000)` pour prototyper le nettoyage sur un échantillon avant de lancer sur les 40 millions de lignes.
- Enregistrer les données nettoyées en Parquet, pas en CSV : les types sont conservés, le fichier est 5 fois plus petit et la relecture 20 fois plus rapide.

## Polars et Parquet
<!-- slug: polars-parquet | difficulte: 2 | prereqs: pandas-manipulation -->

**En une phrase** — Un moteur de DataFrame en Rust qui exécute un plan optimisé au lieu d'opérations une par une, associé à un format de fichier colonne compressé.

**Explication** — Polars propose deux modes. En mode *eager*, il ressemble à pandas mais parallélise sur tous les cœurs. En mode *lazy* (`scan_parquet`, `.lazy()`), il ne calcule rien avant l'appel à `.collect()` : il construit un plan de requête, puis l'optimise — élagage des colonnes inutiles, remontée des filtres au plus près de la lecture, fusion des projections. Sur une requête qui ne touche que 3 colonnes sur 200, il ne lit littéralement que ces 3 colonnes sur le disque.

Parquet est le format qui rend cela possible. Contrairement au CSV qui stocke ligne par ligne, Parquet stocke **colonne par colonne**, avec un type déclaré, une compression par colonne et des statistiques min/max par bloc de lignes. Un filtre `WHERE annee = 2024` peut donc écarter des blocs entiers sans les décompresser — c'est le *predicate pushdown*. Partitionner en dossiers (`annee=2024/mois=03/`) pousse la logique plus loin : les partitions non concernées ne sont jamais ouvertes.

L'autre différence culturelle : Polars n'a **pas d'index**. Toute opération s'exprime avec des *expressions* (`pl.col('x')`) composables et réutilisables. Le code est plus verbeux au début, beaucoup plus prévisible ensuite.

**Cas d'utilisation**
- Datasets de 1 à 100 Go sur une seule machine : la zone où pandas souffre et où Spark est excessif.
- Pipelines de préparation qui tournent tous les jours et dont le temps d'exécution compte.
- Stockage intermédiaire de tout dataset nettoyé, quelle que soit sa taille.
- Pour 10 000 lignes bricolées une fois, pandas reste plus rapide à écrire.

**Algorithme**
```text
1. Convertir la source une fois en Parquet, partitionné sur la colonne
   la plus filtrée (souvent une date).
2. scan_parquet() -> LazyFrame (aucune lecture disque encore).
3. Enchaîner filtres, select, with_columns, group_by.
4. .explain() -> vérifier que les filtres sont bien descendus dans le scan.
5. .collect(streaming=True) si le résultat dépasse la RAM.
```

**Implémentation**
```python
import polars as pl

# Conversion unique CSV -> Parquet partitionné
(pl.scan_csv('ventes.csv', try_parse_dates=True)
   .with_columns(annee=pl.col('date').dt.year())
   .sink_parquet('data/ventes/', partition_by='annee'))

# Requête paresseuse : seules les colonnes et partitions utiles sont lues
q = (pl.scan_parquet('data/ventes/**/*.parquet')
       .filter((pl.col('annee') == 2024) & (pl.col('montant') > 0))
       .group_by('region')
       .agg(total=pl.col('montant').sum(),
            panier=pl.col('montant').mean(),
            n=pl.len())
       .sort('total', descending=True))

print(q.explain())        # lire le plan avant de lancer
resultat = q.collect()    # exécution ici seulement

# Expressions réutilisables, impossible en pandas
marge_nette = (pl.col('montant') - pl.col('cout')) / pl.col('montant')
df = q.collect().with_columns(marge=marge_nette.round(3))

# Passerelle sans copie vers l'écosystème existant
df.to_pandas(use_pyarrow_extension_array=True)
```

**Outils** — `pip install polars pyarrow`. Pour interroger du Parquet en SQL : `pip install duckdb`.

**Alternatives open-source**
- *Bibliothèques* : **DuckDB** couvre presque le même terrain en SQL, avec de meilleures jointures complexes et une intégration directe dans un notebook ; **Apache Arrow** est la couche mémoire commune sous Polars, DuckDB et pandas récent ; **PyArrow Dataset** pour du Parquet partitionné sans moteur de requête ; **fastparquet** en lecteur léger alternatif.
- *Outils graphiques* : **DBeaver** connecté à DuckDB pour explorer des Parquet à la souris ; **Tad** ouvre un Parquet comme un tableur ; **parquet-tools** en ligne de commande pour inspecter le schéma et les statistiques de blocs.

**Astuces**
- Toujours `scan_*` et jamais `read_*` en mode lazy : `read_parquet` charge tout et annule l'intérêt.
- Un fichier Parquet par partition doit peser entre 100 Mo et 1 Go. Des milliers de petits fichiers tuent la performance — c'est le problème des *small files*.
- Ne pas partitionner sur une colonne à forte cardinalité (identifiant client) : cela crée un dossier par valeur.
- Polars remonte des erreurs de type strictes là où pandas convertit silencieusement. C'est pénible une fois, salvateur ensuite.
- `pl.len()` remplace `pl.count()` (déprécié) ; l'API bouge encore vite, épingler la version dans le projet.
- Parquet ne se lit pas avec `head` ni `grep` : garder un petit script d'inspection à portée de main.

## Visualisation
<!-- slug: visualisation-matplotlib | difficulte: 1 | prereqs: pandas-manipulation -->

**En une phrase** — Traduire des nombres en formes pour repérer ce qu'aucun tableau de statistiques ne montre : formes de distribution, valeurs aberrantes, relations non linéaires.

**Explication** — Matplotlib repose sur deux niveaux d'API. L'interface `pyplot` (`plt.plot`) manipule une figure globale implicite, pratique pour un jet rapide. L'interface **orientée objet** (`fig, ax = plt.subplots()`) donne des objets `Figure` et `Axes` explicites : c'est la seule qui reste maîtrisable dès qu'il y a plusieurs sous-graphiques. Prendre l'habitude de la seconde immédiatement évite de tout réapprendre plus tard.

Le choix de la forme n'est pas cosmétique, il détermine ce que tu peux voir. Une distribution demande un histogramme ou un KDE. Une relation entre deux variables continues demande un nuage de points — et dès que les points se superposent, il faut passer à un `hexbin` ou baisser l'alpha. Une comparaison de groupes demande une boîte à moustaches ou un violon, jamais des barres de moyennes qui cachent la dispersion. Une évolution temporelle demande une ligne. Le quartet d'Anscombe — quatre jeux de données avec les mêmes moyennes, variances et corrélations, mais des formes radicalement différentes — reste la meilleure démonstration que la statistique descriptive seule est aveugle.

En ML spécifiquement, quatre graphiques reviennent sans cesse : la courbe d'apprentissage (perte vs époque, entraînement et validation), la matrice de confusion, la courbe ROC ou précision-rappel, et le graphique des résidus en régression.

**Cas d'utilisation**
- Analyse exploratoire : voir la forme avant de choisir un modèle.
- Diagnostic d'entraînement : la courbe de validation qui remonte signale le surapprentissage.
- Communication d'un résultat : une figure exportée en PNG ou SVG.
- Pour de l'exploration interactive avec zoom et survol, matplotlib n'est pas adapté — voir les alternatives.

**Algorithme**
```text
1. Quelle question la figure doit-elle répondre ? (une seule)
2. Combien de variables, de quels types ?
     1 continue          -> histogramme, KDE
     1 catégorielle      -> barres horizontales triées
     2 continues         -> nuage (hexbin si > 5000 points)
     1 cat + 1 continue  -> boîte, violon, ou strip
     temps + continue    -> ligne
     matrice             -> heatmap
3. fig, ax = plt.subplots() ; tracer sur ax.
4. Titre, noms d'axes avec unités, légende si plusieurs séries.
5. Regarder la figure et se demander ce qui surprend. C'est là qu'est l'information.
```

**Implémentation**
```python
import matplotlib.pyplot as plt
import numpy as np

# Deux diagnostics côte à côte : courbe d'apprentissage + résidus
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

epochs = np.arange(1, 51)
ax1.plot(epochs, perte_train, label="entraînement")
ax1.plot(epochs, perte_val, label="validation")
ax1.axvline(np.argmin(perte_val) + 1, ls='--', c='gray', lw=1)
ax1.set(xlabel="époque", ylabel="perte", title="Courbe d'apprentissage")
ax1.legend()

ax2.scatter(y_pred, y_true - y_pred, s=8, alpha=0.3)
ax2.axhline(0, c='black', lw=1)
ax2.set(xlabel="prédiction", ylabel="résidu", title="Résidus")

fig.savefig('diagnostic.png', dpi=150)

# Nuage de points avec 200 000 points : hexbin, pas scatter
fig, ax = plt.subplots()
hb = ax.hexbin(x, y, gridsize=60, cmap='viridis', mincnt=1)
fig.colorbar(hb, ax=ax, label="nombre de points")
```

**Outils** — `pip install matplotlib seaborn`. Seaborn ajoute des graphiques statistiques en une ligne et des thèmes corrects par défaut.

**Alternatives open-source**
- *Bibliothèques* : **seaborn** pour les graphiques statistiques (`pairplot`, `heatmap`, `boxplot`) — construit sur matplotlib, donc combinable ; **plotly** pour l'interactivité (zoom, survol, export HTML autonome) ; **Altair** applique la grammaire graphique de Vega-Lite, très cohérente et concise ; **plotnine** reproduit `ggplot2` pour qui vient de R ; **datashader** pour rendre des centaines de millions de points.
- *Outils graphiques* : **Metabase** et **Apache Superset** pour des tableaux de bord SQL sans code ; **Grafana** pour des séries temporelles et du suivi de production ; **Orange Data Mining** enchaîne visualisation et modèles par blocs visuels, excellent pédagogiquement ; **PyGWalker** transforme un DataFrame en interface type Tableau dans un notebook.

**Astuces**
- Un graphique sans nom d'axes est inutilisable dans trois semaines, y compris pour toi.
- Ne jamais utiliser la palette `jet` : elle crée des frontières visuelles inexistantes. `viridis` est perceptuellement uniforme et lisible en niveaux de gris ou en cas de daltonisme.
- `constrained_layout=True` à la création de la figure règle 90 % des étiquettes coupées.
- Un axe des ordonnées qui ne part pas de zéro sur un graphique en barres exagère les écarts. Sur une ligne, c'est au contraire souvent justifié.
- `plt.close(fig)` en boucle, sinon la mémoire explose après quelques centaines de figures.
- Exporter en SVG ou PDF pour un rapport, en PNG à `dpi=150` pour un écran. Jamais de capture d'écran.
- Avant tout modèle, tracer la cible en fonction de chaque feature. Beaucoup de projets de ML se règlent par une transformation évidente vue sur un nuage de points.

## Environnements et reproductibilité
<!-- slug: environnements-uv | difficulte: 1 | prereqs:  -->

**En une phrase** — Isoler les dépendances de chaque projet et figer leurs versions, pour que le code qui marche aujourd'hui marche encore dans six mois.

**Explication** — Un projet de ML dépend de dizaines de paquets aux versions étroitement couplées : une version de NumPy incompatible avec la version de scikit-learn compilée contre elle, un modèle sérialisé illisible par une version différente. Sans isolement, installer un second projet casse le premier. Un environnement virtuel est simplement un dossier contenant son propre interpréteur et ses propres paquets.

La distinction essentielle est celle entre **déclaration** et **verrouillage**. Le fichier de déclaration (`pyproject.toml`) exprime l'intention : « scikit-learn au moins 1.4 ». Le fichier de verrouillage (`uv.lock`, `requirements.txt` figé) enregistre le résultat exact de la résolution : chaque paquet, chaque version, chaque empreinte. C'est le second qui rend une installation reproductible ; c'est aussi celui qu'on oublie de versionner.

`uv` a remplacé l'outillage historique en 2024-2025 : écrit en Rust, il installe 10 à 100 fois plus vite que `pip`, gère les versions de Python lui-même, et remplace `pip`, `venv`, `pip-tools` et `pyenv` d'un coup. Pour un projet neuf, il n'y a plus de raison de commencer autrement.

**Cas d'utilisation**
- Tout projet, même un exercice d'une heure — l'habitude vaut la discipline.
- Reprise d'un projet abandonné six mois plus tôt sans passer la soirée à réparer les imports.
- Partage d'un dépôt qui doit s'installer chez quelqu'un d'autre en une commande.
- Pour du deep learning avec CUDA, l'environnement Python ne suffit pas : la version du pilote et celle de la bibliothèque compilée doivent aussi correspondre, d'où le recours à Docker.

**Algorithme**
```text
1. uv init            -> pyproject.toml, .python-version, .gitignore
2. uv add <paquet>    -> résout, installe, met à jour le verrou
3. uv run script.py   -> exécute dans l'environnement, sans activation
4. Versionner pyproject.toml ET uv.lock. Jamais .venv/ ni data/ ni *.db.
5. Sur une autre machine : uv sync -> environnement identique au bit près.
```

**Implémentation**
```bash
uv init atelier-ml && cd atelier-ml
uv python pin 3.12
uv add numpy pandas polars scikit-learn matplotlib seaborn
uv add --dev pytest ruff jupyterlab      # outils de dev séparés

uv run python entrainement.py
uv run jupyter lab
uv sync                                   # reproduire l'environnement à l'identique
uv lock --upgrade-package scikit-learn     # mise à jour ciblée et contrôlée
```

```toml
# pyproject.toml — déclaration lisible et minimale
[project]
name = "atelier-ml"
requires-python = ">=3.12"
dependencies = [
  "numpy>=2.0",
  "pandas>=2.2",
  "scikit-learn>=1.5",
]

[tool.ruff]
line-length = 100
```

**Outils** — `curl -LsSf https://astral.sh/uv/install.sh | sh` (ou `winget install astral-sh.uv` sous Windows).

**Alternatives open-source**
- *Bibliothèques et outils* : **venv + pip-tools** est l'approche standard historique, sans dépendance externe ; **Poetry** gère aussi la publication de paquets, plus lourd ; **Conda / Mamba** reste nécessaire quand des binaires non-Python entrent en jeu (CUDA, GDAL, R) — **Micromamba** en est la version légère et rapide ; **Pixi** combine l'écosystème conda et la vitesse de Rust ; **Docker** dès qu'il faut figer autre chose que Python.
- *Outils graphiques* : **VS Code** détecte et bascule les environnements automatiquement ; **JupyterLab** avec `nb_conda_kernels` liste les environnements comme noyaux ; **Docker Desktop** pour visualiser images et conteneurs.

**Astuces**
- Ne jamais installer avec `pip install` global. La tentation revient toujours à 23 h ; c'est toujours celle-là qui casse la machine.
- Versionner le fichier de verrouillage. Un dépôt sans verrou n'est pas reproductible, quoi qu'en dise son `requirements.txt`.
- `uv run` évite complètement l'activation manuelle, donc les erreurs de « mauvais environnement actif ».
- Ajouter `data/`, `*.db`, `.venv/`, `models/`, `*.ckpt` au `.gitignore` dès le premier commit. Un poids de modèle poussé par erreur reste dans l'historique Git pour toujours.
- Fixer les graines aléatoires ne suffit pas à la reproductibilité : sur GPU, certaines opérations restent non déterministes. Enregistrer les métriques obtenues, pas seulement le code.
- Dans un notebook, `%load_ext autoreload` puis `%autoreload 2` recharge le code modifié sans redémarrer le noyau. Gain de temps quotidien considérable.

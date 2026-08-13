---
module: big-data
titre: Big data et passage à l'échelle
ordre: 9
resume: La première question n'est pas « quel outil distribué ? » mais « ai-je vraiment besoin d'un cluster ? ». Souvent non.
---

## Stockage et architecture de données
<!-- slug: architecture-donnees | difficulte: 3 | prereqs: polars-parquet -->

**En une phrase** — Organiser les données sur disque en format colonne, partitionné et transactionnel, pour que les traitements ne lisent que ce dont ils ont besoin.

**Explication** — Le passage du CSV au **format colonne** est le gain le plus important, et il ne coûte rien. Parquet stocke chaque colonne séparément, typée et compressée, avec des statistiques min/max par groupe de lignes. Une requête sur 3 colonnes parmi 200 ne lit que ces 3 colonnes ; un filtre sur une plage de dates écarte des blocs entiers sans les décompresser. Sur des données réelles, la combinaison réduit souvent les entrées-sorties d'un facteur 50.

Le **partitionnement** pousse la logique dans l'arborescence : `ventes/annee=2025/mois=03/` permet au moteur de n'ouvrir que les dossiers concernés. La règle de dimensionnement importe : viser des fichiers de 100 Mo à 1 Go. Des milliers de petits fichiers ruinent les performances — c'est le problème des *small files*, aussi classique que le sur-partitionnement sur une colonne à forte cardinalité.

Les **formats de table** (Delta Lake, Apache Iceberg, Hudi) ajoutent au-dessus de Parquet ce qui manquait : transactions atomiques, évolution de schéma, voyage dans le temps (relire l'état d'il y a trois jours), et suppression de lignes. C'est ce qui permet de traiter un lac de fichiers comme une vraie base — le *lakehouse*. Pour un projet d'apprentissage, Parquet partitionné suffit ; Iceberg et Delta valent d'être connus parce qu'ils sont devenus le standard industriel.

Enfin, un rappel utile : **DuckDB** interroge en SQL des centaines de gigaoctets de Parquet sur un simple portable. Beaucoup de projets « big data » n'ont jamais eu besoin d'un cluster.

**Cas d'utilisation**
- Convertir toute source (CSV, base, API) en Parquet dès l'ingestion : gain immédiat et définitif.
- Historisation d'événements avec partitionnement par date.
- Alimentation d'un entraînement de modèle sur des volumes qui ne tiennent pas en mémoire.
- Delta ou Iceberg quand plusieurs processus écrivent en concurrence, ou quand il faut pouvoir corriger l'historique.
- Inutile en dessous de quelques centaines de mégaoctets : un seul fichier Parquet non partitionné est optimal.

**Algorithme**
```text
1. Choisir la clé de partitionnement : la colonne la plus souvent filtrée,
   presque toujours une date. Cardinalité cible : dizaines à milliers de
   partitions, jamais des millions.
2. Écrire en Parquet, compression zstd (meilleur ratio/vitesse que snappy).
3. Viser 100 Mo - 1 Go par fichier. Compacter périodiquement les petits.
4. Trier les données à l'intérieur de chaque partition sur la deuxième
   colonne de filtrage : les statistiques min/max deviennent alors sélectives.
5. Documenter le schéma et l'unité de chaque colonne. Sans cela, le lac
   devient un marécage en six mois.
6. Interroger avec DuckDB ou Polars en mode paresseux avant d'envisager Spark.
```

**Implémentation**
```python
import polars as pl, duckdb

# 1. Conversion et partitionnement, en flux (mémoire bornée)
(pl.scan_csv('brut/evenements*.csv', try_parse_dates=True)
   .with_columns(annee=pl.col('date').dt.year(), mois=pl.col('date').dt.month())
   .sort('client_id')                       # tri interne : min/max sélectifs
   .sink_parquet('lac/evenements/', partition_by=['annee', 'mois'],
                 compression='zstd'))

# 2. SQL direct sur les fichiers, sans base de données à installer
con = duckdb.connect()
con.sql("""
    SELECT annee, mois, count(*) AS n, avg(montant) AS panier
    FROM read_parquet('lac/evenements/**/*.parquet', hive_partitioning := true)
    WHERE annee = 2025 AND montant > 0
    GROUP BY 1, 2 ORDER BY 1, 2
""").show()

# 3. Inspecter le schéma et les statistiques d'un fichier
import pyarrow.parquet as pq
f = pq.ParquetFile('lac/evenements/annee=2025/mois=3/part-0.parquet')
print(f.schema_arrow)
print(f.metadata.num_rows, f.metadata.num_row_groups)
for i in range(f.metadata.num_row_groups):
    st = f.metadata.row_group(i).column(0).statistics
    print(i, st.min, st.max)                 # ce qui permet le saut de blocs

# 4. Compacter les petits fichiers : maintenance à planifier
(pl.scan_parquet('lac/evenements/annee=2025/mois=3/*.parquet')
   .sink_parquet('lac/evenements/annee=2025/mois=3/compacte.parquet'))

# 5. Delta Lake : transactions, voyage dans le temps, mise à jour
# pip install deltalake
from deltalake import write_deltalake, DeltaTable
write_deltalake('lac/table_delta', df.to_arrow(), mode='append',
                partition_by=['annee'])
dt = DeltaTable('lac/table_delta')
print(dt.history()[:3])                      # journal des versions
ancien = dt.to_pyarrow_table(version=2)      # relire un état passé
dt.delete("montant < 0")                     # suppression transactionnelle
dt.optimize.compact()                        # compaction des petits fichiers
```

**Outils** — `pip install polars duckdb pyarrow deltalake`.

**Alternatives open-source**
- *Bibliothèques* : **DuckDB** est probablement l'outil le plus rentable du domaine — SQL analytique complet, sur Parquet, sans serveur ; **Apache Iceberg** (via `pyiceberg`) est le format de table le plus adopté, indépendant du moteur ; **Delta Lake** est plus simple à démarrer en Python ; **Apache Hudi** pour des mises à jour très fréquentes ; **Apache Arrow** comme couche mémoire commune ; **fsspec** et **s3fs** pour accéder à un stockage objet comme à un système de fichiers ; **MinIO** pour un stockage objet compatible S3 auto-hébergé.
- *Outils graphiques* : **DBeaver** connecté à DuckDB pour explorer un lac Parquet à la souris ; **Tad** ouvre un Parquet comme un tableur ; **Apache Superset** ou **Metabase** pour des tableaux de bord au-dessus de DuckDB ou Trino ; **Trino** (ex-PrestoSQL) pour interroger plusieurs sources en un seul SQL ; **OpenMetadata** ou **DataHub** pour le catalogue et la traçabilité.

**Astuces**
- Convertir en Parquet est le premier réflexe, avant toute optimisation de code. Un CSV de 10 Go devient souvent un Parquet de 800 Mo lu vingt fois plus vite.
- Ne jamais partitionner sur un identifiant client ou une clé à forte cardinalité : cela crée un dossier par valeur et détruit les performances.
- Trier à l'intérieur des partitions rend les statistiques min/max sélectives. Sans tri, elles couvrent toute la plage et ne servent à rien.
- `zstd` offre un meilleur compromis que `snappy` sur du stockage moderne : fichiers 20 à 30 % plus petits pour une décompression comparable.
- Compter avant d'agir : `SELECT count(*)` et la taille sur disque. Beaucoup de projets « trop gros pour pandas » font 4 Go et tiennent parfaitement en mémoire avec Polars.
- Parquet ne se lit pas avec `head` ni `grep`. Garder un script d'inspection du schéma sous la main, sinon le débogage devient pénible.
- Documenter le schéma dans le dépôt. Un lac de données sans catalogue devient inutilisable plus vite qu'on ne le croit.

## PySpark
<!-- slug: pyspark | difficulte: 3 | prereqs: architecture-donnees, pandas-manipulation -->

**En une phrase** — Un moteur distribué qui découpe les données en partitions réparties sur un cluster et exécute un plan de calcul optimisé, avec une API proche de pandas.

**Explication** — Un `DataFrame` Spark est une abstraction sur des données découpées en **partitions**, chacune traitée par un cœur. Les opérations se divisent en **transformations** (`select`, `filter`, `join`, `groupBy`) qui sont *paresseuses* et ne construisent qu'un plan, et **actions** (`count`, `collect`, `write`) qui déclenchent l'exécution. Le moteur Catalyst optimise le plan avant de l'exécuter : élagage des colonnes, descente des filtres, choix de stratégie de jointure.

Le concept qui détermine toute la performance est le **brassage** (*shuffle*) : quand une opération exige de regrouper des lignes par clé (jointure, agrégation, tri), les données doivent traverser le réseau entre exécuteurs. C'est l'opération la plus coûteuse, et la source de presque tous les problèmes. Deux pathologies classiques : le **déséquilibre** (*skew*), quand une clé concentre l'essentiel des lignes et qu'une seule tâche prend dix fois plus de temps que les autres ; et le brassage évitable, quand une jointure avec une petite table aurait pu être diffusée (*broadcast join*) à tous les exécuteurs.

Il faut nommer clairement le seuil d'utilité. Spark apporte de la valeur au-delà de quelques centaines de gigaoctets, ou quand un cluster existe déjà. En dessous, sa surcharge — JVM, sérialisation, planification — le rend **plus lent** que Polars ou DuckDB sur une seule machine. Apprendre Spark reste utile : c'est le standard en entreprise, et les concepts de partitionnement et de brassage éclairent tout le calcul distribué.

**Cas d'utilisation**
- Traitements sur plusieurs téraoctets, ou pipelines déjà déployés sur un cluster.
- Jointures massives entre plusieurs grandes tables.
- Écriture de pipelines ETL industrialisés, avec reprise sur incident.
- Traitement en flux (Structured Streaming) avec la même API que le traitement par lots.
- Mauvais choix en dessous de 50 Go sur une seule machine, où DuckDB ou Polars sont plus rapides à écrire et à exécuter.

**Algorithme**
```text
1. Lire en Parquet (jamais en CSV pour de gros volumes).
2. Filtrer et sélectionner AU PLUS TÔT : moins de données à brasser.
3. Diffuser les petites tables lors des jointures (broadcast si < ~100 Mo).
4. Contrôler le nombre de partitions :
     trop peu  -> pas de parallélisme, mémoire saturée par tâche
     trop      -> surcharge de planification
     cible : 2 à 4 partitions par cœur disponible, ~128 Mo chacune.
5. cache() UNIQUEMENT un DataFrame réutilisé plusieurs fois, et unpersist() après.
6. Lire le plan avec explain() et l'interface web : compter les brassages.
7. Traiter le déséquilibre : activer l'exécution adaptative (AQE), ou saler
   la clé (ajouter un suffixe aléatoire) pour répartir la charge.
```

**Implémentation**
```python
from pyspark.sql import SparkSession, functions as F, Window

spark = (SparkSession.builder
         .appName('atelier-ml')
         .config('spark.sql.adaptive.enabled', 'true')            # AQE : gère le déséquilibre
         .config('spark.sql.adaptive.coalescePartitions.enabled', 'true')
         .config('spark.sql.shuffle.partitions', '200')
         .getOrCreate())

ventes = spark.read.parquet('lac/ventes/')
clients = spark.read.parquet('lac/clients/')      # petite table

# Filtrer tôt, sélectionner tôt : les deux règles qui comptent le plus
resultat = (ventes
    .filter((F.col('annee') == 2025) & (F.col('montant') > 0))
    .select('client_id', 'produit', 'montant', 'date')
    # diffusion de la petite table : supprime un brassage entier
    .join(F.broadcast(clients.select('client_id', 'segment')), on='client_id', how='left')
    .groupBy('segment', F.month('date').alias('mois'))
    .agg(F.sum('montant').alias('total'),
         F.countDistinct('client_id').alias('n_clients'),
         F.expr('percentile_approx(montant, 0.5)').alias('median')))

resultat.explain('formatted')          # compter les Exchange = brassages
resultat.write.mode('overwrite').partitionBy('mois').parquet('lac/resume/')

# Fenêtres : puissant, mais chaque fenêtre implique un brassage
w = Window.partitionBy('client_id').orderBy('date')
ventes_lag = ventes.withColumn('montant_precedent', F.lag('montant').over(w)) \
                   .withColumn('rang', F.row_number().over(w))

# Diagnostiquer un déséquilibre de clé avant de subir une tâche interminable
(ventes.groupBy('client_id').count()
       .orderBy(F.desc('count')).limit(10).show())

# Salage d'une clé déséquilibrée : répartir une clé trop lourde sur N tâches
N = 20
gauche = ventes.withColumn('sel', (F.rand() * N).cast('int'))
droite = clients.withColumn('sel', F.explode(F.array([F.lit(i) for i in range(N)])))
joint = gauche.join(droite, on=['client_id', 'sel'])

# Passerelle vers pandas : uniquement sur un résultat AGRÉGÉ, jamais sur les données brutes
petit = resultat.toPandas()             # collect() sur 100 Go plante le driver

# UDF : préférer les fonctions natives ; sinon Pandas UDF (vectorisée)
from pyspark.sql.functions import pandas_udf
import pandas as pd

@pandas_udf('double')
def normaliser(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std()
```

**Outils** — `pip install pyspark` (Java 17 requis), ou un cluster existant (Databricks, EMR, Kubernetes).

**Alternatives open-source**
- *Bibliothèques* : **DuckDB** et **Polars** couvrent tout ce qui tient sur une machine, avec une productivité bien supérieure ; **Dask** est plus naturel pour du code Python existant ; **Ray** pour distribuer du calcul Python arbitraire, notamment de l'entraînement de modèles ; **Trino** pour du SQL fédéré interactif sur plusieurs sources ; **Apache Flink** pour du flux à faible latence, plus solide que Spark Streaming sur ce point ; **dbt** pour organiser des transformations SQL versionnées et testées, souvent le vrai besoin derrière « pipeline de données ».
- *Outils graphiques* : l'**interface web Spark** (port 4040) est l'outil de diagnostic indispensable — étapes, brassages, déséquilibre, temps par tâche ; **Spark History Server** pour l'analyse après exécution ; **Apache Airflow** ou **Dagster** pour orchestrer les jobs ; **Jupyter** avec un noyau PySpark pour l'exploration ; **Databricks Community Edition** offre un cluster gratuit pour apprendre.

**Astuces**
- Mesurer avant de distribuer. Si les données tiennent sur une machine, Spark sera plus lent que Polars ou DuckDB, pour un code plus verbeux.
- Ne jamais faire `collect()` ou `toPandas()` sur un DataFrame non agrégé : tout remonte dans la mémoire du driver, qui tombe.
- `F.broadcast()` sur les petites tables de jointure supprime un brassage complet. C'est l'optimisation la plus rentable de Spark.
- `spark.sql.shuffle.partitions` vaut 200 par défaut, valeur absurde sur un petit cluster comme sur un très gros. L'ajuster au nombre de cœurs, ou activer l'AQE qui le fait dynamiquement.
- Lire l'interface web systématiquement. Une tâche dix fois plus longue que ses sœurs signale un déséquilibre de clé, pas un manque de ressources.
- `cache()` n'est utile que pour un DataFrame relu plusieurs fois. Utilisé partout, il sature la mémoire et provoque des recalculs.
- Les UDF Python sérialisent ligne par ligne et sont dix à cent fois plus lentes que les fonctions natives. Chercher toujours l'équivalent dans `pyspark.sql.functions`, ou utiliser une Pandas UDF vectorisée.
- Le message « out of memory » vient presque toujours d'un brassage déséquilibré ou d'une partition trop grosse, rarement d'un manque global de RAM.

## Spark MLlib
<!-- slug: spark-mllib | difficulte: 3 | prereqs: pyspark, pipelines -->

**En une phrase** — Entraîner des modèles de machine learning directement sur des données distribuées, avec une API de pipelines analogue à celle de scikit-learn.

**Explication** — MLlib reprend les abstractions de scikit-learn en version distribuée. Un `Transformer` transforme un DataFrame (`VectorAssembler`, `StringIndexer`, `StandardScaler`), un `Estimator` s'ajuste et produit un modèle (`LogisticRegression`, `GBTClassifier`), et un `Pipeline` les enchaîne. La différence essentielle est la représentation : MLlib exige que toutes les features soient rassemblées dans **une seule colonne de type vecteur**, construite par `VectorAssembler`. Cette étape surprend systématiquement à la première utilisation.

Le catalogue est volontairement restreint aux algorithmes qui se parallélisent bien : modèles linéaires, arbres, forêts, gradient boosting (`GBTClassifier`), ALS pour la recommandation, k-means, LDA, et quelques utilitaires de texte. On n'y trouve ni XGBoost natif, ni beaucoup de variantes récentes.

La question stratégique est de savoir **quand** on en a besoin. Trois situations seulement : les données d'entraînement ne tiennent pas en mémoire sur une machine ; le prétraitement distribué est déjà en Spark et l'on veut éviter un transfert ; ou l'on doit scorer des milliards de lignes en lot. Dans tous les autres cas — et ils sont majoritaires — la bonne approche est d'agréger ou d'échantillonner avec Spark, puis d'entraîner un LightGBM sur une seule machine : plus précis, plus rapide, mieux outillé.

**Cas d'utilisation**
- Entraînement sur des centaines de millions de lignes qui ne tiennent pas en mémoire.
- Scoring en lot d'énormes volumes, avec le modèle appliqué en parallèle.
- Recommandation par factorisation matricielle (ALS) sur une matrice utilisateurs-articles massive.
- Pipeline de bout en bout maintenu par une équipe data engineering déjà sur Spark.
- Mauvais choix pour l'expérimentation et la recherche de modèle : itérer sur un échantillon en scikit-learn ou LightGBM est bien plus rapide.

**Algorithme**
```text
1. Décider honnêtement : les données tiennent-elles en mémoire après
   agrégation et sélection ? Si oui, sortir de Spark pour la modélisation.
2. Sinon, pipeline MLlib :
     StringIndexer / OneHotEncoder   sur les catégorielles
     Imputer                          sur les manquants
     VectorAssembler                  -> colonne 'features' (obligatoire)
     StandardScaler                   si modèle linéaire
     Estimateur                       (GBTClassifier, LogisticRegression...)
3. Découpage : randomSplit, ou filtre temporel pour respecter la chronologie.
4. Validation croisée : CrossValidator (coûteux) ou TrainValidationSplit.
5. Évaluer avec BinaryClassificationEvaluator / MulticlassClassificationEvaluator.
6. Sauvegarder le PipelineModel entier (prétraitement inclus).
```

**Implémentation**
```python
from pyspark.ml import Pipeline
from pyspark.ml.feature import (VectorAssembler, StringIndexer, OneHotEncoder,
                                StandardScaler, Imputer)
from pyspark.ml.classification import GBTClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit

df = spark.read.parquet('lac/dataset/')
cols_num, cols_cat = ['age', 'revenu', 'anciennete'], ['region', 'segment']

etapes = []
etapes.append(Imputer(inputCols=cols_num, outputCols=cols_num, strategy='median'))
for c in cols_cat:
    etapes.append(StringIndexer(inputCol=c, outputCol=f'{c}_idx', handleInvalid='keep'))
    etapes.append(OneHotEncoder(inputCol=f'{c}_idx', outputCol=f'{c}_ohe'))
# L'étape spécifique à MLlib : tout rassembler en une colonne vecteur
etapes.append(VectorAssembler(inputCols=cols_num + [f'{c}_ohe' for c in cols_cat],
                              outputCol='features', handleInvalid='keep'))
etapes.append(GBTClassifier(featuresCol='features', labelCol='cible',
                            maxIter=100, maxDepth=5, stepSize=0.05, seed=0))

pipeline = Pipeline(stages=etapes)

# Découpage temporel : indispensable si une date existe
seuil = df.approxQuantile('date_num', [0.8], 0.01)[0]
tr, te = df.filter(f'date_num <= {seuil}'), df.filter(f'date_num > {seuil}')

grille = (ParamGridBuilder()
          .addGrid(etapes[-1].maxDepth, [4, 6])
          .addGrid(etapes[-1].stepSize, [0.05, 0.1])
          .build())
ev = BinaryClassificationEvaluator(labelCol='cible', metricName='areaUnderROC')

tvs = TrainValidationSplit(estimator=pipeline, estimatorParamMaps=grille,
                           evaluator=ev, trainRatio=0.8, parallelism=4, seed=0)
modele = tvs.fit(tr)
print("AUC test :", ev.evaluate(modele.transform(te)))

# Importance des features, avec les noms retrouvés depuis les métadonnées
meilleur = modele.bestModel.stages[-1]
attrs = modele.transform(te).schema['features'].metadata['ml_attr']['attrs']
noms = [a['name'] for grp in attrs.values() for a in grp]
for n, imp in sorted(zip(noms, meilleur.featureImportances.toArray()),
                     key=lambda t: -t[1])[:10]:
    print(f"{n:28s} {imp:.4f}")

# Sauvegarde et rechargement du pipeline complet
modele.bestModel.write().overwrite().save('modeles/gbt_pipeline')
from pyspark.ml import PipelineModel
charge = PipelineModel.load('modeles/gbt_pipeline')

# Recommandation à grande échelle : ALS
from pyspark.ml.recommendation import ALS
als = ALS(userCol='user_id', itemCol='item_id', ratingCol='note',
          rank=32, regParam=0.05, coldStartStrategy='drop', seed=0)
reco = als.fit(interactions).recommendForAllUsers(10)

# L'approche souvent meilleure : agréger en Spark, entraîner en local
echantillon = df.sample(fraction=0.05, seed=0).toPandas()
# puis LightGBM sur echantillon : plus précis, plus rapide à itérer
```

**Outils** — `pyspark.ml` (inclus dans PySpark).

**Alternatives open-source**
- *Bibliothèques* : **XGBoost4J-Spark** et **SynapseML** (Microsoft) apportent XGBoost et LightGBM distribués sur Spark, plus performants que `GBTClassifier` ; **Ray Train** distribue l'entraînement PyTorch ou XGBoost sans passer par Spark ; **Dask-ML** pour la même chose dans l'écosystème Dask ; **H2O Sparkling Water** combine H2O et Spark avec une interface graphique ; **Petastorm** pour alimenter un entraînement PyTorch depuis du Parquet ; **cuML** et **Spark RAPIDS** pour l'accélération GPU.
- *Outils graphiques* : **H2O Flow** entraîne des modèles distribués sans code, avec graphiques d'importance et de dépendance partielle ; **MLflow** (intégré à Databricks) pour le suivi d'expériences Spark ; l'**interface web Spark** pour diagnostiquer un entraînement lent ; **Databricks Community Edition** pour pratiquer gratuitement.

**Astuces**
- `VectorAssembler` est obligatoire et systématiquement oublié la première fois. Le message d'erreur ne le dit pas clairement.
- Ne pas faire de recherche d'hyperparamètres exhaustive en MLlib : chaque essai est un job complet. Chercher sur un échantillon en local, puis réentraîner en distribué avec les paramètres retenus.
- `handleInvalid='keep'` sur `StringIndexer` évite un plantage sur une modalité inconnue en production. Le défaut lève une exception.
- `GBTClassifier` est nettement plus lent et moins précis que LightGBM à configuration comparable. Si le volume le permet, préférer LightGBM.
- Mettre `cache()` sur le DataFrame d'entraînement : il est relu à chaque itération du boosting, et sans cache tout le plan est recalculé.
- `CrossValidator` multiplie le coût par le nombre de blocs. `TrainValidationSplit` suffit largement sur de gros volumes, où la variance d'estimation est faible.
- Sauvegarder le `PipelineModel` complet et non l'estimateur seul, sinon le prétraitement est perdu et les prédictions deviennent incohérentes.

## Dask
<!-- slug: dask | difficulte: 2 | prereqs: pandas-manipulation, architecture-donnees -->

**En une phrase** — Paralléliser du code Python et pandas existant sur tous les cœurs d'une machine ou sur un cluster, en découpant les données en morceaux traités paresseusement.

**Explication** — Dask propose des **collections** qui imitent les API familières : `dask.dataframe` reproduit pandas, `dask.array` reproduit NumPy, `dask.bag` traite des collections d'objets Python. Chaque collection est en réalité un graphe de tâches : les opérations construisent un plan, et rien n'est calculé avant `.compute()`. Un `dask.dataframe` est concrètement une collection de DataFrames pandas (les *partitions*), chacune traitée indépendamment, avec agrégation des résultats.

L'atout est la **continuité** avec l'écosystème Python. Le code pandas existant fonctionne souvent en changeant l'import, ce qui n'est pas vrai de Spark ni de Polars. `dask.delayed` permet en outre de paralléliser n'importe quelle fonction Python, ce qui en fait un outil général de calcul parallèle — utile pour lancer cent entraînements de modèles en même temps.

Ses limites doivent être connues. Dask hérite des lenteurs de pandas partition par partition : sur une seule machine, Polars et DuckDB sont généralement plus rapides, parfois d'un ordre de grandeur. Les opérations exigeant un brassage global (tri, jointure sur une colonne non indexée, `groupby` à forte cardinalité) sont coûteuses et parfois instables en mémoire. Dask brille surtout sur les traitements *embarrassingly parallel* : appliquer la même transformation à des milliers de fichiers, ou paralléliser des calculs indépendants.

**Cas d'utilisation**
- Traiter un dossier de milliers de fichiers CSV ou Parquet dépassant la RAM.
- Paralléliser du code Python existant (simulations, prétraitements, validations croisées) sans le réécrire.
- Calcul sur de gros tableaux NumPy (images satellites, données scientifiques) avec `dask.array` et `xarray`.
- Recherche d'hyperparamètres distribuée via `dask-ml` ou le backend `joblib`.
- Mauvais choix si Polars ou DuckDB suffisent : ils sont plus simples et plus rapides sur une machine.

**Algorithme**
```text
1. Choisir la taille de partition : viser 100-200 Mo en mémoire par partition.
   Trop petites -> surcharge de planification. Trop grandes -> mémoire saturée.
2. Construire le graphe avec l'API familière (dd.read_parquet, filtres, agrégats).
3. persist() les résultats intermédiaires réutilisés (garde en RAM distribuée) ;
   compute() seulement pour rapatrier un résultat final petit.
4. Surveiller le tableau de bord (port 8787) : mémoire par worker, tâches,
   temps de transfert.
5. Éviter les brassages globaux. Si un tri complet est nécessaire, se demander
   si DuckDB ne ferait pas le travail plus simplement.
```

**Implémentation**
```python
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster

# Un client local expose un tableau de bord très informatif
client = Client(LocalCluster(n_workers=4, threads_per_worker=2, memory_limit='4GB'))
print(client.dashboard_link)          # http://localhost:8787

# Lecture de milliers de fichiers comme un seul DataFrame paresseux
df = dd.read_parquet('lac/evenements/**/*.parquet',
                     columns=['client_id', 'date', 'montant', 'produit'])
print(df.npartitions)

resume = (df[df.montant > 0]
          .assign(mois=df.date.dt.to_period('M').astype(str))
          .groupby(['mois', 'produit'])
          .montant.agg(['sum', 'mean', 'count']))
resultat = resume.compute()            # exécution ici, résultat en pandas

# persist() : garder en mémoire distribuée un intermédiaire réutilisé
propre = df[df.montant.between(0, 1e6)].persist()
total = propre.montant.sum().compute()
moyenne = propre.montant.mean().compute()      # ne recalcule pas le filtre

# dask.delayed : paralléliser n'importe quelle fonction Python
from dask import delayed, compute

@delayed
def traiter_fichier(chemin):
    import pandas as pd
    d = pd.read_csv(chemin)
    return {'fichier': chemin, 'n': len(d), 'total': d.montant.sum()}

import glob
resultats = compute(*[traiter_fichier(f) for f in glob.glob('brut/*.csv')])

# dask.array : gros tableaux, calcul par blocs
import dask.array as da
X = da.from_zarr('donnees/images.zarr')        # (100000, 512, 512)
moy = X.mean(axis=0).compute()

# Recherche d'hyperparamètres parallélisée : trois lignes
import joblib
from sklearn.model_selection import GridSearchCV
with joblib.parallel_backend('dask'):
    GridSearchCV(modele, grille, cv=5, n_jobs=-1).fit(X_tr, y_tr)

# Repartitionner quand les partitions sont mal dimensionnées
df = df.repartition(partition_size='128MB')
```

**Outils** — `pip install "dask[complete]" distributed`.

**Alternatives open-source**
- *Bibliothèques* : **Polars** en mode paresseux et streaming couvre la plupart des besoins mono-machine, plus vite et plus simplement ; **DuckDB** pour tout ce qui s'exprime en SQL ; **Ray** pour distribuer du calcul Python arbitraire, avec un meilleur modèle d'exécution pour le ML ; **joblib** pour du parallélisme local simple ; **xarray** au-dessus de `dask.array` pour des tableaux étiquetés multidimensionnels (climat, géospatial) ; **dask-ml** pour du ML sur données distribuées ; **PySpark** si le volume dépasse la centaine de téraoctets.
- *Outils graphiques* : le **tableau de bord Dask** (port 8787) est excellent — graphe des tâches, mémoire par worker, flux de travail en direct ; **Coiled** (partiellement libre) pour déployer un cluster Dask dans le cloud ; **JupyterLab** avec l'extension `dask-labextension` intègre le tableau de bord dans le notebook ; **Prefect** ou **Dagster** pour orchestrer des flux Dask.

**Astuces**
- Toujours ouvrir le tableau de bord. Il montre immédiatement si le problème est la mémoire, le transfert réseau ou un déséquilibre de partitions.
- Viser 100 à 200 Mo par partition. Des milliers de partitions minuscules passent plus de temps en planification qu'en calcul.
- `persist()` garde un intermédiaire en mémoire distribuée ; `compute()` rapatrie chez le client. Confondre les deux fait tomber le processus principal.
- Spécifier `columns=` à la lecture de Parquet : Dask ne lit alors que les colonnes utiles, gain souvent supérieur à toute autre optimisation.
- Éviter `set_index` sur une grande colonne : c'est un tri global, l'opération la plus coûteuse de Dask.
- `df.apply()` sans `meta=` force Dask à inférer les types en exécutant sur une partition. Fournir `meta` explicitement évite des surprises et des avertissements.
- Avant d'installer Dask, vérifier que Polars ou DuckDB ne suffisent pas. Pour un usage mono-machine, ils sont presque toujours le meilleur choix.

## Traitement en flux
<!-- slug: streaming-kafka | difficulte: 4 | prereqs: pyspark, features-temporelles -->

**En une phrase** — Traiter les données au fil de leur arrivée plutôt que par lots, avec un journal d'événements durable comme colonne vertébrale.

**Explication** — **Kafka** est un journal distribué, partitionné et persistant. Les producteurs écrivent des messages dans des *topics* découpés en partitions ; les consommateurs lisent séquentiellement en mémorisant leur position (*offset*). L'ordre est garanti **à l'intérieur d'une partition** seulement, ce qui impose de choisir la clé de partitionnement en fonction de l'ordre dont on a besoin (par exemple la clé client, pour que tous ses événements arrivent dans l'ordre). Les messages étant conservés, un consommateur peut rejouer l'historique — propriété qui change la nature du système : ce n'est pas une file d'attente, c'est un journal.

Le concept le plus délicat est la distinction entre **temps de l'événement** et **temps de traitement**. Un événement produit à 10 h 00 peut arriver à 10 h 03 pour cause de latence réseau ou de mobile hors ligne. Agréger par temps de traitement produit des résultats faux ; agréger par temps d'événement exige d'attendre les retardataires, d'où la notion de **filigrane** (*watermark*) : « je considère avoir tout reçu jusqu'à $t$, moins une tolérance de 10 minutes ». Ce paramètre arbitre entre latence et exhaustivité.

Enfin les **garanties de livraison** : *au plus une fois* (pertes possibles), *au moins une fois* (doublons possibles, le défaut usuel), *exactement une fois* (coûteux, exige de l'idempotence ou des transactions). En pratique, on conçoit des traitements **idempotents** et on accepte l'« au moins une fois » — bien plus simple et robuste que de viser l'exactement une fois.

**Cas d'utilisation**
- Détection de fraude ou d'anomalie en temps quasi réel.
- Features en direct pour un modèle servi en ligne (nombre de transactions dans la dernière heure).
- Journalisation d'événements applicatifs, télémétrie, capteurs.
- Découplage entre services : le producteur ignore qui consomme.
- Mauvais choix quand un traitement toutes les heures suffit. Le flux multiplie la complexité opérationnelle, et l'immense majorité des besoins métier tolère un lot.

**Algorithme**
```text
1. Modéliser l'événement : clé, horodatage de l'ÉVÉNEMENT, charge utile,
   version de schéma. L'horodatage de production est indispensable.
2. Choisir la clé de partition selon l'ordre requis (souvent l'entité métier).
3. Consommer par micro-lots (Spark Structured Streaming) ou événement par
   événement (Flink, Faust).
4. Agréger par fenêtre de temps d'ÉVÉNEMENT, avec filigrane :
     tumbling (fenêtres disjointes), sliding (glissantes), session (par inactivité)
5. Rendre le traitement idempotent : clé de déduplication, écriture upsert.
6. Écrire dans un puits transactionnel (Delta, Iceberg) avec point de reprise.
7. Surveiller : retard de consommation (lag), débit, latence de bout en bout.
```

**Implémentation**
```python
# --- Producteur Kafka ---
# pip install confluent-kafka
from confluent_kafka import Producer
import json, time

p = Producer({'bootstrap.servers': 'localhost:9092',
              'enable.idempotence': True})      # évite les doublons du producteur

def envoyer(evenement):
    p.produce('transactions',
              key=str(evenement['client_id']),   # même client -> même partition -> ordre garanti
              value=json.dumps(evenement).encode(),
              callback=lambda err, msg: err and print('échec :', err))
    p.poll(0)

envoyer({'client_id': 42, 'montant': 129.9, 'ts_evenement': time.time()})
p.flush()

# --- Consommateur simple ---
from confluent_kafka import Consumer
c = Consumer({'bootstrap.servers': 'localhost:9092',
              'group.id': 'scoring-fraude',
              'auto.offset.reset': 'earliest',
              'enable.auto.commit': False})      # valider APRÈS traitement réussi
c.subscribe(['transactions'])
while True:
    msg = c.poll(1.0)
    if msg is None or msg.error():
        continue
    ev = json.loads(msg.value())
    traiter(ev)                                  # doit être idempotent
    c.commit(msg)                                # au moins une fois

# --- Spark Structured Streaming : même API que le traitement par lots ---
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, DoubleType, LongType

schema = StructType([StructField('client_id', LongType()),
                     StructField('montant', DoubleType()),
                     StructField('ts_evenement', DoubleType())])

flux = (spark.readStream.format('kafka')
        .option('kafka.bootstrap.servers', 'localhost:9092')
        .option('subscribe', 'transactions')
        .option('startingOffsets', 'latest')
        .load()
        .select(F.from_json(F.col('value').cast('string'), schema).alias('d'))
        .select('d.*')
        .withColumn('ts', F.to_timestamp(F.col('ts_evenement'))))

# Agrégation par temps d'ÉVÉNEMENT, avec filigrane pour les retardataires
agrege = (flux
    .withWatermark('ts', '10 minutes')
    .groupBy(F.window('ts', '5 minutes', '1 minute'), 'client_id')
    .agg(F.count('*').alias('n'), F.sum('montant').alias('total')))

requete = (agrege.writeStream
    .format('delta').outputMode('append')
    .option('checkpointLocation', 'points_reprise/agrege')   # reprise sur incident
    .trigger(processingTime='30 seconds')
    .start('lac/features_temps_reel'))

# Surveillance : le retard de consommation est LA métrique à suivre
print(requete.lastProgress['sources'][0]['numInputRows'],
      requete.lastProgress['durationMs'])
```

**Outils** — Kafka (ou Redpanda, compatible et plus simple à opérer), `pip install confluent-kafka pyspark`.

**Alternatives open-source**
- *Bibliothèques et systèmes* : **Redpanda** est compatible avec l'API Kafka, sans ZooKeeper ni JVM, bien plus simple à exploiter pour un projet personnel ; **Apache Flink** (avec PyFlink) offre un vrai traitement événement par événement, une gestion d'état supérieure et une latence plus faible que Spark ; **Faust** et **Quix Streams** pour du flux en Python pur, sans cluster ; **Benthos/Redpanda Connect** pour du routage et de la transformation déclaratifs ; **NATS** et **RabbitMQ** pour de la messagerie plus légère quand la persistance n'est pas requise ; **Debezium** pour capturer les changements d'une base relationnelle (CDC) ; **Feast** pour un magasin de features cohérent entre le lot et le temps réel.
- *Outils graphiques* : **Redpanda Console** et **AKHQ** pour inspecter topics, partitions et retard de consommation ; **Kafka UI** (provectus) ; **Grafana** avec Prometheus pour la surveillance ; l'**interface Flink** pour les jobs de flux ; **Apache Superset** pour visualiser les agrégats en quasi temps réel.

**Astuces**
- Se demander d'abord si un traitement par lots toutes les 15 minutes suffit. Il répond à l'immense majorité des besoins pour un dixième de la complexité.
- Toujours inclure l'horodatage de l'**événement** dans le message, jamais se fier à l'heure de réception. C'est irréversible : une information non émise ne se reconstitue pas.
- Concevoir tous les traitements comme idempotents. Viser l'« exactement une fois » coûte cher et échoue de toute façon aux frontières du système.
- Le filigrane arbitre entre latence et exhaustivité. Un filigrane de 10 minutes signifie que les résultats sont définitifs 10 minutes après la fin de la fenêtre.
- Le point de reprise (*checkpoint*) est obligatoire en production : sans lui, un redémarrage recommence de zéro ou perd des données.
- La clé de partitionnement détermine l'ordre. Une clé mal choisie rend impossible tout calcul dépendant de la séquence des événements d'une même entité.
- Surveiller le retard de consommation (*consumer lag*) plus que le débit : c'est lui qui révèle qu'un consommateur décroche.
- Versionner le schéma des messages dès le premier jour (Avro, Protobuf, ou simple champ `version`). Un changement de schéma non anticipé casse tous les consommateurs simultanément.

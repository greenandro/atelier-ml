---
module: non-supervise
titre: Apprentissage non supervisé
ordre: 5
resume: Pas de cible, donc pas de score objectif. La validation est ici un problème en soi, et c'est la vraie difficulté du module.
---

## k-means
<!-- slug: kmeans | difficulte: 2 | prereqs: algebre-lineaire, mise-a-l-echelle -->

**En une phrase** — Partitionner les données en $k$ groupes en alternant deux étapes : affecter chaque point au centre le plus proche, puis replacer chaque centre au barycentre de son groupe.

**Explication** — L'algorithme minimise l'inertie intra-classe $\sum_i \|x_i - \mu_{c(i)}\|^2$. Le problème exact est NP-difficile ; l'algorithme de Lloyd en donne une solution locale, et converge en une dizaine d'itérations. Chaque étape diminue nécessairement l'inertie, ce qui garantit la convergence — mais vers un optimum local dépendant de l'initialisation. D'où **k-means++**, qui tire les centres initiaux éloignés les uns des autres, et `n_init=10` qui relance plusieurs fois pour garder le meilleur.

Les hypothèses implicites sont fortes et souvent oubliées : les groupes sont supposés **sphériques**, de **taille comparable** et de **densité comparable**, parce que le critère est une distance euclidienne. Sur des groupes allongés, imbriqués ou de densités très différentes, k-means échoue de façon spectaculaire — et sans prévenir, puisqu'il retourne toujours une partition.

Le choix de $k$ n'a pas de réponse objective. La méthode du coude (inertie vs $k$) est souvent illisible. Le **score de silhouette** est plus informatif : pour chaque point, il compare la distance moyenne à son propre groupe et au groupe voisin le plus proche, et varie de -1 à 1. Mais le vrai critère reste métier : les groupes obtenus sont-ils actionnables et interprétables ?

**Cas d'utilisation**
- Segmentation de clientèle, de produits, de comportements.
- Quantification vectorielle : compression d'images, réduction d'une palette de couleurs.
- Création d'une feature catégorielle « groupe » pour un modèle supervisé.
- Pré-agrégation avant un traitement coûteux (résumer un million de points par 500 centres).
- Mauvais choix pour des formes non convexes, des densités inégales, ou quand $k$ est réellement inconnu — DBSCAN ou un modèle de mélange sont alors plus adaptés.

**Algorithme**
```text
Entrée : X (n, d), k, n_init
Répéter n_init fois, garder la meilleure inertie :
  1. Initialiser les centres par k-means++ :
       premier centre au hasard ; chaque suivant tiré avec une probabilité
       proportionnelle au carré de la distance au centre le plus proche déjà choisi.
  2. Répéter jusqu'à stabilité (ou max_iter) :
       a. AFFECTATION : c(i) = argmin_j ||x_i - mu_j||²
       b. MISE À JOUR : mu_j = moyenne des points affectés à j
       c. Si aucune affectation ne change -> arrêt
  3. Inertie = somme des distances au carré aux centres.
Prérequis : features standardisées (le critère est une distance euclidienne).
```

**Implémentation**
```python
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

Xs = StandardScaler().fit_transform(X)

# --- from scratch (palier 2) ---
def kmeans(X, k, iters=100, graine=0):
    rng = np.random.default_rng(graine)
    # k-means++ simplifié
    C = [X[rng.integers(len(X))]]
    for _ in range(k - 1):
        d2 = np.min(((X[:, None] - np.array(C)[None]) ** 2).sum(2), axis=1)
        C.append(X[rng.choice(len(X), p=d2 / d2.sum())])
    C = np.array(C)
    for _ in range(iters):
        D = ((X ** 2).sum(1)[:, None] - 2 * X @ C.T + (C ** 2).sum(1)[None, :])
        aff = D.argmin(1)
        Cnew = np.array([X[aff == j].mean(0) if (aff == j).any() else C[j] for j in range(k)])
        if np.allclose(C, Cnew):
            break
        C = Cnew
    inertie = ((X - C[aff]) ** 2).sum()
    return aff, C, inertie

# --- choix de k : silhouette plutôt que coude ---
for k in range(2, 11):
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Xs)
    print(f"k={k:2d}  inertie={km.inertia_:10.1f}  silhouette={silhouette_score(Xs, km.labels_):.3f}")

# Grand volume : MiniBatchKMeans, 10 à 50 fois plus rapide, résultat très proche
mbk = MiniBatchKMeans(n_clusters=8, batch_size=4096, n_init=5, random_state=0).fit(Xs)

# Interprétation : profil moyen de chaque groupe, en unités d'origine
import pandas as pd
profils = pd.DataFrame(X, columns=noms).assign(groupe=mbk.labels_).groupby('groupe').mean()
print(profils.round(2))
print(pd.Series(mbk.labels_).value_counts().sort_index())   # tailles des groupes
```

**Outils** — `scikit-learn.cluster` (`KMeans`, `MiniBatchKMeans`), `yellowbrick` pour les visualisations de sélection de $k$.

**Alternatives open-source**
- *Bibliothèques* : **HDBSCAN** ne demande pas $k$ et trouve des groupes de densités variables — le meilleur remplaçant par défaut ; **GaussianMixture** autorise des groupes ellipsoïdaux et donne des affectations souples ; **AgglomerativeClustering** fournit une hiérarchie complète ; **faiss** contient un k-means GPU capable de traiter des milliards de vecteurs ; **kmodes** pour des données purement catégorielles (k-modes, k-prototypes) ; **tslearn** pour du clustering de séries temporelles avec distance DTW.
- *Outils graphiques* : **Orange** enchaîne visuellement k-means, silhouette et projection, idéal pour comprendre ; **KNIME** pour l'intégrer dans un flux ; **TensorFlow Embedding Projector** pour visualiser les groupes dans un espace d'embeddings ; **Weka** avec son visualiseur de clusters.

**Astuces**
- Sans standardisation, k-means groupe selon la feature de plus grande amplitude. Erreur numéro un, résultat toujours plausible.
- `n_init=10` est indispensable ; le défaut a changé selon les versions de scikit-learn. Vérifier explicitement.
- La méthode du coude ne montre presque jamais de coude sur des données réelles. Utiliser la silhouette, et surtout l'interprétabilité métier des profils.
- Un groupe contenant 2 % des points est souvent un artefact ou un ensemble d'aberrations. Le regarder de près avant de conclure.
- k-means sur des données one-hot n'a pas de sens géométrique clair : la distance euclidienne entre modalités est arbitraire. Utiliser k-modes ou une distance adaptée.
- Ajouter le numéro de groupe comme feature catégorielle dans un modèle supervisé fonctionne étonnamment bien, pour un coût nul.

## Clustering hiérarchique
<!-- slug: clustering-hierarchique | difficulte: 2 | prereqs: kmeans -->

**En une phrase** — Construire un arbre de fusions successives entre les points, puis le couper à la hauteur qui donne le nombre de groupes souhaité.

**Explication** — La version ascendante (agglomérative) part de $n$ groupes d'un point et fusionne à chaque étape les deux groupes les plus proches, jusqu'à n'en avoir qu'un. Le résultat est un **dendrogramme** : un arbre dont la hauteur de chaque fusion indique la dissimilarité à laquelle elle s'est produite. On n'a donc pas besoin de fixer $k$ à l'avance — on regarde l'arbre et on coupe où les branches sont longues.

Tout dépend du **critère de liaison**. `ward` minimise l'augmentation de variance intra-groupe : il produit des groupes compacts et équilibrés, et c'est le meilleur défaut. `complete` (distance maximale) donne des groupes compacts mais sensibles aux aberrations. `average` est intermédiaire. `single` (distance minimale) peut suivre des formes allongées mais souffre de l'effet de chaînage : deux groupes distincts reliés par un pont de quelques points fusionnent.

Le coût est le point faible : $O(n^2)$ en mémoire pour la matrice de distances, $O(n^2 \log n)$ ou $O(n^3)$ en temps. Au-delà d'environ 20 000 points, la méthode devient impraticable sans échantillonnage ou pré-agrégation par k-means.

**Cas d'utilisation**
- Découverte exploratoire quand $k$ est vraiment inconnu : le dendrogramme est un outil de lecture, pas seulement de calcul.
- Taxonomies naturellement hiérarchiques : documents, gènes, produits.
- Regroupement de features corrélées avant sélection (clustering sur la matrice de corrélation).
- Petits jeux de données où l'interprétation compte plus que la vitesse.
- Mauvais choix au-delà de quelques dizaines de milliers de points.

**Algorithme**
```text
1. Calculer la matrice de distances entre tous les points (n², symétrique).
2. Chaque point est un groupe.
3. Répéter n-1 fois :
     a. Trouver les deux groupes les plus proches selon le critère de liaison.
     b. Les fusionner ; enregistrer (i, j, distance, taille) dans la matrice
        de liaison.
     c. Mettre à jour les distances (formule de Lance-Williams).
4. Le dendrogramme se lit dans la matrice de liaison.
5. Couper : fcluster(Z, t, criterion='maxclust' ou 'distance').
```

**Implémentation**
```python
import numpy as np, matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

Xs = StandardScaler().fit_transform(X)

# scipy : accès à la matrice de liaison, donc au dendrogramme
Z = linkage(Xs, method='ward')                # (n-1, 4)
fig, ax = plt.subplots(figsize=(11, 4))
dendrogram(Z, truncate_mode='lastp', p=30, ax=ax, color_threshold=None)
ax.set(xlabel='groupes', ylabel='distance de fusion', title='Dendrogramme (Ward)')

# Couper : par nombre de groupes, ou par seuil de distance
etiquettes = fcluster(Z, t=5, criterion='maxclust')
etiquettes_seuil = fcluster(Z, t=12.0, criterion='distance')

# Vérifier la qualité de l'arbre : corrélation cophénétique (> 0.75 = arbre fidèle)
from scipy.cluster.hierarchy import cophenet
c, _ = cophenet(Z, pdist(Xs))
print(f"corrélation cophénétique : {c:.3f}")

# Usage très utile : regrouper des FEATURES corrélées avant sélection
corr = np.corrcoef(Xs.T)
Zf = linkage(1 - np.abs(corr), method='average')
groupes_features = fcluster(Zf, t=0.3, criterion='distance')
# garder une seule feature représentative par groupe

# Grands volumes : contraintes de connectivité pour éviter la matrice n²
from sklearn.neighbors import kneighbors_graph
conn = kneighbors_graph(Xs, n_neighbors=10, include_self=False)
agg = AgglomerativeClustering(n_clusters=5, linkage='ward', connectivity=conn).fit(Xs)
```

**Outils** — `scipy.cluster.hierarchy` (dendrogramme), `scikit-learn` (`AgglomerativeClustering`).

**Alternatives open-source**
- *Bibliothèques* : **fastcluster** accélère fortement `linkage` et lève la limite pratique à ~100 000 points ; **HDBSCAN** est une méthode hiérarchique fondée sur la densité, sans $k$ et robuste au bruit — le successeur naturel ; **scipy** `optimal_leaf_ordering` améliore beaucoup la lisibilité du dendrogramme ; **seaborn** `clustermap` combine dendrogramme et heatmap en une figure.
- *Outils graphiques* : **seaborn clustermap** est le meilleur outil de lecture pour une matrice regroupée ; **Orange** propose Hierarchical Clustering avec dendrogramme interactif et sélection de branches à la souris ; **iTOL** et **Dendroscope** pour de gros arbres (contexte bio-informatique) ; **KNIME** avec son nœud dédié.

**Astuces**
- `ward` exige des distances euclidiennes ; l'utiliser avec une autre métrique n'a pas de sens mathématique et scipy ne l'empêche pas toujours.
- `single linkage` produit l'effet de chaînage : de longs groupes filiformes reliés par des ponts. Rarement ce qu'on veut, sauf pour détecter des formes allongées.
- Sur plus de 1 000 points, un dendrogramme complet est illisible. Utiliser `truncate_mode='lastp'`.
- La corrélation cophénétique mesure si l'arbre respecte les distances originales. En dessous de 0,7, changer de critère de liaison.
- Pré-agréger par k-means à 1 000 centres puis appliquer un clustering hiérarchique sur ces centres : on récupère le dendrogramme sur des millions de points.
- Le regroupement de features corrélées avant sélection est l'usage le plus sous-estimé de cette méthode, et l'un des plus utiles.

## DBSCAN et HDBSCAN
<!-- slug: dbscan | difficulte: 3 | prereqs: kmeans -->

**En une phrase** — Grouper les points par densité : un groupe est une région où les points sont serrés, et ce qui est isolé est déclaré bruit.

**Explication** — DBSCAN repose sur deux paramètres : un rayon `eps` et un nombre minimum de voisins `min_samples`. Un point est **cœur** s'il a au moins `min_samples` voisins dans son rayon. Les points cœurs directement ou transitivement voisins forment un groupe ; les points non cœurs mais voisins d'un cœur sont des points **frontière** ; le reste est du **bruit**, étiqueté -1. Trois propriétés distinguent radicalement cette méthode de k-means : le nombre de groupes est découvert, les formes peuvent être arbitrairement allongées ou concaves, et les aberrations sont explicitement identifiées au lieu d'être forcées dans un groupe.

La difficulté est le réglage de `eps`, qui suppose une densité homogène. Sur des données comportant un groupe dense et un groupe diffus, aucune valeur unique ne convient : trop grande, les deux fusionnent ; trop petite, le groupe diffus devient du bruit. C'est exactement ce que **HDBSCAN** résout, en construisant une hiérarchie sur toutes les valeurs de `eps` et en extrayant les groupes les plus **stables** à travers cette hiérarchie. Il ne demande que `min_cluster_size`, paramètre bien plus intuitif.

En pratique, en 2026, HDBSCAN devrait être le défaut et DBSCAN le cas particulier qu'on comprend d'abord.

**Cas d'utilisation**
- Groupes de formes non convexes : trajectoires GPS, contours géographiques, structures spatiales.
- Détection d'anomalies comme produit dérivé : le bruit est le résultat intéressant.
- Nombre de groupes réellement inconnu et variable dans le temps.
- Clustering d'embeddings (documents, images) — usage aujourd'hui très courant, associé à UMAP.
- Mauvais choix en très grande dimension sans réduction préalable : la densité perd son sens (malédiction de la dimension).

**Algorithme**
```text
DBSCAN(X, eps, min_samples) :
1. Pour chaque point, compter les voisins à distance <= eps.
2. Marquer comme CŒUR ceux qui en ont au moins min_samples.
3. etiquette = -1 partout ; c = 0
4. Pour chaque point cœur non visité :
     a. c += 1 ; l'affecter au groupe c.
     b. Propager par une file : ajouter ses voisins au groupe ;
        si un voisin est lui-même cœur, empiler ses propres voisins.
5. Les points non cœurs atteints par propagation = frontière.
   Les points jamais atteints = bruit (-1).

Choix de eps : tracer la distance au k-ième voisin (k = min_samples),
triée par ordre croissant. Le coude de cette courbe donne eps.
```

**Implémentation**
```python
import numpy as np, matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

Xs = StandardScaler().fit_transform(X)

# 1. Choisir eps par la courbe des k-distances
k = 5
d, _ = NearestNeighbors(n_neighbors=k).fit(Xs).kneighbors(Xs)
dk = np.sort(d[:, -1])
fig, ax = plt.subplots()
ax.plot(dk)
ax.set(xlabel='points triés', ylabel=f'distance au {k}e voisin',
       title='Le coude donne eps')

db = DBSCAN(eps=0.6, min_samples=k, n_jobs=-1).fit(Xs)
lab = db.labels_
print(f"groupes : {len(set(lab)) - (1 in [-1] and -1 in lab)}  "
      f"bruit : {(lab == -1).mean():.1%}")

# 2. HDBSCAN : un seul paramètre, densités variables gérées
h = HDBSCAN(min_cluster_size=25, min_samples=5).fit(Xs)
print(f"groupes : {h.labels_.max() + 1}  bruit : {(h.labels_ == -1).mean():.1%}")
# probabilities_ donne la force d'appartenance de chaque point
incertains = h.probabilities_ < 0.5

# 3. Combinaison standard pour des embeddings : UMAP puis HDBSCAN
# pip install umap-learn
import umap
Z = umap.UMAP(n_neighbors=15, min_dist=0.0, n_components=5, random_state=0).fit_transform(embeddings)
groupes = HDBSCAN(min_cluster_size=30).fit_predict(Z)

# 4. Le bruit comme détecteur d'anomalies
anomalies = X[h.labels_ == -1]
```

**Outils** — `scikit-learn.cluster` (`DBSCAN`, `HDBSCAN` depuis la version 1.3), `pip install umap-learn`.

**Alternatives open-source**
- *Bibliothèques* : **hdbscan** (paquet original de McInnes) offre plus d'options que la version scikit-learn, dont l'extraction souple et `approximate_predict` pour de nouveaux points ; **OPTICS** (dans scikit-learn) produit un diagramme d'atteignabilité qui permet de lire plusieurs valeurs de `eps` d'un coup ; **cuML** DBSCAN sur GPU ; **BERTopic** enchaîne embeddings, UMAP et HDBSCAN pour du *topic modeling*, et constitue le meilleur exemple d'usage moderne.
- *Outils graphiques* : **Orange** dispose de DBSCAN avec visualisation immédiate en projection ; **QGIS** applique DBSCAN sur des données géographiques avec rendu cartographique ; **hdbscan** fournit `condensed_tree_.plot()`, très éclairant sur la hiérarchie de densité.

**Astuces**
- Standardiser d'abord : `eps` est une distance, donc dépendante des échelles.
- Préférer HDBSCAN par défaut. `min_cluster_size` est un paramètre qu'on peut justifier métier (« un segment de moins de 30 clients ne m'intéresse pas »), contrairement à `eps`.
- Une proportion de bruit supérieure à 30 % signale un `eps` trop petit — ou des données réellement sans structure de densité.
- DBSCAN ne possède pas de méthode `predict` : il ne sait pas classer un nouveau point. Utiliser `hdbscan.approximate_predict` si c'est nécessaire en production.
- En dimension supérieure à ~10, réduire d'abord (UMAP ou ACP). Sur des embeddings à 768 dimensions, appliquer HDBSCAN directement donne des résultats médiocres.
- Le score de silhouette est mal défini en présence de bruit : soit exclure les points -1, soit utiliser l'indice DBCV, conçu pour les méthodes de densité.

## Modèles de mélange gaussien
<!-- slug: melange-gaussien | difficulte: 4 | prereqs: kmeans, probabilites-bayes -->

**En une phrase** — Supposer que les données sont issues d'un mélange de plusieurs gaussiennes et estimer leurs paramètres, ce qui donne des groupes ellipsoïdaux avec des appartenances probabilistes.

**Explication** — Le modèle postule $p(x) = \sum_k \pi_k\, \mathcal{N}(x \mid \mu_k, \Sigma_k)$ : chaque point est tiré d'une des $K$ gaussiennes, avec des poids $\pi_k$. On estime $\pi, \mu, \Sigma$ par l'algorithme **EM** (espérance-maximisation). Étape E : calculer la responsabilité $\gamma_{ik}$, probabilité que le point $i$ vienne de la composante $k$, par la formule de Bayes. Étape M : recalculer chaque $\mu_k$ et $\Sigma_k$ comme moyenne et covariance pondérées par ces responsabilités. On itère ; la log-vraisemblance augmente à chaque tour.

C'est une généralisation de k-means à deux titres. Les groupes peuvent être **ellipsoïdaux** et orientés, grâce à la matrice de covariance, là où k-means ne connaît que des sphères. Et l'affectation est **souple** : un point à la frontière reçoit 55 % / 45 % au lieu d'un choix arbitraire. En faisant tendre les covariances vers zéro, on retrouve exactement k-means.

Avantage décisif sur k-means : le modèle étant génératif et probabiliste, on peut comparer différents $K$ par un critère d'information — **BIC** ou **AIC** — qui pénalise le nombre de paramètres. Cela fournit un critère de choix de $K$ **objectif**, ce que ni k-means ni DBSCAN n'offrent.

**Cas d'utilisation**
- Groupes qui se recouvrent, où une affectation souple est plus honnête qu'une partition dure.
- Estimation de densité : évaluer la vraisemblance d'un nouveau point, donc détecter les anomalies.
- Segmentation où les groupes ont des formes et des orientations différentes.
- Choix de $K$ par BIC quand il faut une justification quantitative.
- Mauvais choix si les groupes ne sont pas approximativement gaussiens (formes en croissant, densités très irrégulières) : HDBSCAN convient mieux.

**Algorithme**
```text
Entrée : X (n, d), K, type de covariance
1. Initialiser par k-means (bien meilleur qu'un tirage aléatoire).
2. Répéter jusqu'à convergence de la log-vraisemblance :
   E : pour tout i, k :
         gamma[i,k] = pi_k N(x_i | mu_k, Sigma_k) / somme_j (...)
   M : N_k = somme_i gamma[i,k]
         pi_k    = N_k / n
         mu_k    = (1/N_k) somme_i gamma[i,k] x_i
         Sigma_k = (1/N_k) somme_i gamma[i,k] (x_i - mu_k)(x_i - mu_k)'
                   + reg_covar * I         (régularisation indispensable)
3. Choix de K : minimiser BIC = -2 logL + p log(n), p = nb de paramètres.
4. Calculs en log avec logsumexp pour éviter les dépassements.
```

**Implémentation**
```python
import numpy as np
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler

Xs = StandardScaler().fit_transform(X)

# Choix de K et du type de covariance par BIC : critère objectif
resultats = []
for K in range(1, 11):
    for cov in ['full', 'tied', 'diag', 'spherical']:
        gm = GaussianMixture(n_components=K, covariance_type=cov,
                             n_init=5, reg_covar=1e-6, random_state=0).fit(Xs)
        resultats.append((gm.bic(Xs), K, cov, gm))
resultats.sort()
bic, K, cov, meilleur = resultats[0]
print(f"BIC minimal : K={K}, covariance={cov}, BIC={bic:.0f}")

# Affectation souple : la nuance que k-means ne donne pas
proba = meilleur.predict_proba(Xs)              # (n, K)
dur = proba.argmax(1)
incertains = proba.max(1) < 0.6                 # points ambigus, à examiner
print(f"{incertains.mean():.1%} de points ambigus")

# Détection d'anomalies : la vraisemblance est directement exploitable
logp = meilleur.score_samples(Xs)
seuil = np.percentile(logp, 1)
anomalies = Xs[logp < seuil]

# Variante bayésienne : élague automatiquement les composantes inutiles
bgm = BayesianGaussianMixture(n_components=20, weight_concentration_prior=0.01,
                              n_init=5, random_state=0).fit(Xs)
print("composantes réellement utilisées :", (bgm.weights_ > 0.01).sum())

# --- EM from scratch, cas diagonal (palier 2) ---
from scipy.special import logsumexp

def em_diag(X, K, iters=100, graine=0):
    rng = np.random.default_rng(graine)
    n, d = X.shape
    mu = X[rng.choice(n, K, replace=False)]
    var = np.tile(X.var(0), (K, 1))
    pi = np.full(K, 1 / K)
    for _ in range(iters):
        # E : log-densités puis normalisation stable
        lp = -0.5 * (((X[:, None] - mu) ** 2 / var).sum(2)
                     + np.log(2 * np.pi * var).sum(1)) + np.log(pi)
        ll = logsumexp(lp, axis=1, keepdims=True)
        g = np.exp(lp - ll)                       # (n, K) responsabilités
        # M
        Nk = g.sum(0)
        pi = Nk / n
        mu = (g.T @ X) / Nk[:, None]
        var = (g.T @ (X ** 2)) / Nk[:, None] - mu ** 2 + 1e-6
    return pi, mu, var, ll.sum()
```

**Outils** — `scikit-learn.mixture` (`GaussianMixture`, `BayesianGaussianMixture`), `scipy.special.logsumexp`.

**Alternatives open-source**
- *Bibliothèques* : **pomegranate** permet des mélanges de distributions quelconques, pas seulement gaussiennes, et des modèles de Markov cachés ; **PyMC** et **NumPyro** pour un traitement pleinement bayésien avec incertitude sur tous les paramètres ; **hmmlearn** pour la version séquentielle (les états cachés d'une série temporelle) ; **scikit-learn** `KernelDensity` si seule l'estimation de densité intéresse.
- *Outils graphiques* : la galerie de démonstration **scikit-learn** sur les types de covariance est la meilleure illustration de l'effet de `covariance_type` ; **Orange** pour combiner mélange et projection ; **ArviZ** pour visualiser les postérieures d'un modèle bayésien de mélange.

**Astuces**
- Toujours initialiser par k-means (`init_params='kmeans'`, le défaut) : une initialisation aléatoire converge souvent vers une solution dégénérée.
- `reg_covar` évite les matrices de covariance singulières. Sans lui, la log-vraisemblance peut diverger vers l'infini en collapsant une gaussienne sur un point unique.
- `covariance_type='full'` coûte $K d(d+1)/2$ paramètres : impraticable au-delà d'une cinquantaine de dimensions. `'diag'` est le compromis usuel.
- Le BIC choisit le modèle le plus vraisemblable, pas le plus interprétable. Il tend à surestimer $K$ sur de gros échantillons — regarder aussi la courbe, pas seulement le minimum.
- Comparer les BIC entre modèles n'a de sens que sur exactement les mêmes données et le même prétraitement.
- Les mélanges gaussiens sont un excellent détecteur d'anomalies sur des données à peu près normales, et un mauvais détecteur sinon. Vérifier la normalité approximative des marges avant.

## Analyse en composantes principales
<!-- slug: pca | difficulte: 3 | prereqs: algebre-lineaire, mise-a-l-echelle -->

**En une phrase** — Trouver les directions orthogonales de plus grande variance et y projeter les données, pour réduire la dimension en perdant le minimum d'information.

**Explication** — On centre les données, on calcule la matrice de covariance $C = \frac{1}{n}X^\top X$, et on prend ses vecteurs propres. Le premier est la direction de plus grande variance ; le second est la direction de plus grande variance parmi celles orthogonales au premier ; et ainsi de suite. Projeter sur les $p$ premiers donne la meilleure approximation de rang $p$ au sens des moindres carrés — c'est un théorème (Eckart-Young), pas une heuristique.

En pratique, on passe par la **SVD** $X = U\Sigma V^\top$ plutôt que par la diagonalisation de la covariance : plus stable numériquement, et cela évite de former une matrice $d \times d$. Les colonnes de $V$ sont les axes principaux, et $\sigma_i^2/\sum\sigma_j^2$ donne la proportion de variance expliquée par l'axe $i$. La courbe cumulée de ces proportions est l'outil de décision : garder assez d'axes pour atteindre 90 ou 95 %.

Deux limites structurelles. L'ACP est **linéaire** : elle ne dépliera pas une structure courbe (spirale, croissant). Et elle est **non supervisée** : les axes de plus grande variance ne sont pas nécessairement les plus prédictifs de la cible — une feature peu variable peut être décisive. Ce n'est donc pas un outil de sélection de features déguisé.

**Cas d'utilisation**
- Décorréler et compresser des features numériques nombreuses et redondantes (capteurs, spectres, pixels).
- Accélérer un modèle sensible à la dimension : kNN, SVM à noyau, k-means.
- Visualiser en 2D pour une première inspection — mais t-SNE ou UMAP sont supérieurs pour cet usage précis.
- Débruiter : reconstruire à partir des premiers axes seulement supprime le bruit de faible variance.
- Détection d'anomalies par erreur de reconstruction.
- Mauvais choix quand l'interprétabilité des features d'origine est requise : une composante principale est une combinaison linéaire de tout, difficile à expliquer.

**Algorithme**
```text
1. Centrer les colonnes (obligatoire). Réduire aussi (standardiser) si les
   unités diffèrent — sinon la variable en euros écrase la variable en années.
2. SVD : X_centré = U S V'.
3. Axes principaux = lignes de V' ; variance expliquée = S² / somme(S²).
4. Projection sur p composantes : Z = X_centré @ V[:, :p]  -> (n, p).
5. Choisir p : cumul de variance expliquée >= 90-95 %, ou coude du scree plot,
   ou par validation croisée sur la performance du modèle en aval.
6. Reconstruction : X_hat = Z @ V[:, :p].T + moyenne.
   Erreur de reconstruction = anomalie.
```

**Implémentation**
```python
import numpy as np, matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA, TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# --- from scratch par SVD (palier 2) ---
def acp(X, p):
    mu = X.mean(0)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var_expl = S ** 2 / (S ** 2).sum()
    Z = Xc @ Vt[:p].T
    return Z, Vt[:p], var_expl, mu

# --- version bibliothèque, dans un pipeline (la standardisation compte) ---
pca = make_pipeline(StandardScaler(), PCA(n_components=0.95, random_state=0)).fit(X_tr)
print("composantes retenues :", pca[-1].n_components_, "/", X_tr.shape[1])

# Scree plot et cumul : la lecture qui décide de p
p = pca[-1]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
a1.bar(range(1, len(p.explained_variance_ratio_) + 1), p.explained_variance_ratio_)
a1.set(xlabel='composante', ylabel='variance expliquée', title='Scree plot')
cum = np.cumsum(p.explained_variance_ratio_)
a2.plot(range(1, len(cum) + 1), cum, 'o-')
a2.axhline(0.95, ls='--', c='gray')
a2.set(xlabel='composantes', ylabel='cumul', title='Variance cumulée')

# Interprétation : quelles features pèsent sur chaque axe ?
import pandas as pd
charges = pd.DataFrame(p.components_[:3].T, index=noms, columns=['CP1', 'CP2', 'CP3'])
print(charges.abs().sort_values('CP1', ascending=False).head(8).round(3))

# Détection d'anomalies par erreur de reconstruction
Z = pca.transform(X_te)
X_rec = pca[-1].inverse_transform(Z)
err = ((pca[0].transform(X_te) - X_rec) ** 2).sum(1)
anomalies = X_te[err > np.percentile(err, 99)]

# Grands volumes ou données creuses
ipca = IncrementalPCA(n_components=50, batch_size=1000)      # par lots, mémoire bornée
tsvd = TruncatedSVD(n_components=100)                        # matrices creuses (TF-IDF)
```

**Outils** — `scikit-learn.decomposition` (`PCA`, `IncrementalPCA`, `TruncatedSVD`, `KernelPCA`).

**Alternatives open-source**
- *Bibliothèques* : **UMAP** et **t-SNE** pour la visualisation, nettement supérieurs à l'ACP sur ce point précis ; **KernelPCA** pour une version non linéaire par astuce de noyau ; **NMF** (factorisation en matrices non négatives) produit des composantes additives et interprétables, adaptées au texte et aux spectres ; **FactorAnalysis** quand on modélise explicitement un bruit par variable ; **prince** implémente ACM et AFC pour données catégorielles et tableaux de contingence ; **TruncatedSVD** est l'équivalent pour matrices creuses (analyse sémantique latente).
- *Outils graphiques* : **Orange** propose ACP avec scree plot et biplot interactif, excellent pour comprendre ; **TensorFlow Embedding Projector** fait de l'ACP et du t-SNE en 3D dans le navigateur ; **FactoMineR + Factoshiny** (R) reste la référence pour l'analyse factorielle avec interprétation détaillée ; **JASP** pour une ACP avec sorties statistiques complètes.

**Astuces**
- Standardiser avant l'ACP dès que les unités diffèrent. Sans cela, une variable en euros domine entièrement les axes. C'est l'erreur la plus fréquente.
- L'ACP doit être **dans le pipeline** : ajustée sur l'entraînement seul. L'ajuster sur tout le dataset est une fuite.
- Les signes des composantes sont arbitraires : un axe peut s'inverser d'une exécution à l'autre sans que rien ne change. Ne pas interpréter le signe absolu.
- `n_components=0.95` laisse scikit-learn choisir le nombre d'axes pour atteindre 95 % de variance. Plus lisible qu'un nombre magique.
- Une ACP suivie d'une régression n'améliore la performance que si les features sont très redondantes. Souvent, elle la dégrade légèrement tout en supprimant l'interprétabilité — vérifier par validation croisée avant de l'adopter.
- L'erreur de reconstruction est un détecteur d'anomalies simple et robuste, très utilisé en surveillance industrielle.

## t-SNE et UMAP
<!-- slug: tsne-umap | difficulte: 3 | prereqs: pca -->

**En une phrase** — Projeter des données de grande dimension en deux dimensions en préservant le voisinage local, pour *voir* la structure des groupes.

**Explication** — Ces méthodes ne cherchent pas à préserver les distances globales mais les **voisinages**. t-SNE convertit les distances en probabilités de voisinage dans l'espace d'origine, fait de même dans l'espace 2D avec une loi de Student à queue lourde, et minimise la divergence de Kullback-Leibler entre les deux par descente de gradient. La queue lourde de la loi de Student est l'astuce clé : elle laisse de la place aux groupes en 2D, ce qui évite l'entassement au centre.

UMAP part d'une construction différente — un graphe de voisinage flou dont on cherche une représentation basse dimension par optimisation d'entropie croisée — mais l'esprit est le même. En pratique UMAP est **10 à 100 fois plus rapide**, préserve mieux la structure globale, sait projeter de nouveaux points (`transform`), et supporte plusieurs dimensions de sortie. C'est le choix par défaut aujourd'hui.

Un avertissement doit accompagner tout usage : **ces projections déforment**. Les distances entre groupes éloignés ne sont pas interprétables, les tailles relatives des amas ne signifient rien, et des groupes peuvent apparaître sur des données purement aléatoires si les hyperparamètres sont mal choisis. `perplexity` (t-SNE) et `n_neighbors` (UMAP) déterminent l'échelle observée : petits, ils montrent une structure locale fragmentée ; grands, une structure globale lissée. Toujours regarder plusieurs valeurs avant de conclure.

**Cas d'utilisation**
- Visualiser des embeddings : documents, images, cellules, utilisateurs. C'est l'usage principal.
- Contrôle qualité d'un modèle : les classes se séparent-elles dans l'espace latent ?
- Étape de réduction avant HDBSCAN pour du clustering d'embeddings (UMAP seulement, en 5 à 15 dimensions).
- Détecter visuellement des lots aberrants, des doublons, une fuite (un groupe qui correspond exactement à une classe cible).
- Mauvais choix comme réduction de dimension pour un modèle supervisé : instable, non interprétable, et t-SNE ne sait pas transformer de nouvelles données.

**Algorithme**
```text
1. Réduire d'abord par ACP à ~50 dimensions : accélère beaucoup et débruite.
2. UMAP :
     a. Construire le graphe des k plus proches voisins (n_neighbors).
     b. Pondérer les arêtes par une distance floue locale, adaptée à la
        densité de chaque point.
     c. Optimiser une disposition 2D par descente stochastique, en
        rapprochant les voisins et repoussant des paires tirées au hasard.
3. Paramètres à balayer :
     n_neighbors : 5 (structure locale) -> 50 (structure globale)
     min_dist    : 0.0 (amas compacts, pour clustering) -> 0.5 (lisible)
4. Fixer random_state, colorer par une variable connue, et JAMAIS
   interpréter les distances entre amas éloignés.
```

**Implémentation**
```python
import numpy as np, matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Étape préalable systématique : ACP à 50 dimensions
X50 = PCA(n_components=50, random_state=0).fit_transform(Xs)

# t-SNE : plusieurs perplexités, car la structure vue en dépend
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
for ax, perp in zip(axes, [5, 30, 100]):
    Z = TSNE(n_components=2, perplexity=perp, init='pca',
             learning_rate='auto', random_state=0).fit_transform(X50)
    ax.scatter(Z[:, 0], Z[:, 1], c=y, s=5, cmap='viridis', alpha=0.7)
    ax.set(title=f'perplexity={perp}', xticks=[], yticks=[])

# UMAP : plus rapide, transforme de nouveaux points, meilleure structure globale
# pip install umap-learn
import umap
red = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2,
                metric='euclidean', random_state=0).fit(X50)
Z = red.embedding_
Z_nouveau = red.transform(PCA_deja_ajustee.transform(X_nouveau))   # impossible en t-SNE

# UMAP supervisé : sépare les classes connues, très utile pour un diagnostic
Z_sup = umap.UMAP(n_neighbors=15, min_dist=0.0, random_state=0).fit_transform(X50, y=y)

# Pour du clustering : sortie en 5-15 dimensions, min_dist=0
Z_clust = umap.UMAP(n_neighbors=15, min_dist=0.0, n_components=8, random_state=0).fit_transform(X50)
from sklearn.cluster import HDBSCAN
groupes = HDBSCAN(min_cluster_size=30).fit_predict(Z_clust)

# Contrôle de fiabilité : la projection préserve-t-elle le voisinage ?
from sklearn.neighbors import NearestNeighbors
def fidelite_voisinage(X, Z, k=15):
    vx = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X, return_distance=False)[:, 1:]
    vz = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z, return_distance=False)[:, 1:]
    return np.mean([len(set(a) & set(b)) / k for a, b in zip(vx, vz)])
print(f"voisins préservés : {fidelite_voisinage(X50, Z):.1%}")
```

**Outils** — `scikit-learn.manifold` (`TSNE`), `pip install umap-learn`.

**Alternatives open-source**
- *Bibliothèques* : **openTSNE** est une implémentation de t-SNE bien plus rapide, avec possibilité de projeter de nouveaux points ; **PaCMAP** et **TriMap** préservent mieux la structure globale qu'UMAP et sont moins sujets aux faux amas ; **PHATE** est conçu pour des trajectoires continues (données biologiques) ; **cuML** propose t-SNE et UMAP sur GPU ; **Ivis** utilise un réseau siamois et généralise bien.
- *Outils graphiques* : **TensorFlow Embedding Projector** est le meilleur explorateur interactif (ACP, t-SNE, UMAP, recherche de voisins, en 3D dans le navigateur) ; **Nomic Atlas** (partiellement libre) visualise des millions de points avec étiquetage automatique ; **BERTopic** fournit des visualisations de topics fondées sur UMAP ; **Orange** avec ses blocs t-SNE et MDS.

**Astuces**
- Ne jamais tirer de conclusion quantitative d'une projection. Les distances entre amas, leurs tailles et leurs densités relatives sont des artefacts.
- Toujours regarder au moins trois valeurs de `perplexity` ou `n_neighbors`. Un amas qui disparaît quand le paramètre change n'existe probablement pas.
- Passer par une ACP à 50 dimensions avant : dix fois plus rapide, et souvent une projection plus propre car débruitée.
- t-SNE ne possède pas de `transform` : impossible de projeter un nouveau point sans tout recalculer. UMAP ou openTSNE si c'est nécessaire.
- Pour du clustering, `min_dist=0.0` et une sortie en 5 à 15 dimensions (pas 2) donnent de bien meilleurs résultats. La 2D est pour l'œil, pas pour l'algorithme.
- Colorer la projection par une variable *non* utilisée pour la construire est le meilleur test : si les couleurs se séparent, la structure est réelle.
- UMAP en mode supervisé sépare artificiellement les classes. Superbe pour une présentation, trompeur comme diagnostic — ne pas confondre les deux usages.

## Détection d'anomalies
<!-- slug: detection-anomalies | difficulte: 3 | prereqs: pca, melange-gaussien -->

**En une phrase** — Identifier les observations qui ne ressemblent pas au reste, sans disposer d'exemples étiquetés de ce qu'on cherche.

**Explication** — Trois cadres distincts, souvent confondus. La **détection d'aberrations** (*outlier detection*) travaille sur un jeu contenant déjà des anomalies et cherche à les repérer. La **détection de nouveauté** (*novelty detection*) apprend sur un jeu propre et signale ce qui s'en écarte. Et la **classification déséquilibrée** dispose d'étiquettes, même rares — dans ce cas, un modèle supervisé bat presque toujours une méthode non supervisée, et il faut le préférer.

Les familles d'approches recoupent les intuitions de ce module. Par **densité** : un point dans une région peu dense est anormal (LOF, mélanges gaussiens). Par **isolement** : `IsolationForest` construit des arbres de coupes aléatoires, et un point isolable en peu de coupes est anormal — approche remarquablement efficace, en $O(n\log n)$, et l'un des rares algorithmes qui fonctionne bien en grande dimension. Par **reconstruction** : ACP ou auto-encodeur reconstruisent mal ce qui sort du domaine appris. Par **frontière** : `OneClassSVM` enveloppe les données normales.

La difficulté centrale est l'**évaluation**. Sans étiquettes, aucune métrique n'est disponible, et le paramètre `contamination` (proportion supposée d'anomalies) est fixé à la main. En pratique, on produit un **score** continu, on inspecte manuellement le haut du classement, et on construit progressivement un jeu étiqueté — qui permettra ensuite de passer en supervisé.

**Cas d'utilisation**
- Surveillance industrielle : panne de capteur, dérive de machine.
- Fraude et sécurité, en amont d'un système supervisé, quand les étiquettes manquent encore.
- Contrôle qualité de données : détecter des lignes corrompues avant l'entraînement d'un modèle.
- Surveillance de production : détecter que les données d'entrée du modèle ont changé.
- Mauvais choix si l'on possède plus d'une centaine d'exemples étiquetés d'anomalies : un classifieur supervisé sera meilleur.

**Algorithme**
```text
1. Définir « anormal » avec le métier. Anormal par rapport à quoi : à
   l'ensemble, au même équipement, à la même période ? Cette question
   détermine tout le reste.
2. Standardiser. Réduire la dimension si d > 50 (ACP).
3. Appliquer plusieurs détecteurs et comparer leurs classements :
     IsolationForest       -> défaut robuste, rapide, grande dimension
     LocalOutlierFactor    -> densités variables, structure locale
     erreur de reconstruction ACP -> corrélations linéaires rompues
     GaussianMixture score_samples -> données approximativement normales
4. Récupérer les SCORES, pas les étiquettes binaires. Trier.
5. Inspecter manuellement les 50 premiers. Étiqueter. Recommencer.
6. Consensus : un point signalé par 3 détecteurs sur 4 est un bon candidat.
7. Une fois ~100 anomalies étiquetées, passer en supervisé.
```

**Implémentation**
```python
import numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

Xs = StandardScaler().fit_transform(X)

# 1. IsolationForest : le meilleur point de départ
iso = IsolationForest(n_estimators=300, contamination='auto',
                      random_state=0, n_jobs=-1).fit(Xs)
s_iso = -iso.score_samples(Xs)            # plus grand = plus anormal

# 2. LOF : densité locale. novelty=False pour de la détection d'aberrations
lof = LocalOutlierFactor(n_neighbors=20, contamination='auto')
lof.fit_predict(Xs)
s_lof = -lof.negative_outlier_factor_

# 3. Erreur de reconstruction ACP
p = PCA(n_components=0.90, random_state=0).fit(Xs)
s_pca = ((Xs - p.inverse_transform(p.transform(Xs))) ** 2).sum(1)

# 4. Consensus par rang : robuste aux échelles incomparables des scores
scores = pd.DataFrame({'iso': s_iso, 'lof': s_lof, 'pca': s_pca})
rangs = scores.rank(pct=True)
scores['consensus'] = rangs.mean(1)
suspects = scores.sort_values('consensus', ascending=False).head(50)
print(suspects.round(3))
# -> inspecter ces 50 lignes à la main, c'est l'étape qui produit de la valeur

# 5. Détection de NOUVEAUTÉ : apprendre sur du propre, surveiller le flux
iso_ref = IsolationForest(n_estimators=300, random_state=0).fit(X_periode_normale)
alerte = -iso_ref.score_samples(X_flux) > seuil_calibre

# 6. Séries temporelles : anomalie = écart au comportement attendu
resid = y_reel - modele_prevision.predict(X_futur)
z = (resid - resid.mean()) / resid.std()
anomalies_temporelles = np.abs(z) > 3
```

**Outils** — `scikit-learn` (`IsolationForest`, `LocalOutlierFactor`, `OneClassSVM`, `EllipticEnvelope`), `pip install pyod`.

**Alternatives open-source**
- *Bibliothèques* : **PyOD** rassemble plus de 40 détecteurs avec une API unifiée et des méthodes d'ensemble — la référence du domaine ; **ADBench** fournit un banc d'essai comparatif sur 57 jeux de données, très utile pour choisir ; **alibi-detect** (Seldon) est orienté production, avec détection de dérive et de contradiction ; **PySAD** pour la détection en flux continu ; **Merlion** (Salesforce) et **Kats** (Meta) pour les anomalies de séries temporelles ; **darts** intègre des détecteurs fondés sur la prévision.
- *Outils graphiques* : **Evidently AI** produit des rapports de dérive et d'anomalie prêts à l'emploi ; **Grafana** avec des seuils et alertes sur séries temporelles ; **Orange** dispose de blocs Outlier Detection ; **Elastic Stack** intègre une détection d'anomalies dans ses tableaux de bord de journaux.

**Astuces**
- Ne jamais fixer `contamination` au hasard. Utiliser `'auto'`, récupérer le score continu, et laisser le seuil au métier — « je peux inspecter 20 alertes par jour » est un critère bien plus solide qu'un pourcentage inventé.
- `IsolationForest` est le meilleur premier essai : rapide, peu de réglage, robuste en grande dimension. Commencer par lui systématiquement.
- Le consensus par **rangs** entre plusieurs détecteurs est bien plus fiable qu'un détecteur unique, car les scores bruts ne sont pas comparables entre méthodes.
- Une anomalie détectée n'est pas forcément un problème métier : ce peut être une valeur légitimement rare. L'inspection manuelle des premiers cas n'est pas une étape facultative.
- Standardiser d'abord, et surtout traiter les manquants : la plupart des détecteurs les refusent, et une ligne à moitié vide sera systématiquement signalée pour de mauvaises raisons.
- En grande dimension, LOF et OneClassSVM se dégradent fortement. Réduire par ACP à 20-50 dimensions, ou s'en tenir à `IsolationForest`.
- Sur des séries temporelles, la bonne approche est presque toujours indirecte : prévoir, puis considérer un résidu anormalement grand comme une anomalie.

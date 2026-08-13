---
module: ml-supervise
titre: ML supervisé classique
ordre: 2
resume: 80 % des problèmes réels sur données tabulaires se règlent avec ce module. Le deep learning ne les bat pas.
---

## Régression linéaire
<!-- slug: regression-lineaire | difficulte: 1 | prereqs: algebre-lineaire, descente-de-gradient -->

**En une phrase** — Prédire une valeur continue comme une somme pondérée des features, en choisissant les poids qui minimisent l'erreur quadratique.

**Explication** — Le modèle est $\hat{y} = w_1x_1 + \dots + w_dx_d + b$. On cherche les $w$ minimisant $\frac{1}{n}\sum(\hat{y}_i - y_i)^2$. Ce problème est convexe et possède une solution analytique, l'équation normale $w = (X^\top X)^{-1}X^\top y$, qui est géométriquement la **projection orthogonale** de $y$ sur l'espace engendré par les colonnes de $X$.

L'intérêt de ce modèle n'est pas sa performance mais son **interprétabilité** : $w_j$ se lit comme « si $x_j$ augmente d'une unité, toutes choses égales, $\hat{y}$ augmente de $w_j$ ». Cette lecture n'est valide que si les features sont peu corrélées entre elles ; sinon les coefficients deviennent instables et changent de signe d'un échantillon à l'autre (*multicolinéarité*).

C'est aussi la **baseline obligatoire**. Un modèle sophistiqué qui ne bat pas une régression linéaire sur des features raisonnables signale un problème de données, pas un besoin de plus de couches.

**Cas d'utilisation**
- Prix, durée, consommation, quantité : toute cible continue et à peu près monotone en ses features.
- Point de référence à battre avant d'essayer autre chose.
- Contexte où il faut *expliquer* la prédiction (crédit, santé, réglementaire).
- Mauvais choix si la relation est fortement non linéaire ou avec de nombreuses interactions, sauf à créer ces features explicitement.

**Algorithme**
```text
Entrée : X (n, d), y (n)
1. Ajouter une colonne de 1 pour l'ordonnée à l'origine (ou centrer y et X).
2. Standardiser les colonnes de X si on compare les coefficients entre eux.
3. Résoudre (X'X) w = X'y — via lstsq, jamais par inversion explicite.
   Ou : descente de gradient si n ou d est très grand.
4. Prédire : y_hat = X @ w.
5. Diagnostiquer : R², RMSE, puis TRACER les résidus contre les prédictions.
```

**Implémentation**
```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# --- from scratch (palier 2) ---
class RegressionLineaire:
    def fit(self, X, y):
        X1 = np.c_[np.ones(len(X)), X]              # colonne de biais
        self.coef_, *_ = np.linalg.lstsq(X1, y, rcond=None)
        return self

    def predict(self, X):
        return np.c_[np.ones(len(X)), X] @ self.coef_

# --- version bibliothèque ---
modele = LinearRegression().fit(X_train, y_train)
pred = modele.predict(X_test)
print(f"RMSE={mean_squared_error(y_test, pred) ** 0.5:.3f}  R²={r2_score(y_test, pred):.3f}")

# Lecture des coefficients : n'a de sens QUE sur features standardisées
for nom, c in sorted(zip(noms_features, modele.coef_), key=lambda t: -abs(t[1])):
    print(f"{nom:20s} {c:+.3f}")

# Détection de multicolinéarité : VIF > 10 = coefficients non interprétables
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = [variance_inflation_factor(X_train, i) for i in range(X_train.shape[1])]
```

**Outils** — `scikit-learn` (`LinearRegression`), `statsmodels` (`OLS`) si on veut les p-valeurs et intervalles de confiance des coefficients.

**Alternatives open-source**
- *Bibliothèques* : **statsmodels** `OLS` fournit la table de régression complète (erreurs types, p-valeurs, tests de spécification) que scikit-learn ne donne pas ; **Ridge** / **Lasso** dès que $d$ est grand ; **HuberRegressor** et **RANSAC** pour résister aux valeurs aberrantes ; **QuantileRegressor** pour prédire une médiane ou un quantile plutôt qu'une moyenne ; **pygam** pour des relations non linéaires tout en restant additif et lisible.
- *Outils graphiques* : **Orange Data Mining** (blocs visuels, très pédagogique) ; **JASP** / **Jamovi** pour une régression avec diagnostics complets sans code ; **KNIME** pour des flux de traitement graphiques ; **LibreOffice Calc** avec `DROITEREG`, utile pour vérifier son implémentation sur cinq lignes.

**Astuces**
- Toujours tracer les résidus contre les prédictions. Une forme en cornet = variance non constante (passer au log de la cible). Une courbure = relation non linéaire manquée. C'est le diagnostic le plus informatif du ML tabulaire.
- Le $R^2$ augmente mécaniquement quand on ajoute des features, même du bruit pur. Comparer avec le $R^2$ ajusté, ou mieux, sur un jeu de test séparé.
- Un $R^2$ négatif sur le test est possible et signifie que le modèle fait moins bien que prédire la moyenne. Presque toujours une fuite de données ou une dérive de distribution.
- Prendre le logarithme d'une cible strictement positive et très asymétrique (prix, revenus) améliore souvent radicalement le modèle. Ne pas oublier `expm1` pour revenir, et que la moyenne des logs n'est pas le log de la moyenne.
- Les valeurs aberrantes ont une influence disproportionnée, car l'erreur est *quadratique*. Un point à 10 écarts types pèse 100 fois plus qu'un point à 1.

## Régularisation : ridge, lasso, elastic net
<!-- slug: regularisation | difficulte: 2 | prereqs: regression-lineaire -->

**En une phrase** — Ajouter une pénalité sur la taille des coefficients pour empêcher le modèle de coller au bruit des données d'entraînement.

**Explication** — Sans contrainte, un modèle avec beaucoup de features trouve toujours une combinaison qui explique parfaitement le bruit de l'échantillon. La régularisation ajoute un terme à minimiser : **ridge** (L2) pénalise $\lambda\sum w_j^2$, **lasso** (L1) pénalise $\lambda\sum|w_j|$. Le paramètre $\lambda$ (nommé `alpha` dans scikit-learn) règle l'arbitrage entre coller aux données et rester simple.

La différence de comportement vient de la géométrie de la pénalité. La boule L2 est ronde : la solution touche sa surface en un point où toutes les coordonnées sont petites mais non nulles — ridge **rétrécit** tous les coefficients. La boule L1 est un losange à coins pointus alignés sur les axes : la solution touche souvent un coin, où certaines coordonnées valent exactement zéro — lasso **sélectionne** les features. L'**elastic net** combine les deux et se comporte mieux quand des features sont fortement corrélées, cas où le lasso choisit arbitrairement l'une et rejette l'autre.

Interprétation bayésienne, utile à connaître : ridge équivaut à un a priori gaussien sur les coefficients, lasso à un a priori de Laplace. Régulariser, c'est déclarer qu'on croit *a priori* les effets petits.

**Cas d'utilisation**
- Plus de features que d'exemples ($d > n$) : génomique, texte vectorisé, capteurs. La régularisation devient obligatoire.
- Features corrélées : ridge stabilise des coefficients qui sinon partent dans tous les sens.
- Sélection automatique de variables parmi des centaines : lasso.
- Inutile avec très peu de features et beaucoup de données : le biais introduit n'apporte rien.

**Algorithme**
```text
1. Standardiser IMPÉRATIVEMENT les features (la pénalité dépend de l'échelle).
2. Définir une grille logarithmique de alpha : np.logspace(-4, 3, 50).
3. Pour chaque alpha : validation croisée en k blocs, mémoriser l'erreur moyenne.
4. Choisir l'alpha de plus faible erreur, ou la règle « une erreur type » :
   le plus grand alpha dont l'erreur reste dans un écart type du minimum
   (modèle plus simple, généralise souvent mieux).
5. Réentraîner sur toutes les données d'entraînement avec cet alpha.
```

**Implémentation**
```python
import numpy as np
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

alphas = np.logspace(-4, 3, 50)

# Le pipeline garantit que la standardisation est apprise sur les seuls
# blocs d'entraînement de la validation croisée : pas de fuite de données.
ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas, cv=5)).fit(X_tr, y_tr)
lasso = make_pipeline(StandardScaler(), LassoCV(alphas=alphas, cv=5, max_iter=10000)).fit(X_tr, y_tr)

print("alpha ridge :", ridge[-1].alpha_)
print("features retenues par lasso :", (lasso[-1].coef_ != 0).sum(), "/", X_tr.shape[1])

# --- ridge from scratch : une ligne de plus que les moindres carrés ---
def ridge_fit(X, y, lam):
    Xc, yc = X - X.mean(0), y - y.mean()
    d = Xc.shape[1]
    w = np.linalg.solve(Xc.T @ Xc + lam * np.eye(d), Xc.T @ yc)
    b = y.mean() - X.mean(0) @ w
    return w, b

# Chemin de régularisation : voir les coefficients s'éteindre un par un
from sklearn.linear_model import lasso_path
alphas_path, coefs, _ = lasso_path(X_std, y, alphas=alphas)
# coefs a la forme (d, len(alphas)) -> tracer chaque ligne
```

**Outils** — `scikit-learn` (`Ridge`, `Lasso`, `ElasticNet`, et leurs variantes `*CV`).

**Alternatives open-source**
- *Bibliothèques* : **glum** implémente des GLM régularisés très rapides, y compris Poisson et Tweedie (assurance) ; **celer** accélère fortement le lasso en grande dimension ; **group-lasso** pour pénaliser des groupes entiers de features (par exemple toutes les modalités d'une variable catégorielle) ; **scikit-learn** `SGDRegressor(penalty='elasticnet')` pour du très grand volume ; **statsmodels** `fit_regularized` pour garder l'appareil inférentiel.
- *Outils graphiques* : **Orange** propose ridge et lasso en blocs avec visualisation du chemin de régularisation ; **KNIME** pour intégrer la sélection de variables dans un flux ; **Weka** (Java, libre) reste utile pour comparer visuellement des méthodes de sélection.

**Astuces**
- Sans standardisation, la pénalité frappe plus fort les features de petite échelle. C'est l'erreur la plus fréquente sur ce sujet, et le résultat reste plausible donc invisible.
- `alpha` dans scikit-learn est le $\lambda$ de la théorie ; dans `LogisticRegression`, c'est `C = 1/lambda` — l'inverse. Confusion classique quand on transpose une valeur trouvée d'un modèle à l'autre.
- Le lasso ne peut pas sélectionner plus de $n$ features. Avec $d \gg n$, l'elastic net est plus raisonnable.
- Ne pas pénaliser l'ordonnée à l'origine. Les bibliothèques le gèrent, une implémentation manuelle doit y penser (d'où le centrage dans le code ci-dessus).
- La « règle d'une erreur type » donne des modèles plus petits et souvent plus robustes en production que le minimum strict de la courbe de validation.
- `LassoCV` qui ne converge pas : augmenter `max_iter`, ou vérifier que les features sont bien standardisées.

## Régression logistique
<!-- slug: regression-logistique | difficulte: 2 | prereqs: descente-de-gradient, probabilites-bayes -->

**En une phrase** — Un modèle linéaire dont la sortie est écrasée entre 0 et 1 par la fonction sigmoïde, ce qui en fait un estimateur de probabilité de classe.

**Explication** — On calcule un score linéaire $z = Xw + b$, puis $p = \sigma(z) = 1/(1 + e^{-z})$. La frontière de décision est l'hyperplan $z = 0$ : le modèle est **linéaire dans l'espace des features**, même si la sortie est courbe. La perte utilisée est l'**entropie croisée** (log-vraisemblance négative) : $L = -\frac{1}{n}\sum [y_i\log p_i + (1-y_i)\log(1-p_i)]$. Elle punit très durement une prédiction confiante et fausse, ce qui est exactement le comportement souhaité.

Il n'existe pas de solution analytique, mais le problème reste convexe : la descente de gradient converge vers l'optimum global. Et le gradient est d'une élégance remarquable — $\nabla_w L = \frac{1}{n}X^\top(p - y)$, la même forme que pour les moindres carrés, avec $p$ au lieu de $\hat{y}$.

L'interprétation des coefficients passe par les **cotes** (*odds*) : $w_j$ est la variation du logarithme de la cote pour une unité de $x_j$. Donc $e^{w_j}$ est un facteur multiplicatif sur la cote. Un coefficient de 0,7 signifie une cote multipliée par 2. C'est ce qui rend ce modèle omniprésent en médecine et en scoring de crédit.

**Cas d'utilisation**
- Toute classification binaire où il faut une probabilité, pas seulement une étiquette : scoring de crédit, risque médical, churn.
- Baseline systématique en classification, y compris sur du texte vectorisé en TF-IDF où elle est difficile à battre.
- Contexte réglementaire exigeant une explication du refus.
- Insuffisante si les classes ne sont pas séparables linéairement et qu'on ne veut pas construire les interactions à la main.

**Algorithme**
```text
Entrée : X (n, d), y binaire, taux eta, pénalité lambda
1. Standardiser X. Initialiser w = 0, b = 0.
2. Répéter jusqu'à convergence :
     z = X @ w + b
     p = sigmoide(z)                      # attention au dépassement
     grad_w = X.T @ (p - y) / n + 2 lambda w
     grad_b = (p - y).mean()
     w -= eta * grad_w ; b -= eta * grad_b
3. Prédire la probabilité p ; l'étiquette = (p > seuil).
4. Choisir le SEUIL selon le coût des erreurs — 0.5 est un défaut arbitraire.
```

**Implémentation**
```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

# --- from scratch (palier 2) ---
def sigmoide(z):
    # forme stable : évite exp d'un grand positif
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))

class RegLogistique:
    def __init__(self, eta=0.5, epoques=2000, lam=0.0):
        self.eta, self.epoques, self.lam = eta, epoques, lam

    def fit(self, X, y):
        n, d = X.shape
        self.w, self.b = np.zeros(d), 0.0
        for _ in range(self.epoques):
            p = sigmoide(X @ self.w + self.b)
            err = p - y
            self.w -= self.eta * (X.T @ err / n + 2 * self.lam * self.w)
            self.b -= self.eta * err.mean()
        return self

    def predict_proba(self, X):
        return sigmoide(X @ self.w + self.b)

# --- version bibliothèque ---
clf = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced').fit(X_tr, y_tr)
p = clf.predict_proba(X_te)[:, 1]
print("AUC :", roc_auc_score(y_te, p))

# Lecture en rapports de cotes
import pandas as pd
print(pd.Series(np.exp(clf.coef_[0]), index=noms).sort_values(ascending=False).head(10))

# Choix du seuil par coût métier : un faux négatif coûte 10 fois un faux positif
seuils = np.linspace(0.01, 0.99, 99)
couts = [((p > s) & (y_te == 0)).sum() * 1 + ((p <= s) & (y_te == 1)).sum() * 10
         for s in seuils]
print("seuil optimal :", seuils[int(np.argmin(couts))])
```

**Outils** — `scikit-learn` (`LogisticRegression`), `statsmodels` (`Logit`) pour les p-valeurs des coefficients.

**Alternatives open-source**
- *Bibliothèques* : **statsmodels** `Logit` fournit erreurs types et intervalles de confiance sur les rapports de cotes ; **LinearSVC** pour la même frontière linéaire mais sans probabilité et avec une perte charnière ; **LightGBM** dès que les interactions comptent plus que l'interprétabilité ; **vowpal-wabbit** pour de l'apprentissage en ligne sur des milliards d'exemples ; **liblinear** (moteur sous-jacent) pour du texte creux de très grande dimension.
- *Outils graphiques* : **Orange** montre la frontière de décision en direct sur deux features ; **Jamovi** pour une régression logistique complète avec sorties de publication ; **Weka** Explorer pour comparer côte à côte plusieurs classifieurs sur le même jeu.

**Astuces**
- Le seuil de 0,5 n'a aucune justification métier. Sur des classes déséquilibrées, il produit un modèle qui prédit toujours la classe majoritaire alors que l'AUC est excellente. Choisir le seuil explicitement, par coût ou par rappel cible.
- `class_weight='balanced'` corrige le déséquilibre pendant l'apprentissage, mais **décalibre** les probabilités : elles ne correspondent plus aux fréquences réelles. À recalibrer si on utilise les valeurs numériques.
- Une séparabilité parfaite fait diverger les coefficients vers l'infini. Si un coefficient dépasse 20, chercher une fuite de données : une feature contient probablement la réponse.
- La sigmoïde naïve `1/(1+np.exp(-z))` déborde en `float32` pour $|z| > 88$. Utiliser la forme par morceaux ci-dessus, ou `scipy.special.expit`.
- `max_iter=100` par défaut est souvent insuffisant et produit un avertissement de convergence qu'il ne faut pas ignorer : le modèle retourné n'est pas l'optimum.
- Pour du multi-classe, `multi_class='multinomial'` (softmax) est préférable au un-contre-tous : les probabilités somment à 1 et sont mieux calibrées.

## k plus proches voisins
<!-- slug: knn | difficulte: 1 | prereqs: algebre-lineaire -->

**En une phrase** — Pour prédire, chercher les $k$ exemples d'entraînement les plus proches et voter (classification) ou moyenner (régression).

**Explication** — C'est le seul algorithme sans phase d'apprentissage : `fit` mémorise simplement les données. Tout le coût est au moment de la prédiction, où il faut calculer $n$ distances. On parle de méthode **non paramétrique** et **paresseuse**. La frontière de décision qui en résulte peut être arbitrairement complexe, ce qui est à la fois sa force et sa faiblesse.

Le paramètre $k$ contrôle directement l'arbitrage biais-variance. À $k=1$, le modèle mémorise parfaitement l'entraînement — variance maximale, erreur nulle sur l'entraînement, généralisation médiocre. À $k = n$, il prédit toujours la classe majoritaire — biais maximal. La valeur utile se trouve par validation croisée, généralement entre 5 et 50, et on privilégie un $k$ impair en binaire pour éviter les égalités.

La faiblesse structurelle est la **malédiction de la dimension**. En grande dimension, les distances entre points se concentrent : le plus proche voisin et le plus lointain finissent à des distances comparables, et la notion de « proche » perd tout sens. Au-delà d'une vingtaine de features informatives, kNN s'effondre — sauf après réduction de dimension.

**Cas d'utilisation**
- Baseline immédiate sur un petit dataset de faible dimension.
- Systèmes de recommandation par similarité entre utilisateurs ou entre articles.
- Recherche de similarité sur des **embeddings** (visages, textes, images) : c'est là que kNN est aujourd'hui le plus utilisé, sous le nom de recherche vectorielle.
- Imputation de valeurs manquantes (`KNNImputer`).
- Mauvais choix avec beaucoup de features, beaucoup de données, ou une contrainte de latence en production.

**Algorithme**
```text
Entraînement : mémoriser X, y. (Rien d'autre.)
Prédiction pour un point x :
1. Calculer la distance de x à tous les points de X.
2. Prendre les indices des k plus petites distances.
3. Classification : vote majoritaire parmi leurs étiquettes.
   Régression : moyenne de leurs valeurs.
4. Variante pondérée : poids = 1/distance, les voisins proches comptent plus.
Prérequis absolu : les features doivent être standardisées.
```

**Implémentation**
```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# --- from scratch, vectorisé (palier 2) ---
class KNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X, self.y = X, y
        return self

    def predict(self, Xq):
        # ||a-b||² = ||a||² - 2ab + ||b||² : évite un tenseur (nq, n, d)
        d2 = ((Xq ** 2).sum(1)[:, None] - 2 * Xq @ self.X.T + (self.X ** 2).sum(1)[None, :])
        idx = np.argpartition(d2, self.k, axis=1)[:, :self.k]   # O(n), pas de tri complet
        votes = self.y[idx]
        return np.array([np.bincount(v).argmax() for v in votes])

# --- version bibliothèque, avec réglage de k ---
pipe = make_pipeline(StandardScaler(), KNeighborsClassifier())
grille = {'kneighborsclassifier__n_neighbors': [1, 3, 5, 9, 15, 25, 51],
          'kneighborsclassifier__weights': ['uniform', 'distance']}
gs = GridSearchCV(pipe, grille, cv=5, scoring='f1_macro').fit(X_tr, y_tr)
print(gs.best_params_, round(gs.best_score_, 3))

# Recherche vectorielle approchée : le vrai usage moderne, à grande échelle
# pip install faiss-cpu
import faiss
index = faiss.IndexFlatIP(dim)          # produit scalaire = cosinus si normalisé
index.add(embeddings.astype('float32'))
distances, voisins = index.search(requete.astype('float32'), k=10)
```

**Outils** — `scikit-learn` (`KNeighborsClassifier`, `KNeighborsRegressor`, `KNNImputer`).

**Alternatives open-source**
- *Bibliothèques* : **FAISS** (Meta) pour la recherche approchée sur des millions de vecteurs, la référence ; **hnswlib** et **Annoy** (Spotify), plus légers, même principe de graphe navigable ; **ScaNN** (Google) pour le meilleur compromis vitesse/rappel ; **Qdrant**, **Milvus**, **Weaviate**, **Chroma** comme bases vectorielles complètes avec filtrage et persistance ; **pgvector** pour rester dans PostgreSQL.
- *Outils graphiques* : **Orange** visualise l'effet de $k$ sur la frontière de décision, très parlant ; **TensorFlow Embedding Projector** explore un espace d'embeddings en 3D avec recherche de voisins ; **Weka** pour comparer $k$ et métriques de distance.

**Astuces**
- Sans standardisation, la feature de plus grande amplitude domine entièrement la distance. Une variable en euros écrase une variable en années.
- Utiliser `np.argpartition` et non `np.argsort` : sélection partielle en $O(n)$ contre un tri complet en $O(n\log n)$.
- Pour des données creuses de grande dimension (texte), préférer la distance **cosinus** à l'euclidienne : la longueur du document devient sans importance.
- `algorithm='kd_tree'` accélère en dessous de ~20 dimensions et devient plus lent que la force brute au-delà. Le défaut `'auto'` choisit correctement.
- Les classes déséquilibrées faussent le vote : la classe majoritaire est plus souvent dans le voisinage. Utiliser `weights='distance'` ou rééquilibrer.
- En production, kNN exige de garder tout le jeu d'entraînement en mémoire. C'est souvent le point bloquant, pas la précision.

## Machines à vecteurs de support
<!-- slug: svm-noyaux | difficulte: 4 | prereqs: regression-logistique, algebre-lineaire -->

**En une phrase** — Trouver l'hyperplan séparateur qui maximise la marge avec les points les plus proches, et rendre le problème non linéaire par une astuce de noyau.

**Explication** — Parmi tous les hyperplans séparant deux classes, le SVM choisit celui dont la distance aux points les plus proches est maximale. Ces points critiques sont les **vecteurs de support** : seuls eux déterminent la solution, tous les autres pourraient être supprimés sans rien changer. C'est une propriété rare et précieuse. La version *marge souple* introduit un paramètre $C$ qui autorise des violations : $C$ grand exige une séparation stricte (risque de surapprentissage), $C$ petit tolère des erreurs pour une frontière plus lisse.

L'**astuce du noyau** est l'idée profonde. Le problème dual ne fait intervenir les données qu'à travers des produits scalaires $x_i \cdot x_j$. On peut donc remplacer ce produit par une fonction noyau $K(x_i, x_j)$ qui correspond à un produit scalaire dans un espace de dimension bien plus grande — **sans jamais calculer les coordonnées dans cet espace**. Le noyau RBF $K(x, x') = \exp(-\gamma\|x - x'\|^2)$ correspond à un espace de dimension infinie, et se calcule en trois opérations. On obtient une frontière très flexible pour le prix d'une distance euclidienne.

Le coût est algorithmique : l'entraînement est en $O(n^2)$ à $O(n^3)$, ce qui plafonne l'usage vers 10 000 à 50 000 exemples. Au-delà, les méthodes d'ensemble d'arbres ont supplanté les SVM sur données tabulaires.

**Cas d'utilisation**
- Datasets petits à moyens (< 20 000 lignes) avec beaucoup de features : le SVM y est souvent excellent.
- Classification de texte en grande dimension avec noyau linéaire (`LinearSVC`), très efficace.
- Problèmes où la frontière est nettement non linéaire mais les données peu nombreuses.
- Détection de nouveauté avec `OneClassSVM`.
- Mauvais choix au-delà de 100 000 lignes, ou quand il faut des probabilités (celles du SVM sont obtenues par un post-traitement coûteux).

**Algorithme**
```text
Problème primal (marge souple) :
  minimiser  (1/2)||w||² + C * somme(xi_i)
  sous       y_i (w·x_i + b) >= 1 - xi_i,  xi_i >= 0

Résolution pratique :
1. Standardiser les features (indispensable : le noyau RBF est une distance).
2. Choisir le noyau : linear si d est grand, rbf sinon.
3. Régler C et gamma ensemble par recherche sur grille logarithmique
   C in [1e-2 .. 1e3], gamma in [1e-4 .. 1e1] — ils interagissent fortement.
4. Le solveur (SMO) renvoie les coefficients duaux alpha_i ; les points
   avec alpha_i > 0 sont les vecteurs de support.
5. Décision : signe de somme(alpha_i y_i K(x_i, x)) + b.
```

**Implémentation**
```python
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

pipe = make_pipeline(StandardScaler(), SVC(kernel='rbf'))
grille = {'svc__C': np.logspace(-2, 3, 6), 'svc__gamma': np.logspace(-4, 1, 6)}
gs = GridSearchCV(pipe, grille, cv=5, n_jobs=-1).fit(X_tr, y_tr)
print(gs.best_params_)
print("vecteurs de support :", gs.best_estimator_[-1].n_support_, "/", len(X_tr))

# Texte de grande dimension : noyau linéaire, solveur dédié, bien plus rapide
from sklearn.feature_extraction.text import TfidfVectorizer
txt = make_pipeline(TfidfVectorizer(min_df=3, ngram_range=(1, 2)),
                    LinearSVC(C=1.0, dual='auto')).fit(textes_tr, y_tr)

# --- comprendre le noyau : projection explicite vs astuce ---
def noyau_rbf(A, B, gamma):
    d2 = (A**2).sum(1)[:, None] - 2 * A @ B.T + (B**2).sum(1)[None, :]
    return np.exp(-gamma * d2)

K = noyau_rbf(X_tr, X_tr, gamma=0.1)     # matrice (n, n) : voilà la limite de scalabilité
```

**Outils** — `scikit-learn` (`SVC`, `LinearSVC`, `SVR`, `OneClassSVM`), appuyé sur `libsvm` et `liblinear`.

**Alternatives open-source**
- *Bibliothèques* : **ThunderSVM** exécute le même algorithme sur GPU, gain de 10 à 100× ; **cuML** (RAPIDS) pour des SVM GPU intégrés à un pipeline scikit-learn ; **LightGBM** / **XGBoost** les remplacent avantageusement dès que $n$ grandit ; **Nystroem** ou **RBFSampler** de scikit-learn approchent le noyau RBF par des features explicites, ce qui permet d'utiliser un modèle linéaire rapide sur des millions de lignes ; **libsvm** en ligne de commande pour reproduire des résultats de papiers anciens.
- *Outils graphiques* : **libsvm applet** et la démo SVM de **scikit-learn** montrent l'effet de $C$ et $\gamma$ sur la frontière — le meilleur moyen de construire l'intuition ; **Orange** propose le SVM en bloc avec visualisation ; **Weka** pour comparer les noyaux.

**Astuces**
- $C$ et $\gamma$ interagissent : les régler séparément mène systématiquement à un mauvais optimum. Toujours une grille croisée.
- `gamma='scale'` (le défaut actuel) vaut $1/(d \cdot \text{Var}(X))$ et constitue un bon point de départ. `gamma='auto'` est l'ancien défaut, moins bon.
- Un nombre de vecteurs de support proche de $n$ signale un surapprentissage ou un $\gamma$ trop grand : le modèle mémorise chaque point.
- `SVC(probability=True)` déclenche une validation croisée interne (Platt scaling) qui multiplie le temps d'entraînement par cinq. Ne l'activer que si les probabilités sont réellement nécessaires.
- Pour un noyau linéaire, `LinearSVC` est bien plus rapide que `SVC(kernel='linear')` : autre solveur, complexité linéaire en $n$.
- Le SVM ne gère pas nativement le multi-classe : scikit-learn fait du un-contre-un, soit $k(k-1)/2$ modèles. Avec 50 classes, cela devient déraisonnable.

## Arbres de décision
<!-- slug: arbres-decision | difficulte: 2 | prereqs: regression-lineaire -->

**En une phrase** — Une suite de questions binaires sur les features, choisies à chaque étape pour rendre les groupes obtenus aussi homogènes que possible.

**Explication** — L'algorithme CART procède de façon **gloutonne** et récursive. À chaque nœud, il essaie tous les couples (feature, seuil) possibles et retient celui qui réduit le plus l'impureté des deux enfants. En classification, l'impureté se mesure par l'indice de **Gini** $1 - \sum p_k^2$ ou par l'**entropie** $-\sum p_k \log p_k$ — les deux donnent des arbres presque identiques. En régression, on minimise la variance intra-nœud. La récursion s'arrête sur un critère d'arrêt, et la prédiction d'une feuille est la classe majoritaire ou la moyenne des exemples qu'elle contient.

Deux propriétés expliquent le succès des arbres sur données tabulaires. D'abord l'**invariance aux transformations monotones** : seul l'ordre des valeurs compte, donc aucune standardisation, aucun traitement de log, aucune sensibilité aux valeurs aberrantes dans les features. Ensuite la capture automatique des **interactions** : un chemin racine-feuille combine plusieurs conditions, ce qu'un modèle linéaire ne fait qu'en explicitant chaque produit.

Le défaut est tout aussi net : un arbre non contraint atteint 100 % sur l'entraînement en isolant chaque exemple, et sa structure change complètement si l'on modifie quelques lignes. Cette **instabilité** est précisément ce que les forêts et le boosting exploitent — un modèle instable et peu biaisé est le composant idéal d'un ensemble.

**Cas d'utilisation**
- Modèle explicable qu'on peut dessiner et montrer à un non-spécialiste : c'est son avantage décisif.
- Données mêlant variables numériques et catégorielles, avec des valeurs manquantes.
- Brique de base des forêts aléatoires et du gradient boosting.
- Extraction de règles métier (« si ancienneté < 6 mois et région = X alors risque élevé »).
- Mauvais choix seul, pour la performance pure : un arbre unique est presque toujours battu par un ensemble.

**Algorithme**
```text
construire(D, profondeur) :
  1. Si critère d'arrêt atteint (profondeur max, |D| < min_samples_split,
     impureté nulle) -> feuille = classe majoritaire ou moyenne.
  2. meilleur_gain = 0
  3. Pour chaque feature j :
       Pour chaque seuil candidat s (milieux des valeurs triées uniques) :
         gauche, droite = D[x_j <= s], D[x_j > s]
         gain = impureté(D) - (|g|/|D|) impureté(g) - (|d|/|D|) impureté(d)
         si gain > meilleur_gain : mémoriser (j, s)
  4. Si meilleur_gain <= min_impurity_decrease -> feuille.
  5. Créer un nœud (j, s) ; récursion sur gauche et droite.
```

**Implémentation**
```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import matplotlib.pyplot as plt

# --- from scratch, l'essentiel (palier 2) ---
def gini(y):
    if len(y) == 0:
        return 0.0
    p = np.bincount(y) / len(y)
    return 1 - (p ** 2).sum()

def meilleure_coupe(X, y):
    n, d = X.shape
    base, meilleur = gini(y), (0.0, None, None)
    for j in range(d):
        ordre = np.argsort(X[:, j])
        xs, ys = X[ordre, j], y[ordre]
        for i in range(1, n):
            if xs[i] == xs[i - 1]:
                continue                    # seuil inutile entre valeurs égales
            g = base - (i / n) * gini(ys[:i]) - ((n - i) / n) * gini(ys[i:])
            if g > meilleur[0]:
                meilleur = (g, j, (xs[i] + xs[i - 1]) / 2)
    return meilleur                          # (gain, feature, seuil)

# --- version bibliothèque, avec élagage par complexité de coût ---
arbre = DecisionTreeClassifier(max_depth=4, min_samples_leaf=20,
                              class_weight='balanced', random_state=0).fit(X_tr, y_tr)

print(export_text(arbre, feature_names=list(noms)))     # les règles en texte
fig, ax = plt.subplots(figsize=(16, 8))
plot_tree(arbre, feature_names=noms, class_names=['non', 'oui'], filled=True, ax=ax)

# Élagage : chercher alpha par validation croisée plutôt que régler max_depth à la main
chemin = DecisionTreeClassifier(random_state=0).cost_complexity_pruning_path(X_tr, y_tr)
from sklearn.model_selection import GridSearchCV
gs = GridSearchCV(DecisionTreeClassifier(random_state=0),
                  {'ccp_alpha': chemin.ccp_alphas[::5]}, cv=5).fit(X_tr, y_tr)
```

**Outils** — `scikit-learn` (`DecisionTreeClassifier`, `DecisionTreeRegressor`, `plot_tree`, `export_text`).

**Alternatives open-source**
- *Bibliothèques* : **LightGBM** et **XGBoost** avec `n_estimators=1` pour un arbre unique doté d'un meilleur traitement des catégorielles ; **dtreeviz** produit des visualisations d'arbres bien supérieures à `plot_tree`, avec distributions par nœud ; **imodels** implémente des modèles à base de règles interprétables (RuleFit, arbres optimaux) ; **PyStruct** et **CHAID** pour les variantes historiques ; **interpret** (Microsoft) propose les Explainable Boosting Machines, aussi précises qu'un boosting et lisibles comme un arbre.
- *Outils graphiques* : **Orange** affiche l'arbre et la répartition des classes en direct ; **KNIME** avec son nœud Decision Tree Learner et son vue interactive ; **Weka** Explorer ; **RapidMiner Studio** (édition libre limitée) ; **dtreeviz** en sortie SVG dans un notebook.

**Astuces**
- Un arbre sans contrainte surapprend toujours. Fixer au minimum `max_depth` et `min_samples_leaf`. `min_samples_leaf=20` est un défaut raisonnable et sous-utilisé.
- L'importance des features par réduction d'impureté (`feature_importances_`) est **biaisée** en faveur des variables à forte cardinalité. Préférer l'importance par permutation.
- Un arbre ne peut pas extrapoler : sa prédiction est constante hors du domaine observé. Rédhibitoire pour une tendance temporelle.
- Les seuils sont des comparaisons `<=` : les features catégorielles encodées en entiers créent des coupes absurdes (« code postal <= 45000 »). Encoder correctement (voir le module features).
- L'élagage par `ccp_alpha` est plus principiel que le réglage manuel de la profondeur, et scikit-learn fournit le chemin complet gratuitement.
- Un arbre profond entraîné sur des identifiants uniques atteint 100 % en apprentissage. C'est le signe le plus visible d'une fuite de données.

## Forêts aléatoires
<!-- slug: forets-aleatoires | difficulte: 2 | prereqs: arbres-decision -->

**En une phrase** — Entraîner des centaines d'arbres sur des échantillons et des sous-ensembles de features tirés au hasard, puis moyenner leurs prédictions.

**Explication** — Deux sources d'aléa décorrèlent les arbres. Le **bagging** (*bootstrap aggregating*) donne à chaque arbre un échantillon tiré avec remise, de même taille que l'original — donc environ 63 % d'exemples distincts, le reste étant dupliqué. Et à chaque nœud, l'arbre ne considère qu'un sous-ensemble aléatoire de features (`max_features`, par défaut $\sqrt{d}$ en classification). Cette seconde idée est la contribution propre de Breiman : sans elle, une feature très prédictive serait choisie à la racine de tous les arbres, qui se ressembleraient trop.

Le mécanisme statistique est simple et puissant. La moyenne de $B$ prédicteurs de variance $\sigma^2$ et de corrélation moyenne $\rho$ a pour variance $\rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$. Augmenter $B$ élimine le second terme ; réduire $\rho$ (par l'aléa sur les features) attaque le premier. Le biais, lui, reste celui d'un arbre profond — c'est-à-dire faible. On obtient donc un modèle à faible biais **et** faible variance, sans compromis à régler.

Conséquence pratique remarquable : les forêts **ne surapprennent pas** quand on ajoute des arbres. Plus d'arbres ne peut que stabiliser. Le seul coût est le temps de calcul, et l'entraînement est parfaitement parallélisable.

**Cas d'utilisation**
- Baseline solide sur n'importe quelles données tabulaires, en quinze secondes et sans réglage.
- Estimation d'importance de features (par permutation).
- Problèmes avec beaucoup de features bruitées : l'aléa sur les features en fait une méthode robuste.
- `ExtraTrees` ou `IsolationForest` pour la détection d'anomalies.
- Mauvais choix si la latence de prédiction est critique (des centaines d'arbres à parcourir) ou si l'on cherche le dernier point de performance — le boosting gagne alors.

**Algorithme**
```text
Entraînement :
  Pour b = 1..B (en parallèle) :
    1. Tirer n exemples avec remise dans le jeu d'entraînement.
    2. Construire un arbre profond, en n'évaluant à chaque nœud que
       max_features features tirées au hasard.
    3. Ne pas élaguer.
Prédiction :
    Classification : moyenne des probabilités des arbres (mieux que le vote).
    Régression      : moyenne des prédictions.
Bonus gratuit :
    Chaque exemple est hors-échantillon pour ~37 % des arbres.
    Prédire avec ces arbres seulement donne le score OOB, une estimation
    de généralisation sans jeu de validation séparé.
```

**Implémentation**
```python
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.inspection import permutation_importance
import numpy as np

rf = RandomForestClassifier(
    n_estimators=500,          # plus = mieux, jusqu'à saturation
    max_features='sqrt',       # l'aléa qui décorrèle les arbres
    min_samples_leaf=2,        # seul garde-fou vraiment utile
    oob_score=True,            # validation gratuite
    class_weight='balanced_subsample',
    n_jobs=-1, random_state=0,
).fit(X_tr, y_tr)

print("OOB :", round(rf.oob_score_, 4))       # à comparer au score de test

# Importance par permutation : fiable, contrairement à feature_importances_
imp = permutation_importance(rf, X_te, y_te, n_repeats=20, random_state=0, n_jobs=-1)
for i in imp.importances_mean.argsort()[::-1][:10]:
    print(f"{noms[i]:25s} {imp.importances_mean[i]:.4f} ± {imp.importances_std[i]:.4f}")

# Incertitude de prédiction : la dispersion entre arbres est exploitable
preds = np.stack([a.predict_proba(X_te)[:, 1] for a in rf.estimators_])
moyenne, ecart = preds.mean(0), preds.std(0)   # ecart élevé = zone mal couverte

# Courbe de saturation : combien d'arbres sont réellement utiles ?
for B in [10, 50, 100, 300, 600]:
    m = RandomForestClassifier(n_estimators=B, n_jobs=-1, random_state=0).fit(X_tr, y_tr)
    print(B, round(m.score(X_te, y_te), 4))
```

**Outils** — `scikit-learn` (`RandomForestClassifier`, `RandomForestRegressor`, `ExtraTreesClassifier`, `IsolationForest`).

**Alternatives open-source**
- *Bibliothèques* : **ranger** (R, très rapide) et son portage Python **skranger** ; **cuML** RandomForest sur GPU pour des dizaines de millions de lignes ; **LightGBM** en mode `boosting_type='rf'` pour du bagging avec un moteur plus rapide ; **ExtraTreesClassifier** tire les seuils au hasard, encore plus rapide et souvent équivalent ; **IsolationForest** pour la détection d'anomalies non supervisée ; **quantile-forest** pour des intervalles de prédiction.
- *Outils graphiques* : **Orange** et **KNIME** proposent la forêt en bloc avec importance des features visualisée ; **H2O Flow** (interface web de H2O-3) entraîne des forêts distribuées sans écrire de code et fournit des graphiques de dépendance partielle ; **Weka** pour la comparaison pédagogique bagging / boosting.

**Astuces**
- `n_estimators` n'a pas d'optimum : plus est toujours au moins aussi bon. Tracer la courbe et s'arrêter à la saturation, généralement entre 200 et 500.
- Le score OOB remplace un jeu de validation quand les données sont rares. Il est légèrement pessimiste, donc conservateur — ce qui convient.
- Ne pas utiliser `feature_importances_` pour des conclusions métier : cette mesure surévalue les variables continues et à haute cardinalité. `permutation_importance` sur le jeu de test est la bonne réponse.
- Sur des features fortement corrélées, l'importance se répartit entre elles et chacune paraît négligeable. Regrouper par corrélation avant d'interpréter.
- `n_jobs=-1` est gratuit et divise le temps par le nombre de cœurs. L'oublier est la cause la plus fréquente de « la forêt est lente ».
- Une forêt de 500 arbres profonds peut peser plusieurs centaines de mégaoctets une fois sérialisée. Limiter `max_depth` si le modèle doit être déployé.

## Gradient boosting
<!-- slug: gradient-boosting | difficulte: 3 | prereqs: forets-aleatoires, descente-de-gradient -->

**En une phrase** — Ajouter des arbres l'un après l'autre, chacun entraîné à corriger les erreurs résiduelles de la somme des précédents.

**Explication** — Le boosting est **séquentiel** là où le bagging est parallèle. On part d'une prédiction constante, puis à chaque itération on calcule le gradient de la perte par rapport aux prédictions actuelles — pour l'erreur quadratique, c'est simplement le résidu $y - \hat{y}$ — et on entraîne un petit arbre à prédire ce gradient. On ajoute ensuite cet arbre à la somme, multiplié par un taux d'apprentissage $\eta$ qui bride sa contribution. Formellement : $F_m(x) = F_{m-1}(x) + \eta h_m(x)$. C'est une descente de gradient, non pas dans l'espace des paramètres, mais dans l'espace des fonctions.

Contrairement aux forêts, le boosting **surapprend** si on ajoute trop d'arbres. Deux hyperparamètres se compensent : un $\eta$ faible (0,01 à 0,05) exige beaucoup d'arbres mais généralise mieux ; un $\eta$ élevé converge vite et surapprend. La bonne pratique est de fixer $\eta$ bas, de mettre `n_estimators` très haut, et de laisser l'**arrêt précoce** sur un jeu de validation décider du nombre réel.

Les implémentations modernes ajoutent ce qui fait toute la différence en pratique : histogrammes de seuils au lieu du tri exact (LightGBM, XGBoost `hist`), croissance par feuille plutôt que par niveau, régularisation L1/L2 sur les poids de feuilles, gestion native des valeurs manquantes et des variables catégorielles. Sur données tabulaires, ces modèles restent en 2026 l'état de l'art — les réseaux de neurones ne les ont pas dépassés.

**Cas d'utilisation**
- Objectif de performance maximale sur données tabulaires : c'est le choix par défaut, et il gagne les compétitions.
- Classement (*ranking*) pour la recherche et la recommandation (`LGBMRanker`).
- Pertes personnalisées : quantile, Poisson, Tweedie pour l'assurance, pertes asymétriques.
- Mauvais choix quand il faut un modèle explicable simplement, ou avec très peu de données (< 500 lignes) où une régression régularisée est plus sûre.

**Algorithme**
```text
Entrée : X, y, perte L, taux eta, M itérations
1. F_0 = argmin_c somme L(y_i, c)          # constante initiale (moyenne, log-odds)
2. Pour m = 1..M :
     a. g_i = -dL/dF évalué en F_{m-1}(x_i)     # pseudo-résidus
     b. Entraîner un arbre peu profond h_m sur (X, g)
     c. Optionnel : recalculer la valeur optimale de chaque feuille (Newton)
     d. F_m = F_{m-1} + eta * h_m
     e. Évaluer sur validation ; si pas d'amélioration depuis p tours, arrêter.
3. Retourner F_m avec le meilleur score de validation.
```

**Implémentation**
```python
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

modele = lgb.LGBMClassifier(
    n_estimators=5000,        # volontairement énorme : l'arrêt précoce tranchera
    learning_rate=0.03,
    num_leaves=31,            # LightGBM contrôle la complexité par feuilles
    min_child_samples=20,
    subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=0, n_jobs=-1,
)
modele.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)], eval_metric='auc',
    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)],
)
print("arbres retenus :", modele.best_iteration_)

# Catégorielles natives : pas d'encodage one-hot, LightGBM gère
for c in colonnes_cat:
    df[c] = df[c].astype('category')
modele.fit(df[features], y, categorical_feature=colonnes_cat)

# --- boosting from scratch, pour comprendre (palier 2) ---
from sklearn.tree import DecisionTreeRegressor

def boosting(X, y, M=200, eta=0.1, profondeur=3):
    F = np.full(len(y), y.mean())
    arbres = []
    for _ in range(M):
        residu = y - F                                # gradient de la MSE
        h = DecisionTreeRegressor(max_depth=profondeur).fit(X, residu)
        F += eta * h.predict(X)
        arbres.append(h)
    return y.mean(), arbres, eta
```

**Outils** — `pip install lightgbm xgboost catboost`. `scikit-learn` fournit `HistGradientBoostingClassifier`, très proche de LightGBM et sans dépendance supplémentaire.

**Alternatives open-source**
- *Bibliothèques* : **LightGBM** (Microsoft) est le meilleur compromis vitesse/précision et le choix par défaut ; **XGBoost** est plus mature, mieux documenté, avec un excellent support GPU et distribué ; **CatBoost** (Yandex) gère les catégorielles à haute cardinalité mieux que tout le monde grâce à l'encodage par cible ordonné, et demande le moins de réglage ; **HistGradientBoosting** de scikit-learn quand on veut zéro dépendance ; **NGBoost** pour des prédictions probabilistes complètes ; **interpret** (EBM) pour un boosting additif entièrement lisible.
- *Outils graphiques* : **H2O Flow** et **H2O AutoML** entraînent et comparent des GBM par interface web ; **Orange** dispose d'un bloc Gradient Boosting ; **Optuna Dashboard** visualise la recherche d'hyperparamètres ; **SHAP** fournit les meilleures visualisations d'explication pour ces modèles ; **Netron** ne les lit pas — utiliser `plot_importance` et `plot_tree` des bibliothèques elles-mêmes.

**Astuces**
- Toujours utiliser l'arrêt précoce. Régler `n_estimators` à la main est une perte de temps et le mauvais réglage coûte plusieurs points.
- Ordre de réglage par rentabilité décroissante : `learning_rate` (bas), `num_leaves` ou `max_depth`, `min_child_samples`, puis les sous-échantillonnages, enfin les régularisations. Les autres paramètres sont du bruit.
- `num_leaves` de LightGBM et `max_depth` de XGBoost ne sont pas équivalents : LightGBM croît par feuille, donc `num_leaves=31` correspond à peu près à une profondeur 5 mais avec des branches inégales. Ne pas transposer les valeurs entre les deux.
- CatBoost sur des catégorielles à 10 000 modalités battra LightGBM avec encodage one-hot, sans réglage. Sur du purement numérique, LightGBM est plus rapide.
- Le jeu de validation de l'arrêt précoce **ne doit pas** servir à comparer des modèles : il a participé à la sélection. Il faut un troisième jeu, ou une validation croisée imbriquée.
- Sur séries temporelles, le découpage aléatoire crée une fuite massive. Utiliser `TimeSeriesSplit` et vérifier qu'aucune feature ne contient d'information future.
- Le boosting ne nécessite ni standardisation ni traitement des valeurs aberrantes dans les features. Il reste en revanche sensible aux aberrations sur la **cible**.

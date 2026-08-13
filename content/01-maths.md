---
module: maths
titre: Socle mathématique
ordre: 1
resume: Ne pas traiter ce module « avant » le ML. Revenir sur chaque brique quand un algorithme la réclame.
---

## Algèbre linéaire
<!-- slug: algebre-lineaire | difficulte: 3 | prereqs: numpy-vectorisation -->

**En une phrase** — Le langage qui permet d'écrire une opération sur des milliers d'exemples et de dimensions en une seule ligne, et de la calculer efficacement.

**Explication** — Une donnée devient un **vecteur** de $\mathbb{R}^d$ : un client, une image, une phrase. Un dataset devient une **matrice** $X$ de forme $(n, d)$ — $n$ exemples en lignes, $d$ features en colonnes. Une matrice n'est pas seulement un tableau : c'est une **transformation linéaire**, une fonction qui envoie un vecteur sur un autre en le tournant, l'étirant et le projetant. Comprendre le produit matriciel $Xw$ comme « appliquer une transformation » plutôt que comme « une somme de produits » change complètement la lecture des modèles.

Quatre notions suffisent pour le ML. Le **produit scalaire** $x \cdot w = \sum_i x_i w_i$ mesure l'alignement entre deux vecteurs : c'est le cœur de tout modèle linéaire, et de l'attention dans les Transformers. La **norme** $\|x\|_2 = \sqrt{x \cdot x}$ mesure une longueur, donc une distance, donc une pénalité de régularisation. La **projection** de $x$ sur un sous-espace donne la meilleure approximation de $x$ dans cet espace — c'est exactement ce que fait la régression des moindres carrés, et ce que fait l'ACP. Enfin les **valeurs et vecteurs propres** : les directions que la transformation ne fait qu'étirer, $Av = \lambda v$. Sur une matrice de covariance, ce sont les axes de plus grande variance des données.

**Cas d'utilisation**
- Écrire un modèle linéaire : $\hat{y} = Xw + b$, un produit matriciel pour tous les exemples à la fois.
- Calculer une similarité entre documents ou entre utilisateurs (cosinus = produit scalaire normalisé).
- Réduire la dimension (ACP, SVD) et compresser des données.
- Comprendre pourquoi une matrice mal conditionnée rend une régression instable.

**Algorithme**
```text
Vérifications de forme à faire avant tout calcul :
1. (n, d) @ (d, k) -> (n, k). Les dimensions intérieures doivent coïncider.
2. Transposer avec .T pour aligner : X.T @ X est (d, d), X @ X.T est (n, n).
3. Une réduction sur un axe fait disparaître cet axe.
4. Produit scalaire = X @ w ; norme = np.linalg.norm ; projection = X @ V @ V.T.
5. Ne jamais inverser une matrice explicitement : résoudre le système.
```

**Implémentation**
```python
import numpy as np

X = np.random.randn(200, 5)          # 200 exemples, 5 features
w = np.array([1.5, -2.0, 0.0, 0.5, 3.0])
y = X @ w + 0.1 * np.random.randn(200)

# Moindres carrés : équation normale (X'X)w = X'y
# À NE PAS écrire : np.linalg.inv(X.T @ X) @ X.T @ y  (instable)
w_hat = np.linalg.solve(X.T @ X, X.T @ y)
# Encore mieux, robuste au mauvais conditionnement :
w_hat, residus, rang, sv = np.linalg.lstsq(X, y, rcond=None)

# Similarité cosinus entre toutes les paires de lignes
Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
S = Xn @ Xn.T                        # (200, 200) dans [-1, 1]

# Axes principaux = vecteurs propres de la covariance
C = np.cov(X, rowvar=False)          # (5, 5)
val, vec = np.linalg.eigh(C)         # eigh : matrice symétrique, valeurs triées
variance_expliquee = val[::-1] / val.sum()

# Conditionnement : > 1e4 annonce des ennuis numériques
print(np.linalg.cond(X.T @ X))
```

**Outils** — `numpy` (`numpy.linalg`), `scipy.linalg` pour les décompositions avancées et les matrices creuses.

**Alternatives open-source**
- *Bibliothèques* : **SciPy** (`scipy.sparse`) pour des matrices creuses de plusieurs millions de lignes ; **PyTorch** / **JAX** pour le même algèbre avec dérivation automatique et GPU ; **SymPy** pour vérifier une dérivation à la main symboliquement.
- *Outils graphiques* : la série **3Blue1Brown « Essence of Linear Algebra »** est le meilleur support d'intuition existant, gratuit ; **GeoGebra** pour manipuler transformations et vecteurs propres à la souris ; **Immersive Math** (livre interactif en ligne) ; **Octave** comme bac à sable de calcul matriciel.

**Astuces**
- Utiliser `eigh` et non `eig` sur une matrice symétrique (covariance, gramienne) : plus rapide, plus stable, valeurs réelles garanties.
- Ne jamais calculer $A^{-1}b$ : utiliser `solve`. L'inversion explicite perd en précision et coûte plus cher.
- `@` est le produit matriciel, `*` est le produit élément par élément. La confusion produit un résultat de forme plausible et de valeur fausse — le pire type de bug.
- Le conditionnement élevé se traite en centrant-réduisant les features, ou en ajoutant une régularisation ridge $(X^\top X + \lambda I)$ qui rend le système inversible.
- Retenir : le rang de $X^\top X$ ne peut pas dépasser $\min(n, d)$. Si $d > n$, le système est sous-déterminé — d'où l'obligation de régulariser en grande dimension.

## Dérivées et gradients
<!-- slug: derivees-gradients | difficulte: 3 | prereqs: algebre-lineaire -->

**En une phrase** — Le gradient d'une fonction est le vecteur qui pointe dans la direction de plus forte augmentation ; l'apprentissage consiste à marcher dans le sens inverse.

**Explication** — Pour une fonction $f$ de plusieurs variables, la dérivée partielle $\partial f / \partial w_j$ mesure la variation de $f$ quand on bouge $w_j$ seul. Le **gradient** $\nabla f$ rassemble toutes ces dérivées partielles en un vecteur. Sa direction est celle de la plus forte pente montante, et sa norme dit à quel point la pente est raide. Un gradient nul signale un point critique : minimum, maximum, ou point selle.

Toute la mécanique de l'apprentissage tient dans la **règle de la chaîne**. Une perte $L$ dépend de la prédiction $\hat{y}$, qui dépend d'une activation $z$, qui dépend des poids $w$. Alors $\frac{\partial L}{\partial w} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial w}$. On multiplie les dérivées locales le long du chemin. Cette formule, appliquée en remontant un graphe de calcul, *est* la rétropropagation — il n'y a rien de plus.

Trois gradients à connaître par cœur, parce qu'ils reviennent partout. Pour l'erreur quadratique $L = \frac{1}{n}\|Xw - y\|^2$ : $\nabla_w L = \frac{2}{n} X^\top (Xw - y)$. Pour la régression logistique avec entropie croisée : $\nabla_w L = \frac{1}{n} X^\top (\sigma(Xw) - y)$ — remarquablement, la même forme. Pour la pénalité ridge $\lambda\|w\|^2$ : $2\lambda w$.

**Cas d'utilisation**
- Entraîner n'importe quel modèle paramétrique, du modèle linéaire au Transformer.
- Diagnostiquer un entraînement qui ne converge pas (gradients qui explosent ou disparaissent).
- Calculer l'importance d'une feature ou d'un pixel par sensibilité (`saliency maps`).
- Vérifier une implémentation *from scratch* par comparaison au gradient numérique.

**Algorithme**
```text
Dérivation d'un gradient à la main, méthode sûre :
1. Écrire la perte pour UN exemple, avec des indices explicites.
2. Dériver par rapport à un seul paramètre w_j, en appliquant la chaîne
   terme par terme, de la perte vers le paramètre.
3. Reconnaître la forme vectorielle : une somme sur i devient un produit
   matriciel avec X.T.
4. Vérifier les formes : grad doit avoir exactement la forme de w.
5. Vérifier numériquement : (f(w + eps) - f(w - eps)) / (2 eps).
```

**Implémentation**
```python
import numpy as np

def perte_et_gradient(w, X, y, lam=0.0):
    """MSE + ridge. Retourne (perte, gradient)."""
    n = len(y)
    residu = X @ w - y                      # (n,)
    perte = (residu @ residu) / n + lam * (w @ w)
    grad = 2 / n * X.T @ residu + 2 * lam * w
    return perte, grad

def verifier_gradient(f, w, eps=1e-6):
    """Différences finies centrées : indispensable pour valider tout gradient."""
    _, g_analytique = f(w)
    g_num = np.zeros_like(w)
    for j in range(len(w)):
        wp, wm = w.copy(), w.copy()
        wp[j] += eps
        wm[j] -= eps
        g_num[j] = (f(wp)[0] - f(wm)[0]) / (2 * eps)
    err = np.abs(g_analytique - g_num).max()
    print(f"écart max = {err:.2e}")          # doit être < 1e-6
    return err

X, y = np.random.randn(50, 4), np.random.randn(50)
verifier_gradient(lambda w: perte_et_gradient(w, X, y, lam=0.1), np.random.randn(4))
```

**Outils** — `numpy` pour le calcul manuel ; `torch.autograd` ou `jax.grad` pour la dérivation automatique.

**Alternatives open-source**
- *Bibliothèques* : **PyTorch** (`loss.backward()`) construit le graphe à l'exécution et dérive tout seul ; **JAX** (`jax.grad`) dérive une fonction Python pure et se compose (`grad(grad(f))` pour la hessienne) ; **SymPy** pour obtenir la formule symbolique et la comparer à sa dérivation papier ; **micrograd** (Karpathy, 150 lignes) est le meilleur support pour comprendre l'autograd de l'intérieur.
- *Outils graphiques* : **Wolfram Alpha** pour vérifier une dérivée en une seconde ; **Desmos** pour visualiser une fonction et sa dérivée ; **TensorFlow Playground** pour voir un réseau apprendre en direct dans le navigateur ; **Netron** pour inspecter un graphe de calcul exporté.

**Astuces**
- Toujours vérifier un gradient codé à la main par différences finies avant de l'utiliser. Un gradient faux ne plante pas : il apprend mal, et on accuse les hyperparamètres pendant deux jours.
- `eps = 1e-6` est le bon compromis en `float64`. En `float32` la vérification numérique n'est pas fiable.
- Un gradient dont la norme dépasse $10^3$ annonce une divergence : réduire le pas, ou écrêter (`clip_grad_norm_`).
- Le gradient de la perte moyenne dépend de la taille du lot ; celui de la perte sommée non. Cette confusion explique bien des pas d'apprentissage à recalibrer après un changement de `batch_size`.
- La règle de la chaîne se lit de gauche à droite en avant, de droite à gauche en arrière. Dessiner le graphe de calcul avant de dériver, systématiquement.

## Descente de gradient
<!-- slug: descente-de-gradient | difficulte: 2 | prereqs: derivees-gradients -->

**En une phrase** — Partir d'un point au hasard et faire des petits pas dans la direction opposée au gradient jusqu'à atteindre un minimum de la perte.

**Explication** — La règle de mise à jour est $w \leftarrow w - \eta \nabla L(w)$, où $\eta$ est le **taux d'apprentissage**. Trop petit, la convergence prend des heures ; trop grand, l'algorithme rebondit et diverge. C'est l'hyperparamètre le plus important de tout le ML, et le premier à régler — par recherche logarithmique ($10^{-1}, 10^{-2}, 10^{-3}...$), jamais linéaire.

Trois variantes se distinguent par la quantité de données utilisée à chaque pas. La descente **par lot complet** calcule le gradient sur tout le dataset : trajectoire lisse, mais un pas coûte un passage complet. La descente **stochastique** (SGD) utilise un seul exemple : très bruitée, très rapide, et ce bruit aide paradoxalement à s'échapper des mauvais minima. La descente **par mini-lots** (32 à 512 exemples) est le compromis universel : assez stable, et surtout elle exploite la parallélisation matérielle.

Sur une fonction **convexe** (moindres carrés, régression logistique), il n'existe qu'un minimum et la convergence est garantie pour $\eta$ assez petit. Sur un réseau de neurones, la surface est non convexe, criblée de points selles et de minima locaux — l'expérience montre que ce n'est pas un problème pratique en grande dimension, parce que la plupart des minima locaux sont de qualité comparable.

**Cas d'utilisation**
- Tout modèle sans solution analytique : régression logistique, SVM, réseaux de neurones.
- Datasets trop gros pour une solution fermée, même quand elle existe.
- Apprentissage en ligne, où les données arrivent en flux continu.
- À éviter quand une solution exacte est disponible et le dataset petit : `lstsq` est plus précis et instantané.

**Algorithme**
```text
Entrée : X, y, taux eta, nb_epoques, taille_lot
1. Initialiser w (zéros pour un modèle linéaire, aléatoire pour un réseau).
2. Répéter nb_epoques fois :
     a. Mélanger les indices du dataset.
     b. Pour chaque mini-lot (X_b, y_b) :
          g = gradient de la perte sur ce lot
          w = w - eta * g
     c. Calculer la perte de validation ; l'enregistrer.
3. Arrêter si la perte de validation ne s'améliore plus depuis p époques
   (arrêt précoce) et restaurer les meilleurs poids.
Sortie : w
```

**Implémentation**
```python
import numpy as np

def descente(X, y, eta=0.1, epoques=100, taille_lot=32, patience=10, graine=0):
    rng = np.random.default_rng(graine)
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    historique, meilleur, attente = [], (np.inf, w, b), 0

    for ep in range(epoques):
        idx = rng.permutation(n)
        for debut in range(0, n, taille_lot):
            lot = idx[debut:debut + taille_lot]
            Xb, yb = X[lot], y[lot]
            residu = Xb @ w + b - yb
            w -= eta * (2 / len(lot)) * Xb.T @ residu
            b -= eta * (2 / len(lot)) * residu.sum()

        perte = ((X @ w + b - y) ** 2).mean()
        historique.append(perte)
        if perte < meilleur[0] - 1e-6:
            meilleur, attente = (perte, w.copy(), b), 0
        else:
            attente += 1
            if attente >= patience:
                break                      # arrêt précoce
    return meilleur[1], meilleur[2], historique

# Décroissance du pas : souvent plus efficace qu'un pas fixe bien réglé
eta_t = lambda t, eta0=0.1, k=0.01: eta0 / (1 + k * t)
```

**Outils** — `numpy` pour l'implémentation manuelle ; `scikit-learn` (`SGDRegressor`, `SGDClassifier`) ; `torch.optim` pour les variantes modernes.

**Alternatives open-source**
- *Bibliothèques* : **scikit-learn** `SGDClassifier` avec `learning_rate='optimal'` évite tout réglage manuel ; **torch.optim** fournit Adam, AdamW, RMSprop et les ordonnanceurs de pas ; **scipy.optimize** propose L-BFGS, quasi-newtonien, excellent sur des problèmes convexes de dimension moyenne ; **Optuna** pour chercher le taux d'apprentissage automatiquement.
- *Outils graphiques* : **TensorFlow Playground** montre l'effet du taux d'apprentissage en temps réel ; **losslandscape.com** visualise des surfaces de perte réelles ; **TensorBoard** ou **Weights & Biases** (`wandb`, cœur open-source) pour suivre des courbes d'entraînement.

**Astuces**
- **Toujours standardiser les features avant une descente de gradient.** Sur des échelles inégales, la surface de perte devient une vallée étroite et l'algorithme zigzague. C'est la première cause de non-convergence, et elle passe pour un mauvais choix de $\eta$.
- Tracer la courbe de perte à chaque fois. Une perte qui remonte = pas trop grand. Une perte plate = pas trop petit ou gradient nul. Une perte en escalier = taille de lot trop petite.
- Faire un test de sur-apprentissage volontaire sur 10 exemples : si le modèle n'atteint pas une perte quasi nulle, il y a un bug, pas un problème de données.
- Le mélange des données à chaque époque n'est pas facultatif : sans lui, un dataset trié par classe fait osciller le modèle.
- Un `NaN` dans la perte apparaît presque toujours au premier pas trop grand, ou via un $\log(0)$. Diviser $\eta$ par 10 et vérifier les bornes des logarithmes.

## Probabilités et théorème de Bayes
<!-- slug: probabilites-bayes | difficulte: 3 | prereqs:  -->

**En une phrase** — Le cadre qui permet de raisonner sur l'incertain, et la formule qui met à jour une croyance à la lumière d'une observation.

**Explication** — Une probabilité **conditionnelle** $P(A \mid B)$ est la probabilité de $A$ sachant que $B$ est réalisé. La règle du produit $P(A, B) = P(A \mid B) P(B)$ mène directement au théorème de Bayes :

$$P(H \mid D) = \frac{P(D \mid H)\, P(H)}{P(D)}$$

En mots : la probabilité de l'hypothèse sachant les données (*postérieure*) est proportionnelle à la vraisemblance des données sous l'hypothèse, multipliée par la croyance initiale (*a priori*). Le dénominateur ne sert qu'à normaliser. Tout le ML probabiliste est une déclinaison de cette ligne : un classifieur estime $P(\text{classe} \mid \text{features})$.

Le piège central, et il est constant en pratique : $P(A \mid B) \neq P(B \mid A)$. Un test médical détectant 99 % des malades, avec 1 % de faux positifs, sur une maladie touchant 1 personne sur 10 000, donne un test positif juste dans moins de 1 % des cas. La raison est l'a priori : les faux positifs des 9 999 bien-portants écrasent le vrai positif. C'est exactement le problème de la détection de fraude, du diagnostic rare et de toute classe minoritaire — et cela explique pourquoi l'exactitude (*accuracy*) est un indicateur trompeur.

L'**indépendance** $P(A, B) = P(A)P(B)$ est ce qui rend les calculs faisables. Elle est presque toujours fausse et souvent utile quand même : c'est l'hypothèse « naïve » de Naive Bayes.

**Cas d'utilisation**
- Interpréter correctement les métriques d'un classifieur sur une classe rare.
- Naive Bayes pour la classification de texte, les filtres anti-spam.
- Choix d'un seuil de décision selon le coût des erreurs et la prévalence.
- Modélisation bayésienne : intégrer une connaissance experte sous forme d'a priori, obtenir des intervalles de crédibilité au lieu de points.

**Algorithme**
```text
Résolution d'un problème bayésien, en 5 lignes :
1. Nommer l'hypothèse H et l'observation D sans ambiguïté.
2. Écrire l'a priori P(H) — souvent la prévalence de base.
3. Écrire la vraisemblance P(D|H) et P(D|non H).
4. P(D) = P(D|H)P(H) + P(D|non H)P(non H)   (loi des probabilités totales)
5. Diviser. Puis vérifier l'ordre de grandeur sur 10 000 cas fictifs :
   raisonner en effectifs, pas en pourcentages, rend l'erreur visible.
```

**Implémentation**
```python
def bayes(p_h, p_d_si_h, p_d_si_non_h):
    """Probabilité postérieure de H sachant D."""
    p_d = p_d_si_h * p_h + p_d_si_non_h * (1 - p_h)
    return p_d_si_h * p_h / p_d

# Test médical : sensibilité 99 %, faux positifs 1 %, prévalence 1/10 000
print(bayes(1e-4, 0.99, 0.01))       # 0.0098 -> moins de 1 % de chances d'être malade

# Vérification par effectifs sur 1 000 000 de personnes
malades = 100
vrais_pos = 0.99 * malades                      # 99
faux_pos = 0.01 * (1_000_000 - malades)         # 9999
print(vrais_pos / (vrais_pos + faux_pos))       # même résultat, intuition plus claire

# Naive Bayes gaussien from scratch : un classifieur en 12 lignes
import numpy as np

class NaiveBayesGaussien:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.prior = np.array([(y == c).mean() for c in self.classes])
        self.mu = np.array([X[y == c].mean(axis=0) for c in self.classes])
        self.var = np.array([X[y == c].var(axis=0) + 1e-9 for c in self.classes])
        return self

    def log_vraisemblance(self, X):
        # somme des log-densités gaussiennes : l'hypothèse naïve d'indépendance
        z = (X[:, None, :] - self.mu[None]) ** 2 / self.var[None]
        return -0.5 * (z + np.log(2 * np.pi * self.var[None])).sum(axis=2)

    def predict(self, X):
        scores = self.log_vraisemblance(X) + np.log(self.prior)[None]
        return self.classes[scores.argmax(axis=1)]
```

**Outils** — `scipy.stats` pour les distributions ; `scikit-learn` (`GaussianNB`, `MultinomialNB`) ; `numpy`.

**Alternatives open-source**
- *Bibliothèques* : **PyMC** pour l'inférence bayésienne complète par MCMC, API Python très lisible ; **NumPyro** / **Stan** pour la même chose en plus rapide et plus scalable ; **ArviZ** pour diagnostiquer et visualiser des postérieures ; **pomegranate** pour les modèles graphiques probabilistes et les chaînes de Markov cachées.
- *Outils graphiques* : **Seeing Theory** (Brown University) est une introduction visuelle interactive remarquable ; **Bayes' theorem visualisation** de 3Blue1Brown ; **JASP** et **Jamovi**, interfaces libres de statistiques avec modules bayésiens ; **BayesiaLab** (partiellement libre) pour construire des réseaux bayésiens à la souris.

**Astuces**
- Toujours retraduire un énoncé probabiliste en effectifs sur une population de 10 000. L'erreur de raisonnement devient immédiatement visible.
- Travailler en **log-probabilités** dès qu'on multiplie plus de quelques termes : le produit de 500 probabilités vaut 0 en `float64`. `scipy.special.logsumexp` gère la normalisation sans dépassement.
- Ajouter un lissage de Laplace (`alpha=1`) dans Naive Bayes multinomial : sans lui, un mot jamais vu dans une classe annule toute la probabilité.
- Les probabilités renvoyées par `predict_proba` ne sont *pas* des probabilités fiables pour la plupart des modèles (voir la fiche calibration). Naive Bayes est particulièrement mal calibré, tout en classant correctement.
- La prévalence change tout : un modèle entraîné sur des données ré-équilibrées produit des probabilités biaisées qu'il faut recalibrer avant usage.

## Statistiques inférentielles
<!-- slug: statistiques-inferentielles | difficulte: 3 | prereqs: probabilites-bayes -->

**En une phrase** — Ce qu'on peut affirmer sur une population entière à partir d'un échantillon, et avec quelle marge d'erreur.

**Explication** — Une statistique calculée sur un échantillon (une moyenne, une exactitude de modèle) est elle-même une variable aléatoire : un autre échantillon aurait donné un autre nombre. Sa dispersion s'appelle l'**erreur type**, et pour une moyenne elle vaut $\sigma/\sqrt{n}$. Le $\sqrt{n}$ explique une règle empirique majeure : diviser l'incertitude par deux exige quatre fois plus de données. Le **théorème central limite** garantit que la moyenne d'un échantillon suffisamment grand suit une loi normale, quelle que soit la distribution d'origine — c'est ce qui rend possible tout intervalle de confiance.

Un **intervalle de confiance** à 95 % est un intervalle produit par une procédure qui, répétée sur de nombreux échantillons, contiendrait la vraie valeur 95 fois sur 100. Ce n'est pas « 95 % de chances que la vraie valeur soit dedans » — cette formulation est bayésienne et correspond à un *intervalle de crédibilité*. En pratique, la distinction change peu de choses ; ce qui change tout est de **donner un intervalle** plutôt qu'un nombre nu.

Un **test d'hypothèse** calcule la probabilité d'observer un écart au moins aussi grand si l'hypothèse nulle était vraie : c'est la p-valeur. Une p-valeur de 0,03 ne dit rien sur la taille de l'effet, ni sur son intérêt pratique. Avec 10 millions de lignes, tout écart devient « significatif ». D'où l'insistance moderne sur la **taille d'effet** et les intervalles, plutôt que sur le seuil de 0,05.

Pour le ML, la technique la plus utile est le **bootstrap** : rééchantillonner avec remise dans son propre jeu de test, recalculer la métrique 1 000 fois, et lire les percentiles 2,5 et 97,5. Aucune hypothèse de distribution, applicable à n'importe quelle métrique, dix lignes de code.

**Cas d'utilisation**
- Donner un intervalle de confiance sur le score d'un modèle au lieu d'un « 87,3 % » faussement précis.
- Décider si le modèle B est réellement meilleur que le modèle A, ou si l'écart tient au hasard du découpage.
- Dimensionner un test A/B : combien d'utilisateurs pour détecter une amélioration de 1 % ?
- Détecter une dérive de distribution en production (test de Kolmogorov-Smirnov, PSI).

**Algorithme**
```text
Bootstrap d'une métrique (le couteau suisse) :
1. Soit (y_vrai, y_pred) de taille n sur le jeu de test.
2. Répéter B = 1000 fois :
     a. Tirer n indices avec remise.
     b. Calculer la métrique sur ce rééchantillon ; stocker.
3. Intervalle à 95 % = percentiles 2.5 et 97.5 des B valeurs.
4. Comparer deux modèles : bootstrapper la DIFFÉRENCE des métriques
   sur les mêmes indices. Si l'intervalle contient 0, l'écart n'est pas établi.
```

**Implémentation**
```python
import numpy as np
from sklearn.metrics import roc_auc_score

def bootstrap_metrique(y, p, metrique=roc_auc_score, B=1000, graine=0):
    rng = np.random.default_rng(graine)
    n = len(y)
    vals = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:      # garde-fou : classe absente du tirage
            vals[b] = np.nan
            continue
        vals[b] = metrique(y[idx], p[idx])
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return np.nanmean(vals), lo, hi

def comparer(y, pa, pb, B=1000, graine=0):
    """Bootstrap apparié : mêmes indices pour les deux modèles."""
    rng = np.random.default_rng(graine)
    n, diffs = len(y), np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        diffs[b] = roc_auc_score(y[idx], pb[idx]) - roc_auc_score(y[idx], pa[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    verdict = "B meilleur" if lo > 0 else "A meilleur" if hi < 0 else "indécidable"
    return diffs.mean(), lo, hi, verdict

# Dérive de distribution entre entraînement et production
from scipy.stats import ks_2samp
stat, p = ks_2samp(x_train, x_prod)
```

**Outils** — `scipy.stats`, `numpy`, `statsmodels` pour les modèles statistiques et les diagnostics.

**Alternatives open-source**
- *Bibliothèques* : **statsmodels** donne des tables de régression complètes avec p-valeurs, intervalles et tests de diagnostic — irremplaçable quand l'inférence compte plus que la prédiction ; **pingouin** simplifie les tests classiques avec tailles d'effet incluses par défaut ; **scipy.stats.bootstrap** implémente déjà le bootstrap avec correction BCa ; **PyMC** pour l'approche bayésienne.
- *Outils graphiques* : **JASP** et **Jamovi** (interfaces libres, sorties APA propres, modules bayésiens) ; **R + RStudio** reste la référence pour l'analyse statistique fine ; **Evidently AI** pour surveiller la dérive de données en production avec rapports visuels ; **G*Power** pour le calcul de puissance.

**Astuces**
- Ne jamais rapporter une métrique sans son intervalle. Sur 500 exemples de test, une exactitude de 87 % a un intervalle d'environ ±3 points : la différence avec un modèle à 85 % n'est pas établie.
- Comparer deux modèles par bootstrap **apparié** (mêmes rééchantillons) : la variance de la différence est bien plus faible que celle des deux métriques séparées, le test est donc plus puissant.
- Attention au multiple testing : tester 20 hypothèses à 5 % donne en moyenne une « découverte » fausse. Corriger (Bonferroni, Benjamini-Hochberg) ou assumer l'exploration.
- La p-valeur n'est pas la probabilité que l'hypothèse nulle soit vraie. Cette confusion est si répandue qu'elle a un nom (*sophisme du procureur*).
- Un test de normalité sur 100 000 points rejette toujours la normalité. Regarder un QQ-plot vaut mieux qu'un test.
- Pour dimensionner un test A/B : $n \approx 16\sigma^2/\delta^2$ par groupe pour détecter un écart $\delta$ avec 80 % de puissance. Utile de tête.

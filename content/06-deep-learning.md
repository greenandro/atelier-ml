---
module: deep-learning
titre: Deep learning — fondations
ordre: 6
resume: Sur données tabulaires, le boosting reste meilleur. Le deep learning gagne sur image, texte, audio, et sur tout ce qui a une structure.
---

## Perceptron multicouche
<!-- slug: perceptron-mlp | difficulte: 3 | prereqs: regression-logistique, descente-de-gradient -->

**En une phrase** — Empiler des transformations linéaires séparées par des non-linéarités, ce qui permet d'approcher n'importe quelle fonction continue.

**Explication** — Une couche calcule $h = \phi(Wx + b)$ : une transformation affine suivie d'une fonction d'activation appliquée élément par élément. Empiler des couches sans activation ne sert à rien — la composition de deux applications linéaires est linéaire. C'est la non-linéarité $\phi$ qui crée la capacité d'expression. Le **théorème d'approximation universelle** garantit qu'une seule couche cachée suffisamment large approche n'importe quelle fonction continue sur un compact ; en pratique, plusieurs couches étroites sont bien plus efficaces qu'une couche énorme, car elles composent des représentations hiérarchiques.

L'architecture se lit par ses dimensions. L'entrée a $d$ neurones, la sortie en a 1 pour une régression ou une classification binaire, $K$ pour $K$ classes. Les couches cachées sont le choix libre : deux ou trois couches de 64 à 512 neurones couvrent l'immense majorité des besoins sur données tabulaires. La sortie n'a **pas** d'activation en régression, une sigmoïde en binaire, une softmax en multi-classe — et en pratique on garde les logits bruts en laissant la fonction de perte appliquer la transformation, pour la stabilité numérique.

Une réalité à connaître : sur des données tabulaires, un MLP est presque toujours battu par un gradient boosting, à effort égal ou supérieur. Le deep learning s'impose quand les données ont une **structure** exploitable — voisinage spatial des pixels, ordre des mots, continuité du signal — et que cette structure peut être encodée dans l'architecture.

**Cas d'utilisation**
- Briques de sortie de toute architecture profonde (après un CNN, après un Transformer).
- Apprentissage de représentations : embeddings d'entités à très haute cardinalité (millions d'identifiants), là où le one-hot est impossible.
- Systèmes de recommandation à deux tours.
- Auto-encodeurs pour compression et détection d'anomalies.
- Sorties multiples ou pertes personnalisées difficiles à exprimer dans un modèle classique.
- Mauvais choix comme premier modèle sur un CSV de 50 000 lignes : plus lent à régler et moins performant qu'un LightGBM.

**Algorithme**
```text
Passe avant :
  a0 = x
  pour l = 1..L :  z_l = W_l a_{l-1} + b_l ;  a_l = phi(z_l)
  sortie = a_L (logits)
  perte = L(sortie, y)

Entraînement (une époque) :
  pour chaque mini-lot :
    1. passe avant, calcul de la perte
    2. rétropropagation : gradients de tous les W_l, b_l
    3. optimizer.step() puis remise à zéro des gradients
  évaluer sur validation ; arrêt précoce sur cette métrique

Réglages, par ordre d'importance :
  taux d'apprentissage >> architecture > régularisation > taille de lot
```

**Implémentation**
```python
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

def faire_mlp(d_entree, couches=(256, 128), n_sorties=1, p_dropout=0.2):
    modules, prec = [], d_entree
    for h in couches:
        modules += [nn.Linear(prec, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p_dropout)]
        prec = h
    modules.append(nn.Linear(prec, n_sorties))     # pas d'activation : logits bruts
    return nn.Sequential(*modules)

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
modele = faire_mlp(X_tr.shape[1]).to(dev)
perte_fn = nn.BCEWithLogitsLoss()                  # sigmoïde + BCE fusionnées, stable
opt = torch.optim.AdamW(modele.parameters(), lr=1e-3, weight_decay=1e-2)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)

dl_tr = DataLoader(TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                                 torch.tensor(y_tr, dtype=torch.float32)),
                   batch_size=256, shuffle=True, drop_last=True)

meilleur, attente = float('inf'), 0
for epoque in range(200):
    modele.train()
    for xb, yb in dl_tr:
        xb, yb = xb.to(dev), yb.to(dev)
        opt.zero_grad()
        perte = perte_fn(modele(xb).squeeze(-1), yb)
        perte.backward()
        torch.nn.utils.clip_grad_norm_(modele.parameters(), 1.0)
        opt.step()

    modele.eval()
    with torch.no_grad():
        p_val = perte_fn(modele(X_val_t).squeeze(-1), y_val_t).item()
    sched.step(p_val)
    if p_val < meilleur - 1e-4:
        meilleur, attente = p_val, 0
        torch.save(modele.state_dict(), 'meilleur.pt')
    else:
        attente += 1
        if attente >= 15:
            break
modele.load_state_dict(torch.load('meilleur.pt'))

# Embeddings d'entités : le vrai atout du deep learning sur données tabulaires
class ModeleAvecEmbeddings(nn.Module):
    def __init__(self, cardinalites, d_num):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c, min(50, (c + 1) // 2)) for c in cardinalites])
        d_tot = sum(e.embedding_dim for e in self.embs) + d_num
        self.tete = faire_mlp(d_tot, (256, 128), 1)

    def forward(self, x_cat, x_num):
        e = [emb(x_cat[:, i]) for i, emb in enumerate(self.embs)]
        return self.tete(torch.cat(e + [x_num], dim=1))
```

**Outils** — `pip install torch` (choisir la variante CUDA sur le site officiel si GPU).

**Alternatives open-source**
- *Bibliothèques* : **PyTorch Lightning** supprime la boucle d'entraînement répétitive tout en gardant PyTorch dessous ; **fastai** propose d'excellents défauts et un modèle tabulaire prêt à l'emploi ; **skorch** enveloppe un modèle PyTorch dans l'API scikit-learn, donc compatible `GridSearchCV` et `Pipeline` ; **pytorch-tabular** implémente TabNet, FT-Transformer et NODE, les architectures spécifiquement conçues pour le tabulaire ; **JAX + Flax** pour la recherche et la compilation ; **Keras 3** fonctionne au-dessus de PyTorch, JAX ou TensorFlow.
- *Outils graphiques* : **TensorBoard** pour suivre pertes, gradients et poids en direct ; **Netron** visualise une architecture exportée en ONNX, très utile pour comprendre un modèle repris ; **TensorFlow Playground** pour l'intuition sur l'effet des couches et activations ; **Weights & Biases** (cœur libre) pour comparer des dizaines d'exécutions.

**Astuces**
- Standardiser les entrées est encore plus critique qu'en ML classique : un réseau sur données non normalisées ne converge simplement pas.
- Toujours faire le test de sur-apprentissage sur 100 exemples avant l'entraînement complet. Si le modèle n'atteint pas une perte quasi nulle, le bug est dans le code ou les données, pas dans les hyperparamètres.
- Utiliser `BCEWithLogitsLoss` et `CrossEntropyLoss` sur les **logits**, jamais une sigmoïde ou softmax suivie d'un log : les versions fusionnées sont numériquement stables.
- `CrossEntropyLoss` de PyTorch applique déjà la softmax. La double application est un bug fréquent qui produit un modèle qui apprend mal sans erreur visible.
- Oublier `opt.zero_grad()` accumule les gradients d'un lot au suivant. Symptôme : divergence brutale après quelques itérations.
- Oublier `modele.eval()` à l'évaluation laisse le dropout actif et la batch norm en mode entraînement : les scores de validation deviennent bruités et pessimistes.
- Commencer par `lr=1e-3` avec AdamW. Si la perte diverge, diviser par 10 ; si elle stagne, multiplier par 3.

## Fonctions d'activation
<!-- slug: fonctions-activation | difficulte: 2 | prereqs: perceptron-mlp -->

**En une phrase** — La non-linéarité appliquée après chaque couche, qui détermine ce que le réseau peut représenter et la facilité avec laquelle il s'entraîne.

**Explication** — Historiquement, la **sigmoïde** et la **tangente hyperbolique** dominaient. Elles ont un défaut fatal en profondeur : leur dérivée vaut au plus 0,25 (sigmoïde) et sature à zéro pour de grandes entrées. En rétropropageant à travers dix couches, on multiplie dix dérivées inférieures à 1, et le gradient **disparaît**. C'est ce qui a bloqué le deep learning pendant deux décennies.

**ReLU** ($\max(0, x)$) a levé le verrou : sa dérivée vaut exactement 1 sur les positifs, donc le gradient traverse intact. Elle est aussi triviale à calculer, et elle produit une **parcimonie** naturelle (environ la moitié des neurones inactifs). Son défaut est le neurone « mort » : un neurone dont l'entrée reste négative pour tous les exemples a un gradient nul et ne se réactive jamais. **LeakyReLU** ($\max(\alpha x, x)$ avec $\alpha = 0{,}01$) et **ELU** corrigent cela en laissant passer une petite pente négative.

Les activations modernes sont lisses : **GELU** pondère l'entrée par la probabilité gaussienne cumulée, **SiLU/Swish** vaut $x \cdot \sigma(x)$. Elles gagnent un point ou deux sur les grandes architectures et sont le standard dans les Transformers. **GLU** et **SwiGLU** ajoutent un mécanisme de porte multiplicative et équipent la plupart des grands modèles de langage actuels.

Pour la couche de sortie, le choix n'est pas libre : rien en régression, sigmoïde en binaire, softmax en multi-classe exclusif, sigmoïdes indépendantes en multi-étiquette.

**Cas d'utilisation**
- ReLU par défaut pour tout MLP et tout CNN : rapide, éprouvée, sans réglage.
- GELU ou SiLU dans les Transformers et les architectures profondes récentes.
- LeakyReLU quand on observe beaucoup de neurones morts (activations nulles sur tout un lot).
- Tanh dans les RNN classiques et là où une sortie bornée et centrée est requise.
- Softmax uniquement en sortie multi-classe, jamais dans une couche cachée.

**Algorithme**
```text
Choix, en pratique :
  couches cachées, cas général     -> ReLU
  Transformer, réseau très profond -> GELU ou SiLU
  neurones morts observés          -> LeakyReLU(0.01) ou ELU
  RNN, sortie bornée               -> tanh
  porte (LSTM, GRU, GLU)           -> sigmoïde

Couche de sortie :
  régression                       -> aucune activation
  régression positive              -> softplus, ou log de la cible
  binaire                          -> logits + BCEWithLogitsLoss
  multi-classe exclusif            -> logits + CrossEntropyLoss
  multi-étiquette                  -> logits + BCEWithLogitsLoss (K sorties)

Diagnostic : tracer la distribution des activations par couche.
  Tout à zéro       -> neurones morts, lr trop grand ou init mauvaise
  Tout saturé       -> revoir la normalisation d'entrée
  Écart croissant   -> ajouter une normalisation de couche
```

**Implémentation**
```python
import torch, torch.nn as nn, numpy as np

# --- from scratch : fonctions et dérivées (palier 2) ---
relu       = lambda z: np.maximum(0, z)
d_relu     = lambda z: (z > 0).astype(z.dtype)
leaky      = lambda z, a=0.01: np.where(z > 0, z, a * z)
d_leaky    = lambda z, a=0.01: np.where(z > 0, 1.0, a)
sigmoide   = lambda z: np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))
d_sigmoide = lambda z: sigmoide(z) * (1 - sigmoide(z))     # <= 0.25 : voilà le problème
gelu       = lambda z: 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z ** 3)))

# Illustration du gradient qui disparaît
for nom, d in [('sigmoide', d_sigmoide), ('tanh', lambda z: 1 - np.tanh(z) ** 2),
               ('relu', d_relu)]:
    z = np.random.randn(1000)
    print(f"{nom:9s} produit de 10 dérivées : {np.prod([d(z).mean()] * 10):.2e}")

# --- diagnostic sur un vrai réseau : distribution des activations ---
activations = {}

def sonde(nom):
    def hook(_module, _entree, sortie):
        activations[nom] = sortie.detach()
    return hook

for i, m in enumerate(modele):
    if isinstance(m, nn.ReLU):
        m.register_forward_hook(sonde(f'relu_{i}'))

modele.eval()
with torch.no_grad():
    modele(x_lot)
for nom, a in activations.items():
    morts = (a == 0).float().mean().item()
    print(f"{nom:10s} moy={a.mean():.3f} std={a.std():.3f} morts={morts:.1%}")
    # morts > 50 % de façon persistante -> passer à LeakyReLU ou baisser le lr

# Bloc moderne typique
bloc = nn.Sequential(nn.Linear(256, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.1))
```

**Outils** — `torch.nn` (`ReLU`, `LeakyReLU`, `GELU`, `SiLU`, `ELU`, `Softplus`, `Mish`).

**Alternatives open-source**
- *Bibliothèques* : **torch.nn.functional** donne les versions fonctionnelles de toutes les activations ; **timm** (Ross Wightman) implémente les activations exotiques et permet de comparer leurs effets sur des modèles de vision ; **Keras 3** pour la même palette dans un autre cadre ; **JAX/Flax** avec `nn.gelu`, `nn.silu`.
- *Outils graphiques* : **TensorFlow Playground** montre en direct l'effet du choix d'activation sur la frontière apprise ; **TensorBoard** avec les histogrammes d'activations par couche, exactement le diagnostic décrit ci-dessus ; **Netron** pour lire les activations d'une architecture importée.

**Astuces**
- Ne pas chercher de gain dans le choix d'activation avant d'avoir réglé le taux d'apprentissage, la normalisation et l'architecture. Le gain y est de l'ordre de 1 %, contre 20 % pour le taux d'apprentissage.
- Un taux de neurones morts durablement supérieur à 50 % indique un taux d'apprentissage trop élevé, pas un problème d'activation.
- ReLU n'est pas dérivable en zéro ; les bibliothèques prennent la convention 0 et cela n'a jamais posé de problème pratique.
- GELU est plus coûteuse que ReLU. Sur un petit réseau, la différence de temps ne se justifie pas par le gain.
- Ne jamais mettre de softmax dans une couche cachée : elle force une somme à 1 qui n'a aucun sens là, et bloque l'apprentissage.
- L'ordre `Linear → Norm → Activation → Dropout` est la convention. Mettre le dropout avant la normalisation dégrade les statistiques de lot.

## Rétropropagation
<!-- slug: retropropagation | difficulte: 4 | prereqs: derivees-gradients, perceptron-mlp -->

**En une phrase** — Calculer efficacement les gradients de la perte par rapport à tous les poids, en appliquant la règle de la chaîne de la sortie vers l'entrée.

**Explication** — L'idée essentielle est la **réutilisation**. Calculer naïvement le gradient de chaque poids séparément recalculerait les mêmes produits des milliers de fois. La rétropropagation calcule une seule fois, pour chaque couche, la quantité $\delta_l = \partial L/\partial z_l$ — la sensibilité de la perte à l'entrée pré-activation de la couche — et la propage vers l'arrière. Le coût total d'une passe arrière est du même ordre que celui d'une passe avant, quel que soit le nombre de paramètres. C'est ce qui rend l'entraînement de milliards de paramètres possible.

Les équations tiennent en trois lignes. À la dernière couche, $\delta_L = \nabla_{a}L \odot \phi'(z_L)$. Puis on remonte : $\delta_l = (W_{l+1}^\top \delta_{l+1}) \odot \phi'(z_l)$. Et les gradients des paramètres se lisent directement : $\partial L/\partial W_l = \delta_l a_{l-1}^\top$, $\partial L/\partial b_l = \delta_l$. Cas particulier remarquable : avec softmax et entropie croisée, $\delta_L = p - y$, d'une simplicité qui n'est pas un hasard.

La **différentiation automatique** généralise cela. PyTorch construit à l'exécution un graphe orienté des opérations, chacune sachant calculer sa dérivée locale. `loss.backward()` parcourt ce graphe en sens inverse et accumule les gradients dans `.grad`. Ce n'est pas de la dérivation symbolique (pas de formule produite) ni numérique (pas de différences finies) : c'est une application mécanique de la règle de la chaîne sur un graphe.

**Cas d'utilisation**
- Entraînement de tout réseau de neurones : c'est le mécanisme, pas une option.
- Débogage d'un entraînement qui ne converge pas : inspecter les normes de gradients par couche localise le problème.
- Attaques adverses et cartes de saillance : gradient de la perte par rapport à l'**entrée** au lieu des poids.
- Optimisation de l'entrée à modèle figé : transfert de style, inversion de modèle.

**Algorithme**
```text
Passe avant, en mémorisant les z_l et a_l (nécessaires à la passe arrière) :
  a_0 = x
  pour l = 1..L : z_l = W_l a_{l-1} + b_l ; a_l = phi(z_l)

Passe arrière :
  delta_L = dL/da_L * phi'(z_L)
            (softmax + entropie croisée : delta_L = p - y, directement)
  pour l = L..1 :
     grad_W_l = delta_l @ a_{l-1}.T
     grad_b_l = somme de delta_l sur le lot
     si l > 1 : delta_{l-1} = (W_l.T @ delta_l) * phi'(z_{l-1})

Vérification obligatoire :
  comparer chaque gradient aux différences finies centrées.
  Écart relatif < 1e-6 en float64 -> implémentation correcte.
```

**Implémentation**
```python
import numpy as np

class MLP:
    """MLP à une couche cachée, rétropropagation écrite à la main (palier 2)."""

    def __init__(self, d, h, k, graine=0):
        rng = np.random.default_rng(graine)
        # initialisation de He : variance 2/n_entrees, adaptée à ReLU
        self.W1 = rng.normal(0, np.sqrt(2 / d), (d, h))
        self.b1 = np.zeros(h)
        self.W2 = rng.normal(0, np.sqrt(2 / h), (h, k))
        self.b2 = np.zeros(k)

    def avant(self, X):
        self.X = X
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)              # ReLU
        self.z2 = self.a1 @ self.W2 + self.b2         # logits
        # softmax stable
        e = np.exp(self.z2 - self.z2.max(1, keepdims=True))
        self.p = e / e.sum(1, keepdims=True)
        return self.p

    def perte(self, Y):
        return -np.log(np.clip(self.p[np.arange(len(Y)), Y], 1e-12, None)).mean()

    def arriere(self, Y):
        n = len(Y)
        d2 = self.p.copy()
        d2[np.arange(n), Y] -= 1
        d2 /= n                                        # delta_L = (p - y) / n
        gW2 = self.a1.T @ d2
        gb2 = d2.sum(0)
        d1 = (d2 @ self.W2.T) * (self.z1 > 0)          # chaîne + dérivée de ReLU
        gW1 = self.X.T @ d1
        gb1 = d1.sum(0)
        return gW1, gb1, gW2, gb2

    def pas(self, grads, eta):
        gW1, gb1, gW2, gb2 = grads
        self.W1 -= eta * gW1; self.b1 -= eta * gb1
        self.W2 -= eta * gW2; self.b2 -= eta * gb2

# Vérification par différences finies : l'étape que personne ne saute deux fois
def verifier(net, X, Y, eps=1e-6):
    net.avant(X); net.perte(Y)
    gW1 = net.arriere(Y)[0]
    num = np.zeros_like(net.W1)
    it = np.nditer(net.W1, flags=['multi_index'])
    for _ in it:
        i = it.multi_index
        v = net.W1[i]
        net.W1[i] = v + eps; net.avant(X); lp = net.perte(Y)
        net.W1[i] = v - eps; net.avant(X); lm = net.perte(Y)
        net.W1[i] = v
        num[i] = (lp - lm) / (2 * eps)
    print("écart max :", np.abs(gW1 - num).max())      # doit être < 1e-7

net = MLP(5, 8, 3)
verifier(net, np.random.randn(20, 5), np.random.randint(0, 3, 20))

# --- équivalent PyTorch : autograd fait tout ---
import torch
x = torch.randn(20, 5, requires_grad=False)
W = torch.randn(5, 3, requires_grad=True)
perte = torch.nn.functional.cross_entropy(x @ W, torch.randint(0, 3, (20,)))
perte.backward()
print(W.grad.shape)          # gradient calculé automatiquement
```

**Outils** — `numpy` pour l'implémentation manuelle, `torch.autograd` pour la version automatique.

**Alternatives open-source**
- *Bibliothèques* : **micrograd** (Karpathy, ~150 lignes) est le meilleur support pour comprendre l'autograd de l'intérieur — le lire entièrement vaut mieux que n'importe quel cours ; **tinygrad** est un cadre complet en quelques milliers de lignes, lisible de bout en bout ; **JAX** propose un autograd fonctionnel composable (`grad`, `vmap`, `jit`) et gère les dérivées d'ordre supérieur naturellement ; **autograd** (HIPS) pour dériver du NumPy pur ; **torchviz** dessine le graphe de calcul construit par PyTorch.
- *Outils graphiques* : **torchviz** / `make_dot` visualise le graphe et révèle les branches détachées par erreur ; **TensorBoard** avec les histogrammes de gradients par couche ; **Netron** pour un graphe ONNX exporté ; **PyTorch profiler** pour voir où passe le temps entre avant et arrière.

**Astuces**
- Vérifier tout gradient écrit à la main par différences finies. Un gradient faux n'émet aucune erreur : il apprend mal, et on accuse les hyperparamètres pendant des jours.
- `.detach()` coupe le graphe : indispensable pour stocker une valeur sans fuite mémoire, catastrophique si on le met par erreur dans le chemin d'apprentissage.
- `torch.no_grad()` autour de l'évaluation économise mémoire et temps. L'oublier fait exploser la mémoire GPU sur de gros lots de validation.
- Les gradients s'**accumulent** dans PyTorch. `opt.zero_grad()` avant chaque `backward()`. Cette accumulation est volontaire : elle permet de simuler de gros lots (*gradient accumulation*).
- Inspecter la norme des gradients par couche : une norme qui décroît d'un facteur 10 par couche signale un gradient qui disparaît (ajouter des connexions résiduelles ou une normalisation) ; une norme qui explose demande un écrêtage.
- La passe arrière consomme la mémoire des activations mémorisées. `torch.utils.checkpoint` les recalcule au besoin : deux fois plus lent, mais permet des modèles bien plus gros.

## Optimiseurs et taux d'apprentissage
<!-- slug: optimiseurs | difficulte: 3 | prereqs: descente-de-gradient, retropropagation -->

**En une phrase** — Comment transformer un gradient en mise à jour de poids : avec de l'inertie, une échelle adaptative par paramètre, et un taux qui évolue au cours de l'entraînement.

**Explication** — Le SGD pur suit le gradient du mini-lot, ce qui zigzague dans les vallées étroites. Le **momentum** ajoute une inertie : $v \leftarrow \beta v + g$, puis $w \leftarrow w - \eta v$. Les composantes cohérentes du gradient s'accumulent, les composantes oscillantes s'annulent. Avec $\beta = 0{,}9$, la vitesse effective est environ dix fois celle du gradient instantané.

Les méthodes **adaptatives** normalisent chaque paramètre par l'amplitude historique de son gradient. **Adam** combine momentum sur le gradient (moment d'ordre 1) et sur son carré (moment d'ordre 2), avec correction de biais au démarrage. Résultat : les paramètres à petits gradients avancent autant que les autres, et le réglage du taux d'apprentissage devient beaucoup moins critique. **AdamW** corrige un défaut subtil d'Adam : la décroissance de poids (*weight decay*) y était mélangée au gradient adaptatif, donc mal appliquée. AdamW la découple, et c'est aujourd'hui le défaut universel.

L'**ordonnancement** du taux compte souvent autant que l'optimiseur. Trois schémas dominent. L'**échauffement** (*warmup*) augmente linéairement le taux sur les premières centaines d'itérations : indispensable pour les Transformers, dont les statistiques initiales sont instables. La **décroissance cosinus** ramène doucement le taux vers zéro et donne presque toujours un meilleur point final qu'un taux constant. Le **cycle unique** (*one-cycle*) monte puis descend, et permet d'utiliser un taux maximal élevé — c'est la méthode de la « convergence super-rapide ».

**Cas d'utilisation**
- AdamW avec cosinus et échauffement : le défaut à appliquer partout, tout le temps.
- SGD avec momentum et décroissance par paliers : encore supérieur sur les CNN de vision quand on a le budget de réglage.
- Un cycle unique quand le budget d'entraînement est court et fixé.
- Taux différenciés par groupe de paramètres (*discriminative fine-tuning*) : taux faible sur les couches pré-entraînées, élevé sur la tête neuve.

**Algorithme**
```text
Adam (une itération, pour chaque paramètre) :
  m = b1 m + (1-b1) g                 # moment 1 : direction
  v = b2 v + (1-b2) g²                # moment 2 : amplitude
  m_hat = m / (1 - b1^t)              # correction du biais initial
  v_hat = v / (1 - b2^t)
  w = w - eta * m_hat / (sqrt(v_hat) + eps)
AdamW ajoute :  w = w - eta * lambda * w      (découplé du gradient)

Trouver le taux d'apprentissage (test de plage, 1 minute) :
  1. Partir de 1e-7, multiplier par ~1.1 à chaque lot, sur ~100 lots.
  2. Tracer la perte en fonction du taux, échelle log.
  3. Prendre le taux à la descente la plus raide, divisé par 3.

Ordonnancement recommandé :
  échauffement linéaire sur 5 % des pas, puis cosinus jusqu'à ~0.
```

**Implémentation**
```python
import torch, math
import matplotlib.pyplot as plt

# Défaut recommandé : AdamW + échauffement + cosinus
opt = torch.optim.AdamW(modele.parameters(), lr=3e-4, weight_decay=0.01,
                        betas=(0.9, 0.999))

pas_total = len(dl_tr) * n_epoques
pas_echauffement = int(0.05 * pas_total)

def facteur(pas):
    if pas < pas_echauffement:
        return pas / max(1, pas_echauffement)
    p = (pas - pas_echauffement) / max(1, pas_total - pas_echauffement)
    return 0.5 * (1 + math.cos(math.pi * p))

sched = torch.optim.lr_scheduler.LambdaLR(opt, facteur)

for xb, yb in dl_tr:
    opt.zero_grad()
    perte_fn(modele(xb), yb).backward()
    torch.nn.utils.clip_grad_norm_(modele.parameters(), 1.0)
    opt.step()
    sched.step()                       # à chaque PAS, pas à chaque époque

# Test de plage : trouver le taux en une minute plutôt qu'en une grille
def test_plage(modele, dl, perte_fn, lr_min=1e-7, lr_max=1, n=100):
    etat = {k: v.clone() for k, v in modele.state_dict().items()}
    opt = torch.optim.AdamW(modele.parameters(), lr=lr_min)
    gamma = (lr_max / lr_min) ** (1 / n)
    lrs, pertes = [], []
    for i, (xb, yb) in enumerate(dl):
        if i >= n:
            break
        opt.zero_grad()
        p = perte_fn(modele(xb), yb)
        p.backward(); opt.step()
        lrs.append(opt.param_groups[0]['lr']); pertes.append(p.item())
        opt.param_groups[0]['lr'] *= gamma
        if p.item() > 4 * min(pertes):
            break                       # divergence : inutile de continuer
    modele.load_state_dict(etat)
    plt.semilogx(lrs, pertes); plt.xlabel('taux'); plt.ylabel('perte')
    return lrs[int(torch.tensor(pertes).argmin())] / 3

# Taux différenciés : ajustement fin d'un modèle pré-entraîné
opt = torch.optim.AdamW([
    {'params': modele.encodeur.parameters(), 'lr': 1e-5},
    {'params': modele.tete.parameters(), 'lr': 1e-3},
], weight_decay=0.01)

# Simuler un gros lot avec peu de mémoire
accum = 4
for i, (xb, yb) in enumerate(dl_tr):
    (perte_fn(modele(xb), yb) / accum).backward()
    if (i + 1) % accum == 0:
        opt.step(); opt.zero_grad(); sched.step()
```

**Outils** — `torch.optim` (`AdamW`, `SGD`, `RMSprop`), `torch.optim.lr_scheduler`.

**Alternatives open-source**
- *Bibliothèques* : **timm** fournit `Lion`, `AdaBelief`, `LAMB` et des ordonnanceurs éprouvés en vision ; **bitsandbytes** propose des optimiseurs 8 bits qui divisent par quatre la mémoire d'état — décisif pour l'ajustement fin de grands modèles ; **Lion** (Google) surpasse parfois AdamW avec moins de mémoire ; **Sophia** et **Shampoo** exploitent une information de second ordre ; **Optuna** ou **Ray Tune** pour chercher les hyperparamètres d'optimisation ; **fastai** implémente `lr_find` et le cycle unique clé en main.
- *Outils graphiques* : **TensorBoard** pour tracer le taux effectif et la norme des gradients au fil des pas ; **Weights & Biases** pour comparer visuellement des courbes entre configurations ; **losslandscape.com** pour l'intuition sur les surfaces d'optimisation.

**Astuces**
- Le taux d'apprentissage est l'hyperparamètre dominant, d'un ordre de grandeur devant les autres. Le test de plage prend une minute et remplace une grille de recherche entière.
- `sched.step()` s'appelle après chaque `opt.step()` pour les ordonnanceurs par pas, une fois par époque pour ceux par époque. Se tromper produit un ordonnancement 100 fois trop rapide ou trop lent.
- L'échauffement n'est pas facultatif sur un Transformer : sans lui, l'entraînement diverge dans les premières centaines d'itérations.
- Une décroissance cosinus jusqu'à zéro donne presque toujours un meilleur point final qu'un taux constant, pour un coût nul.
- Ne pas appliquer de décroissance de poids aux biais ni aux paramètres de normalisation : cela dégrade les résultats. Les bibliothèques bien écrites les excluent, une implémentation manuelle doit y penser.
- Doubler la taille de lot autorise environ un taux $\sqrt{2}$ fois plus élevé (règle de la racine), ou 2 fois selon la règle linéaire utilisée en vision. Les deux existent ; l'important est de réajuster.
- `clip_grad_norm_(1.0)` coûte presque rien et évite qu'un seul lot pathologique détruise l'entraînement. À mettre par défaut.

## Régularisation en deep learning
<!-- slug: regularisation-dl | difficulte: 3 | prereqs: perceptron-mlp, biais-variance -->

**En une phrase** — Les techniques qui empêchent un réseau surparamétré de mémoriser ses données d'entraînement.

**Explication** — Un réseau moderne a souvent plus de paramètres que d'exemples : il *peut* mémoriser parfaitement, y compris des étiquettes aléatoires. La régularisation est donc structurelle, pas accessoire. Quatre familles, par ordre de rentabilité.

L'**arrêt précoce** est le plus simple et le plus efficace : surveiller la validation, garder les meilleurs poids. Gratuit, et il rend inutile une grande partie du réglage restant. L'**augmentation de données** vient ensuite : transformer les entrées de façon à préserver l'étiquette (rotations, recadrages, bruit) multiplie effectivement la taille du jeu. Sur image et audio, c'est le levier dominant, souvent devant l'architecture.

Le **dropout** met à zéro une fraction $p$ des activations à chaque passe d'entraînement, ce qui empêche les neurones de dépendre les uns des autres et équivaut à moyenner un ensemble exponentiel de sous-réseaux. À l'inférence, il est désactivé et les activations sont mises à l'échelle. Attention : dans les architectures modernes avec normalisation par lot, le dropout apporte peu et peut nuire ; il reste précieux dans les Transformers et les MLP tabulaires.

La **décroissance de poids** (L2 découplée dans AdamW) rétrécit les poids en continu. Et la **normalisation** — par lot (*batch norm*) ou par couche (*layer norm*) — a un effet régularisant secondaire tout en accélérant fortement la convergence, son objectif principal. Enfin, le **lissage d'étiquettes** remplace les cibles dures 0/1 par 0,05/0,95 : le modèle devient moins sur-confiant et mieux calibré.

**Cas d'utilisation**
- Arrêt précoce : toujours, sans exception.
- Augmentation de données : dès qu'on travaille sur image, audio, signal, texte.
- Dropout de 0,1 à 0,3 dans les MLP tabulaires et les Transformers.
- Décroissance de poids de 0,01 à 0,1 avec AdamW : défaut raisonnable.
- Lissage d'étiquettes de 0,1 en classification multi-classe avec beaucoup de classes.
- Inutile si le modèle sous-apprend : ajouter de la régularisation à un modèle en sous-capacité aggrave le problème.

**Algorithme**
```text
Ordre de mise en place :
1. Arrêt précoce sur la métrique de validation, avec restauration des
   meilleurs poids. Coût nul.
2. Augmentation de données adaptée à la modalité et à l'invariance réelle
   du problème (ne pas retourner un chiffre écrit à l'envers).
3. Décroissance de poids : 0.01 avec AdamW.
4. Dropout : 0.1 puis 0.3 si l'écart entraînement/validation persiste.
5. Normalisation de couche (Transformer) ou de lot (CNN).
6. Lissage d'étiquettes 0.1 si le modèle est sur-confiant (ECE élevée).
7. Si l'écart persiste malgré tout : plus de données, ou modèle plus petit.

Diagnostic : perte d'entraînement qui continue de baisser tandis que la
perte de validation remonte -> surapprentissage, appliquer 1 à 6.
Les deux stagnent haut -> sous-apprentissage, RETIRER de la régularisation.
```

**Implémentation**
```python
import torch, torch.nn as nn, copy

# 1. Arrêt précoce avec restauration : le patron à réutiliser partout
class ArretPrecoce:
    def __init__(self, patience=15, delta=1e-4):
        self.patience, self.delta = patience, delta
        self.meilleur, self.attente, self.etat = float('inf'), 0, None

    def __call__(self, perte_val, modele):
        if perte_val < self.meilleur - self.delta:
            self.meilleur, self.attente = perte_val, 0
            self.etat = copy.deepcopy(modele.state_dict())
            return False
        self.attente += 1
        if self.attente >= self.patience:
            modele.load_state_dict(self.etat)     # restaurer le meilleur
            return True
        return False

# 2. Dropout et normalisation dans un bloc
bloc = nn.Sequential(
    nn.Linear(256, 512),
    nn.BatchNorm1d(512),      # LayerNorm pour un Transformer ou des lots variables
    nn.ReLU(),
    nn.Dropout(0.2),          # après l'activation, jamais avant la normalisation
)

# 3. Décroissance de poids découplée, en excluant biais et normalisations
decay, no_decay = [], []
for nom, p in modele.named_parameters():
    if not p.requires_grad:
        continue
    (no_decay if p.ndim <= 1 or 'bias' in nom else decay).append(p)
opt = torch.optim.AdamW([{'params': decay, 'weight_decay': 0.01},
                         {'params': no_decay, 'weight_decay': 0.0}], lr=3e-4)

# 4. Lissage d'étiquettes : intégré à la perte
perte_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

# 5. Mixup : deux exemples mélangés, très efficace et presque gratuit
def mixup(x, y, alpha=0.2):
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    perm = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[perm], y, y[perm], lam

xm, ya, yb, lam = mixup(xb, yb)
perte = lam * perte_fn(modele(xm), ya) + (1 - lam) * perte_fn(modele(xm), yb)

# 6. Augmentation image : le levier le plus rentable en vision
from torchvision import transforms as T
aug = T.Compose([
    T.RandomResizedCrop(224, scale=(0.7, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.3, 0.3, 0.3),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    T.RandomErasing(p=0.25),
])
```

**Outils** — `torch.nn` (`Dropout`, `BatchNorm1d`, `LayerNorm`), `torchvision.transforms`.

**Alternatives open-source**
- *Bibliothèques* : **albumentations** est la référence pour l'augmentation d'images, bien plus rapide et complète que torchvision, avec support des masques et boîtes ; **audiomentations** pour l'audio ; **nlpaug** et **TextAttack** pour le texte ; **timm** implémente Mixup, CutMix, RandAugment et la moyenne mobile de poids (EMA) prêts à l'emploi ; **torch-ema** pour la moyenne exponentielle des poids, gain fréquent de 0,5 à 1 point ; **stochastic-depth** (dans timm) pour les réseaux très profonds.
- *Outils graphiques* : **TensorBoard** ou **Weights & Biases** pour voir l'écart entraînement/validation se creuser en direct — le diagnostic se lit sur cette courbe ; **albumentations demo** (application web) pour prévisualiser les transformations ; **FiftyOne** pour inspecter visuellement un jeu d'images augmenté et repérer des transformations absurdes.

**Astuces**
- Commencer par l'arrêt précoce et l'augmentation. Empiler du dropout, de la décroissance et du lissage avant d'avoir mesuré l'écart entraînement/validation revient à traiter un symptôme absent.
- Le dropout et la batch norm interagissent mal. Dans un CNN moderne avec batch norm, le dropout apporte peu. Dans un Transformer avec layer norm, il est utile.
- `modele.eval()` désactive le dropout et fige la batch norm. L'oublier à l'évaluation fausse toutes les mesures — erreur la plus fréquente en PyTorch.
- La batch norm devient instable pour des lots de moins de 16 exemples : les statistiques sont trop bruitées. Utiliser `GroupNorm` ou `LayerNorm` dans ce cas.
- L'augmentation doit préserver l'étiquette. Retourner horizontalement une radiographie ou un chiffre manuscrit change le sens de l'image et détruit l'apprentissage.
- Mixup et CutMix améliorent souvent de 1 à 2 points en vision pour trois lignes de code, et améliorent aussi la calibration.
- Si le modèle sous-apprend, la bonne action est de **retirer** de la régularisation. C'est contre-intuitif et régulièrement manqué.

## PyTorch en pratique
<!-- slug: pytorch-bases | difficulte: 3 | prereqs: perceptron-mlp, optimiseurs -->

**En une phrase** — Le cadre de travail standard du deep learning : des tenseurs avec autograd, des modules composables, et une boucle d'entraînement qu'on écrit soi-même.

**Explication** — Trois abstractions suffisent. Le **tenseur** est un tableau NumPy qui sait vivre sur GPU et enregistrer son historique d'opérations (`requires_grad`). Le **module** (`nn.Module`) encapsule des paramètres et une méthode `forward` ; les modules se composent en arbre, et `parameters()` remonte récursivement tout ce qui doit être appris. Le **DataLoader** transforme un `Dataset` en flux de mini-lots, avec mélange, chargement parallèle et collation.

La particularité de PyTorch est que **la boucle d'entraînement n'est pas cachée**. On écrit explicitement : mettre les gradients à zéro, passe avant, calcul de la perte, passe arrière, pas d'optimiseur. C'est plus verbeux que `model.fit()`, et c'est ce qui permet de tout modifier — pertes multiples, entraînement adverse, accumulation de gradients, boucles imbriquées. Ce coût initial est vite amorti.

Les points d'attention pratiques sont presque toujours les mêmes : le **périphérique** (les tenseurs doivent être sur le même appareil que le modèle), le **mode** (`train()` / `eval()`), les **formes** (une erreur de dimension silencieuse via le broadcasting est le bug le plus long à trouver), et la **mémoire GPU** (les activations conservées pour la passe arrière dominent la consommation).

**Cas d'utilisation**
- Tout projet de deep learning, du prototype à la production.
- Ajustement fin de modèles pré-entraînés (vision, texte, audio) — l'usage le plus courant.
- Calcul tensoriel sur GPU même sans réseau de neurones : PyTorch remplace avantageusement NumPy pour des opérations matricielles lourdes.
- Exportation vers ONNX ou TorchScript pour un déploiement hors Python.

**Algorithme**
```text
Squelette de projet, dans cet ordre :
1. Dataset  : __len__ et __getitem__ retournant (x, y) en tenseurs.
2. DataLoader : batch_size, shuffle=True à l'entraînement, num_workers > 0.
3. Modèle   : nn.Module avec __init__ (couches) et forward (calcul).
4. Perte + optimiseur + ordonnanceur.
5. Boucle par époque :
     train() ; pour chaque lot : zero_grad, avant, perte, arriere, clip, step
     eval() + no_grad : métriques de validation
     ordonnanceur, arrêt précoce, sauvegarde du meilleur état
6. Charger le meilleur état, évaluer sur le test une seule fois.
7. Sauvegarder state_dict (pas le modèle entier), avec la config d'architecture.
```

**Implémentation**
```python
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class JeuTabulaire(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

dl_tr = DataLoader(JeuTabulaire(X_tr, y_tr), batch_size=256, shuffle=True,
                   num_workers=4, pin_memory=True, drop_last=True)
dl_val = DataLoader(JeuTabulaire(X_val, y_val), batch_size=1024)

class Reseau(nn.Module):
    def __init__(self, d, h=256, p=0.2):
        super().__init__()
        self.corps = nn.Sequential(
            nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p),
            nn.Linear(h, h // 2), nn.BatchNorm1d(h // 2), nn.ReLU(), nn.Dropout(p),
        )
        self.tete = nn.Linear(h // 2, 1)

    def forward(self, x):
        return self.tete(self.corps(x)).squeeze(-1)

modele = Reseau(X_tr.shape[1]).to(dev)
perte_fn = nn.BCEWithLogitsLoss()
opt = torch.optim.AdamW(modele.parameters(), lr=1e-3, weight_decay=0.01)
scaler = torch.amp.GradScaler('cuda', enabled=(dev.type == 'cuda'))

def epoque_entrainement():
    modele.train()
    total = 0.0
    for xb, yb in dl_tr:
        xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        # précision mixte : 2x plus rapide, 2x moins de mémoire sur GPU récent
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=(dev.type == 'cuda')):
            perte = perte_fn(modele(xb), yb)
        scaler.scale(perte).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(modele.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        total += perte.item() * len(yb)
    return total / len(dl_tr.dataset)

@torch.no_grad()
def evaluer(dl):
    modele.eval()
    logits, cibles = [], []
    for xb, yb in dl:
        logits.append(modele(xb.to(dev)).cpu())
        cibles.append(yb)
    return torch.cat(logits), torch.cat(cibles)

# Sauvegarde : le state_dict, avec la configuration nécessaire pour reconstruire
torch.save({'etat': modele.state_dict(), 'd': X_tr.shape[1], 'h': 256}, 'modele.pt')
ck = torch.load('modele.pt', map_location=dev, weights_only=True)
m2 = Reseau(ck['d'], ck['h']).to(dev)
m2.load_state_dict(ck['etat'])

# Débogage des formes : le réflexe qui économise des heures
def tracer_formes(modele, x):
    for nom, m in modele.named_children():
        x = m(x)
        print(f"{nom:12s} -> {tuple(x.shape)}")
    return x
```

**Outils** — `pip install torch torchvision` (variante CUDA depuis le site officiel si GPU).

**Alternatives open-source**
- *Bibliothèques* : **PyTorch Lightning** supprime la boucle et gère multi-GPU, points de reprise et journalisation sans changer le modèle ; **Hugging Face Accelerate** est plus léger et rend un script portable CPU/GPU/multi-GPU en trois lignes ; **fastai** offre les meilleurs défauts pour aller vite ; **skorch** rend un modèle PyTorch compatible scikit-learn ; **JAX + Flax** pour la recherche et la compilation XLA ; **Keras 3** si l'on préfère une API déclarative ; **ONNX Runtime** et **TorchScript** pour déployer sans Python.
- *Outils graphiques* : **TensorBoard** (intégré via `torch.utils.tensorboard`) pour les courbes, histogrammes et graphes ; **Netron** pour lire une architecture exportée ; **PyTorch Profiler** avec sa vue chronologique pour trouver les goulots ; **FiftyOne** pour inspecter visuellement les erreurs d'un modèle de vision ; **MLflow** pour versionner modèles et exécutions.

**Astuces**
- `zero_grad(set_to_none=True)` est légèrement plus rapide et évite d'accumuler des gradients nuls.
- Tout `.item()`, `.cpu()` ou `print` dans la boucle force une synchronisation GPU et ralentit fortement. Accumuler sur GPU et transférer une fois par époque.
- `num_workers=4` et `pin_memory=True` accélèrent souvent plus que n'importe quelle optimisation du modèle : le goulot est fréquemment le chargement des données, pas le calcul.
- Sauvegarder `state_dict()` et non le modèle entier : un pickle de modèle dépend de la structure du code et casse au premier refactoring.
- `torch.load(..., weights_only=True)` évite l'exécution de code arbitraire à la désérialisation. À utiliser systématiquement, surtout pour un poids téléchargé.
- La précision mixte (`autocast` + `bfloat16`) double la vitesse et divise la mémoire par deux sur GPU récent, sans perte de qualité mesurable.
- « CUDA out of memory » : réduire la taille de lot, activer la précision mixte, ou utiliser `torch.utils.checkpoint`. Vérifier aussi qu'aucun tenseur avec graphe n'est stocké dans une liste — cause fréquente d'une fuite mémoire progressive.
- Fixer `torch.manual_seed()` ne suffit pas à la reproductibilité exacte sur GPU. Enregistrer les métriques obtenues, pas seulement la graine.

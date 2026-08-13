---
module: vision-sequences
titre: Vision et séquences
ordre: 7
resume: Les architectures qui encodent une structure — voisinage spatial pour l'image, ordre et dépendances pour les séquences.
---

## Convolutions et CNN
<!-- slug: convolutions-cnn | difficulte: 4 | prereqs: perceptron-mlp, retropropagation -->

**En une phrase** — Faire glisser de petits filtres appris sur une image pour détecter des motifs locaux, indépendamment de leur position.

**Explication** — Un MLP appliqué à une image de 224×224×3 exigerait 150 000 poids pour un seul neurone de la première couche, et n'apprendrait rien sur la structure spatiale : deux pixels voisins n'ont pour lui aucune relation particulière. La convolution résout les deux problèmes par deux principes. La **localité** : chaque neurone ne regarde qu'une petite fenêtre (3×3 typiquement). Le **partage de poids** : le même filtre est appliqué partout, donc un détecteur de bord appris en haut à gauche fonctionne aussi en bas à droite. Un filtre 3×3 sur 3 canaux vers 64 canaux ne coûte que $3 \times 3 \times 3 \times 64 = 1728$ poids.

Les dimensions se calculent avec une formule à connaître : pour une entrée de taille $H$, un noyau $k$, un remplissage $p$ et un pas $s$, la sortie vaut $\lfloor (H + 2p - k)/s \rfloor + 1$. Avec $k=3, p=1, s=1$, la taille est conservée — c'est la configuration standard. Le **pooling** (ou un pas de 2) réduit la résolution spatiale tout en augmentant le nombre de canaux : l'information passe progressivement du « où » au « quoi ».

La notion clé pour comprendre la profondeur est le **champ réceptif** : la zone de l'image d'origine qui influence un neurone donné. Il croît linéairement avec la profondeur pour des convolutions 3×3, et double à chaque pooling. Les premières couches voient des bords et des textures, les dernières des objets entiers. Cette hiérarchie n'est pas programmée, elle émerge de l'entraînement.

**Cas d'utilisation**
- Classification, détection et segmentation d'images.
- Signaux 1D : audio, ECG, séries temporelles (convolution causale, TCN).
- Volumes 3D : imagerie médicale, vidéo.
- Extraction de features pour toute tâche visuelle, via un modèle pré-entraîné.
- Mauvais choix sur des données tabulaires : il n'y a aucune structure de voisinage entre colonnes, donc rien à exploiter.

**Algorithme**
```text
Convolution 2D, pour chaque position (i, j) et chaque canal de sortie o :
  sortie[o, i, j] = biais[o] + somme_{c, u, v} noyau[o, c, u, v] * entree[c, i*s+u-p, j*s+v-p]

Bloc CNN classique :
  Conv(3x3, padding=1) -> BatchNorm -> ReLU -> [Conv -> BN -> ReLU] -> pooling(2x2)
  À chaque bloc : résolution / 2, canaux x 2.

Tête de classification :
  GlobalAvgPool -> Linear(canaux, n_classes)
  (le pooling global remplace le Flatten et rend le réseau indépendant
   de la taille d'entrée, avec beaucoup moins de paramètres)

Formule de sortie : H_out = (H + 2p - k) / s + 1, partie entière.
```

**Implémentation**
```python
import torch, torch.nn as nn, numpy as np

# --- convolution from scratch, par im2col (palier 2) ---
def conv2d_naive(X, K, pad=1, stride=1):
    """X: (C, H, W)  K: (O, C, k, k)  ->  (O, H', W')"""
    C, H, W = X.shape
    O, _, k, _ = K.shape
    Xp = np.pad(X, ((0, 0), (pad, pad), (pad, pad)))
    Ho = (H + 2 * pad - k) // stride + 1
    Wo = (W + 2 * pad - k) // stride + 1
    # im2col : chaque fenêtre devient une colonne -> la convolution est un produit matriciel
    col = np.empty((C * k * k, Ho * Wo))
    idx = 0
    for i in range(Ho):
        for j in range(Wo):
            fen = Xp[:, i * stride:i * stride + k, j * stride:j * stride + k]
            col[:, idx] = fen.reshape(-1)
            idx += 1
    return (K.reshape(O, -1) @ col).reshape(O, Ho, Wo)

# --- CNN complet en PyTorch ---
def bloc(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )

class PetitCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.corps = nn.Sequential(bloc(3, 32), bloc(32, 64), bloc(64, 128))
        self.tete = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(0.2), nn.Linear(128, n_classes))

    def forward(self, x):
        return self.tete(self.corps(x))

# Vérifier les formes couche par couche : le réflexe indispensable
x = torch.randn(2, 3, 64, 64)
m = PetitCNN()
for nom, couche in m.corps.named_children():
    x = couche(x)
    print(f"bloc {nom} -> {tuple(x.shape)}")

# Champ réceptif : le calculer évite de concevoir un réseau qui ne voit rien
def champ_receptif(couches):
    """couches : liste de (k, s). Retourne la taille du champ réceptif."""
    r, saut = 1, 1
    for k, s in couches:
        r += (k - 1) * saut
        saut *= s
    return r

print(champ_receptif([(3, 1), (3, 1), (2, 2)] * 3))    # à comparer à la taille des objets
```

**Outils** — `torch.nn` (`Conv2d`, `BatchNorm2d`, `MaxPool2d`, `AdaptiveAvgPool2d`), `torchvision`.

**Alternatives open-source**
- *Bibliothèques* : **timm** donne accès à plus de mille architectures de vision pré-entraînées avec une API uniforme — indispensable ; **torchvision.models** pour les classiques ; **albumentations** pour l'augmentation ; **MMDetection** et **Detectron2** pour la détection d'objets ; **segmentation_models_pytorch** pour la segmentation avec encodeurs interchangeables ; **kornia** implémente des opérations de vision différentiables (transformations géométriques, filtres) directement dans le graphe.
- *Outils graphiques* : **Netron** pour visualiser une architecture ONNX couche par couche ; **CNN Explainer** (Georgia Tech, dans le navigateur) montre les activations d'un CNN en direct — le meilleur support pédagogique du sujet ; **FiftyOne** pour explorer visuellement les erreurs de prédiction sur un jeu d'images ; **Captum** et **grad-cam** produisent des cartes de saillance ; **Label Studio** et **CVAT** pour annoter.

**Astuces**
- `padding=1` avec un noyau 3×3 conserve la taille. C'est la configuration à utiliser par défaut, et elle évite de recalculer les dimensions à chaque couche.
- `bias=False` sur une convolution suivie d'une `BatchNorm` : le biais est redondant car la normalisation recentre déjà.
- Remplacer `Flatten` par `AdaptiveAvgPool2d(1)` supprime des millions de paramètres et rend le réseau tolérant à la taille d'entrée.
- Vérifier que le champ réceptif final couvre bien la taille des objets à reconnaître. Un réseau trop peu profond ne peut pas voir un objet plus grand que son champ réceptif.
- Normaliser les images avec les moyennes et écarts types d'ImageNet quand on part d'un modèle pré-entraîné : ce sont les statistiques sur lesquelles il a été calibré.
- L'augmentation de données est le levier le plus rentable en vision, souvent devant l'architecture. La régler avant de changer de modèle.
- `channels_last` (`memory_format=torch.channels_last`) accélère de 20 à 30 % sur GPU récent avec précision mixte, pour une ligne de code.

## Architectures de vision
<!-- slug: architectures-vision | difficulte: 4 | prereqs: convolutions-cnn -->

**En une phrase** — Les schémas d'assemblage qui ont rendu les réseaux profonds entraînables : connexions résiduelles, blocs répétés, et aujourd'hui les Transformers de vision.

**Explication** — Jusqu'en 2015, empiler plus de vingt couches **dégradait** les performances, y compris sur l'entraînement — donc ce n'était pas du surapprentissage mais un problème d'optimisation. **ResNet** l'a résolu par la connexion résiduelle : $y = F(x) + x$. Le bloc n'apprend plus la transformation complète mais l'**écart** à l'identité, ce qui est bien plus facile. Surtout, le gradient dispose désormais d'un chemin direct par l'addition, ce qui supprime l'atténuation en profondeur. Cette idée d'une ligne a rendu possibles les réseaux à 100 couches, et elle est aujourd'hui présente dans toutes les architectures, Transformers inclus.

Les autres briques structurantes valent d'être connues. La convolution **1×1** mélange les canaux sans toucher à l'espace, ce qui permet de réduire la dimension avant une convolution coûteuse (goulot d'étranglement). La convolution **séparable en profondeur** (MobileNet) décompose une convolution en un filtrage par canal suivi d'un mélange 1×1, divisant le coût par 8 ou 9 — c'est ce qui rend la vision possible sur téléphone. L'**EfficientNet** met en évidence qu'il faut augmenter profondeur, largeur et résolution *conjointement*.

Depuis 2020, le **Vision Transformer** découpe l'image en tuiles de 16×16, les traite comme une séquence de jetons et applique de l'auto-attention. Il dépasse les CNN au-delà de dizaines de millions d'images, car il n'a aucun biais spatial intégré : très bon quand les données abondent, moins bon en régime restreint. Les **ConvNeXt** ont ensuite montré qu'un CNN modernisé rattrapait les ViT. En pratique le choix se fait par le volume de données, non par la mode.

**Cas d'utilisation**
- ResNet-50 ou ConvNeXt-Tiny : bases fiables pour toute tâche de classification.
- MobileNet ou EfficientNet-Lite : contrainte d'embarqué, mobile, temps réel.
- ViT ou Swin Transformer : gros volumes de données, ou pour exploiter un modèle pré-entraîné à très grande échelle (CLIP, DINOv2).
- U-Net pour la segmentation, YOLO ou DETR pour la détection.
- En pratique, on ne conçoit presque jamais d'architecture soi-même : on prend un modèle pré-entraîné dans `timm` et on l'ajuste.

**Algorithme**
```text
Bloc résiduel (ResNet) :
  y = x
  h = ReLU(BN(Conv3x3(x)))
  h = BN(Conv3x3(h))
  si les formes diffèrent : y = BN(Conv1x1(x))     # projection
  sortie = ReLU(h + y)

Bloc à goulot (ResNet-50 et au-delà) :
  Conv1x1 (réduit C à C/4) -> Conv3x3 -> Conv1x1 (rétablit C) + raccourci

Convolution séparable (MobileNet) :
  DepthwiseConv3x3 (un filtre par canal) -> PointwiseConv1x1 (mélange)
  coût divisé par ~k² comparé à une convolution standard

Vision Transformer :
  image -> tuiles 16x16 -> projection linéaire -> + encodage de position
        -> N blocs (LayerNorm, attention multi-têtes, MLP, résiduels)
        -> jeton [CLS] -> tête de classification
```

**Implémentation**
```python
import torch, torch.nn as nn, timm

# --- bloc résiduel from scratch (palier 2) ---
class BlocResiduel(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False), nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False), nn.BatchNorm2d(c_out),
        )
        self.raccourci = (nn.Sequential() if stride == 1 and c_in == c_out else
                          nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False),
                                        nn.BatchNorm2d(c_out)))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv(x) + self.raccourci(x))

# --- en pratique : timm, plus de 1000 modèles pré-entraînés ---
print(timm.list_models('convnext*', pretrained=True)[:5])

modele = timm.create_model('convnext_tiny', pretrained=True,
                           num_classes=10, drop_rate=0.1)

# La configuration de prétraitement du modèle : à respecter absolument
cfg = timm.data.resolve_data_config({}, model=modele)
transformation = timm.data.create_transform(**cfg, is_training=True)
print(cfg)   # taille d'entrée, moyennes, écarts types, interpolation

# Extraction de features plutôt que classification
extracteur = timm.create_model('resnet50', pretrained=True, num_classes=0)  # sans tête
with torch.no_grad():
    features = extracteur(images)        # (N, 2048) — utilisables en kNN, clustering, SVM

# Comparer coût et taille avant de choisir
for nom in ['resnet18', 'resnet50', 'convnext_tiny', 'efficientnet_b0', 'vit_small_patch16_224']:
    m = timm.create_model(nom, pretrained=False)
    n = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"{nom:26s} {n:6.1f} M paramètres")
```

**Outils** — `pip install timm torch torchvision`.

**Alternatives open-source**
- *Bibliothèques* : **timm** est la référence absolue pour les modèles de vision pré-entraînés ; **torchvision.models** pour les classiques avec poids officiels ; **segmentation_models_pytorch** combine n'importe quel encodeur timm avec U-Net, FPN ou DeepLab ; **ultralytics** (YOLO) pour la détection et la segmentation en temps réel, très simple d'emploi ; **Detectron2** et **MMDetection** pour la détection avancée ; **transformers** (Hugging Face) pour ViT, Swin, DINOv2, CLIP ; **open_clip** pour des modèles vision-langage.
- *Outils graphiques* : **Netron** pour explorer une architecture exportée ; **FiftyOne** pour l'analyse d'erreurs sur jeux d'images, indispensable en pratique ; **CVAT** et **Label Studio** pour l'annotation ; **Roboflow** (partiellement libre) pour la préparation de jeux de détection ; **grad-cam** et **Captum** pour visualiser où le modèle regarde.

**Astuces**
- Ne pas concevoir d'architecture soi-même. Prendre un modèle pré-entraîné et l'ajuster gagne des dizaines de points par rapport à un réseau entraîné de zéro sur un jeu de taille modeste.
- Respecter la configuration de prétraitement du modèle pré-entraîné (taille, normalisation, interpolation). Une normalisation différente de celle de l'entraînement fait chuter les performances sans erreur visible.
- `resnet18` reste un excellent point de départ : rapide à entraîner, difficile à battre significativement sur un petit jeu.
- Les ViT nécessitent beaucoup de données ou un pré-entraînement massif. Sur 5 000 images, un ConvNeXt ou un ResNet ajusté sera meilleur.
- Compter les paramètres et mesurer la latence **avant** de choisir. Un gain de 1 % d'exactitude pour 10 fois plus de calcul est rarement le bon arbitrage.
- Les connexions résiduelles sont la raison pour laquelle les réseaux profonds fonctionnent. Si un réseau maison à plus de 15 couches n'apprend pas, la première chose à ajouter est un raccourci.

## Transfert d'apprentissage
<!-- slug: transfert-apprentissage | difficulte: 3 | prereqs: architectures-vision, optimiseurs -->

**En une phrase** — Réutiliser un modèle entraîné sur un immense jeu de données générique et l'adapter à sa propre tâche avec peu d'exemples.

**Explication** — Les premières couches d'un réseau de vision apprennent des détecteurs de bords, de textures, de motifs — universels, indépendants de la tâche. Les dernières apprennent des concepts spécifiques au jeu d'origine. Le transfert consiste à garder les premières et remplacer les dernières. Avec 500 images étiquetées, un modèle pré-entraîné ajusté atteint des performances qu'un réseau entraîné de zéro n'atteindrait pas avec 50 000.

Deux régimes selon le volume disponible. En **extraction de features**, on gèle tout l'encodeur et on n'entraîne qu'une tête neuve — très rapide, peu de risque de surapprentissage, idéal en dessous de 1 000 images. En **ajustement fin** (*fine-tuning*), on dégèle tout ou partie de l'encodeur avec un taux d'apprentissage **beaucoup plus faible** (10 à 100 fois moins que pour la tête) : les poids pré-entraînés sont déjà bons, on ne veut que les nuancer. Un taux trop élevé détruit ce qui a été appris — c'est l'*oubli catastrophique*.

La stratégie qui fonctionne le mieux en pratique est le **dégel progressif** : entraîner d'abord la tête seule pendant quelques époques, puis dégeler par blocs en partant de la fin, avec des taux croissants du début vers la fin du réseau. Cela s'applique identiquement au texte (BERT, LLM) et à l'audio.

**Cas d'utilisation**
- Jeu de données étiqueté petit ou moyen (100 à 100 000 exemples) : c'est le cas normal en entreprise.
- Domaine spécialisé sans modèle dédié : imagerie médicale, industriel, agricole.
- Extraction d'embeddings pour de la recherche par similarité, du clustering ou une baseline kNN.
- Peu utile si le domaine est très éloigné des données de pré-entraînement (spectrogrammes exotiques, images satellites multispectrales) — le gain diminue, sans disparaître.

**Algorithme**
```text
1. Choisir un modèle pré-entraîné proche du domaine (ImageNet pour des photos
   naturelles, DINOv2 ou CLIP pour de la robustesse générale).
2. Remplacer la tête par une couche adaptée au nombre de classes.
3. Étape A — tête seule :
     geler l'encodeur (requires_grad = False)
     entraîner la tête, lr ~ 1e-3, 3 à 10 époques
4. Étape B — ajustement fin :
     dégeler tout (ou les derniers blocs)
     taux différenciés : encodeur 1e-5, tête 1e-3
     cosinus + arrêt précoce, augmentation active
5. Si moins de 500 images : s'arrêter à l'étape A, ou dégeler les 2 derniers
   blocs seulement.
6. Évaluer sur un jeu de test issu de la MÊME distribution que la production.
```

**Implémentation**
```python
import torch, torch.nn as nn, timm

modele = timm.create_model('convnext_tiny', pretrained=True, num_classes=n_classes)

# --- Étape A : geler l'encodeur, entraîner la tête ---
for p in modele.parameters():
    p.requires_grad = False
for p in modele.get_classifier().parameters():
    p.requires_grad = True

opt = torch.optim.AdamW([p for p in modele.parameters() if p.requires_grad], lr=1e-3)
# ... quelques époques ...

# --- Étape B : dégel complet avec taux différenciés ---
for p in modele.parameters():
    p.requires_grad = True

tete = set(id(p) for p in modele.get_classifier().parameters())
opt = torch.optim.AdamW([
    {'params': [p for p in modele.parameters() if id(p) not in tete], 'lr': 1e-5},
    {'params': list(modele.get_classifier().parameters()), 'lr': 1e-3},
], weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=pas_total)

# --- Extraction de features : la baseline à faire en premier, en 5 minutes ---
extracteur = timm.create_model('resnet50', pretrained=True, num_classes=0).eval().cuda()
feats, cibles = [], []
with torch.no_grad():
    for xb, yb in dl:
        feats.append(extracteur(xb.cuda()).cpu())
        cibles.append(yb)
F = torch.cat(feats).numpy()

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
print("baseline linéaire sur features gelées :",
      cross_val_score(LogisticRegression(max_iter=2000), F, torch.cat(cibles).numpy(), cv=5).mean())
# Souvent déjà excellent. Si l'ajustement fin ne bat pas ce chiffre, il y a un bug.

# --- Ajustement fin efficace en paramètres (LoRA), utile si le modèle est gros ---
# pip install peft
from peft import LoraConfig, get_peft_model
cfg = LoraConfig(r=8, lora_alpha=16, target_modules=['qkv', 'proj'], lora_dropout=0.05)
modele_lora = get_peft_model(modele, cfg)
modele_lora.print_trainable_parameters()      # souvent < 1 % des paramètres
```

**Outils** — `pip install timm torch`, `pip install peft` pour l'ajustement efficace.

**Alternatives open-source**
- *Bibliothèques* : **timm** et **transformers** pour les modèles pré-entraînés ; **peft** (Hugging Face) implémente LoRA, QLoRA et les adaptateurs — indispensable pour les gros modèles ; **fastai** propose le dégel progressif et les taux différenciés en une ligne (`fit_one_cycle`, `freeze_to`) ; **open_clip** et **DINOv2** fournissent des encodeurs visuels remarquablement robustes pour de l'extraction de features ; **sentence-transformers** pour l'équivalent textuel ; **lightly** pour de l'apprentissage auto-supervisé quand les étiquettes manquent.
- *Outils graphiques* : **FiftyOne** pour comparer visuellement les erreurs avant et après ajustement ; **TensorBoard** pour suivre les deux étapes ; **Hugging Face Hub** pour parcourir et essayer des modèles avant de les télécharger ; **Label Studio** pour compléter les annotations là où le modèle échoue (apprentissage actif).

**Astuces**
- Toujours commencer par la baseline « features gelées + régression logistique ». Elle prend cinq minutes et fixe le niveau à battre. Un ajustement fin qui ne la dépasse pas révèle un problème d'entraînement.
- Le taux d'apprentissage de l'encodeur doit être 10 à 100 fois plus faible que celui de la tête. C'est l'erreur numéro un du transfert : un taux uniforme détruit les poids pré-entraînés dès le premier lot.
- Respecter la normalisation d'entrée du modèle d'origine, sans exception.
- Avec la batch norm gelée, penser à mettre l'encodeur en mode `eval()` même pendant l'entraînement de la tête : sinon les statistiques de lot dérivent et polluent les features.
- En dessous de 200 images par classe, ne pas dégeler tout le réseau : se limiter aux deux derniers blocs, ou rester en extraction de features.
- DINOv2 et CLIP produisent des embeddings souvent meilleurs qu'un ResNet ImageNet pour du kNN ou du clustering, sans aucun entraînement.
- Enregistrer le nom exact du modèle et sa version de poids. « resnet50 pré-entraîné » ne suffit pas à reproduire un résultat : les jeux de poids diffèrent.

## RNN, LSTM et GRU
<!-- slug: rnn-lstm | difficulte: 4 | prereqs: retropropagation, fonctions-activation -->

**En une phrase** — Traiter une séquence élément par élément en maintenant un état caché qui résume tout ce qui a été vu, avec des portes pour contrôler ce qui est retenu ou oublié.

**Explication** — Un RNN applique la même transformation à chaque pas de temps : $h_t = \phi(W_h h_{t-1} + W_x x_t + b)$. L'état $h_t$ est une mémoire compressée du passé. L'entraînement se fait par rétropropagation à travers le temps : on déplie la récurrence sur $T$ pas et on applique la chaîne. Le problème est mécanique — le gradient traverse $T$ multiplications par $W_h$, donc il tend vers zéro si les valeurs propres sont inférieures à 1, et explose sinon. En pratique, un RNN simple ne retient rien au-delà d'une dizaine de pas.

Le **LSTM** résout cela par une seconde voie, l'état de cellule $c_t$, mise à jour de façon **additive** : $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$. Le gradient traverse cette addition sans être multiplié par une matrice, donc il se conserve. Trois portes sigmoïdes le contrôlent : la porte d'oubli $f$ décide ce qu'on jette, la porte d'entrée $i$ ce qu'on ajoute, la porte de sortie $o$ ce qu'on expose. Le **GRU** simplifie à deux portes et un seul état, avec des performances généralement équivalentes et un tiers de paramètres en moins.

Depuis 2018, les Transformers ont largement remplacé les RNN en traitement du langage, parce que leur calcul est parallélisable sur toute la séquence là où un RNN est séquentiel par nature. Les RNN restent néanmoins pertinents : sur séquences très longues à mémoire bornée, en flux temps réel avec état persistant, et sur de petits jeux de données où un Transformer surapprendrait. Les modèles à espace d'état (Mamba, S4) reprennent d'ailleurs l'idée récurrente avec un entraînement parallélisable.

**Cas d'utilisation**
- Prévision de séries temporelles multivariées avec état continu.
- Traitement en flux où la latence par élément compte (un pas = un calcul, contre $O(T)$ pour l'attention).
- Petits jeux de données séquentielles, où le Transformer manque de données.
- Modélisation de capteurs, de signaux, de sessions utilisateur.
- Mauvais choix pour du texte de plus de quelques centaines de jetons, ou quand un modèle de langage pré-entraîné existe : le Transformer gagne largement.

**Algorithme**
```text
Cellule LSTM, à chaque pas t :
  f = sigmoide(W_f [h_{t-1}, x_t] + b_f)         # oubli
  i = sigmoide(W_i [h_{t-1}, x_t] + b_i)         # entrée
  o = sigmoide(W_o [h_{t-1}, x_t] + b_o)         # sortie
  g = tanh(W_g [h_{t-1}, x_t] + b_g)             # candidat
  c_t = f * c_{t-1} + i * g                      # ADDITIF : le gradient survit
  h_t = o * tanh(c_t)

Entraînement :
  1. Déplier sur T pas, rétropropagation à travers le temps.
  2. ÉCRÊTER la norme du gradient (clip 1.0) — non facultatif.
  3. Séquences de longueurs variables : pack_padded_sequence, sinon le
     modèle apprend sur le remplissage.
  4. Bidirectionnel si toute la séquence est disponible à l'avance ;
     impossible en flux ou en génération.
```

**Implémentation**
```python
import torch, torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ClassifieurSequence(nn.Module):
    def __init__(self, d_entree, h=128, n_couches=2, n_classes=2, bidir=True):
        super().__init__()
        self.lstm = nn.LSTM(d_entree, h, n_couches, batch_first=True,
                            dropout=0.2 if n_couches > 1 else 0.0, bidirectional=bidir)
        self.tete = nn.Linear(h * (2 if bidir else 1), n_classes)

    def forward(self, x, longueurs):
        # pack : le LSTM ignore le remplissage, ce qui change réellement les résultats
        emb = pack_padded_sequence(x, longueurs.cpu(), batch_first=True, enforce_sorted=False)
        sorties, (h, c) = self.lstm(emb)
        sorties, _ = pad_packed_sequence(sorties, batch_first=True)
        # pooling sur les pas valides uniquement
        masque = (torch.arange(sorties.size(1), device=x.device)[None] < longueurs[:, None])
        pooled = (sorties * masque.unsqueeze(-1)).sum(1) / longueurs[:, None]
        return self.tete(pooled)

modele = ClassifieurSequence(d_entree=10).cuda()
opt = torch.optim.AdamW(modele.parameters(), lr=1e-3)

for xb, lb, yb in dl:
    opt.zero_grad()
    perte = nn.functional.cross_entropy(modele(xb.cuda(), lb.cuda()), yb.cuda())
    perte.backward()
    nn.utils.clip_grad_norm_(modele.parameters(), 1.0)   # indispensable pour un RNN
    opt.step()

# --- cellule LSTM from scratch (palier 2) ---
class CelluleLSTM(nn.Module):
    def __init__(self, d_in, d_h):
        super().__init__()
        self.W = nn.Linear(d_in + d_h, 4 * d_h)     # les 4 portes en une matrice
        self.d_h = d_h

    def forward(self, x, etat):
        h, c = etat
        z = self.W(torch.cat([h, x], dim=1))
        f, i, o, g = z.chunk(4, dim=1)
        f, i, o, g = torch.sigmoid(f), torch.sigmoid(i), torch.sigmoid(o), torch.tanh(g)
        c = f * c + i * g                           # mise à jour additive
        h = o * torch.tanh(c)
        return h, (h, c)

# Prévision de série temporelle : fenêtres glissantes en entrée
def fenetres(serie, taille=48, horizon=1):
    X = torch.stack([serie[i:i + taille] for i in range(len(serie) - taille - horizon + 1)])
    y = torch.stack([serie[i + taille:i + taille + horizon]
                     for i in range(len(serie) - taille - horizon + 1)])
    return X.unsqueeze(-1), y
```

**Outils** — `torch.nn` (`LSTM`, `GRU`, `RNN`), `torch.nn.utils.rnn` pour les longueurs variables.

**Alternatives open-source**
- *Bibliothèques* : **darts** (Unit8) unifie une vingtaine de modèles de prévision, des ARIMA aux Transformers, avec une API cohérente et une validation temporelle correcte ; **NeuralForecast** (Nixtla) implémente NHITS, TFT et PatchTST, aujourd'hui l'état de l'art en prévision ; **sktime** pour un cadre scikit-learn appliqué aux séries ; **mamba-ssm** pour les modèles à espace d'état, successeurs récurrents modernes ; **tsai** pour la classification de séries temporelles ; **Prophet** pour une baseline rapide et lisible.
- *Outils graphiques* : **Grafana** pour explorer les séries et les prévisions ; **TensorBoard** pour l'entraînement ; **darts** intègre des fonctions de tracé de prévisions avec intervalles ; **Kats** (Meta) pour la détection de changement et la décomposition saisonnière.

**Astuces**
- L'écrêtage du gradient n'est pas optionnel sur un RNN. Sans lui, un seul lot pathologique fait diverger l'entraînement.
- `batch_first=True` évite les transpositions mentales incessantes. Le défaut PyTorch est `False`, source d'erreurs de forme silencieuses.
- Sans `pack_padded_sequence`, le modèle traite le remplissage comme du signal et apprend sur du vide. L'effet est très net sur des longueurs hétérogènes.
- Un LSTM bidirectionnel est impossible en génération ou en flux : il exige la séquence complète. Vérifier la contrainte avant de l'utiliser.
- Le `dropout` de `nn.LSTM` ne s'applique qu'entre les couches, pas à l'intérieur de la récurrence. Pour du dropout récurrent, il faut l'implémenter soi-même.
- Avant tout modèle profond sur une série temporelle, comparer à la baseline naïve (« demain = aujourd'hui ») et à un LightGBM sur features de décalages. Ces deux références battent souvent un LSTM mal réglé.
- Standardiser la série, et surtout apprendre les statistiques sur le seul passé : les normaliser sur toute la série est une fuite temporelle.

## Attention et Transformers
<!-- slug: attention-transformer | difficulte: 5 | prereqs: rnn-lstm, perceptron-mlp -->

**En une phrase** — Chaque élément d'une séquence regarde tous les autres et en agrège l'information selon des poids calculés, ce qui supprime la récurrence et rend le calcul parallélisable.

**Explication** — L'auto-attention projette chaque jeton en trois vecteurs : une **requête** $Q$ (ce que je cherche), une **clé** $K$ (ce que je propose), une **valeur** $V$ (ce que je transmets). Le poids d'attention du jeton $i$ vers le jeton $j$ est le produit scalaire $q_i \cdot k_j$, normalisé par $\sqrt{d_k}$ puis passé en softmax sur tous les $j$. La sortie est la moyenne pondérée des valeurs. En une formule : $\text{Attn}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$. La division par $\sqrt{d_k}$ n'est pas cosmétique : sans elle, les produits scalaires en grande dimension ont une variance élevée, la softmax sature et les gradients disparaissent.

Le mécanisme est appliqué en **plusieurs têtes** en parallèle, chacune sur un sous-espace : une tête peut suivre la syntaxe, une autre la coréférence. Un bloc Transformer complet enchaîne une normalisation de couche, l'attention multi-têtes, une connexion résiduelle, puis un MLP position par position avec un autre résiduel. On empile ces blocs, de 12 dans un petit modèle à plus de 100 dans les grands.

Deux conséquences importantes. D'abord, l'attention est **invariante à l'ordre** : sans information supplémentaire, une phrase mélangée donnerait le même résultat. Il faut donc ajouter un **encodage de position** (sinusoïdal, appris, ou RoPE aujourd'hui standard). Ensuite, le coût est **quadratique** en longueur de séquence — la matrice $QK^\top$ est de taille $T \times T$. C'est la limite fondamentale du contexte long, et l'objet de nombreux travaux (FlashAttention pour la mémoire, attention creuse ou linéaire pour la complexité).

**Cas d'utilisation**
- Tout traitement de langage moderne : c'est l'architecture de tous les grands modèles.
- Vision (ViT), audio (Whisper), multimodal (CLIP), biologie (AlphaFold).
- Séries temporelles longues (PatchTST, TFT) où les dépendances lointaines comptent.
- Données tabulaires via FT-Transformer, mais le boosting reste supérieur.
- Mauvais choix sur des séquences très longues sans optimisation dédiée, ou sur peu de données sans pré-entraînement : le Transformer est très gourmand en données.

**Algorithme**
```text
Attention par produit scalaire mis à l'échelle :
  1. Q = X W_q ; K = X W_k ; V = X W_v          (T, d_k)
  2. S = Q K' / sqrt(d_k)                        (T, T)
  3. Masquage si nécessaire :
       causal    -> S[i, j] = -inf pour j > i    (génération)
       remplissage -> -inf sur les positions de padding
  4. A = softmax(S, dim=-1)
  5. sortie = A V

Multi-têtes : h têtes de dimension d_k = d_modele / h, en parallèle,
concaténées puis projetées par W_o.

Bloc Transformer (pré-normalisation, standard actuel) :
  x = x + Attention(LayerNorm(x))
  x = x + MLP(LayerNorm(x))              MLP : Linear(d, 4d) -> GELU -> Linear(4d, d)

Encodeur (BERT)  : attention bidirectionnelle, tâche de masquage.
Décodeur (GPT)   : attention causale, prédiction du jeton suivant.
```

**Implémentation**
```python
import torch, torch.nn as nn, torch.nn.functional as F, math

# --- attention multi-têtes from scratch (palier 2) ---
class AttentionMultiTetes(nn.Module):
    def __init__(self, d_modele, n_tetes, causal=False, p=0.1):
        super().__init__()
        assert d_modele % n_tetes == 0
        self.h, self.dk, self.causal = n_tetes, d_modele // n_tetes, causal
        self.qkv = nn.Linear(d_modele, 3 * d_modele, bias=False)
        self.proj = nn.Linear(d_modele, d_modele)
        self.drop = nn.Dropout(p)

    def forward(self, x, masque_pad=None):
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        # (B, T, D) -> (B, h, T, dk)
        q, k, v = [t.view(B, T, self.h, self.dk).transpose(1, 2) for t in (q, k, v)]

        s = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)          # (B, h, T, T)
        if self.causal:
            m = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
            s = s.masked_fill(m, float('-inf'))
        if masque_pad is not None:
            s = s.masked_fill(~masque_pad[:, None, None, :], float('-inf'))
        a = self.drop(s.softmax(-1))
        y = (a @ v).transpose(1, 2).reshape(B, T, D)
        return self.proj(y), a          # a : poids d'attention, interprétables

class BlocTransformer(nn.Module):
    def __init__(self, d, h, p=0.1):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = AttentionMultiTetes(d, h, causal=True, p=p)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                 nn.Linear(4 * d, d), nn.Dropout(p))

    def forward(self, x):
        h, _ = self.att(self.n1(x))
        x = x + h                        # résiduel : le gradient passe intact
        return x + self.mlp(self.n2(x))

# --- en production : la version optimisée, 2 à 4x plus rapide et moins gourmande ---
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # FlashAttention si dispo

# --- en pratique : un modèle pré-entraîné, jamais entraîné de zéro ---
from transformers import AutoTokenizer, AutoModel
tok = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-base')
enc = AutoModel.from_pretrained('intfloat/multilingual-e5-base')
lots = tok(["première phrase", "deuxième phrase"], padding=True,
           truncation=True, max_length=512, return_tensors='pt')
with torch.no_grad():
    sortie = enc(**lots).last_hidden_state
# pooling moyen SUR LE MASQUE, jamais sur le remplissage
m = lots['attention_mask'].unsqueeze(-1)
embeddings = (sortie * m).sum(1) / m.sum(1)
embeddings = F.normalize(embeddings, dim=-1)
```

**Outils** — `torch.nn` (`MultiheadAttention`, `TransformerEncoderLayer`), `F.scaled_dot_product_attention`, `pip install transformers`.

**Alternatives open-source**
- *Bibliothèques* : **transformers** (Hugging Face) donne accès à des dizaines de milliers de modèles pré-entraînés — le point d'entrée obligé ; **nanoGPT** (Karpathy, ~300 lignes lisibles) est le meilleur support pour comprendre un GPT complet ; **x-transformers** implémente les variantes de recherche (RoPE, ALiBi, attention creuse) ; **flash-attn** pour l'attention optimisée en mémoire ; **xformers** (Meta) pour des blocs efficaces ; **mamba-ssm** pour les modèles à espace d'état, alternative sous-quadratique ; **sentence-transformers** pour des embeddings de phrases prêts à l'emploi.
- *Outils graphiques* : **BertViz** visualise les têtes d'attention couche par couche, remarquablement instructif ; **exBERT** pour explorer les représentations internes ; **Netron** pour l'architecture ; **The Illustrated Transformer** (Jay Alammar) et **Transformer Explainer** (Georgia Tech, interactif) sont les meilleures ressources d'intuition ; **TensorBoard** pour l'entraînement.

**Astuces**
- Utiliser `F.scaled_dot_product_attention` plutôt qu'une implémentation manuelle en production : elle bascule automatiquement sur FlashAttention, avec une mémoire linéaire au lieu de quadratique.
- La division par $\sqrt{d_k}$ est indispensable. L'omettre fait saturer la softmax et bloque l'apprentissage — bug classique d'une implémentation maison.
- L'échauffement du taux d'apprentissage est obligatoire sur un Transformer. Sans lui, la divergence survient dans les premières centaines d'itérations.
- La **pré-normalisation** (LayerNorm avant l'attention) est le standard actuel : elle rend l'entraînement bien plus stable que la post-normalisation du papier original.
- Ne jamais faire de pooling sur les positions de remplissage. C'est le bug le plus fréquent quand on calcule des embeddings de phrases à la main, et il dégrade silencieusement la qualité.
- Le coût mémoire de l'attention croît en $T^2$ : doubler la longueur de contexte quadruple la mémoire. C'est presque toujours la contrainte réelle, avant le nombre de paramètres.
- Les poids d'attention sont interprétables avec prudence : ils montrent où le modèle regarde, pas pourquoi. Ne pas les présenter comme une explication causale.
- N'entraîner un Transformer de zéro que pour apprendre. En production, on part toujours d'un modèle pré-entraîné.

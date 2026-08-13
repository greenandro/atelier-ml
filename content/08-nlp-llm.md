---
module: nlp-llm
titre: NLP et modèles de langage
ordre: 8
resume: Le domaine où le pré-entraînement a tout changé. On n'entraîne plus, on adapte et on évalue.
---

## Tokenisation
<!-- slug: tokenisation | difficulte: 2 | prereqs: pandas-manipulation -->

**En une phrase** — Découper du texte en unités entières que le modèle sait manipuler, en équilibrant taille du vocabulaire et longueur des séquences.

**Explication** — Deux extrêmes sont mauvais. Découper par **mots** produit un vocabulaire immense, ne gère pas les fautes ni les mots inconnus, et dépend fortement de la langue. Découper par **caractères** donne un vocabulaire minuscule mais des séquences très longues, coûteuses pour une attention quadratique. La solution universelle est la tokenisation en **sous-mots** : les mots fréquents restent entiers, les mots rares sont décomposés en morceaux réutilisables.

**BPE** (*byte-pair encoding*) construit le vocabulaire par fusions successives : on part des caractères, et on fusionne itérativement la paire adjacente la plus fréquente du corpus, jusqu'à atteindre la taille visée. C'est l'algorithme de GPT et de la plupart des modèles actuels, dans sa variante *byte-level* qui garantit qu'aucun caractère n'est hors vocabulaire — même un emoji ou un idéogramme inconnu. **WordPiece** (BERT) fusionne selon la vraisemblance plutôt que la fréquence brute ; **SentencePiece** avec Unigram (T5, Llama) part d'un grand vocabulaire et l'élague, et traite l'espace comme un caractère ordinaire, ce qui le rend indépendant de la langue.

Conséquences pratiques trop souvent ignorées. Le français consomme environ 1,3 fois plus de jetons que l'anglais pour le même contenu, et certaines langues jusqu'à 3 fois — cela affecte directement le coût d'appel d'une API et la longueur utile du contexte. Les nombres sont découpés de façon irrégulière, ce qui explique une partie des faiblesses arithmétiques des modèles. Et un jeton ne correspond pas à un mot : compter en mots pour estimer un contexte est une erreur systématique.

**Cas d'utilisation**
- Toute préparation de texte pour un modèle de langage.
- Estimation d'un coût d'API ou d'un dépassement de contexte avant l'appel.
- Découpage de documents en fragments (*chunking*) pour un RAG, où la longueur doit être mesurée en jetons.
- Analyse d'un corpus spécialisé pour décider s'il faut étendre le vocabulaire (médical, juridique, code).

**Algorithme**
```text
Entraînement BPE :
1. Vocabulaire initial = tous les octets (256 symboles) — aucun inconnu possible.
2. Pré-découper le texte (espaces, ponctuation) pour éviter les fusions
   traversant les frontières de mots.
3. Répéter jusqu'à |vocabulaire| = taille visée :
     a. Compter toutes les paires de symboles adjacents dans le corpus.
     b. Fusionner la plus fréquente ; l'ajouter au vocabulaire.
     c. Enregistrer la règle de fusion (l'ordre compte).
4. Encoder un texte nouveau = appliquer les fusions dans l'ordre appris.

Règle pratique : utiliser TOUJOURS le tokeniseur livré avec le modèle.
Un tokeniseur différent rend les identifiants de jetons incohérents.
```

**Implémentation**
```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-base')

texte = "L'apprentissage automatique n'est pas de la magie : 42 % de préparation."
ids = tok.encode(texte)
print(len(ids), "jetons")
print(tok.convert_ids_to_tokens(ids))          # voir le découpage réel

# Comparer le coût en jetons entre langues : information budgétaire concrète
for langue, phrase in [('fr', "Le modèle apprend à partir des données."),
                       ('en', "The model learns from the data.")]:
    print(langue, len(tok.encode(phrase)), "jetons")

# Découpage de documents pour un RAG : mesurer en JETONS, pas en caractères
def fragmenter(texte, tok, taille=400, chevauchement=60):
    ids = tok.encode(texte, add_special_tokens=False)
    pas = taille - chevauchement
    return [tok.decode(ids[i:i + taille]) for i in range(0, len(ids), pas)]

# Vérifier la longueur maximale du modèle avant d'envoyer
print(tok.model_max_length)
lots = tok(textes, padding=True, truncation=True, max_length=512, return_tensors='pt')

# --- BPE from scratch, version minimale (palier 2) ---
from collections import Counter

def entrainer_bpe(corpus, n_fusions=50):
    mots = Counter(' '.join(list(m)) + ' </w>' for m in corpus.split())
    fusions = []
    for _ in range(n_fusions):
        paires = Counter()
        for mot, n in mots.items():
            s = mot.split()
            for a, b in zip(s, s[1:]):
                paires[(a, b)] += n
        if not paires:
            break
        (a, b), _ = paires.most_common(1)[0]
        fusions.append((a, b))
        mots = Counter({mot.replace(f'{a} {b}', a + b): n for mot, n in mots.items()})
    return fusions

print(entrainer_bpe("le chat le chien le chaton le chiot", 8))

# Entraîner un vocabulaire sur un corpus spécialisé (code, médical, juridique)
# pip install tokenizers
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
t = Tokenizer(models.BPE())
t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
t.train_from_iterator(corpus_specialise,
                      trainers.BpeTrainer(vocab_size=32000,
                                          special_tokens=['<pad>', '<unk>', '<s>', '</s>']))
```

**Outils** — `pip install transformers tokenizers`, `pip install tiktoken` pour les tokeniseurs OpenAI.

**Alternatives open-source**
- *Bibliothèques* : **tokenizers** (Hugging Face, écrit en Rust) est la référence pour entraîner et utiliser des tokeniseurs rapides ; **SentencePiece** (Google) reste le standard pour les modèles multilingues et ne suppose aucune segmentation en mots ; **tiktoken** (OpenAI) pour compter exactement les jetons facturés par leurs API ; **spaCy** pour une tokenisation linguistique classique avec lemmatisation et analyse morphosyntaxique — encore le meilleur choix hors modèles de langage ; **NLTK** pour l'enseignement et les corpus classiques.
- *Outils graphiques* : le **Tokenizer Playground** de Hugging Face et le tokeniseur en ligne d'OpenAI montrent le découpage jeton par jeton, très éclairant ; **spaCy displaCy** visualise dépendances syntaxiques et entités nommées ; **Label Studio** et **Doccano** pour annoter du texte.

**Astuces**
- Ne jamais changer de tokeniseur entre l'entraînement et l'inférence : les identifiants de jetons ne signifient plus rien.
- Compter en jetons, pas en mots ni en caractères. Une règle grossière pour le français : environ 3 caractères par jeton.
- `truncation=True` sans vérifier ce qui est coupé fait perdre silencieusement la fin des documents longs. Toujours journaliser le taux de troncature.
- Le remplissage doit être associé à un masque d'attention. Sans masque, le modèle traite le remplissage comme du contenu.
- Sur un domaine très spécialisé (chimie, code, langue rare), un tokeniseur générique fragmente excessivement. Étendre le vocabulaire ou en entraîner un dédié peut réduire de 30 % la longueur des séquences.
- Les modèles arithmétiquement faibles le sont en partie à cause du découpage des nombres. Ne pas leur confier de calcul : leur donner un outil.

## Embeddings de texte
<!-- slug: embeddings | difficulte: 3 | prereqs: tokenisation, knn -->

**En une phrase** — Représenter un texte par un vecteur dense de quelques centaines de dimensions, tel que deux textes de sens proche donnent deux vecteurs proches.

**Explication** — Historiquement, on représentait un document par un vecteur creux de comptes de mots (sac de mots, **TF-IDF**). C'est encore très efficace pour la recherche par mots-clés et la classification de textes, mais deux formulations synonymes n'ont aucune similarité. **Word2Vec** a introduit les vecteurs denses appris par contexte, avec la propriété fameuse $\text{roi} - \text{homme} + \text{femme} \approx \text{reine}$. Sa limite est d'attribuer un vecteur unique à chaque mot, indépendamment du contexte : « avocat » a un seul vecteur pour le fruit et le juriste.

Les modèles contextuels résolvent cela : le vecteur d'un jeton dépend de toute la phrase. Pour obtenir un vecteur **de phrase** ou de document, on agrège les vecteurs de jetons — mais un pooling naïf sur un BERT brut donne des embeddings médiocres. C'est pourquoi on utilise des modèles spécifiquement entraînés pour cela (**Sentence-BERT**, E5, BGE, GTE), par apprentissage contrastif : rapprocher les paires de textes liés, éloigner les autres.

La mesure de proximité standard est la **similarité cosinus**, c'est-à-dire le produit scalaire de vecteurs normalisés. Elle ignore la norme, donc la longueur du texte. Point crucial et fréquemment oublié : de nombreux modèles exigent un **préfixe d'instruction** différent pour les requêtes et les documents (`query:` / `passage:` chez E5). L'omettre dégrade les résultats de plusieurs points sans qu'aucune erreur ne soit levée.

**Cas d'utilisation**
- Recherche sémantique et RAG : trouver les passages pertinents pour une question.
- Déduplication et détection de quasi-doublons dans un corpus.
- Clustering et découverte de thématiques (avec UMAP puis HDBSCAN).
- Classification avec très peu d'exemples : embeddings gelés plus régression logistique.
- Recommandation de contenu par similarité.
- Mauvais choix pour une recherche par identifiant, référence exacte ou terme rare : le lexical (BM25) y est supérieur, d'où l'intérêt de l'hybridation.

**Algorithme**
```text
1. Choisir un modèle adapté à la langue et à la tâche (voir le classement MTEB).
2. Fragmenter les documents : 200 à 500 jetons, chevauchement de 10 à 20 %,
   en respectant les frontières de paragraphes.
3. Encoder avec le préfixe attendu par le modèle (query: / passage:).
4. Pooling moyen SUR LE MASQUE d'attention, puis normalisation L2.
5. Indexer (FAISS, HNSW, pgvector). Le produit scalaire devient le cosinus.
6. À la recherche : encoder la requête avec SON préfixe, chercher les k voisins.
7. Améliorer : fusionner avec BM25 (recherche hybride), puis réordonner les
   30 premiers résultats avec un modèle de reclassement croisé.
```

**Implémentation**
```python
import numpy as np, torch, torch.nn.functional as F

# --- voie simple et recommandée ---
# pip install sentence-transformers
from sentence_transformers import SentenceTransformer
modele = SentenceTransformer('intfloat/multilingual-e5-base')

docs = ["Le gradient indique la direction de plus forte pente.",
        "Les arbres de décision découpent l'espace par des seuils.",
        "La cuisson des pâtes demande de l'eau salée."]
# Les préfixes ne sont pas décoratifs : ce modèle a été entraîné avec eux.
E = modele.encode([f"passage: {d}" for d in docs], normalize_embeddings=True)
q = modele.encode("query: comment fonctionne la descente de gradient ?",
                  normalize_embeddings=True)
scores = E @ q                                   # cosinus, car vecteurs normalisés
print(np.argsort(-scores), np.sort(-scores) * -1)

# --- pooling à la main : le piège du remplissage ---
from transformers import AutoTokenizer, AutoModel
tok = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-base')
enc = AutoModel.from_pretrained('intfloat/multilingual-e5-base').eval()

lots = tok([f"passage: {d}" for d in docs], padding=True, truncation=True,
           max_length=512, return_tensors='pt')
with torch.no_grad():
    h = enc(**lots).last_hidden_state
m = lots['attention_mask'].unsqueeze(-1).float()
emb = (h * m).sum(1) / m.sum(1)                  # moyenne sur les jetons VALIDES
emb = F.normalize(emb, dim=-1)

# --- index vectoriel pour des volumes réels ---
import faiss
index = faiss.IndexFlatIP(E.shape[1])            # produit scalaire = cosinus
index.add(E.astype('float32'))
d, i = index.search(q[None].astype('float32'), k=5)

# Au-delà d'un million de vecteurs : index approché
quant = faiss.IndexHNSWFlat(E.shape[1], 32)
quant.hnsw.efConstruction = 200
quant.add(E.astype('float32'))

# --- recherche hybride : dense + lexical, presque toujours meilleure ---
# pip install rank_bm25
from rank_bm25 import BM25Okapi
bm25 = BM25Okapi([d.lower().split() for d in docs])

def hybride(requete, k=5, alpha=0.6):
    s_dense = modele.encode(f"query: {requete}", normalize_embeddings=True) @ E.T
    s_lex = np.array(bm25.get_scores(requete.lower().split()))
    norm = lambda v: (v - v.min()) / (v.ptp() + 1e-9)
    s = alpha * norm(s_dense) + (1 - alpha) * norm(s_lex)
    return np.argsort(-s)[:k]

# --- reclassement : le gain le plus rentable d'un RAG ---
from sentence_transformers import CrossEncoder
rerank = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
candidats = [docs[j] for j in hybride("descente de gradient", k=30)]
paires = [[requete, c] for c in candidats]
ordre = np.argsort(-rerank.predict(paires))[:5]
```

**Outils** — `pip install sentence-transformers faiss-cpu rank_bm25`.

**Alternatives open-source**
- *Bibliothèques* : **sentence-transformers** est la voie la plus simple et couvre l'essentiel ; le classement **MTEB** (Hugging Face) permet de choisir un modèle sur des critères mesurés, par langue et par tâche ; **FAISS** pour l'index vectoriel local ; **hnswlib** plus léger ; **Qdrant**, **Milvus**, **Weaviate**, **Chroma** comme bases vectorielles avec filtrage et persistance ; **pgvector** pour rester dans PostgreSQL — souvent le bon choix pragmatique ; **BGE-M3** et **E5** pour le multilingue ; **model2vec** pour des embeddings statiques ultra-rapides sans GPU.
- *Outils graphiques* : **TensorFlow Embedding Projector** pour explorer un espace d'embeddings en 3D ; **Nomic Atlas** pour visualiser des millions de documents avec étiquetage automatique ; **BERTopic** produit des cartes de thématiques interactives ; **Qdrant** et **Weaviate** offrent des interfaces web de collections ; **RAGAS** pour évaluer une chaîne de recherche.

**Astuces**
- Vérifier dans la fiche du modèle s'il attend des préfixes (`query:` / `passage:`). C'est l'erreur la plus fréquente et la plus coûteuse, entièrement silencieuse.
- Normaliser les vecteurs puis utiliser le produit scalaire : c'est le cosinus, et c'est ce que les index attendent.
- Ne jamais faire de pooling sur le remplissage. Utiliser le masque d'attention.
- Le TF-IDF avec une régression logistique reste une baseline très solide en classification de texte, souvent à un ou deux points d'un modèle neuronal, pour mille fois moins de calcul. Le mesurer avant d'investir.
- La recherche hybride (dense + BM25) bat presque toujours le dense seul, surtout sur des noms propres, des références et des termes techniques rares.
- Le reclassement croisé des 30 premiers résultats est le meilleur rapport gain/effort d'un RAG : quelques points de pertinence pour dix lignes de code.
- La taille de fragment est un hyperparamètre réel : 200 jetons pour des questions factuelles précises, 800 pour du raisonnement contextuel. À mesurer, pas à supposer.

## RAG — génération augmentée par la recherche
<!-- slug: rag | difficulte: 4 | prereqs: embeddings, attention-transformer -->

**En une phrase** — Chercher les passages pertinents dans une base documentaire, les injecter dans le contexte du modèle, et lui demander de répondre en s'appuyant uniquement sur eux.

**Explication** — Un modèle de langage ne connaît que ses données d'entraînement, ne peut pas citer ses sources et invente avec assurance quand il ignore. Le RAG déplace la connaissance du modèle vers une base interrogeable. Le schéma comporte deux temps : **hors ligne**, on fragmente les documents, on les encode et on les indexe ; **en ligne**, on encode la question, on récupère les $k$ meilleurs passages, on les place dans le prompt avec une consigne stricte, et le modèle rédige une réponse sourcée.

Le point le plus important à comprendre est que **la qualité d'un RAG est dominée par la recherche, pas par le modèle génératif**. Si le bon passage n'est pas dans le contexte, aucun modèle ne produira la bonne réponse. L'ordre des efforts est donc : qualité du découpage, qualité de la recherche (hybride + reclassement), puis seulement la formulation du prompt, et en dernier le choix du modèle.

Trois difficultés reviennent systématiquement. Le **découpage** : trop petit, le passage perd son contexte ; trop grand, il dilue le signal et remplit le contexte de bruit. La **perte au milieu** (*lost in the middle*) : les modèles exploitent mal l'information située au centre d'un long contexte, donc il vaut mieux 5 passages bien classés que 30 approximatifs. Et l'**évaluation** : il faut mesurer séparément le rappel de la recherche et la fidélité de la réponse aux sources, sinon on ne sait pas quoi corriger.

**Cas d'utilisation**
- Questions-réponses sur une documentation interne, un corpus juridique, une base de connaissances.
- Assistance qui doit citer ses sources et rester vérifiable.
- Domaines évoluant vite, où un réentraînement serait absurde.
- Alternative à l'ajustement fin quand l'objectif est d'apporter des **connaissances** (le fine-tuning sert à modifier le *comportement*, pas à ajouter des faits).
- Mauvais choix pour des questions agrégatives (« combien de contrats au total ? ») : c'est une requête de base de données, pas une recherche sémantique.

**Algorithme**
```text
Hors ligne (indexation) :
1. Extraire le texte proprement (PDF, HTML) en conservant la structure.
2. Fragmenter : 200-500 jetons, chevauchement 10-20 %, aux frontières de
   paragraphes. Conserver les métadonnées (titre, source, page, date).
3. Encoder avec le préfixe passage. Indexer (FAISS / pgvector / Qdrant).
4. Indexer aussi en lexical (BM25) pour l'hybride.

En ligne (réponse) :
5. Recherche hybride, k = 20 à 30 candidats.
6. Reclassement croisé -> garder les 3 à 5 meilleurs.
7. Construire le prompt : consigne stricte + passages numérotés + question.
8. Générer avec température basse (0 à 0.3).
9. Vérifier que chaque affirmation cite un passage. Sinon, répondre
   « information non trouvée » plutôt que d'inventer.

Évaluation, séparément :
  recherche  -> rappel@k, MRR, nDCG sur un jeu de questions annotées
  génération -> fidélité aux sources, pertinence, taux d'abstention correct
```

**Implémentation**
```python
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import faiss

encodeur = SentenceTransformer('intfloat/multilingual-e5-base')
rerank = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')

# --- 1. Fragmentation en respectant les paragraphes ---
def fragmenter(texte, tok, taille=400, chevauchement=60):
    paras = [p.strip() for p in texte.split('\n\n') if p.strip()]
    frags, courant, n = [], [], 0
    for p in paras:
        np_ = len(tok.encode(p, add_special_tokens=False))
        if n + np_ > taille and courant:
            frags.append('\n\n'.join(courant))
            # chevauchement : garder le dernier paragraphe
            courant, n = [courant[-1]], len(tok.encode(courant[-1], add_special_tokens=False))
        courant.append(p); n += np_
    if courant:
        frags.append('\n\n'.join(courant))
    return frags

# --- 2. Index hybride ---
class BaseDocs:
    def __init__(self, fragments, metas):
        self.frags, self.metas = fragments, metas
        E = encodeur.encode([f"passage: {f}" for f in fragments],
                            normalize_embeddings=True, batch_size=32,
                            show_progress_bar=True).astype('float32')
        self.index = faiss.IndexFlatIP(E.shape[1]); self.index.add(E)
        self.bm25 = BM25Okapi([f.lower().split() for f in fragments])

    def chercher(self, question, k_candidats=25, k_final=5, alpha=0.6):
        q = encodeur.encode(f"query: {question}", normalize_embeddings=True)
        s_d, _ = self.index.search(q[None].astype('float32'), len(self.frags))
        s_dense = np.zeros(len(self.frags)); s_dense[_[0]] = s_d[0]
        s_lex = np.array(self.bm25.get_scores(question.lower().split()))
        norm = lambda v: (v - v.min()) / (np.ptp(v) + 1e-9)
        cand = np.argsort(-(alpha * norm(s_dense) + (1 - alpha) * norm(s_lex)))[:k_candidats]
        # reclassement croisé : le gain le plus rentable
        scores = rerank.predict([[question, self.frags[i]] for i in cand])
        garde = [cand[j] for j in np.argsort(-scores)[:k_final]]
        return [(self.frags[i], self.metas[i]) for i in garde]

# --- 3. Prompt strict : c'est lui qui limite les hallucinations ---
GABARIT = """Réponds à la question en t'appuyant UNIQUEMENT sur les passages ci-dessous.
Cite le numéro du passage entre crochets après chaque affirmation.
Si les passages ne contiennent pas la réponse, écris exactement :
"Information non trouvée dans les documents fournis."

{passages}

Question : {question}
Réponse :"""

def construire(question, resultats):
    passages = "\n\n".join(
        f"[{i+1}] (source : {m['source']}, p.{m.get('page','?')})\n{t}"
        for i, (t, m) in enumerate(resultats))
    return GABARIT.format(passages=passages, question=question)

# --- 4. Évaluation de la RECHERCHE : à faire avant de toucher au prompt ---
def rappel_at_k(base, jeu_test, k=5):
    """jeu_test : [(question, id_fragment_attendu), ...]"""
    ok = 0
    for q, attendu in jeu_test:
        trouves = [m['id'] for _, m in base.chercher(q, k_final=k)]
        ok += attendu in trouves
    return ok / len(jeu_test)
```

**Outils** — `pip install sentence-transformers faiss-cpu rank_bm25 pymupdf`.

**Alternatives open-source**
- *Bibliothèques* : **LlamaIndex** est le cadre le plus complet pour construire un RAG (chargeurs, découpage, index, requêtes) ; **LangChain** est plus généraliste et très répandu, au prix d'une abstraction lourde ; **Haystack** (deepset) est solide et orienté production ; **RAGAS** et **TruLens** pour l'évaluation automatisée ; **unstructured** et **PyMuPDF** pour extraire proprement du texte de PDF, étape sous-estimée qui détermine tout le reste ; **Docling** (IBM) pour convertir des documents complexes en Markdown structuré ; **vLLM** ou **Ollama** pour servir le modèle génératif localement.
- *Outils graphiques* : **Qdrant**, **Weaviate** et **Chroma** proposent des interfaces web pour inspecter collections et résultats de recherche ; **Nomic Atlas** pour visualiser le corpus indexé et repérer les zones mal couvertes ; **Langfuse** (libre) pour tracer et évaluer les appels d'une chaîne RAG en production ; **Open WebUI** pour une interface de discussion prête à l'emploi sur un modèle local.

**Astuces**
- Investir d'abord dans la recherche, pas dans le prompt. Mesurer le rappel@5 sur 50 questions annotées : si le bon passage n'est pas récupéré, rien d'autre ne compte.
- L'extraction du texte des PDF est la source d'échec la plus fréquente et la plus négligée. Un tableau mal extrait produit du texte incohérent que la recherche ne retrouvera jamais.
- Conserver les métadonnées (source, page, date) avec chaque fragment. Sans elles, aucune citation vérifiable n'est possible, et le RAG perd son principal avantage.
- Cinq bons passages valent mieux que trente moyens. Le contexte long dégrade la qualité et coûte plus cher.
- Une consigne explicite d'abstention (« si l'information n'est pas là, dis-le ») réduit fortement les hallucinations. Le taux d'abstention correct fait partie des métriques à suivre.
- Le chevauchement entre fragments évite qu'une réponse soit coupée en deux. 10 à 20 % suffisent ; au-delà, on duplique l'index pour rien.
- Une question agrégative ou comptable ne relève pas du RAG. Router ce type de question vers une requête SQL est la bonne réponse d'architecture.

## Ajustement fin et LoRA
<!-- slug: finetuning-lora | difficulte: 4 | prereqs: transfert-apprentissage, attention-transformer -->

**En une phrase** — Adapter un modèle de langage pré-entraîné à une tâche ou un style, en n'entraînant qu'une infime fraction de ses paramètres.

**Explication** — Il faut d'abord savoir **quand** ajuster. Le fine-tuning modifie le **comportement** : format de sortie, ton, respect d'un schéma, tâche de classification spécialisée, langue ou jargon métier. Il n'est pas le bon outil pour ajouter des **connaissances** — c'est le rôle du RAG, plus fiable, moins coûteux et mis à jour instantanément. Beaucoup de projets ajustent un modèle là où un bon prompt et une recherche documentaire auraient suffi.

Ajuster tous les paramètres d'un modèle de 7 milliards de poids demande environ 80 Go de mémoire, en comptant les états de l'optimiseur. **LoRA** rend l'opération accessible : on gèle le modèle et on ajoute, à côté de chaque matrice de poids $W$, deux petites matrices $A$ et $B$ telles que la mise à jour effective soit $W + BA$ avec un rang $r$ très faible (8 à 64). On n'entraîne que $A$ et $B$, soit moins de 1 % des paramètres. L'hypothèse — vérifiée empiriquement — est que l'adaptation nécessaire est de rang faible. **QLoRA** ajoute la quantification du modèle gelé en 4 bits, ce qui permet d'ajuster un modèle de 7 milliards de paramètres sur un GPU de 16 Go.

Ce qui détermine réellement le résultat n'est pas la technique mais les **données**. Cinq cents exemples propres, cohérents dans leur format et représentatifs de l'usage réel valent mieux que cinquante mille exemples bruités. C'est là qu'il faut passer le temps.

**Cas d'utilisation**
- Format de sortie strict : JSON conforme à un schéma, extraction structurée.
- Style, ton, registre de langue propres à une organisation.
- Classification ou extraction spécialisée où un petit modèle ajusté bat un gros modèle générique, pour cent fois moins cher.
- Jargon métier très spécifique (médical, juridique, industriel).
- Réduction de coût : remplacer un gros modèle par un petit modèle ajusté sur les traces du premier (distillation).
- Mauvais choix pour ajouter des faits, ou quand moins de 200 exemples de qualité sont disponibles — préférer alors le prompt avec exemples.

**Algorithme**
```text
1. Vérifier qu'un prompt bien construit avec 5 exemples ne suffit pas.
   Cette vérification économise souvent tout le projet.
2. Construire le jeu de données :
     format conversationnel uniforme, 500 à 5000 exemples,
     jeu de validation séparé, cohérence de format vérifiée exemple par exemple.
3. Choisir le modèle de base : le plus petit qui atteint l'objectif.
4. Configurer LoRA : r = 16, alpha = 32, cibles = toutes les projections
   d'attention et du MLP, dropout 0.05.
5. Entraîner : 1 à 3 époques seulement, lr ~ 1e-4 à 2e-4, cosinus,
   perte calculée sur la RÉPONSE uniquement (masquer le prompt).
6. Évaluer sur des exemples jamais vus, avec une métrique de tâche —
   pas seulement la perte.
7. Fusionner les poids LoRA pour le déploiement, ou les charger à chaud
   (plusieurs adaptateurs sur un même modèle de base).
```

**Implémentation**
```python
# pip install transformers peft trl bitsandbytes datasets accelerate
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

modele_base = 'Qwen/Qwen2.5-1.5B-Instruct'

# QLoRA : quantification 4 bits du modèle gelé
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                         bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_use_double_quant=True)

tok = AutoTokenizer.from_pretrained(modele_base)
modele = AutoModelForCausalLM.from_pretrained(modele_base, quantization_config=bnb,
                                             device_map='auto')
modele = prepare_model_for_kbit_training(modele)

lora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias='none', task_type='CAUSAL_LM',
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                    'gate_proj', 'up_proj', 'down_proj'],
)
modele = get_peft_model(modele, lora)
modele.print_trainable_parameters()      # typiquement 0.3 % des paramètres

# Données : format conversationnel homogène. C'est ici que se joue le résultat.
exemples = [{'messages': [
    {'role': 'system', 'content': "Tu extrais les informations en JSON strict."},
    {'role': 'user', 'content': "Facture 4821 du 12/03/2025, montant 1 240,50 € HT."},
    {'role': 'assistant', 'content': '{"numero":"4821","date":"2025-03-12","montant_ht":1240.50}'},
]}]
jeu = Dataset.from_list(exemples).train_test_split(test_size=0.1, seed=0)

trainer = SFTTrainer(
    model=modele,
    train_dataset=jeu['train'], eval_dataset=jeu['test'],
    processing_class=tok,
    args=SFTConfig(
        output_dir='sorties', num_train_epochs=2,
        per_device_train_batch_size=2, gradient_accumulation_steps=8,
        learning_rate=2e-4, lr_scheduler_type='cosine', warmup_ratio=0.05,
        bf16=True, logging_steps=10, eval_strategy='steps', eval_steps=50,
        save_total_limit=2, max_length=1024,
        # ne calculer la perte que sur la réponse de l'assistant
        assistant_only_loss=True,
    ),
)
trainer.train()

# Sauvegarde de l'adaptateur seul : quelques dizaines de Mo
modele.save_pretrained('adaptateur-extraction')

# Déploiement : fusionner pour supprimer le surcoût d'inférence
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained(modele_base, torch_dtype=torch.bfloat16)
fusionne = PeftModel.from_pretrained(base, 'adaptateur-extraction').merge_and_unload()
fusionne.save_pretrained('modele-fusionne')

# Contrôle qualité : vérifier la métrique de TÂCHE, pas la perte
import json
def taux_json_valide(modele, tok, questions):
    ok = 0
    for q in questions:
        s = generer(modele, tok, q, temperature=0.0)
        try:
            json.loads(s); ok += 1
        except json.JSONDecodeError:
            pass
    return ok / len(questions)
```

**Outils** — `pip install transformers peft trl bitsandbytes datasets accelerate`.

**Alternatives open-source**
- *Bibliothèques* : **Unsloth** accélère l'entraînement LoRA de 2 à 5 fois avec une empreinte mémoire réduite, sans changement d'API — le meilleur choix sur un seul GPU ; **Axolotl** pilote l'ajustement par fichier YAML, très pratique pour itérer sans écrire de code ; **LLaMA-Factory** offre une interface graphique et couvre SFT, DPO et évaluation ; **peft** pour LoRA, DoRA, adaptateurs et *prompt tuning* ; **trl** pour SFT, DPO, GRPO et l'alignement par préférences ; **DeepSpeed** et **FSDP** pour l'entraînement multi-GPU d'un modèle complet ; **MLX** pour ajuster sur puce Apple.
- *Outils graphiques* : **LLaMA-Factory WebUI** pour lancer un ajustement à la souris ; **Weights & Biases** ou **TensorBoard** pour suivre les courbes ; **Argilla** (libre) pour construire et nettoyer collaborativement le jeu d'entraînement — l'étape qui détermine réellement la qualité ; **Ollama** et **LM Studio** pour tester le modèle obtenu localement ; **Langfuse** pour comparer les versions en production.

**Astuces**
- Toujours essayer un bon prompt avec quelques exemples avant d'ajuster. Dans une majorité de cas, cela suffit, et cela ne coûte ni GPU ni maintenance.
- La qualité et l'homogénéité des données dominent tout. Cinq cents exemples relus valent mieux que cinquante mille collectés automatiquement.
- Ne pas calculer la perte sur le prompt. Le modèle apprendrait à générer les questions au lieu des réponses — erreur fréquente et discrète.
- Une à trois époques suffisent. Au-delà, le modèle mémorise le jeu et perd sa capacité de généralisation ainsi que ses aptitudes générales (*oubli catastrophique*).
- `alpha = 2r` est la convention usuelle. Augmenter $r$ au-delà de 64 apporte rarement quelque chose et coûte de la mémoire.
- Cibler toutes les projections (attention **et** MLP) donne de bien meilleurs résultats que les seules matrices $q$ et $v$ du papier original.
- Évaluer avec une métrique de tâche sur des exemples jamais vus. Une perte qui baisse ne prouve pas qu'un JSON sera valide.
- Conserver l'adaptateur séparé pendant les itérations : quelques dizaines de mégaoctets contre plusieurs gigaoctets, et on peut charger plusieurs adaptateurs sur un même modèle de base.

## Évaluation des modèles génératifs
<!-- slug: evaluation-generatif | difficulte: 4 | prereqs: metriques-classification, rag -->

**En une phrase** — Mesurer la qualité de textes générés, alors qu'il n'existe pas une seule bonne réponse et qu'aucune métrique automatique ne suffit.

**Explication** — Les métriques historiques comparent la sortie à une référence : **BLEU** (précision des n-grammes, traduction), **ROUGE** (rappel des n-grammes, résumé), **METEOR**. Elles sont peu corrélées au jugement humain dès que plusieurs formulations sont acceptables — une réponse parfaite avec d'autres mots obtient un mauvais score. **BERTScore** améliore les choses en comparant des embeddings au lieu de mots exacts, mais le problème de fond demeure.

L'approche dominante aujourd'hui est le **modèle juge** : un modèle de langage puissant évalue la sortie selon une grille explicite. C'est efficace et scalable, avec des biais documentés qu'il faut connaître — préférence pour les réponses longues, pour son propre style, sensibilité à l'ordre de présentation lors d'une comparaison par paires. On les atténue par une grille de critères précise, l'inversion aléatoire de l'ordre, et une calibration sur un échantillon annoté par un humain.

Le principe méthodologique le plus important est de **décomposer**. Une évaluation « le RAG est-il bon ? » n'apprend rien. Il faut mesurer séparément le rappel de la recherche, la **fidélité** de la réponse aux passages fournis (chaque affirmation est-elle appuyée ?), la **pertinence** vis-à-vis de la question, et le **taux d'abstention correct** (le modèle dit-il « je ne sais pas » quand il faut ?). Chacune de ces mesures pointe vers une correction différente.

Et rien ne remplace un petit **jeu d'évaluation** propre, construit à la main : 50 à 200 cas représentatifs, avec les réponses attendues, versionné dans le dépôt. C'est ce qui permet de savoir si une modification améliore ou dégrade — sans cela, on itère à l'aveugle.

**Cas d'utilisation**
- Comparer deux prompts, deux modèles, deux configurations de RAG.
- Non-régression avant mise en production : un changement de modèle ne doit pas dégrader les cas connus.
- Surveillance en production : dérive de qualité, apparition d'hallucinations.
- Choisir entre un gros modèle coûteux et un petit modèle ajusté.

**Algorithme**
```text
1. Construire un jeu d'évaluation à la main : 50 à 200 cas couvrant les
   usages réels, y compris les cas limites et les questions sans réponse.
   Le versionner. C'est l'actif le plus précieux du projet.
2. Choisir les métriques par composant :
     recherche   -> rappel@k, MRR, nDCG
     génération  -> fidélité, pertinence, format valide, abstention correcte
     déterministe-> taux de JSON valide, conformité au schéma, exactitude
                    d'extraction (ces mesures sont les plus fiables : les privilégier)
3. Automatiser : script exécutable en une commande, résultats comparables
   entre versions.
4. Modèle juge pour les critères subjectifs, avec grille explicite,
   sortie structurée (note + justification), ordre randomisé.
5. Calibrer le juge : annoter 30 cas à la main, mesurer l'accord.
   En dessous d'un accord raisonnable, revoir la grille.
6. Suivre aussi le coût et la latence : ils font partie de la qualité.
```

**Implémentation**
```python
import json, numpy as np

# --- 1. Métriques déterministes : les plus fiables, à privilégier ---
def conforme_schema(sortie, cles_requises):
    try:
        d = json.loads(sortie)
    except json.JSONDecodeError:
        return False
    return all(k in d for k in cles_requises)

def exactitude_extraction(predictions, verites, cles):
    """Exactitude champ par champ : bien plus informatif qu'un score global."""
    res = {}
    for k in cles:
        ok = [str(p.get(k, '')).strip().lower() == str(v.get(k, '')).strip().lower()
              for p, v in zip(predictions, verites)]
        res[k] = float(np.mean(ok))
    return res

# --- 2. Métriques de recherche ---
def mrr(resultats, attendus):
    """Rang réciproque moyen du premier document pertinent."""
    s = 0.0
    for res, att in zip(resultats, attendus):
        for rang, doc in enumerate(res, 1):
            if doc in att:
                s += 1 / rang
                break
    return s / len(resultats)

def rappel_at_k(resultats, attendus, k=5):
    return float(np.mean([len(set(r[:k]) & set(a)) / max(1, len(a))
                          for r, a in zip(resultats, attendus)]))

# --- 3. Modèle juge, avec grille explicite et sortie structurée ---
GRILLE_JUGE = """Tu évalues une réponse produite à partir de passages fournis.

Passages :
{passages}

Question : {question}
Réponse à évaluer : {reponse}

Note chaque critère de 1 à 5 :
- fidelite : chaque affirmation de la réponse est-elle appuyée par les passages ?
             (5 = tout est appuyé, 1 = affirmations inventées)
- pertinence : la réponse répond-elle à la question posée ?
- completude : les éléments disponibles dans les passages sont-ils utilisés ?
- concision : absence de remplissage et de répétitions.

Réponds en JSON strict :
{{"fidelite": n, "pertinence": n, "completude": n, "concision": n,
  "justification": "une phrase", "affirmations_non_appuyees": ["..."]}}"""

def juger(client, question, passages, reponse):
    msg = GRILLE_JUGE.format(passages=passages, question=question, reponse=reponse)
    # appel au modèle juge, température 0, sortie JSON
    return json.loads(appeler_modele(client, msg, temperature=0.0))

# --- 4. Comparaison par paires, avec ordre randomisé contre le biais de position ---
def comparer_paire(client, question, rep_a, rep_b, graine=0):
    rng = np.random.default_rng(graine)
    inverse = bool(rng.integers(2))
    p, s = (rep_b, rep_a) if inverse else (rep_a, rep_b)
    verdict = appeler_juge_comparatif(client, question, p, s)   # "premiere" | "seconde" | "egal"
    if verdict == 'egal':
        return 'egal'
    gagne_premiere = verdict == 'premiere'
    return 'A' if gagne_premiere != inverse else 'B'

# --- 5. Harnais d'évaluation : une commande, un tableau comparable ---
def evaluer(jeu, pipeline, nom_version):
    lignes = []
    for cas in jeu:
        sortie = pipeline(cas['question'])
        lignes.append({
            'question': cas['question'],
            'json_valide': conforme_schema(sortie['texte'], cas.get('cles', [])),
            'rappel5': rappel_at_k([sortie['docs']], [cas['docs_attendus']], 5),
            'abstention_correcte': (sortie['abstenu'] == cas['sans_reponse']),
            'latence_ms': sortie['latence'],
            'cout_jetons': sortie['jetons'],
        })
    import pandas as pd
    df = pd.DataFrame(lignes)
    print(f"--- {nom_version} ---")
    print(df.select_dtypes('number').mean().round(3))
    print("taux JSON valide :", df['json_valide'].mean().round(3))
    df.to_csv(f'eval_{nom_version}.csv', index=False)     # comparer entre versions
    return df
```

**Outils** — `pip install ragas deepeval promptfoo` (le dernier via npm), plus un modèle juge local ou distant.

**Alternatives open-source**
- *Bibliothèques* : **RAGAS** implémente fidélité, pertinence de réponse et précision de contexte spécifiquement pour le RAG ; **DeepEval** propose une suite de tests façon `pytest` pour modèles de langage, très pratique en intégration continue ; **promptfoo** compare des prompts et des modèles côte à côte par fichier de configuration ; **lm-evaluation-harness** (EleutherAI) exécute les bancs d'essai académiques standards ; **Giskard** détecte biais et vulnérabilités ; **BERTScore** et **evaluate** (Hugging Face) pour les métriques classiques ; **Langfuse** et **Phoenix** (Arize) pour la traçabilité et l'évaluation en production.
- *Outils graphiques* : **Langfuse** offre une interface de traces, d'annotation humaine et de comparaison de versions — le meilleur choix libre pour un usage sérieux ; **Phoenix** (Arize) pour explorer les traces et les embeddings ; **Argilla** pour l'annotation humaine collaborative ; **promptfoo** génère un rapport web comparatif ; **Chainlit** et **Open WebUI** pour tester interactivement.

**Astuces**
- Construire le jeu d'évaluation **avant** d'optimiser quoi que ce soit. Sans lui, chaque modification est une intuition non vérifiée.
- Privilégier les métriques déterministes quand la tâche le permet : validité du JSON, exactitude d'extraction champ par champ, conformité au schéma. Elles sont fiables, gratuites et reproductibles.
- Décomposer l'évaluation par composant. « La qualité a baissé » n'est pas actionnable ; « le rappel@5 est passé de 0,82 à 0,61 » l'est.
- Les modèles juges préfèrent les réponses longues. Inclure explicitement un critère de concision et vérifier la corrélation avec un jugement humain sur 30 cas.
- Randomiser l'ordre en comparaison par paires : le biais de position est important et fausse tout classement.
- Inclure dans le jeu des questions **sans réponse** dans le corpus. Le taux d'abstention correct est souvent la métrique la plus révélatrice de la fiabilité réelle.
- Suivre le coût et la latence au même titre que la qualité. Un gain de 2 % pour un coût multiplié par dix n'est pas une amélioration.
- Fixer la température à 0 pour toute évaluation. Sinon le bruit d'échantillonnage dépasse l'effet mesuré.

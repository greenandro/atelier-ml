---
module: mlops
titre: MLOps et mise en production
ordre: 10
resume: Un modèle dans un notebook n'a aucune valeur. Ce module transforme un score en service qui tient dans le temps.
---

## Suivi d'expériences
<!-- slug: suivi-experiences | difficulte: 2 | prereqs: validation-croisee -->

**En une phrase** — Enregistrer automatiquement paramètres, métriques, code et artefacts de chaque entraînement, pour pouvoir comparer et reproduire.

**Explication** — Sans outil, on perd le fil au bout de vingt essais : quel jeu de features donnait 0,87 ? avec quel `learning_rate` ? le fichier a-t-il été écrasé ? Le suivi d'expériences résout ce problème en enregistrant chaque exécution comme un objet consultable : hyperparamètres, métriques, version du code (empreinte Git), dépendances, et artefacts (modèle sérialisé, figures, importances de features).

**MLflow** est la référence libre. Quatre composants : le *tracking* (journalisation et interface de comparaison), les *projects* (empaquetage reproductible), les *models* (format standard avec signature d'entrée/sortie), et le *registry* (versions d'un modèle avec étapes : `staging`, `production`, `archived`). En pratique, le tracking et le registry couvrent l'essentiel du besoin.

L'apport dépasse le confort : c'est ce qui rend un résultat **défendable**. Six mois plus tard, on peut retrouver exactement quelle version du modèle a produit un score, avec quelles données et quel code. Sans cela, un modèle en production est une boîte dont l'origine est perdue — situation courante, et sérieusement problématique dès qu'un audit ou une régression survient.

**Cas d'utilisation**
- Dès le troisième entraînement d'un projet : le seuil de rentabilité est très bas.
- Comparer des dizaines de configurations et retrouver la meilleure de façon fiable.
- Reproduire un résultat ancien, ou expliquer un écart entre deux versions.
- Passer un modèle du prototype à la production avec une traçabilité complète.

**Algorithme**
```text
1. Un dépôt Git par projet. Aucun entraînement sans commit.
2. Une expérience MLflow par problème métier, une exécution par entraînement.
3. À chaque exécution, journaliser :
     paramètres  : hyperparamètres, version des données, jeu de features
     métriques   : validation ET test, avec écart type de validation croisée
     artefacts   : modèle, figures de diagnostic, importances, config complète
     étiquettes  : commit Git, auteur, motif de l'essai
4. Comparer dans l'interface, trier par métrique, exporter le tableau.
5. Enregistrer les candidats sérieux au registry, avec une signature.
6. Promouvoir en production explicitement, jamais par copie de fichier.
```

**Implémentation**
```python
# pip install mlflow
import mlflow, mlflow.sklearn, subprocess, numpy as np
from sklearn.model_selection import cross_validate, StratifiedKFold

mlflow.set_tracking_uri('file:./mlruns')        # ou un serveur distant
mlflow.set_experiment('churn-clients')

commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                        capture_output=True, text=True).stdout.strip()

params = {'n_estimators': 500, 'learning_rate': 0.05, 'num_leaves': 31}

with mlflow.start_run(run_name='lgbm-features-v3') as run:
    mlflow.set_tags({'commit': commit, 'jeu_features': 'v3',
                     'motif': 'ajout des features de récence'})
    mlflow.log_params(params)

    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    res = cross_validate(pipeline, X_tr, y_tr, cv=cv,
                         scoring=['roc_auc', 'average_precision'], n_jobs=-1)
    for m in ['roc_auc', 'average_precision']:
        v = res[f'test_{m}']
        mlflow.log_metric(f'cv_{m}', v.mean())
        mlflow.log_metric(f'cv_{m}_std', v.std())     # l'écart type compte autant

    pipeline.fit(X_tr, y_tr)
    p_te = pipeline.predict_proba(X_te)[:, 1]
    mlflow.log_metric('test_roc_auc', roc_auc_score(y_te, p_te))

    # Artefacts : figures de diagnostic et modèle avec signature
    fig = tracer_courbes(y_te, p_te)
    mlflow.log_figure(fig, 'diagnostic.png')
    from mlflow.models import infer_signature
    mlflow.sklearn.log_model(pipeline, name='modele',
                             signature=infer_signature(X_tr, p_te),
                             input_example=X_tr[:3])

    print('run_id :', run.info.run_id)

# Comparer par programme, pas seulement dans l'interface
runs = mlflow.search_runs(experiment_names=['churn-clients'],
                          order_by=['metrics.cv_roc_auc DESC'], max_results=10)
print(runs[['run_id', 'params.learning_rate', 'metrics.cv_roc_auc', 'tags.jeu_features']])

# Enregistrer et promouvoir
mlflow.register_model(f'runs:/{run.info.run_id}/modele', 'churn')
from mlflow import MlflowClient
c = MlflowClient()
c.set_registered_model_alias('churn', 'production', version=3)

# Recharger la version de production, où que ce soit
modele = mlflow.pyfunc.load_model('models:/churn@production')
predictions = modele.predict(nouvelles_donnees)

# Journalisation automatique : une ligne, capture presque tout
mlflow.autolog()
```

**Outils** — `pip install mlflow`, puis `mlflow ui` pour l'interface sur `localhost:5000`.

**Alternatives open-source**
- *Bibliothèques* : **Weights & Biases** a la meilleure interface de comparaison (le cœur client est libre, le service est hébergé) ; **Aim** est entièrement libre, auto-hébergeable et très rapide sur des milliers d'exécutions ; **Neptune** et **Comet** sont des équivalents commerciaux avec offres gratuites ; **DVC** avec `dvc exp` intègre le suivi dans Git sans serveur ; **ClearML** couvre suivi, orchestration et données ; **Optuna** enregistre nativement ses essais et se combine avec MLflow ; **TensorBoard** suffit si l'on ne suit que des courbes d'entraînement.
- *Outils graphiques* : l'**interface MLflow** pour comparer, trier et filtrer les exécutions ; **Optuna Dashboard** pour l'importance des hyperparamètres ; **Aim UI** pour explorer de très gros volumes d'exécutions ; **DVC Studio** (partiellement libre) pour visualiser les expériences liées à Git.

**Astuces**
- Journaliser l'empreinte du commit Git à chaque exécution. Sans elle, la reproductibilité est illusoire, même avec tous les paramètres enregistrés.
- `mlflow.autolog()` capture automatiquement paramètres, métriques et modèle pour scikit-learn, PyTorch, LightGBM et XGBoost. À activer par défaut, quitte à compléter à la main.
- Enregistrer aussi l'écart type de la validation croisée. Comparer deux moyennes sans dispersion conduit à choisir du bruit.
- Journaliser la version des **données**, pas seulement du code. Un identifiant de dataset ou une empreinte du fichier suffit.
- La signature du modèle (`infer_signature`) documente les colonnes attendues et fait échouer proprement une entrée malformée plutôt que de produire une prédiction absurde.
- Enregistrer les figures de diagnostic comme artefacts : dans six mois, la courbe de calibration sera plus parlante que le score.
- Écrire une étiquette « motif de l'essai ». C'est ce qui manque le plus quand on relit son propre historique.

## Versionnement des données
<!-- slug: versionnement-donnees | difficulte: 3 | prereqs: suivi-experiences -->

**En une phrase** — Suivre l'évolution des jeux de données et des modèles comme on suit le code, sans mettre des gigaoctets dans Git.

**Explication** — Git est conçu pour du texte : il compare ligne par ligne et conserve tout l'historique. Y placer un fichier de 2 Go rend le dépôt inutilisable pour toujours — même supprimé, le fichier reste dans l'historique. Or un projet de ML a besoin de savoir quelle version des données a produit quel modèle. Trois éléments doivent être versionnés ensemble : le **code**, les **données** et le **modèle**.

La solution standard, portée par **DVC**, consiste à mettre dans Git non pas les données mais leur **empreinte** : un petit fichier `.dvc` contenant un hachage MD5 et un chemin. Les données réelles vivent dans un stockage distant (S3, MinIO, un disque réseau), organisé par contenu. `git checkout` d'un ancien commit suivi de `dvc checkout` restaure l'état exact des données de l'époque. On obtient le versionnement sans le poids.

DVC ajoute un mécanisme précieux : les **pipelines** déclaratifs (`dvc.yaml`). On décrit les étapes avec leurs dépendances et leurs sorties ; `dvc repro` ne réexécute que ce qui a changé, en cascade. C'est un `make` conscient des données, qui remplace la collection de notebooks exécutés dans un ordre approximatif — source majeure d'irreproductibilité.

**Cas d'utilisation**
- Tout projet dont les données évoluent : réétiquetage, nouvelles périodes, corrections.
- Reproduire un résultat ancien à l'identique, y compris les données.
- Travail en équipe sur un même jeu de données, sans se l'envoyer par messagerie.
- Chaîne de traitement à étapes coûteuses, où l'on veut éviter de tout recalculer.
- Superflu pour un dataset public figé et un projet solo : un identifiant de version et un `README` suffisent alors.

**Algorithme**
```text
1. git init ; dvc init. Ajouter .dvc/cache au .gitignore (fait automatiquement).
2. Configurer un stockage distant (S3, MinIO, dossier réseau, Google Drive).
3. dvc add data/brut.parquet -> crée data/brut.parquet.dvc (petit, versionné)
   git add data/brut.parquet.dvc .gitignore ; git commit
4. dvc push -> envoie les données réelles vers le stockage distant.
5. Décrire le pipeline dans dvc.yaml : étapes, dépendances (deps), sorties (outs),
   métriques (metrics), paramètres (params.yaml).
6. dvc repro -> n'exécute que les étapes dont une dépendance a changé.
7. Reproduire un état ancien :
     git checkout <commit> && dvc checkout
8. dvc metrics diff / dvc plots diff -> comparer deux versions.
```

**Implémentation**
```bash
pip install "dvc[s3]"
git init && dvc init

# Stockage distant (MinIO auto-hébergé, S3, ou simple dossier)
dvc remote add -d stockage s3://mon-bucket/atelier-ml
dvc remote modify stockage endpointurl http://localhost:9000   # MinIO

# Versionner un jeu de données
dvc add data/brut.parquet
git add data/brut.parquet.dvc data/.gitignore
git commit -m "données brutes v1"
dvc push

# Reproduire un état passé, données incluses
git checkout v1.0 && dvc checkout
```

```yaml
# dvc.yaml — le pipeline déclaratif remplace l'enchaînement de notebooks
stages:
  preparer:
    cmd: python src/preparer.py
    deps:
      - src/preparer.py
      - data/brut.parquet
    params:
      - preparation.seuil_rare
    outs:
      - data/propre.parquet

  features:
    cmd: python src/features.py
    deps:
      - src/features.py
      - data/propre.parquet
    outs:
      - data/features.parquet

  entrainer:
    cmd: python src/entrainer.py
    deps:
      - src/entrainer.py
      - data/features.parquet
    params:
      - modele.learning_rate
      - modele.num_leaves
    outs:
      - modeles/modele.joblib
    metrics:
      - metriques.json:
          cache: false
    plots:
      - courbes.csv:
          x: seuil
          y: precision
```

```yaml
# params.yaml — tous les hyperparamètres en un seul endroit versionné
preparation:
  seuil_rare: 20
modele:
  learning_rate: 0.05
  num_leaves: 31
```

```bash
dvc repro                       # n'exécute que ce qui a changé
dvc metrics show
dvc metrics diff HEAD~1         # comparer à la version précédente
dvc plots diff HEAD~1           # rapport HTML comparatif

# Expériences sans polluer l'historique Git
dvc exp run -S modele.learning_rate=0.1
dvc exp show                    # tableau comparatif
dvc exp apply <nom>             # adopter la meilleure
```

**Outils** — `pip install "dvc[s3]"`, plus un stockage objet (MinIO en local, S3 sinon).

**Alternatives open-source**
- *Bibliothèques et systèmes* : **Git LFS** est plus simple mais sans pipelines ni cache par contenu — acceptable pour quelques fichiers moyens ; **lakeFS** apporte des branches et des commits sur un lac de données entier, très puissant à l'échelle d'une organisation ; **Delta Lake** et **Iceberg** offrent le voyage dans le temps natif sur les tables, ce qui couvre une partie du besoin ; **Pachyderm** pour des pipelines versionnés sur Kubernetes ; **Quilt** pour des paquets de données ; **DoltHub** pour une base relationnelle versionnée comme Git ; **Hugging Face Datasets** avec versionnement par révision pour des jeux publics.
- *Outils graphiques* : **DVC Studio** et l'extension **VS Code DVC** pour visualiser expériences et métriques ; **lakeFS UI** pour explorer branches et différences de données ; **MinIO Console** pour inspecter le stockage ; **DataHub** ou **OpenMetadata** pour le catalogue et la traçabilité en aval.

**Astuces**
- Ne jamais commiter de données ni de poids de modèle dans Git. Un fichier de 500 Mo poussé par erreur reste dans l'historique définitivement, et `git clone` devient interminable.
- Mettre `data/`, `models/`, `*.db`, `*.ckpt`, `.venv/` dans `.gitignore` dès le premier commit.
- Le pipeline `dvc.yaml` est le vrai gain : il supprime la question « dans quel ordre exécuter les notebooks ? », première cause d'irreproductibilité.
- Externaliser tous les hyperparamètres dans `params.yaml`. DVC détecte alors les changements et sait quoi réexécuter.
- `dvc exp run` permet d'essayer des variantes sans créer de commits, puis d'adopter la meilleure. Bien plus propre que dix branches abandonnées.
- Sans stockage distant, DVC ne protège de rien : le cache local disparaît avec la machine. Configurer le distant dès le début.
- Versionner aussi le jeu de **test**. Un score n'est comparable dans le temps que si le jeu d'évaluation est identique.

## Conteneurisation
<!-- slug: docker | difficulte: 3 | prereqs: environnements-uv -->

**En une phrase** — Emballer le code, ses dépendances et son environnement système dans une image exécutable à l'identique partout.

**Explication** — Un environnement Python ne fige que Python. Un projet de ML dépend aussi de bibliothèques système (BLAS, OpenCV, pilotes CUDA), d'une version de système d'exploitation, de variables d'environnement. Docker capture tout cela dans une **image** : un système de fichiers en couches, construit par un `Dockerfile`, dont chaque instruction crée une couche mise en cache. Un **conteneur** est une instance en cours d'exécution de cette image.

Le mécanisme de cache dicte la façon d'écrire un `Dockerfile`. Les couches sont invalidées en cascade dès qu'une change : il faut donc placer ce qui change rarement au début. En pratique : copier d'abord le fichier de dépendances et les installer, **puis** copier le code. Inversé, chaque modification d'une ligne de code réinstalle toutes les dépendances — erreur qui transforme une reconstruction de 5 secondes en 4 minutes.

Deux pratiques réduisent fortement la taille et le risque. La **construction multi-étapes** compile dans une image lourde et ne copie que le résultat dans une image légère : une image passe couramment de 1,5 Go à 300 Mo. Et l'exécution en **utilisateur non privilégié** évite qu'une faille du service donne les droits root dans le conteneur.

Pour le GPU, l'image doit correspondre à la version du pilote de l'hôte, et il faut le runtime NVIDIA (`--gpus all`). C'est précisément le cas où Docker devient indispensable plutôt que confortable.

**Cas d'utilisation**
- Déployer une API de prédiction de façon identique en développement et en production.
- Figer un environnement GPU avec ses versions de CUDA et cuDNN.
- Exécuter un entraînement reproductible sur une autre machine ou dans une chaîne d'intégration.
- Faire tourner des services d'accompagnement (MLflow, MinIO, PostgreSQL) sans les installer.
- Superflu pour un script local sans dépendance système : `uv` suffit alors.

**Algorithme**
```text
Dockerfile efficace, dans cet ordre :
1. Image de base minimale et ÉPINGLÉE (python:3.12-slim, pas :latest).
2. Dépendances système (apt) si nécessaire, puis nettoyage des listes.
3. Copier UNIQUEMENT le fichier de dépendances ; installer.   <- couche mise en cache
4. Copier le code.                                            <- change souvent
5. Créer un utilisateur non root et basculer dessus.
6. EXPOSE, HEALTHCHECK, CMD.

Puis :
  .dockerignore (données, .git, .venv, __pycache__, notebooks)
  docker build -t image:tag .
  docker run --rm -p 8000:8000 image:tag
  docker compose pour plusieurs services
```

**Implémentation**
```dockerfile
# Dockerfile — API de prédiction, construction multi-étapes
FROM python:3.12-slim AS base
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1

# --- étape de construction ---
FROM base AS build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
# Dépendances AVANT le code : la couche reste en cache tant qu'elles ne changent pas
COPY requirements.txt .
RUN python -m venv /opt/venv && /opt/venv/bin/pip install -r requirements.txt

# --- étape finale, légère ---
FROM base AS runtime
COPY --from=build /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /app
COPY src/ ./src/
COPY modeles/modele.joblib ./modeles/
# utilisateur non privilégié
RUN useradd -m -u 1000 service && chown -R service:service /app
USER service
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

```text
# .dockerignore — sans lui, le contexte de build embarque tout le dataset
.git
.venv
data/
mlruns/
notebooks/
__pycache__/
*.ipynb
*.db
```

```yaml
# compose.yaml — la pile complète en une commande
services:
  api:
    build: .
    ports: ["8000:8000"]
    environment:
      MLFLOW_TRACKING_URI: http://mlflow:5000
    depends_on: [mlflow]

  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.16.0
    command: mlflow server --host 0.0.0.0 --backend-store-uri sqlite:///mlflow.db
    ports: ["5000:5000"]
    volumes: ["mlflow_data:/mlflow"]

  minio:
    image: quay.io/minio/minio
    command: server /data --console-address ":9001"
    ports: ["9000:9000", "9001:9001"]
    environment:
      MINIO_ROOT_USER: admin
      MINIO_ROOT_PASSWORD: changez-moi
    volumes: ["minio_data:/data"]

volumes:
  mlflow_data:
  minio_data:
```

```bash
docker build -t atelier-api:0.1.0 .
docker run --rm -p 8000:8000 atelier-api:0.1.0
docker compose up -d
docker compose logs -f api

# Entraînement GPU
docker run --gpus all --rm -v "$PWD/data:/app/data:ro" atelier-train:0.1.0
```

**Outils** — Docker Engine ou Podman (sans démon, sans privilèges root).

**Alternatives open-source**
- *Bibliothèques et outils* : **Podman** est compatible avec Docker sans démon privilégié, plus sûr par conception ; **BuildKit** et **buildx** pour des constructions parallèles et multi-architecture ; **Apptainer** (ex-Singularity) est le standard en calcul scientifique partagé ; **Nix** ou **Guix** pour une reproductibilité encore plus stricte que Docker, au prix d'une courbe d'apprentissage sérieuse ; **Kubernetes** ou **Nomad** pour orchestrer plusieurs conteneurs en production ; **Skaffold** et **Tilt** pour la boucle de développement ; **Trivy** et **Grype** pour analyser les vulnérabilités d'une image.
- *Outils graphiques* : **Docker Desktop** et **Podman Desktop** pour gérer images et conteneurs visuellement ; **Portainer** pour administrer un hôte Docker depuis un navigateur ; **Lazydocker** en interface terminal, très pratique ; **Dive** pour explorer les couches d'une image et repérer ce qui la gonfle ; **k9s** pour Kubernetes.

**Astuces**
- Copier `requirements.txt` avant le code. Inverser cet ordre réinstalle toutes les dépendances à chaque modification d'une ligne de code.
- Épingler la version de l'image de base. `python:3.12-slim` et non `python:latest` : sinon la reconstruction d'un mois plus tard produit une image différente.
- Le `.dockerignore` est indispensable. Sans lui, le contexte de build envoie `data/` et `.git` au démon, ce qui peut représenter des gigaoctets.
- Utiliser `-slim` plutôt que l'image complète, et une construction multi-étapes : d'une image de 1,5 Go on descend à 300 Mo, ce qui accélère chaque déploiement.
- Ne jamais mettre de secret dans un `Dockerfile` ni dans une variable `ENV`. Les couches sont inspectables par quiconque a l'image.
- Ne pas embarquer les données dans l'image : les monter en volume. Une image doit contenir du code, pas des jeux de données.
- Exécuter en utilisateur non root. Une ligne, et cela supprime toute une classe de risques.
- Sur GPU, la version de CUDA de l'image doit être compatible avec le pilote de l'hôte. C'est la première chose à vérifier devant un « CUDA driver version is insufficient ».

## Servir un modèle
<!-- slug: serving-api | difficulte: 3 | prereqs: docker, pipelines -->

**En une phrase** — Exposer un modèle derrière une interface HTTP fiable, validée, observable et suffisamment rapide.

**Explication** — Le passage du notebook au service impose des exigences nouvelles. La **validation d'entrée** d'abord : un champ manquant, une chaîne au lieu d'un nombre, une modalité inconnue ne doivent pas produire une prédiction silencieusement absurde mais une erreur claire. Pydantic, dans FastAPI, s'en charge déclarativement et génère la documentation OpenAPI au passage.

Le **chargement du modèle** doit se faire une fois au démarrage, pas à chaque requête — erreur qui multiplie la latence par cent. Le modèle chargé doit être le **pipeline complet**, prétraitement inclus : réimplémenter le prétraitement côté service est la principale cause d'écart entre entraînement et production (*training-serving skew*).

Trois régimes de service à distinguer. Le **temps réel synchrone** (une requête, une prédiction) est le cas usuel : optimiser la latence du 99e centile, pas la moyenne. Le **lot** (scorer des millions de lignes la nuit) n'a pas besoin d'API du tout : un script suffit et sera bien plus efficace. Et le **flux** pour du scoring événementiel.

Enfin, l'**observabilité** n'est pas optionnelle. Journaliser chaque prédiction avec ses entrées, sa sortie, sa latence et la version du modèle est ce qui permettra plus tard de détecter une dérive, d'expliquer une décision contestée, et de constituer un jeu de données de production. C'est à faire dès la première version, car ces données ne se reconstituent pas.

**Cas d'utilisation**
- Scoring en ligne appelé par une application : recommandation, risque, tarification.
- Mise à disposition d'un modèle à d'autres équipes via un contrat d'interface stable.
- Prototype démontrable rapidement (FastAPI plus Gradio ou Streamlit).
- Inutile si le besoin est un scoring nocturne : un script planifié écrivant dans une table est plus simple et plus robuste.

**Algorithme**
```text
1. Charger le pipeline COMPLET au démarrage (prétraitement + modèle).
2. Définir le schéma d'entrée et de sortie avec Pydantic. Rejeter proprement
   ce qui ne s'y conforme pas.
3. Endpoints minimaux :
     POST /predict   une ou plusieurs lignes
     GET  /health    le service répond
     GET  /ready     le modèle est chargé et fonctionnel
     GET  /metrics   compteurs et latences (format Prometheus)
4. Journaliser : identifiant de requête, entrées, prédiction, latence,
   version du modèle. En base ou en fichier structuré (JSON par ligne).
5. Traiter les lots dans une seule requête plutôt que N appels : le coût
   par ligne s'effondre.
6. Conteneuriser. Tester la latence sous charge avant de promettre un SLA.
7. Déployer la nouvelle version en parallèle (canari ou ombre) avant de basculer.
```

**Implémentation**
```python
# src/api.py
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
import joblib, pandas as pd, time, uuid, logging, json

logging.basicConfig(level=logging.INFO)
etat = {}

@asynccontextmanager
async def cycle_vie(app: FastAPI):
    etat['modele'] = joblib.load('modeles/modele.joblib')      # UNE fois au démarrage
    etat['version'] = '3.2.0'
    etat['colonnes'] = list(etat['modele'].feature_names_in_)
    yield
    etat.clear()

app = FastAPI(title='API de scoring', version='3.2.0', lifespan=cycle_vie)

class Client(BaseModel):
    age: int = Field(ge=18, le=120)
    revenu: float = Field(ge=0)
    anciennete_mois: int = Field(ge=0)
    region: str
    segment: str

    @field_validator('region')
    @classmethod
    def region_connue(cls, v):
        if v not in {'nord', 'sud', 'est', 'ouest'}:
            raise ValueError(f"région inconnue : {v}")
        return v

class Requete(BaseModel):
    lignes: list[Client] = Field(min_length=1, max_length=1000)

class Prediction(BaseModel):
    probabilites: list[float]
    version_modele: str
    id_requete: str
    latence_ms: float

@app.post('/predict', response_model=Prediction)
def predire(req: Requete):
    t0 = time.perf_counter()
    rid = str(uuid.uuid4())
    try:
        df = pd.DataFrame([l.model_dump() for l in req.lignes])[etat['colonnes']]
        p = etat['modele'].predict_proba(df)[:, 1].tolist()
    except Exception as e:
        logging.exception('échec de prédiction')
        raise HTTPException(status_code=422, detail=str(e))

    dt = (time.perf_counter() - t0) * 1000
    # Journalisation structurée : la base de la détection de dérive future
    logging.info(json.dumps({'id': rid, 'n': len(p), 'latence_ms': round(dt, 2),
                             'version': etat['version'],
                             'entrees': [l.model_dump() for l in req.lignes],
                             'sorties': p}))
    return Prediction(probabilites=p, version_modele=etat['version'],
                      id_requete=rid, latence_ms=round(dt, 2))

@app.get('/health')
def health():
    return {'statut': 'ok'}

@app.get('/ready')
def ready():
    if 'modele' not in etat:
        raise HTTPException(503, 'modèle non chargé')
    return {'statut': 'pret', 'version': etat['version']}
```

```python
# tests/test_api.py — un test de contrat vaut mieux qu'une surveillance tardive
from fastapi.testclient import TestClient
from src.api import app

def test_predict_nominal():
    with TestClient(app) as c:
        r = c.post('/predict', json={'lignes': [{'age': 35, 'revenu': 42000,
                    'anciennete_mois': 24, 'region': 'nord', 'segment': 'A'}]})
        assert r.status_code == 200
        assert 0.0 <= r.json()['probabilites'][0] <= 1.0

def test_rejette_entree_invalide():
    with TestClient(app) as c:
        r = c.post('/predict', json={'lignes': [{'age': 5, 'revenu': -1,
                    'anciennete_mois': 0, 'region': 'lune', 'segment': 'A'}]})
        assert r.status_code == 422
```

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 4
# documentation interactive générée : http://localhost:8000/docs

# Mesurer la latence sous charge AVANT de promettre quoi que ce soit
pip install locust    # ou : hey -n 2000 -c 50 -m POST -D corps.json http://localhost:8000/predict
```

**Outils** — `pip install fastapi uvicorn pydantic joblib`.

**Alternatives open-source**
- *Bibliothèques et serveurs* : **BentoML** empaquette modèle, dépendances et API en un artefact déployable, avec traitement par lots adaptatif intégré ; **MLflow models serve** sert directement un modèle du registry, idéal pour un prototype ; **KServe** et **Seldon Core** pour du service sur Kubernetes avec canari et test en ombre natifs ; **NVIDIA Triton** pour du service GPU multi-modèles à haute performance ; **ONNX Runtime** convertit et accélère l'inférence, souvent 2 à 5 fois plus rapide que scikit-learn ou PyTorch ; **vLLM** ou **Ollama** pour servir des modèles de langage ; **LitServe** comme alternative légère et moderne ; **Gradio** et **Streamlit** pour une démonstration en quelques lignes.
- *Outils graphiques* : la **documentation OpenAPI** générée par FastAPI (`/docs`) sert de banc d'essai interactif ; **Grafana** avec **Prometheus** pour latences et taux d'erreur ; **Locust** offre une interface web de test de charge ; **Portainer** pour surveiller les conteneurs ; **Langfuse** si le service enveloppe un modèle de langage.

**Astuces**
- Charger le modèle au démarrage, jamais dans le gestionnaire de requête. C'est la cause numéro un des latences catastrophiques.
- Servir le **pipeline complet**. Réimplémenter le prétraitement dans l'API garantit un écart avec l'entraînement, d'autant plus insidieux qu'il est partiel.
- Optimiser le 99e centile de latence, pas la moyenne. Les utilisateurs perçoivent la queue de distribution.
- Accepter des lots dans une seule requête : scorer 500 lignes en un appel coûte à peine plus qu'une seule, et divise le coût unitaire.
- Journaliser les entrées et sorties dès la première version. Sans ces traces, la détection de dérive est impossible, et l'explication d'une décision contestée aussi.
- Versionner le modèle dans la réponse. Quand deux versions coexistent, c'est la seule façon d'attribuer une prédiction.
- Écrire un test de contrat qui vérifie le rejet des entrées invalides. Il attrape les régressions de schéma bien avant la production.
- Tester la charge avant de communiquer un engagement de latence. L'écart entre l'intuition et la mesure est systématiquement grand.

## Surveillance et dérive
<!-- slug: monitoring-derive | difficulte: 4 | prereqs: serving-api, statistiques-inferentielles -->

**En une phrase** — Détecter que les données d'entrée ou la relation apprise ont changé, avant que la dégradation de performance ne soit constatée par les utilisateurs.

**Explication** — Un modèle en production se dégrade nécessairement, parce que le monde change. Trois formes de dérive, aux conséquences distinctes. La **dérive de covariables** : la distribution des features change ($P(X)$), par exemple une nouvelle population d'utilisateurs. La relation apprise reste peut-être valide, mais le modèle extrapole hors de son domaine d'entraînement. La **dérive de concept** : la relation elle-même change ($P(y|X)$) — les mêmes caractéristiques n'impliquent plus le même comportement. C'est la plus grave, et la seule qui rende vraiment le modèle faux. Et la **dérive d'étiquette** : la prévalence change, ce qui décalibre les probabilités.

La difficulté opérationnelle majeure est le **délai d'étiquetage**. On ne saura si les prédictions d'aujourd'hui étaient bonnes que dans des semaines, voire des mois (défaut de paiement, résiliation). Pendant ce temps, on ne peut surveiller que des **indicateurs indirects** : distribution des features, distribution des scores prédits, taux de valeurs manquantes, apparition de modalités inconnues, latence.

Les tests statistiques utiles sont peu nombreux. **Kolmogorov-Smirnov** compare deux distributions continues. Le **PSI** (*population stability index*) est le standard en scoring de crédit, avec des seuils empiriques bien établis : moins de 0,1 négligeable, 0,1 à 0,25 à surveiller, au-delà de 0,25 dérive significative. Un piège à connaître : sur des millions de lignes, tout test devient significatif — il faut donc regarder la **taille d'effet**, pas la p-valeur.

**Cas d'utilisation**
- Tout modèle en production, sans exception, dès le premier jour.
- Alerte sur une panne d'amont : un capteur muet, une API changée, une colonne devenue vide.
- Décider quand réentraîner, plutôt que réentraîner à date fixe sans savoir si c'est utile.
- Justifier auprès du métier une baisse de performance par une cause identifiée.

**Algorithme**
```text
1. Constituer une référence : les données d'entraînement, ou une fenêtre de
   production stable et validée.
2. Surveiller par fenêtre (jour ou semaine) :
     a. distribution de chaque feature -> KS ou PSI contre la référence
     b. distribution des scores prédits -> souvent le signal le plus précoce
     c. taux de manquants et de modalités inconnues par colonne
     d. volume de requêtes, latence, taux d'erreur
3. Quand les étiquettes arrivent, calculer les métriques réelles par cohorte
   temporelle et les comparer à la validation d'origine.
4. Seuils d'alerte :
     PSI > 0.25 sur une feature importante  -> enquêter
     PSI > 0.1 sur plusieurs features        -> enquêter
     dérive des scores prédits               -> enquêter immédiatement
     chute de performance réelle             -> réentraîner
5. Distinguer une dérive réelle d'un incident de données. Une colonne
   subitement à 100 % de manquants est une panne, pas une dérive.
6. Automatiser un rapport hebdomadaire. Une surveillance non automatisée
   n'est pas consultée.
```

**Implémentation**
```python
import numpy as np, pandas as pd
from scipy.stats import ks_2samp

def psi(reference, courant, bins=10):
    """Population Stability Index. < 0.1 stable, 0.1-0.25 à surveiller, > 0.25 dérive."""
    bornes = np.percentile(reference, np.linspace(0, 100, bins + 1))
    bornes[0], bornes[-1] = -np.inf, np.inf
    r = np.histogram(reference, bins=bornes)[0] / len(reference)
    c = np.histogram(courant, bins=bornes)[0] / len(courant)
    r, c = np.clip(r, 1e-6, None), np.clip(c, 1e-6, None)
    return float(((c - r) * np.log(c / r)).sum())

def psi_categoriel(reference, courant):
    r = reference.value_counts(normalize=True)
    c = courant.value_counts(normalize=True)
    idx = r.index.union(c.index)
    r = r.reindex(idx, fill_value=1e-6).clip(1e-6)
    c = c.reindex(idx, fill_value=1e-6).clip(1e-6)
    return float(((c - r) * np.log(c / r)).sum())

def rapport_derive(ref: pd.DataFrame, cur: pd.DataFrame, importances: dict):
    lignes = []
    for col in ref.columns:
        if pd.api.types.is_numeric_dtype(ref[col]):
            v = psi(ref[col].dropna(), cur[col].dropna())
            ks = ks_2samp(ref[col].dropna(), cur[col].dropna()).statistic
        else:
            v = psi_categoriel(ref[col].dropna(), cur[col].dropna())
            ks = np.nan
        nouvelles = (set(cur[col].dropna().unique()) - set(ref[col].dropna().unique())
                     if not pd.api.types.is_numeric_dtype(ref[col]) else set())
        lignes.append({
            'feature': col, 'psi': round(v, 4), 'ks': round(ks, 4) if ks == ks else None,
            'manquants_ref': round(ref[col].isna().mean(), 4),
            'manquants_cur': round(cur[col].isna().mean(), 4),
            'modalites_nouvelles': len(nouvelles),
            'importance': importances.get(col, 0.0),
            'alerte': v > 0.25 or (v > 0.1 and importances.get(col, 0) > 0.05),
        })
    return pd.DataFrame(lignes).sort_values(['alerte', 'psi'], ascending=[False, False])

rapport = rapport_derive(df_reference, df_semaine, importances)
print(rapport.head(12).to_string(index=False))

# Le signal le plus précoce : la distribution des SCORES prédits
psi_scores = psi(scores_reference, scores_semaine)
print(f"PSI des scores : {psi_scores:.4f}")
# Il bouge avant les métriques de performance, car il n'attend pas les étiquettes.

# Performance réelle par cohorte, quand les étiquettes arrivent
from sklearn.metrics import roc_auc_score
suivi = (journal.dropna(subset=['etiquette_reelle'])
         .assign(semaine=lambda d: d['ts'].dt.to_period('W'))
         .groupby('semaine')
         .apply(lambda g: pd.Series({
             'n': len(g),
             'auc': roc_auc_score(g['etiquette_reelle'], g['score']) if g['etiquette_reelle'].nunique() > 1 else np.nan,
             'taux_positif': g['etiquette_reelle'].mean(),
             'score_moyen': g['score'].mean(),
         })))
print(suivi.round(4))
# score_moyen qui s'écarte de taux_positif = décalibration

# Détecteur de dérive par classifieur : peut-on distinguer référence et courant ?
def derive_par_classifieur(ref, cur):
    """AUC ~ 0.5 = distributions indistinguables. AUC > 0.75 = dérive nette."""
    import lightgbm as lgb
    from sklearn.model_selection import cross_val_score
    X = pd.concat([ref, cur], ignore_index=True)
    y = np.r_[np.zeros(len(ref)), np.ones(len(cur))]
    auc = cross_val_score(lgb.LGBMClassifier(n_estimators=100, verbose=-1),
                          X, y, cv=3, scoring='roc_auc').mean()
    return auc
```

**Outils** — `pip install evidently scipy scikit-learn`, plus Prometheus et Grafana pour les métriques techniques.

**Alternatives open-source**
- *Bibliothèques* : **Evidently AI** est la référence libre — rapports HTML de dérive, de qualité de données et de performance, avec suites de tests exécutables en intégration continue ; **NannyML** estime la performance **sans étiquettes**, ce qui répond directement au problème du délai d'étiquetage ; **alibi-detect** (Seldon) propose des détecteurs de dérive avancés, y compris pour images et texte ; **deepchecks** pour la validation de données et de modèles ; **whylogs** produit des profils statistiques légers, adaptés à un flux ; **Great Expectations** ou **Soda** pour des contrats de qualité de données en amont ; **river** pour de l'apprentissage incrémental s'adaptant à la dérive.
- *Outils graphiques* : **Evidently** génère des tableaux de bord HTML autonomes, parfaits pour un rapport hebdomadaire ; **Grafana** avec **Prometheus** pour latence, débit et erreurs ; **Streamlit** pour bâtir un tableau de bord de surveillance sur mesure en une heure ; **Apache Superset** ou **Metabase** au-dessus de la table de journalisation ; **Airflow** ou **Dagster** pour planifier le calcul du rapport.

**Astuces**
- Surveiller d'abord la distribution des **scores prédits**. Elle ne nécessite aucune étiquette, bouge avant les métriques de performance, et détecte la plupart des incidents.
- Pondérer les alertes par l'importance des features. Une dérive sur une variable négligeable n'a pas de conséquence ; une dérive sur la variable dominante en a immédiatement.
- Sur des millions de lignes, tout test statistique est significatif. Utiliser le PSI et ses seuils empiriques plutôt que des p-valeurs.
- Distinguer panne et dérive. Une colonne à 100 % de manquants, une modalité soudain absente, un format de date changé : ce sont des incidents d'ingénierie, à corriger en amont et non par un réentraînement.
- Journaliser les prédictions dès le premier jour de production. Ces données ne se reconstituent pas, et sans elles aucune analyse rétrospective n'est possible.
- Réentraîner sur détection, pas sur calendrier. Un réentraînement mensuel systématique consomme du temps sans garantie, et peut même dégrader si les données récentes sont atypiques.
- NannyML mérite d'être connu : estimer la performance sans étiquettes est exactement ce qui manque dans la plupart des dispositifs de surveillance.
- Automatiser le rapport et le pousser vers un canal consulté (courriel, messagerie d'équipe). Un tableau de bord qu'il faut penser à ouvrir n'est jamais ouvert.

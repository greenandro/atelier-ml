---
module: avance
titre: Sujets avancés et recherche
ordre: 11
resume: Les sujets qui séparent celui qui applique des recettes de celui qui comprend et sait lire la littérature.
---

## Interprétabilité et SHAP
<!-- slug: interpretabilite-shap | difficulte: 4 | prereqs: gradient-boosting, selection-features -->

**En une phrase** — Attribuer à chaque feature sa contribution exacte à une prédiction donnée, avec une méthode qui possède des garanties mathématiques.

**Explication** — Deux niveaux d'interprétation. L'interprétabilité **globale** répond à « qu'a appris le modèle ? » : importance des features, dépendances partielles. L'interprétabilité **locale** répond à « pourquoi *cette* prédiction ? » : c'est ce qu'exigent un client dont le crédit est refusé, un médecin, un régulateur.

**SHAP** apporte une réponse fondée. L'idée vient de la théorie des jeux : la valeur de Shapley répartit équitablement le gain d'une coalition entre ses joueurs. Ici, les joueurs sont les features, le gain est l'écart entre la prédiction et la prédiction moyenne. La valeur de Shapley d'une feature est sa contribution marginale moyennée sur **tous les ordres d'ajout possibles** des features. Elle satisfait des propriétés uniques : l'additivité (la somme des contributions égale exactement l'écart à la moyenne), la symétrie, et l'absence d'attribution aux features non utilisées. Aucune autre méthode d'attribution ne les vérifie toutes.

Le calcul exact est exponentiel, mais `TreeSHAP` l'obtient en temps polynomial pour les modèles à base d'arbres — d'où l'omniprésence de SHAP avec XGBoost et LightGBM. Pour les autres modèles, `KernelSHAP` approche par échantillonnage, bien plus lentement.

Une mise en garde essentielle : SHAP explique **le modèle**, pas la réalité. Une contribution élevée signifie que le modèle s'appuie sur cette feature, pas qu'elle cause le phénomène. Confondre les deux est l'erreur la plus répandue sur ce sujet, et elle conduit à des décisions métier erronées.

**Cas d'utilisation**
- Justifier une décision individuelle : crédit, assurance, diagnostic, priorisation.
- Déboguer un modèle : une feature au poids anormal révèle souvent une fuite de données.
- Communiquer avec le métier : les figures SHAP sont comprises sans formation statistique.
- Vérifier l'équité : les contributions diffèrent-elles systématiquement entre sous-groupes ?
- Mauvais choix comme preuve de causalité, ou sur des features fortement corrélées où l'attribution se répartit arbitrairement entre elles.

**Algorithme**
```text
1. Choisir l'explainer adapté :
     arbres et ensembles       -> TreeExplainer (exact, rapide)
     réseaux de neurones       -> DeepExplainer / GradientExplainer
     modèle quelconque         -> KernelExplainer (lent) ou PermutationExplainer
2. Calculer les valeurs SHAP sur un échantillon représentatif (1000 à 5000
   lignes suffisent pour les vues globales).
3. Lectures globales :
     beeswarm  -> importance ET sens de l'effet, la figure la plus riche
     bar       -> importance moyenne absolue
     scatter   -> forme de la relation, équivalent d'une dépendance partielle
4. Lectures locales :
     waterfall -> décomposition d'une prédiction unique
     force     -> même chose en vue compacte
5. Vérifier l'additivité : somme des valeurs + valeur de base = prédiction.
   Si ce n'est pas le cas, l'explainer est mal configuré.
6. Confronter aux connaissances métier. Un effet contre-intuitif est soit une
   découverte, soit un bug — le plus souvent un bug.
```

**Implémentation**
```python
# pip install shap
import shap, numpy as np, lightgbm as lgb
import matplotlib.pyplot as plt

modele = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05).fit(X_tr, y_tr)

# TreeSHAP : exact et rapide sur les modèles à base d'arbres
explainer = shap.TreeExplainer(modele)
sv = explainer(X_te)                       # objet Explanation

# --- vues globales ---
shap.plots.beeswarm(sv, max_display=15)    # importance + sens de l'effet
shap.plots.bar(sv, max_display=15)
shap.plots.scatter(sv[:, 'anciennete'], color=sv[:, 'revenu'])   # effet + interaction

# --- explication individuelle : ce qu'on montre à un client ---
i = 42
shap.plots.waterfall(sv[i], max_display=12)

# Vérification de l'additivité : contrôle de cohérence indispensable
base = explainer.expected_value
somme = base + sv.values[i].sum()
brut = modele.predict(X_te[i:i+1], raw_score=True)[0]
print(f"base {base:.4f} + contributions {sv.values[i].sum():.4f} = {somme:.4f}  vs  {brut:.4f}")

# --- détection de fuite : une feature qui domine tout est suspecte ---
imp = np.abs(sv.values).mean(0)
part = imp / imp.sum()
for nom, p in sorted(zip(X_te.columns, part), key=lambda t: -t[1])[:5]:
    print(f"{nom:28s} {p:.1%}")
# Une seule feature au-delà de 60 % : chercher une fuite avant de continuer.

# --- équité : les explications diffèrent-elles selon le groupe ? ---
import pandas as pd
groupes = X_te['region']
comp = pd.DataFrame(sv.values, columns=X_te.columns).assign(g=groupes.values) \
         .groupby('g').mean()
print(comp[['revenu', 'anciennete']].round(4))

# --- interactions : quelles paires de features agissent conjointement ---
inter = explainer.shap_interaction_values(X_te[:500])     # (n, d, d), coûteux
force = np.abs(inter).mean(0)
np.fill_diagonal(force, 0)
i, j = np.unravel_index(force.argmax(), force.shape)
print("interaction dominante :", X_te.columns[i], "x", X_te.columns[j])

# --- alternatives globales, complémentaires de SHAP ---
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
PartialDependenceDisplay.from_estimator(modele, X_te, ['anciennete', 'revenu'],
                                        kind='both')      # PDP + ICE
```

**Outils** — `pip install shap`, `sklearn.inspection` pour les dépendances partielles.

**Alternatives open-source**
- *Bibliothèques* : **interpret** (Microsoft) propose les Explainable Boosting Machines — presque aussi précises qu'un boosting et **intrinsèquement** lisibles, souvent la meilleure réponse quand l'explicabilité est une exigence ; **LIME** explique localement par un modèle linéaire de substitution, plus intuitif mais sans garanties ; **Captum** (Meta) pour l'attribution dans les réseaux PyTorch ; **grad-cam** et **pytorch-grad-cam** pour les cartes de saillance en vision ; **alibi** (Seldon) pour les explications contrefactuelles (« que faudrait-il changer pour être accepté ? »), souvent plus actionnables ; **dalex** pour une approche unifiée à la R ; **fairlearn** et **AIF360** pour l'audit d'équité ; **PiML** pour des modèles interprétables par conception.
- *Outils graphiques* : **SHAP** produit lui-même les meilleures visualisations du domaine ; **InterpretML dashboard** offre une exploration interactive globale et locale ; **Responsible AI Dashboard** (Microsoft) combine erreurs, équité et interprétabilité ; **Evidently** intègre des sections d'importance ; **Streamlit** pour construire un explicateur destiné au métier en quelques heures.

**Astuces**
- SHAP explique le modèle, jamais la causalité. Le dire explicitement dans tout rapport, sinon la conclusion sera mal utilisée.
- Le graphique `beeswarm` est le plus informatif : il montre l'importance *et* le sens de l'effet *et* la dispersion, en une figure.
- Vérifier l'additivité systématiquement. Un écart signale une mauvaise configuration de l'explainer (mauvais lien, mauvaise sortie).
- Une feature concentrant plus de 60 % de l'importance est presque toujours une fuite de données. C'est l'usage de débogage le plus rentable de SHAP.
- Sur des features corrélées, l'attribution se répartit arbitrairement. Grouper les features corrélées avant d'interpréter, sinon les conclusions sont instables.
- `KernelExplainer` est très lent : le limiter à quelques centaines de lignes. Sur un modèle d'arbres, toujours utiliser `TreeExplainer`.
- Pour un besoin réglementaire fort, envisager un modèle **intrinsèquement** interprétable (EBM, régression logistique, arbre peu profond) plutôt qu'un boosting expliqué après coup. La perte de performance est souvent inférieure à ce qu'on croit.
- Les explications contrefactuelles (« +6 mois d'ancienneté aurait suffi ») sont plus utiles à un utilisateur final qu'une liste de contributions.

## Inférence causale
<!-- slug: inference-causale | difficulte: 5 | prereqs: statistiques-inferentielles, interpretabilite-shap -->

**En une phrase** — Estimer l'effet d'une action plutôt que prédire une corrélation, ce qui exige des hypothèses explicites que la prédiction n'exige pas.

**Explication** — Le ML supervisé répond à « quelle est la valeur probable de $y$ étant donné $x$ ? ». La décision d'entreprise pose une autre question : « que se passerait-il si j'intervenais ? ». Les deux diffèrent radicalement. Un modèle peut prédire parfaitement le churn à partir du fait qu'un client a appelé le support, sans qu'appeler le support ne cause quoi que ce soit.

Le cadre standard est celui des **résultats potentiels**. Pour chaque individu $i$ existent deux résultats : $Y_i(1)$ s'il est traité, $Y_i(0)$ sinon. L'effet individuel est leur différence — et il est **fondamentalement inobservable**, puisqu'on ne voit qu'un des deux. C'est le problème du contrefactuel. On peut néanmoins estimer des **moyennes** (l'effet moyen du traitement, ATE) sous certaines hypothèses.

L'expérience randomisée (test A/B) résout tout : l'assignation aléatoire garantit que les groupes sont comparables, donc la différence de moyennes estime l'effet causal sans hypothèse supplémentaire. C'est pourquoi il faut toujours préférer une expérience quand elle est possible. Sur données **observationnelles**, il faut supposer l'absence de confusion non observée (*ignorabilité*) : tous les facteurs influençant à la fois le traitement et le résultat sont mesurés. Hypothèse forte, invérifiable, et qu'il faut donc énoncer et tester par analyse de sensibilité.

Le piège technique le plus fréquent est le **contrôle d'un collisionneur** : ajouter comme covariable une variable causée par le traitement *et* par le résultat introduit un biais au lieu de le corriger. Contrairement à l'intuition du praticien du ML, en inférence causale **ajouter des variables peut dégrader l'estimation**. Un graphe causal (DAG) dessiné avant l'analyse est ce qui permet de choisir correctement.

**Cas d'utilisation**
- Mesurer l'effet d'une campagne, d'une remise, d'un changement d'interface, d'un traitement.
- *Uplift modeling* : cibler les clients dont le comportement sera **modifié** par l'action, pas ceux qui achèteraient de toute façon.
- Estimer un effet quand l'expérience est impossible (éthique, coût, rétrospectif).
- Évaluer une politique passée à partir de journaux d'observation.
- Mauvais choix si l'objectif est uniquement de prédire : le ML supervisé est plus simple et plus performant pour cela.

**Algorithme**
```text
1. Formuler la question causale : traitement T, résultat Y, population, horizon.
2. DESSINER le graphe causal (DAG) avant toute donnée. Y placer les
   confondants supposés, les médiateurs, les collisionneurs.
3. Identifier l'ensemble d'ajustement par le critère de la porte dérobée :
     contrôler les confondants (causes communes de T et Y)
     NE PAS contrôler les médiateurs (sur le chemin causal) ni les collisionneurs
4. Estimer :
     expérience randomisée   -> différence de moyennes, régression avec covariables
     observationnel          -> appariement, pondération par le score de propension,
                                double ML, forêts causales
5. Vérifier le chevauchement : les distributions de covariables des deux
   groupes doivent se recouvrir. Sans chevauchement, aucune extrapolation valide.
6. Analyse de sensibilité : quelle intensité de confusion non observée
   suffirait à annuler l'effet estimé ?
7. Rapporter un intervalle de confiance, jamais un effet ponctuel.
```

**Implémentation**
```python
# pip install dowhy econml
import numpy as np, pandas as pd

# --- 1. Déclarer le modèle causal explicitement (DoWhy) ---
from dowhy import CausalModel

modele = CausalModel(
    data=df,
    treatment='a_recu_remise',
    outcome='montant_achat',
    # le graphe est le cœur de l'analyse : il rend les hypothèses discutables
    graph="""digraph {
        anciennete -> a_recu_remise;  anciennete -> montant_achat;
        segment    -> a_recu_remise;  segment    -> montant_achat;
        a_recu_remise -> montant_achat;
    }""",
)
estimande = modele.identify_effect(proceed_when_unidentifiable=False)
print(estimande)                       # montre l'ensemble d'ajustement retenu

estime = modele.estimate_effect(estimande,
    method_name='backdoor.propensity_score_weighting')
print("ATE :", round(estime.value, 4))

# Réfutations : la brique la plus utile de DoWhy
for methode in ['random_common_cause', 'placebo_treatment_refuter',
                'data_subset_refuter']:
    r = modele.refute_estimate(estimande, estime, method_name=methode)
    print(methode, '->', r)

# --- 2. Double machine learning : ML pour les nuisances, estimation propre ---
from econml.dml import LinearDML
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

X = df[['anciennete', 'segment_code', 'frequence']].values     # confondants
T = df['a_recu_remise'].values
Y = df['montant_achat'].values

dml = LinearDML(model_y=GradientBoostingRegressor(),
                model_t=GradientBoostingClassifier(),
                discrete_treatment=True, cv=5, random_state=0).fit(Y, T, X=X)
print("ATE :", dml.ate(X).round(4), "IC 95 % :", np.round(dml.ate_interval(X), 4))

# --- 3. Effets hétérogènes : à qui l'action profite-t-elle ? ---
from econml.dr import ForestDRLearner
cate = ForestDRLearner(model_regression=GradientBoostingRegressor(),
                       model_propensity=GradientBoostingClassifier(),
                       cv=5, random_state=0).fit(Y, T, X=X)
effets = cate.effect(X)
df['effet_estime'] = effets
# Cibler le décile d'effet le plus élevé plutôt que le décile de propension à acheter
print(df.groupby(pd.qcut(df['effet_estime'], 10, labels=False))['effet_estime'].mean().round(3))

# --- 4. Vérifier le chevauchement AVANT toute estimation ---
from sklearn.linear_model import LogisticRegression
ps = LogisticRegression(max_iter=1000).fit(X, T).predict_proba(X)[:, 1]
print("scores de propension, traités :", np.percentile(ps[T == 1], [1, 50, 99]).round(3))
print("scores de propension, témoins :", np.percentile(ps[T == 0], [1, 50, 99]).round(3))
# Sans recouvrement, l'estimation extrapole et n'est pas fiable.

# --- 5. Test A/B : quand c'est possible, rien ne le remplace ---
def taille_echantillon(ecart_min, sigma, puissance=0.8, alpha=0.05):
    from scipy.stats import norm
    z_a, z_b = norm.ppf(1 - alpha / 2), norm.ppf(puissance)
    return int(np.ceil(2 * ((z_a + z_b) * sigma / ecart_min) ** 2))

print("par groupe :", taille_echantillon(ecart_min=2.0, sigma=15.0))
```

**Outils** — `pip install dowhy econml`, `statsmodels` pour les modèles de régression classiques.

**Alternatives open-source**
- *Bibliothèques* : **DoWhy** (Microsoft) impose d'expliciter le graphe causal et fournit des tests de réfutation — c'est sa vraie valeur pédagogique ; **EconML** (Microsoft) implémente le double ML, les forêts causales et l'estimation d'effets hétérogènes ; **CausalML** (Uber) est orienté *uplift modeling* marketing ; **causal-learn** pour la découverte de structure causale à partir des données ; **grf** (R) est la référence pour les forêts causales ; **pgmpy** pour les réseaux bayésiens ; **linearmodels** pour les variables instrumentales et les données de panel ; **pymc** pour une approche bayésienne des effets ; **CausalPy** pour les inférences quasi-expérimentales (différences de différences, contrôle synthétique).
- *Outils graphiques* : **DAGitty** (application web) construit un graphe causal et calcule automatiquement les ensembles d'ajustement valides — outil indispensable, et gratuit ; **dowhy** produit des visualisations de graphe ; **GeNIe** (version académique gratuite) pour les réseaux bayésiens ; **CausalNex** (QuantumBlack) pour la modélisation causale structurelle.

**Astuces**
- Une expérience randomisée bien menée vaut mieux que la méthode observationnelle la plus sophistiquée. Toujours demander si un test A/B est possible avant d'investir ailleurs.
- Dessiner le DAG avant de toucher aux données. C'est ce qui rend les hypothèses explicites, discutables, et corrigibles.
- Contrairement au réflexe du ML, **ajouter des covariables peut biaiser** l'estimation. Médiateurs et collisionneurs doivent être exclus, pas inclus.
- Vérifier le chevauchement des scores de propension. Sans recouvrement entre traités et témoins, l'estimation extrapole silencieusement.
- Toujours rapporter un intervalle de confiance. Un effet causal ponctuel sans incertitude n'est pas une estimation, c'est une affirmation.
- En marketing, cibler le décile d'**effet** estimé et non le décile de propension à acheter. Cibler ceux qui auraient acheté de toute façon gaspille le budget — c'est tout l'enjeu de l'uplift.
- Les tests de réfutation de DoWhy (cause commune aléatoire, traitement placebo, sous-échantillon) sont une discipline saine : ils rendent visible la fragilité d'une estimation.
- La corrélation entre `SHAP` et causalité est nulle par construction. Ne jamais présenter une importance de feature comme un effet causal.

## Apprentissage par renforcement
<!-- slug: apprentissage-renforcement | difficulte: 5 | prereqs: perceptron-mlp, probabilites-bayes -->

**En une phrase** — Apprendre une politique d'action par essais et erreurs dans un environnement qui renvoie des récompenses, sans exemples d'actions correctes.

**Explication** — Le cadre formel est le **processus de décision markovien** : un agent observe un état $s$, choisit une action $a$, reçoit une récompense $r$ et transite vers $s'$. L'objectif est de maximiser la somme des récompenses futures actualisées, $\sum_t \gamma^t r_t$, où $\gamma \in [0,1)$ règle l'horizon. Deux fonctions structurent tout : $V(s)$, la valeur attendue depuis l'état $s$, et $Q(s,a)$, la valeur d'y jouer l'action $a$. L'**équation de Bellman** les relie récursivement : $Q(s,a) = r + \gamma \max_{a'} Q(s',a')$, et presque tous les algorithmes en sont une variante.

Le **Q-learning** apprend $Q$ par différence temporelle : on met à jour l'estimation vers la cible de Bellman observée. Avec un réseau de neurones pour approcher $Q$, on obtient le **DQN**, qui nécessite deux astuces indispensables : un **tampon de rejeu** (échantillonner des transitions passées pour casser la corrélation temporelle) et un **réseau cible** figé périodiquement (sinon la cible bouge en même temps que l'estimation, et l'apprentissage diverge). Les méthodes de **gradient de politique** (REINFORCE, A2C, **PPO**) optimisent directement la politique, ce qui gère les actions continues ; PPO est aujourd'hui le choix par défaut pour sa stabilité.

Le dilemme central est l'**exploration contre l'exploitation** : exploiter ce qu'on sait rapporte à court terme, explorer permet de découvrir mieux. La stratégie $\varepsilon$-glouton, ou l'entropie ajoutée à l'objectif, gèrent cet arbitrage.

Il faut être franc sur les limites : le RL est **gourmand en interactions** (des millions d'épisodes), **instable** (deux graines aléatoires donnent des résultats très différents), et exige un simulateur. En dehors des jeux, de la robotique, de l'optimisation de systèmes et de l'alignement de modèles de langage (RLHF, DPO), il est rarement le bon outil. Un bandit contextuel — cas particulier sans état persistant — suffit à la plupart des problèmes industriels de décision séquentielle.

**Cas d'utilisation**
- Jeux et simulations, domaine de naissance et d'excellence du RL.
- Robotique et contrôle : trajectoires, équilibre, préhension.
- Optimisation de systèmes : refroidissement de datacenter, gestion d'énergie, allocation de ressources.
- Alignement de modèles de langage : RLHF et ses successeurs (DPO, GRPO).
- **Bandits contextuels** pour la recommandation, les tests A/B adaptatifs, la tarification — c'est l'application industrielle réaliste la plus fréquente.
- Mauvais choix quand un modèle supervisé ou une optimisation classique répond au problème, ou quand chaque essai coûte cher dans le monde réel.

**Algorithme**
```text
Q-learning tabulaire (le socle à comprendre) :
1. Initialiser Q(s, a) arbitrairement (zéros).
2. Pour chaque épisode :
     s = état initial
     tant que non terminé :
       a = argmax_a Q(s,a) avec probabilité 1-eps, sinon action aléatoire
       observer r, s'
       Q(s,a) += alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))    # erreur TD
       s = s'
     décroître eps
3. Politique finale : pi(s) = argmax_a Q(s,a)

DQN ajoute :
  - réseau Q(s,·) au lieu d'une table
  - tampon de rejeu : échantillonner des transitions passées au hasard
  - réseau cible figé, synchronisé toutes les N étapes
  - écrêtage du gradient

PPO (gradient de politique, défaut moderne) :
  - collecter des trajectoires avec la politique courante
  - estimer l'avantage A(s,a) (GAE)
  - maximiser un objectif de ratio ÉCRÊTÉ, plusieurs époques sur les mêmes données
  - le clip empêche une mise à jour destructrice de la politique
```

**Implémentation**
```python
# pip install gymnasium stable-baselines3
import numpy as np, gymnasium as gym

# --- 1. Q-learning tabulaire from scratch (palier 2) ---
def q_learning(env, episodes=5000, alpha=0.1, gamma=0.99, eps0=1.0, eps_min=0.05):
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    rng = np.random.default_rng(0)
    for ep in range(episodes):
        eps = max(eps_min, eps0 * (1 - ep / episodes))
        s, _ = env.reset()
        fini = False
        while not fini:
            a = rng.integers(env.action_space.n) if rng.random() < eps else int(Q[s].argmax())
            s2, r, termine, tronque, _ = env.step(a)
            fini = termine or tronque
            cible = r + gamma * Q[s2].max() * (not termine)
            Q[s, a] += alpha * (cible - Q[s, a])       # erreur de différence temporelle
            s = s2
    return Q

env = gym.make('FrozenLake-v1', is_slippery=False)
Q = q_learning(env)
print("politique apprise :", Q.argmax(1).reshape(4, 4))

# --- 2. PPO en pratique : ne pas réimplémenter, utiliser une référence ---
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

venv = make_vec_env('CartPole-v1', n_envs=8)          # paralléliser accélère beaucoup
agent = PPO('MlpPolicy', venv, learning_rate=3e-4, n_steps=256, batch_size=256,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
            verbose=0, seed=0)
agent.learn(total_timesteps=200_000)

moy, ecart = evaluate_policy(agent, gym.make('CartPole-v1'), n_eval_episodes=30)
print(f"récompense : {moy:.1f} ± {ecart:.1f}")
# Toujours évaluer sur PLUSIEURS graines : la variance du RL est considérable.

# --- 3. Bandit contextuel : l'application industrielle réaliste ---
class BanditLinUCB:
    """Choisir parmi K actions selon un contexte, en équilibrant exploration
    et exploitation. Recommandation, tarification, tests adaptatifs."""

    def __init__(self, n_actions, d_contexte, alpha=1.0):
        self.A = [np.eye(d_contexte) for _ in range(n_actions)]
        self.b = [np.zeros(d_contexte) for _ in range(n_actions)]
        self.alpha = alpha

    def choisir(self, x):
        scores = []
        for A, b in zip(self.A, self.b):
            Ainv = np.linalg.inv(A)
            theta = Ainv @ b
            # espérance + borne de confiance : l'incertitude pousse à explorer
            scores.append(theta @ x + self.alpha * np.sqrt(x @ Ainv @ x))
        return int(np.argmax(scores))

    def mettre_a_jour(self, a, x, recompense):
        self.A[a] += np.outer(x, x)
        self.b[a] += recompense * x

bandit = BanditLinUCB(n_actions=5, d_contexte=10)
for x, recompense_fn in flux_utilisateurs:
    a = bandit.choisir(x)
    bandit.mettre_a_jour(a, x, recompense_fn(a))
```

**Outils** — `pip install gymnasium stable-baselines3[extra]`.

**Alternatives open-source**
- *Bibliothèques* : **Stable-Baselines3** est la référence pour des implémentations fiables et documentées (PPO, SAC, DQN, TD3) ; **CleanRL** propose une implémentation par fichier unique, lisible de bout en bout — le meilleur support pour comprendre ; **Ray RLlib** pour l'entraînement distribué à grande échelle ; **Gymnasium** (successeur de Gym) pour les environnements standards ; **PettingZoo** pour le multi-agent ; **Stable-Retro** et **MiniGrid** pour l'apprentissage ; **trl** (Hugging Face) pour le RLHF, DPO et GRPO sur modèles de langage ; **Vowpal Wabbit** et **contextualbandits** pour les bandits en production ; **MuJoCo** (libre depuis 2021) pour la simulation physique.
- *Outils graphiques* : **TensorBoard** intégré à Stable-Baselines3 pour les courbes de récompense ; **Weights & Biases** pour comparer plusieurs graines, indispensable ici ; **RL Baselines3 Zoo** fournit des hyperparamètres réglés et des scripts d'évaluation ; les enregistrements vidéo de `gymnasium` (`RecordVideo`) pour visualiser la politique apprise — souvent la façon la plus rapide de comprendre ce qui ne marche pas.

**Astuces**
- Ne pas réimplémenter PPO pour un usage sérieux. Les détails d'implémentation (normalisation des avantages, écrêtage, initialisation) font l'essentiel de la performance et sont faciles à rater.
- Évaluer sur au moins cinq graines aléatoires. La variance du RL est telle qu'un résultat sur une seule graine n'a aucune valeur informative.
- Normaliser les observations et les récompenses (`VecNormalize`). C'est souvent la différence entre un agent qui apprend et un agent qui stagne.
- Commencer par des environnements jouets (CartPole, FrozenLake) pour valider le code, avant tout problème réel. Un bug de signe de récompense est invisible sur un problème complexe.
- Le façonnage de récompense (*reward shaping*) est le levier dominant et le piège principal : une récompense mal conçue produit un agent qui l'optimise littéralement, en contournant l'objectif visé.
- Un bandit contextuel suffit à la majorité des problèmes industriels de décision. Le RL complet n'est nécessaire que si les actions ont des conséquences à long terme sur l'état.
- Le facteur d'actualisation $\gamma$ définit l'horizon effectif ($\approx 1/(1-\gamma)$ pas). $\gamma = 0{,}99$ signifie un horizon d'environ 100 pas — à confronter à la durée réelle des épisodes.

## Modèles génératifs
<!-- slug: modeles-generatifs | difficulte: 5 | prereqs: melange-gaussien, attention-transformer -->

**En une phrase** — Apprendre la distribution des données plutôt qu'une frontière de décision, pour pouvoir en produire de nouveaux échantillons.

**Explication** — Un modèle discriminatif apprend $P(y|x)$ ; un modèle génératif apprend $P(x)$, ou $P(x|y)$. C'est bien plus difficile : il faut capturer toute la structure des données, non seulement ce qui sépare des classes. Quatre familles ont dominé successivement.

Les **auto-encodeurs variationnels** (VAE) encodent une donnée en une distribution latente gaussienne, en échantillonnent (via l'astuce de reparamétrisation, qui rend l'échantillonnage différentiable) et décodent. L'objectif combine une erreur de reconstruction et une divergence KL qui contraint l'espace latent à rester régulier — ce qui permet d'interpoler entre deux points. Leur défaut est un rendu flou.

Les **GAN** opposent un générateur et un discriminateur dans un jeu à somme nulle. Ils produisent des images très nettes mais leur entraînement est notoirement instable (effondrement de mode, non-convergence). Ils ont dominé de 2015 à 2021.

Les **modèles de diffusion** ont pris la relève. L'idée est élégante : on ajoute progressivement du bruit gaussien à une image jusqu'à obtenir du bruit pur (processus direct, fixé et connu), puis on entraîne un réseau à **prédire le bruit ajouté** à chaque étape. Générer consiste à partir du bruit et à débruiter itérativement. L'entraînement est stable — c'est une simple régression — et la qualité surpasse celle des GAN. C'est la base de Stable Diffusion.

Les **modèles autorégressifs** enfin factorisent $P(x) = \prod_t P(x_t | x_{<t})$ et prédisent un élément à la fois. C'est le principe des modèles de langage, et il s'applique aussi à l'image et à l'audio.

**Cas d'utilisation**
- Augmentation de données quand les exemples réels sont rares ou coûteux.
- Génération de données synthétiques préservant la confidentialité (santé, finance).
- Détection d'anomalies : ce que le modèle reconstruit mal est anormal (VAE, diffusion).
- Débruitage, super-résolution, complétion d'images manquantes.
- Apprentissage de représentations non supervisé, réutilisables en aval.
- Mauvais choix pour de la simple classification : un modèle discriminatif est plus simple et plus performant.

**Algorithme**
```text
VAE :
1. Encodeur : x -> mu(x), log sigma²(x)
2. Échantillonner z = mu + sigma * eps, eps ~ N(0, I)   (reparamétrisation)
3. Décodeur : z -> x_reconstruit
4. Perte = reconstruction + beta * KL(N(mu, sigma²) || N(0, I))
   beta règle l'arbitrage netteté / régularité du latent.

Diffusion (DDPM) :
Entraînement :
1. Tirer une image x0, un pas t uniforme, un bruit eps ~ N(0, I).
2. Construire x_t = sqrt(alpha_barre_t) x0 + sqrt(1 - alpha_barre_t) eps
   (fermeture analytique : pas besoin de simuler pas à pas)
3. Perte = || eps - reseau(x_t, t) ||²      -- une simple régression
Échantillonnage :
4. Partir de x_T ~ N(0, I).
5. Pour t = T..1 : prédire le bruit, retirer une partie, ajouter un peu
   de bruit résiduel.
6. Guidage sans classifieur : interpoler entre prédiction conditionnelle
   et inconditionnelle pour renforcer le respect de la consigne.
```

**Implémentation**
```python
import torch, torch.nn as nn, torch.nn.functional as F

# --- VAE from scratch : le générateur le plus simple à comprendre (palier 2) ---
class VAE(nn.Module):
    def __init__(self, d_entree=784, d_cache=400, d_latent=20):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_entree, d_cache), nn.ReLU())
        self.mu = nn.Linear(d_cache, d_latent)
        self.logvar = nn.Linear(d_cache, d_latent)
        self.dec = nn.Sequential(nn.Linear(d_latent, d_cache), nn.ReLU(),
                                 nn.Linear(d_cache, d_entree))

    def encoder(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    def echantillonner(self, mu, logvar):
        # reparamétrisation : rend l'échantillonnage différentiable
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.echantillonner(mu, logvar)
        return self.dec(z), mu, logvar

def perte_vae(x_logits, x, mu, logvar, beta=1.0):
    rec = F.binary_cross_entropy_with_logits(x_logits, x, reduction='sum')
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (rec + beta * kl) / x.size(0)

# Génération : échantillonner dans le latent et décoder
modele = VAE().eval()
with torch.no_grad():
    images = torch.sigmoid(modele.dec(torch.randn(16, 20))).view(-1, 28, 28)

# --- Diffusion : l'entraînement tient en quelques lignes ---
T = 1000
betas = torch.linspace(1e-4, 0.02, T)
alphas_barre = torch.cumprod(1 - betas, dim=0)

def etape_entrainement(reseau, x0, opt):
    B = x0.size(0)
    t = torch.randint(0, T, (B,), device=x0.device)
    eps = torch.randn_like(x0)
    ab = alphas_barre.to(x0.device)[t].view(-1, 1, 1, 1)
    x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * eps        # fermeture analytique
    perte = F.mse_loss(reseau(x_t, t), eps)             # prédire le BRUIT
    opt.zero_grad(); perte.backward(); opt.step()
    return perte.item()

@torch.no_grad()
def echantillonner(reseau, forme, device):
    x = torch.randn(forme, device=device)
    for t in reversed(range(T)):
        tt = torch.full((forme[0],), t, device=device, dtype=torch.long)
        eps_pred = reseau(x, tt)
        a, ab = 1 - betas[t], alphas_barre[t]
        x = (x - (betas[t] / (1 - ab).sqrt()) * eps_pred) / a.sqrt()
        if t > 0:
            x = x + betas[t].sqrt() * torch.randn_like(x)
    return x

# --- En pratique : utiliser diffusers, jamais réimplémenter ---
# pip install diffusers transformers accelerate
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained('stabilityai/stable-diffusion-2-1-base',
                                              torch_dtype=torch.float16).to('cuda')
image = pipe("un carnet ouvert sur une table en bois, lumière du matin",
             guidance_scale=7.5, num_inference_steps=30).images[0]

# --- Données tabulaires synthétiques : usage le plus concret en entreprise ---
# pip install sdv
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import Metadata
meta = Metadata.detect_from_dataframe(df)
synth = CTGANSynthesizer(meta, epochs=300)
synth.fit(df)
df_synthetique = synth.sample(num_rows=10_000)

from sdv.evaluation.single_table import evaluate_quality
print(evaluate_quality(df, df_synthetique, meta))     # fidélité des distributions
```

**Outils** — `pip install torch diffusers transformers`, `pip install sdv` pour le tabulaire.

**Alternatives open-source**
- *Bibliothèques* : **diffusers** (Hugging Face) est la référence pour les modèles de diffusion, entraînement et inférence ; **SDV** (Synthetic Data Vault) pour des données tabulaires synthétiques avec métriques de qualité et de confidentialité — l'usage le plus courant en entreprise ; **ydata-synthetic** comme alternative ; **torchsde** et **k-diffusion** pour les échantillonneurs avancés ; **ComfyUI** et **InvokeAI** pour des flux de génération d'images ; **audiocraft** (Meta) pour l'audio ; **nanoGPT** pour comprendre le génératif autorégressif ; **pythae** rassemble une vingtaine de variantes d'auto-encodeurs ; **Opacus** pour de la génération sous confidentialité différentielle.
- *Outils graphiques* : **ComfyUI** expose le pipeline de diffusion sous forme de graphe manipulable — excellent pour comprendre chaque étape ; **AUTOMATIC1111 WebUI** pour l'expérimentation d'images ; **SDV** produit des rapports de qualité HTML ; **FiftyOne** pour inspecter des jeux d'images générées ; **TensorBoard** pour suivre les pertes de reconstruction.

**Astuces**
- Ne pas réimplémenter Stable Diffusion. Écrire un DDPM sur MNIST pour comprendre, puis utiliser `diffusers` pour tout usage réel.
- L'entraînement d'un GAN est instable par nature. Sur un projet neuf, choisir la diffusion : l'objectif est une simple régression, donc l'entraînement converge.
- Le $\beta$ d'un VAE arbitre entre netteté et régularité du latent. $\beta$ trop grand provoque l'effondrement postérieur : le latent devient du bruit pur et le décodeur ignore $z$.
- Évaluer un modèle génératif est un problème en soi. La perte ne dit presque rien : utiliser FID pour les images, et des métriques de fidélité de distribution pour le tabulaire.
- Les données synthétiques ne protègent pas automatiquement la confidentialité : un modèle peut mémoriser et régurgiter des enregistrements réels. Vérifier avec les métriques de confidentialité de SDV, ou entraîner sous confidentialité différentielle.
- Un modèle génératif entraîné pour l'augmentation de données n'améliore les performances que si le modèle génératif est meilleur que le classifieur en aval. Souvent, une augmentation classique bien réglée suffit et coûte cent fois moins.
- L'astuce de reparamétrisation est le point technique à comprendre absolument dans un VAE : sans elle, aucun gradient ne traverse l'échantillonnage.
- Le guidage sans classifieur (*classifier-free guidance*) est le levier de qualité principal en diffusion conditionnelle. Un facteur entre 5 et 9 est la plage utile ; au-delà, les images se saturent.

## Lire un papier de recherche
<!-- slug: lecture-papiers | difficulte: 3 | prereqs: statistiques-inferentielles, biais-variance -->

**En une phrase** — Extraire l'idée et évaluer la solidité d'un article scientifique en trois passes de lecture, sans le lire linéairement.

**Explication** — Un article n'est pas fait pour être lu du début à la fin. La méthode des **trois passes** (Keshav) est la référence. **Première passe**, cinq à dix minutes : titre, résumé, introduction, titres de sections, conclusion, et un coup d'œil aux figures. Objectif : décider si l'article mérite plus de temps, et pouvoir dire en une phrase ce qu'il propose. **Deuxième passe**, une heure : lire le corps en ignorant les preuves et les détails d'implémentation, regarder attentivement figures et tableaux, noter les références à suivre. On doit alors pouvoir résumer la contribution et la méthode expérimentale à quelqu'un d'autre. **Troisième passe**, plusieurs heures, réservée aux articles qui comptent vraiment : reconstruire mentalement le travail, vérifier les hypothèses, identifier ce qui manque.

L'esprit critique se concentre sur quelques questions récurrentes. À quoi la méthode est-elle **comparée** ? Une baseline faible ou mal réglée invalide tout gain annoncé — c'est le défaut le plus fréquent de la littérature. Le gain est-il **significatif** ? Combien de graines aléatoires, quel écart type ? Un demi-point sur une seule exécution ne veut rien dire. Y a-t-il une **étude d'ablation** montrant quelle partie de la méthode produit réellement l'effet ? Et le coût : un gain de 1 % pour dix fois plus de calcul n'est pas une avancée, c'est un arbitrage à énoncer.

Enfin, savoir **où chercher** compte autant que savoir lire. arXiv publie sans relecture par les pairs : la qualité y est extrêmement variable. Les conférences de référence (NeurIPS, ICML, ICLR, CVPR, ACL) filtrent, mais avec du bruit. Un article très cité et reproduit indépendamment vaut mieux qu'une prépublication spectaculaire de la semaine.

**Cas d'utilisation**
- Décider si une méthode récente vaut d'être essayée dans un projet.
- Reproduire un résultat pour valider sa propre compréhension — c'est l'exercice le plus formateur de tout le domaine.
- Se tenir à jour sans se noyer : le volume de publication rend la sélection indispensable.
- Justifier un choix technique par un état de l'art documenté.

**Algorithme**
```text
Passe 1 (5-10 min) — trier :
  titre, résumé, introduction, conclusion, figures, légendes
  -> Quelle catégorie de problème ? Quelle idée principale ? Est-ce pour moi ?

Passe 2 (~1 h) — comprendre :
  corps du texte sans les preuves ; toutes les figures et tableaux
  -> Quelle est la contribution exacte ? À quoi est-ce comparé ?
     Sur quels jeux de données ? Avec quel protocole ?
  Noter les références à lire ensuite.

Passe 3 (plusieurs heures) — évaluer et reproduire :
  refaire les dérivations, examiner les hypothèses implicites,
  chercher le code, tenter une reproduction partielle.

Grille de lecture critique, à appliquer systématiquement :
  [ ] baselines fortes et correctement réglées ?
  [ ] même budget de calcul et de réglage pour toutes les méthodes comparées ?
  [ ] nombre de graines aléatoires, écart type rapporté ?
  [ ] étude d'ablation isolant chaque composant ?
  [ ] jeux de données standards ou choisis à l'avantage de la méthode ?
  [ ] code et poids publiés ?
  [ ] coût en calcul, mémoire, latence indiqué ?
  [ ] limites explicitement discutées ?
```

**Implémentation**
```text
Gabarit de fiche de lecture, à remplir pour chaque article retenu
(à coller dans la note du concept correspondant) :

# Titre — Auteurs, conférence, année
Lien : arxiv.org/abs/XXXX    Code : github.com/...    Cité : N fois

## En une phrase
(la contribution, avec tes mots — si tu n'y arrives pas, tu n'as pas compris)

## Problème
Que ne savait-on pas faire avant ? Pourquoi c'est important ?

## Idée clé
Le mécanisme, en 3 à 5 lignes. Un schéma vaut mieux qu'un paragraphe.

## Méthode
Formulation, perte, architecture. Ce qui diffère de l'existant.

## Expériences
Jeux de données | baselines | métriques | gain annoncé | graines | écart type

## Verdict critique
Solide :
Fragile :
Non testé :
Coût réel (calcul, mémoire, latence) :

## Applicable à mon travail ?
Oui / Non / Plus tard, parce que...

## À lire ensuite
[références citées qui semblent fondatrices]
```

```python
# Automatiser la veille : récupérer les métadonnées arXiv d'un article
# pip install arxiv
import arxiv

def fiche_arxiv(id_arxiv):
    art = next(arxiv.Client().results(arxiv.Search(id_list=[id_arxiv])))
    return {
        'titre': art.title,
        'auteurs': [a.name for a in art.authors][:5],
        'annee': art.published.year,
        'resume': art.summary.replace('\n', ' '),
        'categories': art.categories,
        'lien': art.entry_id,
    }

print(fiche_arxiv('1706.03762')['titre'])      # Attention Is All You Need

# Suivre les nouveautés d'un domaine, filtrées
recherche = arxiv.Search(query='cat:stat.ML AND abs:"calibration"',
                         max_results=20,
                         sort_by=arxiv.SortCriterion.SubmittedDate)
for r in arxiv.Client().results(recherche):
    print(r.published.date(), r.title)
```

**Outils** — Un gestionnaire de références (Zotero), un carnet de fiches de lecture, `pip install arxiv`.

**Alternatives open-source**
- *Bibliothèques et services* : **Zotero** est le gestionnaire de références libre de référence, avec extension navigateur et synchronisation ; **Papers with Code** relie articles, code et classements — le meilleur filtre pratique contre les résultats non reproductibles ; **Semantic Scholar** et son API pour explorer les citations et l'influence ; **Connected Papers** et **Litmaps** pour visualiser un graphe de littérature autour d'un article ; **arxiv-sanity-lite** (Karpathy) pour une veille filtrée par similarité ; **alphaxiv** et **arXiv Labs** pour les discussions annotées ; **OpenReview** donne accès aux relectures de ICLR et NeurIPS, souvent plus instructives que l'article lui-même.
- *Outils graphiques* : **Connected Papers** produit une carte de littérature en une requête, idéale pour situer un article ; **Zotero** avec l'extension Better BibTeX pour l'écriture ; **Obsidian** ou **Logseq** pour un carnet de fiches liées ; **Hypothesis** pour annoter des PDF collaborativement ; **PDF.js** et **Sioyek** comme lecteurs annotables.

**Astuces**
- Ne jamais lire linéairement. La première passe sert à **abandonner** la plupart des articles : c'est sa fonction, pas un échec.
- Lire les relectures sur OpenReview quand elles existent. Les objections des relecteurs pointent directement les faiblesses, ce que l'article ne fait jamais.
- Vérifier systématiquement la force des baselines. Une méthode qui bat une baseline mal réglée n'a rien démontré, et c'est le défaut le plus courant.
- Le nombre de graines aléatoires et l'écart type sont le meilleur indicateur de sérieux expérimental. Leur absence justifie à elle seule la méfiance.
- Regarder les figures et les tableaux avant le texte. Ils contiennent l'essentiel du contenu vérifiable, et révèlent souvent ce que le texte enjolive.
- Reproduire un résultat, même partiellement, apprend plus que dix articles lus. L'écart entre ce qui est écrit et ce qui est nécessaire pour que ça marche est très instructif.
- Préférer les articles cités et reproduits aux prépublications de la semaine. Le taux de survie des idées nouvelles est faible.
- Tenir un carnet de fiches selon le gabarit ci-dessus. Un article lu sans fiche est un article oublié en trois semaines.

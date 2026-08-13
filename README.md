# Atelier ML

Site personnel pour passer des bases à la maîtrise du machine learning.
Version simple : **React (JavaScript) + Node.js + SQLite**. Pas de labo, pas de
Python côté serveur, pas de service cloud.

---

## 1. Principe

Le contenu pédagogique vit dans des **fichiers Markdown versionnés** (`content/`).
La base SQLite ne stocke que **ce qui t'appartient** : ta progression, tes notes,
tes révisions.

```
content/*.md   -->  lu au démarrage du serveur  -->  API  -->  React
   (le savoir)                                          |
                                                        v
                                                  data.db (SQLite)
                                              progression + notes + révisions
```

Conséquences directes :

- pas de CMS à écrire, pas d'éditeur de contenu, pas de migrations de contenu ;
- les fiches sont lisibles et modifiables dans n'importe quel éditeur ;
- `git log` te donne l'historique du contenu gratuitement ;
- si l'appli meurt, le savoir reste. Seule la progression est en base.

## 2. Fonctionnalités retenues

| Fonction | Description |
|---|---|
| Curriculum | 12 modules, ~65 concepts, chargés depuis `content/` |
| Palier de maîtrise | 0 rencontré · 1 compris · 2 implémenté · 3 appliqué · 4 expliqué |
| Difficulté | théorique (dans la fiche) vs ressentie (saisie par toi) → écart = lacune |
| Fiche concept | explication, cas d'usage, algorithme, code, outils, alternatives, astuces |
| Notes | Markdown par concept, autosauvegarde, maths LaTeX |
| Révisions | flashcards + répétition espacée (SM-2 simplifié) |
| Sessions | minuteur, durée, difficulté ressentie, résumé |
| Tableau de bord | maîtrise par module, heatmap des jours actifs, file du jour |
| Export | Markdown (+ front-matter), JSON complet, un fichier par module |
| Recherche | plein texte sur fiches et notes (SQLite FTS5) |

Volontairement absent : labo d'expériences, exécution de code, registre de projets,
graphe interactif, assistance IA. Ajoutables plus tard sans rien casser.

## 3. Stack

**Front** — React 18 + Vite (JavaScript, pas de TypeScript), React Router,
`react-markdown` + `remark-math` + `rehype-katex` pour le rendu des fiches,
`recharts` pour les 2 graphiques du tableau de bord, CSS simple (variables +
quelques classes, pas de framework).

**Back** — Node 20 + Express 4, `better-sqlite3` (synchrone, un fichier, zéro
config), `gray-matter` pour le front-matter, `marked` pour l'export HTML/PDF.

**Base** — SQLite, un fichier `server/data.db`. Sauvegarde = copie du fichier.

Pourquoi ces choix : un seul langage côté application, aucune dépendance réseau,
démarrage en deux commandes, et un total de 12 dépendances de production.

## 4. Arborescence

```
big-data/
├── README.md
├── package.json                scripts racine (concurrently), dev uniquement
├── content/                    le savoir, en Markdown
│   ├── 00-outillage.md
│   ├── 01-maths.md
│   ├── 02-ml-supervise.md
│   ├── 03-methodologie.md
│   ├── 04-features-pipelines.md
│   ├── 05-non-supervise.md
│   ├── 06-deep-learning.md
│   ├── 07-vision-sequences.md
│   ├── 08-nlp-llm.md
│   ├── 09-big-data.md
│   ├── 10-mlops.md
│   └── 11-avance.md
├── docs/
│   ├── gabarit-fiche.md        le format d'une fiche concept
│   └── modele-de-donnees.md    schéma SQLite + routes API
├── server/
│   ├── index.js                Express
│   ├── db.js                   schéma + requêtes
│   ├── content.js              parseur des fichiers Markdown
│   ├── srs.js                  répétition espacée
│   ├── export.js               md / json / html + écriture du zip
│   ├── test-api.mjs           parcours complet de l'API, sans dépendance
│   └── data.db
└── web/
    ├── index.html
    ├── vite.config.js
    └── src/
        ├── main.jsx
        ├── App.jsx
        ├── api.js
        ├── styles.css
        ├── pages/
        │   ├── Dashboard.jsx
        │   ├── Module.jsx
        │   ├── Concept.jsx
        │   ├── Review.jsx
        │   └── Search.jsx
        └── components/
            ├── MasteryBadge.jsx
            ├── DifficultyBar.jsx
            ├── NoteEditor.jsx
            ├── SessionTimer.jsx
            ├── Heatmap.jsx
            └── Markdown.jsx     rendu commun : GFM + LaTeX
```

## 5. Format du contenu

Chaque fichier de `content/` est un module. Il commence par un front-matter
YAML, puis chaque concept est un bloc `##` suivi d'une ligne de métadonnées :

```md
---
module: ml-supervise
titre: ML supervisé classique
ordre: 2
---

## Régression logistique
<!-- slug: regression-logistique | difficulte: 2 | prereqs: descente-de-gradient -->

**En une phrase** — ...
```

Le parseur (`server/content.js`) fait 40 lignes : `gray-matter` pour l'en-tête,
un `split` sur `/^## /m`, une regex sur le commentaire de métadonnées. Voir
[docs/gabarit-fiche.md](docs/gabarit-fiche.md) pour le format complet des blocs.

## 6. Démarrage

```bash
# une fois
npm install && npm run setup   # racine, puis server/ et web/

# au quotidien : un seul terminal
npm run dev                    # concurrently : API 3001 + front 5173
```

Les deux processus sont préfixés `[api]` et `[web]` dans la même sortie, et
`Ctrl+C` les arrête tous les deux. Séparément si besoin : `npm run dev:api`,
`npm run dev:web`.

`web/vite.config.js` proxifie `/api` vers `localhost:3001` : aucune question de
CORS en développement.

En un seul processus : `npm run build` puis `npm start`. Le serveur sert alors
`web/dist` et tout tient sur <http://localhost:3001>.

Le contenu est lu au démarrage. Après avoir modifié un fichier de `content/`,
« Relire le contenu » en bas de la barre latérale (ou `POST /api/reload`) suffit :
pas besoin de redémarrer.

Vérifier que tout répond, sans toucher à ta base :

```bash
cd server
ATELIER_DB=/tmp/test.db PORT=3999 node index.js &
node test-api.mjs        # 45 assertions : paliers, SRS, FTS, exports
```

## 7. Ordre de construction

1. **Parseur + lecture** — `content.js`, route `GET /api/modules`, pages Module
   et Concept en lecture seule. À ce stade le site est déjà utile : c'est ton
   manuel de référence consultable.
2. **Progression** — table `progress`, paliers, difficulté ressentie, sessions
   minutées, tableau de bord.
3. **Notes + export** — éditeur Markdown par concept, autosauvegarde, export
   Markdown et JSON. À faire avant tout confort : c'est ton filet de sécurité.
4. **Révisions** — flashcards, SM-2, file du jour, recherche FTS5.

## 8. Règle de discipline

80 % du temps sur le ML, 20 % sur le site. Pas de nouvelle fonctionnalité tant
que la précédente n'a pas servi cinq fois. Le risque réel de ce projet est de
coder un joli outil au lieu d'apprendre la rétropropagation.

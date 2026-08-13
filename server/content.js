// Lecture du contenu pédagogique : content/*.md -> objets JS gardés en mémoire.
// Rien n'est écrit en base : le Markdown reste la source de vérité.
import fs from 'node:fs';
import path from 'node:path';
import matter from 'gray-matter';

const META =
  /<!--\s*slug:\s*([\w-]+)\s*\|\s*difficulte:\s*(\d)\s*\|\s*prereqs:\s*([^>]*?)\s*-->/;

const LABELS = {
  'En une phrase': 'phrase',
  Explication: 'explication',
  "Cas d'utilisation": 'casUsage',
  Algorithme: 'algorithme',
  Implémentation: 'implementation',
  Outils: 'outils',
  'Alternatives open-source': 'alternatives',
  Astuces: 'astuces',
};

// On ne découpe que sur les libellés connus : un **gras** en début de ligne
// à l'intérieur d'un bloc ne doit pas couper la fiche en deux.
const DECOUPE = new RegExp(
  `^\\*\\*(${Object.keys(LABELS).map(echapper).join('|')})\\*\\*\\s*(?:—|-)?\\s*`,
  'm'
);

export function loadContent(dir) {
  const modules = [];
  const fichiers = fs
    .readdirSync(dir)
    .filter((f) => f.endsWith('.md'))
    .sort();

  for (const fichier of fichiers) {
    const { data, content } = matter(
      fs.readFileSync(path.join(dir, fichier), 'utf8')
    );
    const concepts = content
      .split(/^## /m)
      .slice(1)
      .map((bloc, i) => {
        const [ligneTitre, ...reste] = bloc.split('\n');
        const corps = reste.join('\n');
        const m = corps.match(META);
        const titre = ligneTitre.trim();
        return {
          slug: m?.[1] ?? slugify(titre),
          titre,
          module: data.module,
          moduleTitre: data.titre,
          ordre: data.ordre ?? 0,
          rang: i,
          difficulte: Number(m?.[2] ?? 3),
          prereqs: (m?.[3] ?? '')
            .split(',')
            .map((s) => s.trim())
            .filter(Boolean),
          blocs: decouper(corps.replace(META, '')),
        };
      });

    modules.push({
      module: data.module ?? path.basename(fichier, '.md'),
      titre: data.titre ?? fichier,
      ordre: data.ordre ?? 0,
      resume: data.resume ?? '',
      fichier,
      concepts,
    });
  }

  modules.sort((a, b) => a.ordre - b.ordre);

  // Index plats, utilisés partout ailleurs.
  const concepts = modules.flatMap((m) => m.concepts);
  const parSlug = Object.fromEntries(concepts.map((c) => [c.slug, c]));

  const doublons = concepts.length - Object.keys(parSlug).length;
  if (doublons > 0) {
    console.warn(`[contenu] ${doublons} slug(s) en double — le dernier gagne.`);
  }
  for (const c of concepts) {
    for (const r of c.prereqs) {
      if (!parSlug[r]) console.warn(`[contenu] ${c.slug} : prérequis inconnu « ${r} »`);
    }
  }

  return { modules, concepts, parSlug };
}

// Découpe sur les libellés en gras : **Explication** — ...
function decouper(corps) {
  const blocs = {};
  const parts = corps.split(DECOUPE);
  for (let i = 1; i < parts.length; i += 2) {
    const cle = LABELS[parts[i].trim()];
    if (cle) blocs[cle] = (parts[i + 1] ?? '').trim();
  }
  return blocs;
}

// Version « liste » : l'arborescence sans le corps des fiches.
export function alleger(modules) {
  return modules.map((m) => ({
    module: m.module,
    titre: m.titre,
    ordre: m.ordre,
    resume: m.resume,
    concepts: m.concepts.map(({ blocs, ...c }) => ({
      ...c,
      phrase: blocs.phrase ?? '',
    })),
  }));
}

// Texte brut d'une fiche, pour la recherche et l'export.
export function texteConcept(c) {
  return [c.titre, ...Object.values(c.blocs)].join('\n');
}

export const slugify = (s) =>
  s
    .toLowerCase()
    .normalize('NFD')
    .replace(/[̀-ͯ]/g, '')
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '');

function echapper(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

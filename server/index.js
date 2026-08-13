// API de l'Atelier ML. Le contenu est lu au démarrage depuis ../content,
// la progression vit dans data.db.
import fs from 'node:fs';
import path from 'node:path';
import express from 'express';
import cors from 'cors';

import { loadContent, alleger, texteConcept } from './content.js';
import * as bd from './db.js';
import { planifier, echeance, aujourdhui } from './srs.js';
import {
  exportMarkdownZip,
  exportJson,
  importJson,
  exportHtml,
} from './export.js';

const RACINE = path.resolve(import.meta.dirname, '..');
const DOSSIER_CONTENU = process.env.ATELIER_CONTENT ?? path.join(RACINE, 'content');
const PORT = Number(process.env.PORT ?? 3001);

let contenu = loadContent(DOSSIER_CONTENU);
console.log(
  `[contenu] ${contenu.modules.length} module(s), ${contenu.concepts.length} concept(s) depuis ${DOSSIER_CONTENU}`
);

const app = express();
app.use(cors());
app.use(express.json({ limit: '8mb' }));

const erreur = (res, code, message) => res.status(code).json({ erreur: message });

// Petit utilitaire : retrouver un concept ou répondre 404.
function concept(req, res) {
  const c = contenu.parSlug[req.params.slug];
  if (!c) {
    erreur(res, 404, `concept inconnu : ${req.params.slug}`);
    return null;
  }
  return c;
}

/* ------------------------------------------------------------------ */
/* Contenu                                                             */
/* ------------------------------------------------------------------ */

app.get('/api/modules', (req, res) => {
  const prog = bd.progressParSlug();
  const modules = alleger(contenu.modules).map((m) => ({
    ...m,
    concepts: m.concepts.map((c) => ({
      ...c,
      palier: prog[c.slug]?.palier ?? 0,
      difficulte_ressentie: prog[c.slug]?.difficulte_ressentie ?? null,
      minutes_total: prog[c.slug]?.minutes_total ?? 0,
    })),
  }));
  res.json(modules);
});

app.get('/api/concepts/:slug', (req, res) => {
  const c = concept(req, res);
  if (!c) return;
  bd.marquerVu(c.slug);
  const prog = bd.progressParSlug();
  res.json({
    ...c,
    prereqsDetail: c.prereqs.map((r) => ({
      slug: r,
      titre: contenu.parSlug[r]?.titre ?? r,
      palier: prog[r]?.palier ?? 0,
      connu: Boolean(contenu.parSlug[r]),
    })),
    suivants: contenu.concepts
      .filter((x) => x.prereqs.includes(c.slug))
      .map((x) => ({ slug: x.slug, titre: x.titre })),
    progress: bd.getProgress(c.slug),
    note: bd.getNote(c.slug),
    cartes: bd.cartesDuSlug(c.slug),
    sessions: bd.sessionsDuSlug(c.slug),
    preuves: bd.preuves(c.slug),
    exigences: bd.EXIGENCES,
  });
});

// Relire content/ sans redémarrer le serveur.
app.post('/api/reload', (req, res) => {
  contenu = loadContent(DOSSIER_CONTENU);
  res.json({ modules: contenu.modules.length, concepts: contenu.concepts.length });
});

/* ------------------------------------------------------------------ */
/* Progression et sessions                                             */
/* ------------------------------------------------------------------ */

app.put('/api/progress/:slug', (req, res) => {
  const c = concept(req, res);
  if (!c) return;

  const actuel = bd.getProgress(c.slug)?.palier ?? 0;
  const { palier, difficulte_ressentie } = req.body ?? {};
  const cible = palier === undefined || palier === null ? actuel : Number(palier);

  if (!Number.isInteger(cible) || cible < 0 || cible > 4)
    return erreur(res, 400, 'palier attendu entre 0 et 4');
  if (
    difficulte_ressentie != null &&
    (!Number.isInteger(difficulte_ressentie) ||
      difficulte_ressentie < 1 ||
      difficulte_ressentie > 5)
  )
    return erreur(res, 400, 'difficulte_ressentie attendue entre 1 et 5');

  const bloquant = bd.palierBloquant(c.slug, cible, actuel);
  if (bloquant !== null) {
    return res.status(422).json({
      erreur: `palier ${bloquant} refusé : preuve manquante`,
      manquant: bloquant,
      exigence: bd.EXIGENCES[bloquant],
      preuves: bd.preuves(c.slug),
    });
  }

  res.json(bd.majProgress(c.slug, { palier: cible, difficulte_ressentie }));
});

app.post('/api/sessions', (req, res) => {
  const { slug, minutes, difficulte_ressentie, resume, debut } = req.body ?? {};
  if (!contenu.parSlug[slug]) return erreur(res, 400, 'slug inconnu');
  if (!Number.isFinite(minutes) || minutes < 0)
    return erreur(res, 400, 'minutes attendues');
  res.json(
    bd.creerSession({ slug, minutes, difficulte_ressentie, resume, debut })
  );
});

app.get('/api/next', (req, res) => {
  res.json(bd.fileDuJour(contenu.concepts, contenu.parSlug));
});

/* ------------------------------------------------------------------ */
/* Notes                                                               */
/* ------------------------------------------------------------------ */

app.get('/api/notes/:slug', (req, res) => {
  const c = concept(req, res);
  if (!c) return;
  res.json(bd.getNote(c.slug) ?? { slug: c.slug, titre: c.titre, corps_md: '', en_mes_mots: '' });
});

app.put('/api/notes/:slug', (req, res) => {
  const c = concept(req, res);
  if (!c) return;
  const { corps_md, en_mes_mots, titre } = req.body ?? {};
  const note = bd.majNote(c.slug, { titre: titre ?? c.titre, corps_md, en_mes_mots });
  res.json({ ...note, preuves: bd.preuves(c.slug) });
});

/* ------------------------------------------------------------------ */
/* Cartes et révisions                                                 */
/* ------------------------------------------------------------------ */

app.get('/api/review/queue', (req, res) => {
  const cartes = bd.cartesDues(req.query.date ?? aujourdhui()).map((c) => ({
    ...c,
    titre: contenu.parSlug[c.slug]?.titre ?? c.slug,
    module: contenu.parSlug[c.slug]?.module ?? null,
  }));
  res.json(cartes);
});

app.post('/api/cards', (req, res) => {
  const { slug, recto_md, verso_md } = req.body ?? {};
  if (!contenu.parSlug[slug]) return erreur(res, 400, 'slug inconnu');
  if (!recto_md?.trim() || !verso_md?.trim())
    return erreur(res, 400, 'recto et verso obligatoires');
  res.json(bd.creerCarte({ slug, recto_md: recto_md.trim(), verso_md: verso_md.trim() }));
});

// Carte par défaut d'un concept : titre au recto, « en une phrase » au verso.
app.post('/api/cards/generer/:slug', (req, res) => {
  const c = concept(req, res);
  if (!c) return;
  if (!c.blocs.phrase) return erreur(res, 422, 'ce concept n’a pas de bloc « En une phrase »');
  res.json(bd.creerCarte({ slug: c.slug, recto_md: c.titre, verso_md: c.blocs.phrase }));
});

app.delete('/api/cards/:id', (req, res) => {
  const ok = bd.supprimerCarte(Number(req.params.id));
  if (!ok) return erreur(res, 404, 'carte inconnue');
  res.json({ supprime: true });
});

app.post('/api/review/:cardId', (req, res) => {
  const carte = bd.getCarte(Number(req.params.cardId));
  if (!carte) return erreur(res, 404, 'carte inconnue');
  const qualite = Number(req.body?.qualite);
  if (!Number.isInteger(qualite) || qualite < 0 || qualite > 3)
    return erreur(res, 400, 'qualite attendue entre 0 et 3');

  const plan = planifier(carte, qualite);
  const majee = bd.enregistrerRevision(
    carte,
    { ...plan, echeance: echeance(plan.intervalle) },
    qualite,
    req.body?.ms
  );
  res.json(majee);
});

/* ------------------------------------------------------------------ */
/* Statistiques et recherche                                           */
/* ------------------------------------------------------------------ */

app.get('/api/stats', (req, res) => {
  res.json(bd.stats(contenu.modules, contenu.concepts));
});

app.get('/api/search', (req, res) => {
  const q = String(req.query.q ?? '').trim();
  if (q.length < 2) return res.json({ q, fiches: [], notes: [] });

  const besoin = normaliser(q);
  const fiches = [];
  for (const c of contenu.concepts) {
    const texte = normaliser(texteConcept(c));
    const i = texte.indexOf(besoin);
    if (i === -1) continue;
    const brut = texteConcept(c);
    fiches.push({
      slug: c.slug,
      titre: c.titre,
      module: c.module,
      moduleTitre: c.moduleTitre,
      difficulte: c.difficulte,
      extrait: brut.slice(Math.max(0, i - 60), i + 140).replace(/\s+/g, ' ').trim(),
      dansLeTitre: normaliser(c.titre).includes(besoin),
    });
  }
  fiches.sort((a, b) => Number(b.dansLeTitre) - Number(a.dansLeTitre));

  const notes = bd.chercherNotes(q).map((n) => ({
    ...n,
    conceptTitre: n.slug ? (contenu.parSlug[n.slug]?.titre ?? n.slug) : null,
  }));

  res.json({ q, fiches: fiches.slice(0, 40), notes });
});

const normaliser = (s) =>
  s.toLowerCase().normalize('NFD').replace(/[̀-ͯ]/g, '');

/* ------------------------------------------------------------------ */
/* Ressources                                                          */
/* ------------------------------------------------------------------ */

app.get('/api/resources', (req, res) => {
  res.json('slug' in req.query ? bd.ressources(req.query.slug || null) : bd.ressources());
});
app.post('/api/resources', (req, res) => {
  if (!req.body?.titre) return erreur(res, 400, 'titre obligatoire');
  res.json(bd.creerRessource(req.body));
});
app.put('/api/resources/:id', (req, res) => {
  const r = bd.majRessource(Number(req.params.id), req.body ?? {});
  if (!r) return erreur(res, 404, 'ressource inconnue');
  res.json(r);
});
app.delete('/api/resources/:id', (req, res) => {
  if (!bd.supprimerRessource(Number(req.params.id)))
    return erreur(res, 404, 'ressource inconnue');
  res.json({ supprime: true });
});

/* ------------------------------------------------------------------ */
/* Exports                                                             */
/* ------------------------------------------------------------------ */

app.get('/api/export/markdown', (req, res) => {
  const zip = exportMarkdownZip(contenu.modules);
  res
    .type('application/zip')
    .set(
      'Content-Disposition',
      `attachment; filename="atelier-ml-${aujourdhui()}.zip"`
    )
    .send(zip);
});

app.get('/api/export/json', (req, res) => {
  res
    .set(
      'Content-Disposition',
      `attachment; filename="atelier-ml-${aujourdhui()}.json"`
    )
    .json(exportJson(contenu.modules));
});

app.post('/api/import/json', (req, res) => {
  try {
    res.json({ importe: importJson(req.body) });
  } catch (e) {
    erreur(res, 400, `import impossible : ${e.message}`);
  }
});

app.get('/api/export/html', (req, res) => {
  res.type('html').send(exportHtml(contenu.modules, { slug: req.query.slug }));
});

/* ------------------------------------------------------------------ */
/* Front compilé (optionnel) et démarrage                              */
/* ------------------------------------------------------------------ */

const DIST = path.join(RACINE, 'web', 'dist');
if (fs.existsSync(DIST)) {
  app.use(express.static(DIST));
  app.get(/^(?!\/api\/).*/, (req, res) => res.sendFile(path.join(DIST, 'index.html')));
}

app.use((err, req, res, next) => {
  console.error(err);
  erreur(res, 500, err.message);
});

app.listen(PORT, () => {
  console.log(`[api] http://localhost:${PORT}`);
});

// Schéma SQLite et requêtes. La base ne contient que ce qui appartient à
// l'utilisateur : progression, notes, sessions, cartes, ressources.
import path from 'node:path';
import Database from 'better-sqlite3';
import { aujourdhui, retard } from './srs.js';

export const db = new Database(
  process.env.ATELIER_DB ?? path.join(import.meta.dirname, 'data.db')
);

db.pragma('journal_mode = WAL');
db.pragma('foreign_keys = ON');

db.exec(`
CREATE TABLE IF NOT EXISTS progress (
  slug            TEXT PRIMARY KEY,
  palier          INTEGER NOT NULL DEFAULT 0,
  difficulte_ressentie INTEGER,
  minutes_total   INTEGER NOT NULL DEFAULT 0,
  premiere_vue    TEXT,
  derniere_vue    TEXT,
  maj_le          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS note (
  id        INTEGER PRIMARY KEY AUTOINCREMENT,
  slug      TEXT,
  titre     TEXT NOT NULL,
  corps_md  TEXT NOT NULL DEFAULT '',
  en_mes_mots TEXT NOT NULL DEFAULT '',
  type      TEXT NOT NULL DEFAULT 'cours',
  maj_le    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_note_slug ON note(slug);

CREATE TABLE IF NOT EXISTS session (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  slug       TEXT NOT NULL,
  debut      TEXT NOT NULL,
  minutes    INTEGER NOT NULL,
  difficulte_ressentie INTEGER,
  resume     TEXT NOT NULL DEFAULT '',
  palier_avant INTEGER,
  palier_apres INTEGER
);
CREATE INDEX IF NOT EXISTS idx_session_slug ON session(slug);

CREATE TABLE IF NOT EXISTS card (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  slug       TEXT NOT NULL,
  recto_md   TEXT NOT NULL,
  verso_md   TEXT NOT NULL,
  intervalle INTEGER NOT NULL DEFAULT 0,
  facilite   REAL    NOT NULL DEFAULT 2.5,
  echeance   TEXT    NOT NULL,
  lapses     INTEGER NOT NULL DEFAULT 0,
  cree_le    TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_card_echeance ON card(echeance);
CREATE INDEX IF NOT EXISTS idx_card_slug ON card(slug);

CREATE TABLE IF NOT EXISTS review (
  id       INTEGER PRIMARY KEY AUTOINCREMENT,
  card_id  INTEGER NOT NULL REFERENCES card(id) ON DELETE CASCADE,
  date     TEXT NOT NULL,
  qualite  INTEGER NOT NULL,
  ms       INTEGER
);
CREATE INDEX IF NOT EXISTS idx_review_card ON review(card_id);

CREATE TABLE IF NOT EXISTS resource (
  id      INTEGER PRIMARY KEY AUTOINCREMENT,
  slug    TEXT,
  type    TEXT NOT NULL,
  titre   TEXT NOT NULL,
  url     TEXT,
  statut  TEXT NOT NULL DEFAULT 'a_lire',
  minutes_reelles INTEGER NOT NULL DEFAULT 0
);

-- Écart assumé avec docs/modele-de-donnees.md : « en mes mots » est indexé lui
-- aussi (ne pas retrouver sa propre reformulation serait absurde), et le
-- tokeniseur ignore les accents — « memoire » trouve « mémoire ».
CREATE VIRTUAL TABLE IF NOT EXISTS note_fts USING fts5(
  titre, corps_md, en_mes_mots, content='note', content_rowid='id',
  tokenize="unicode61 remove_diacritics 2"
);

CREATE TRIGGER IF NOT EXISTS note_ai AFTER INSERT ON note BEGIN
  INSERT INTO note_fts(rowid, titre, corps_md, en_mes_mots)
    VALUES (new.id, new.titre, new.corps_md, new.en_mes_mots);
END;
CREATE TRIGGER IF NOT EXISTS note_ad AFTER DELETE ON note BEGIN
  INSERT INTO note_fts(note_fts, rowid, titre, corps_md, en_mes_mots)
    VALUES('delete', old.id, old.titre, old.corps_md, old.en_mes_mots);
END;
CREATE TRIGGER IF NOT EXISTS note_au AFTER UPDATE ON note BEGIN
  INSERT INTO note_fts(note_fts, rowid, titre, corps_md, en_mes_mots)
    VALUES('delete', old.id, old.titre, old.corps_md, old.en_mes_mots);
  INSERT INTO note_fts(rowid, titre, corps_md, en_mes_mots)
    VALUES (new.id, new.titre, new.corps_md, new.en_mes_mots);
END;
`);

const maintenant = () => new Date().toISOString();

/* ------------------------------------------------------------------ */
/* Progression                                                         */
/* ------------------------------------------------------------------ */

const S = {
  progress: db.prepare('SELECT * FROM progress WHERE slug = ?'),
  tousProgress: db.prepare('SELECT * FROM progress'),
  voir: db.prepare(`
    INSERT INTO progress (slug, premiere_vue, derniere_vue, maj_le)
    VALUES (@slug, @t, @t, @t)
    ON CONFLICT(slug) DO UPDATE SET derniere_vue = @t, maj_le = @t
  `),
  majProgress: db.prepare(`
    INSERT INTO progress (slug, palier, difficulte_ressentie, maj_le, premiere_vue, derniere_vue)
    VALUES (@slug, @palier, @difficulte_ressentie, @t, @t, @t)
    ON CONFLICT(slug) DO UPDATE SET
      palier = @palier,
      difficulte_ressentie = COALESCE(@difficulte_ressentie, difficulte_ressentie),
      maj_le = @t
  `),
  ajouterMinutes: db.prepare(`
    INSERT INTO progress (slug, minutes_total, difficulte_ressentie, premiere_vue, derniere_vue, maj_le)
    VALUES (@slug, @minutes, @difficulte_ressentie, @t, @t, @t)
    ON CONFLICT(slug) DO UPDATE SET
      minutes_total = minutes_total + @minutes,
      difficulte_ressentie = COALESCE(@difficulte_ressentie, difficulte_ressentie),
      derniere_vue = @t,
      maj_le = @t
  `),
};

export const getProgress = (slug) => S.progress.get(slug) ?? null;
export const tousProgress = () => S.tousProgress.all();
export const progressParSlug = () =>
  Object.fromEntries(S.tousProgress.all().map((p) => [p.slug, p]));
export const marquerVu = (slug) => S.voir.run({ slug, t: maintenant() });

export function majProgress(slug, { palier, difficulte_ressentie }) {
  S.majProgress.run({
    slug,
    palier,
    difficulte_ressentie: difficulte_ressentie ?? null,
    t: maintenant(),
  });
  return getProgress(slug);
}

/* ------------------------------------------------------------------ */
/* Preuves de palier — une montée ne se décrète pas                    */
/* ------------------------------------------------------------------ */

const P = {
  note: db.prepare(
    "SELECT * FROM note WHERE slug = ? AND type = 'cours' ORDER BY id LIMIT 1"
  ),
  sessionAvecResume: db.prepare(
    "SELECT COUNT(*) n FROM session WHERE slug = ? AND TRIM(resume) <> ''"
  ),
  carteMure: db.prepare(
    'SELECT COUNT(*) n FROM card WHERE slug = ? AND intervalle >= 21'
  ),
};

export const EXIGENCES = {
  1: 'une note non vide dans « en mes mots »',
  2: 'un bloc de code collé dans la note (refait sans regarder)',
  3: 'une session avec un résumé : dataset, métrique, conclusion',
  4: 'une carte du concept réussie à 21 jours d’intervalle ou plus',
};

export function preuves(slug) {
  const note = P.note.get(slug);
  return {
    1: (note?.en_mes_mots ?? '').trim().length > 0,
    2: /```/.test(note?.corps_md ?? '') || /```/.test(note?.en_mes_mots ?? ''),
    3: P.sessionAvecResume.get(slug).n > 0,
    4: P.carteMure.get(slug).n > 0,
  };
}

// Retourne null si la montée est permise, sinon le palier qui bloque.
export function palierBloquant(slug, cible, actuel) {
  if (cible <= actuel) return null;
  const p = preuves(slug);
  for (let n = actuel + 1; n <= cible; n++) if (!p[n]) return n;
  return null;
}

/* ------------------------------------------------------------------ */
/* Notes                                                               */
/* ------------------------------------------------------------------ */

const N = {
  principale: P.note,
  creer: db.prepare(`
    INSERT INTO note (slug, titre, corps_md, en_mes_mots, type, maj_le)
    VALUES (@slug, @titre, @corps_md, @en_mes_mots, @type, @maj_le)
  `),
  majPrincipale: db.prepare(`
    UPDATE note SET corps_md = @corps_md, en_mes_mots = @en_mes_mots, titre = @titre, maj_le = @maj_le
    WHERE id = @id
  `),
  toutes: db.prepare('SELECT * FROM note ORDER BY id'),
  parSlug: db.prepare('SELECT * FROM note WHERE slug = ? ORDER BY id'),
};

export const getNote = (slug) => N.principale.get(slug) ?? null;
export const toutesNotes = () => N.toutes.all();
export const notesDuSlug = (slug) => N.parSlug.all(slug);

export function majNote(slug, { titre, corps_md, en_mes_mots }) {
  const existante = N.principale.get(slug);
  const maj_le = maintenant();
  if (existante) {
    N.majPrincipale.run({
      id: existante.id,
      titre: titre ?? existante.titre,
      corps_md: corps_md ?? existante.corps_md,
      en_mes_mots: en_mes_mots ?? existante.en_mes_mots,
      maj_le,
    });
  } else {
    N.creer.run({
      slug,
      titre: titre ?? slug,
      corps_md: corps_md ?? '',
      en_mes_mots: en_mes_mots ?? '',
      type: 'cours',
      maj_le,
    });
  }
  return getNote(slug);
}

/* ------------------------------------------------------------------ */
/* Sessions                                                            */
/* ------------------------------------------------------------------ */

const SE = {
  creer: db.prepare(`
    INSERT INTO session (slug, debut, minutes, difficulte_ressentie, resume, palier_avant, palier_apres)
    VALUES (@slug, @debut, @minutes, @difficulte_ressentie, @resume, @palier_avant, @palier_apres)
  `),
  parSlug: db.prepare('SELECT * FROM session WHERE slug = ? ORDER BY debut DESC'),
  toutes: db.prepare('SELECT * FROM session ORDER BY debut'),
  parJour: db.prepare(`
    SELECT substr(debut, 1, 10) AS date, SUM(minutes) AS minutes, COUNT(*) AS n
    FROM session GROUP BY date ORDER BY date
  `),
};

export const sessionsDuSlug = (slug) => SE.parSlug.all(slug);
export const toutesSessions = () => SE.toutes.all();
export const minutesParJour = () => SE.parJour.all();

export const creerSession = db.transaction((s) => {
  const avant = getProgress(s.slug)?.palier ?? 0;
  const info = SE.creer.run({
    slug: s.slug,
    debut: s.debut ?? maintenant(),
    minutes: Math.max(0, Math.round(s.minutes ?? 0)),
    difficulte_ressentie: s.difficulte_ressentie ?? null,
    resume: s.resume ?? '',
    palier_avant: avant,
    palier_apres: s.palier_apres ?? avant,
  });
  S.ajouterMinutes.run({
    slug: s.slug,
    minutes: Math.max(0, Math.round(s.minutes ?? 0)),
    difficulte_ressentie: s.difficulte_ressentie ?? null,
    t: maintenant(),
  });
  return db.prepare('SELECT * FROM session WHERE id = ?').get(info.lastInsertRowid);
});

/* ------------------------------------------------------------------ */
/* Cartes et révisions                                                 */
/* ------------------------------------------------------------------ */

const C = {
  creer: db.prepare(`
    INSERT INTO card (slug, recto_md, verso_md, echeance, cree_le)
    VALUES (@slug, @recto_md, @verso_md, @echeance, @cree_le)
  `),
  parId: db.prepare('SELECT * FROM card WHERE id = ?'),
  parSlug: db.prepare('SELECT * FROM card WHERE slug = ? ORDER BY id'),
  toutes: db.prepare('SELECT * FROM card ORDER BY id'),
  memeRecto: db.prepare('SELECT * FROM card WHERE slug = ? AND recto_md = ?'),
  dues: db.prepare('SELECT * FROM card WHERE echeance <= ? ORDER BY echeance, id'),
  supprimer: db.prepare('DELETE FROM card WHERE id = ?'),
  planifier: db.prepare(
    'UPDATE card SET intervalle = @intervalle, facilite = @facilite, echeance = @echeance, lapses = @lapses WHERE id = @id'
  ),
  historique: db.prepare(
    'INSERT INTO review (card_id, date, qualite, ms) VALUES (?, ?, ?, ?)'
  ),
  revisions: db.prepare('SELECT * FROM review ORDER BY id'),
};

export const cartesDuSlug = (slug) => C.parSlug.all(slug);
export const toutesCartes = () => C.toutes.all();
export const toutesRevisions = () => C.revisions.all();
export const getCarte = (id) => C.parId.get(id) ?? null;
export const supprimerCarte = (id) => C.supprimer.run(id).changes > 0;
export const cartesDues = (date = aujourdhui()) =>
  C.dues.all(date).map((c) => ({ ...c, retard: retard(c.echeance, date) }));

export function creerCarte({ slug, recto_md, verso_md, echeance: e }) {
  const deja = C.memeRecto.get(slug, recto_md);
  if (deja) return deja;
  const info = C.creer.run({
    slug,
    recto_md,
    verso_md,
    echeance: e ?? aujourdhui(),
    cree_le: maintenant(),
  });
  return C.parId.get(info.lastInsertRowid);
}

export const enregistrerRevision = db.transaction((carte, plan, qualite, ms) => {
  C.planifier.run({
    id: carte.id,
    intervalle: plan.intervalle,
    facilite: plan.facilite,
    echeance: plan.echeance,
    lapses: plan.lapses,
  });
  C.historique.run(carte.id, maintenant(), qualite, ms ?? null);
  return C.parId.get(carte.id);
});

/* ------------------------------------------------------------------ */
/* Ressources                                                          */
/* ------------------------------------------------------------------ */

const R = {
  toutes: db.prepare('SELECT * FROM resource ORDER BY id'),
  parSlug: db.prepare('SELECT * FROM resource WHERE slug IS ? ORDER BY id'),
  creer: db.prepare(`
    INSERT INTO resource (slug, type, titre, url, statut, minutes_reelles)
    VALUES (@slug, @type, @titre, @url, @statut, @minutes_reelles)
  `),
  maj: db.prepare(
    'UPDATE resource SET statut = COALESCE(@statut, statut), minutes_reelles = COALESCE(@minutes_reelles, minutes_reelles) WHERE id = @id'
  ),
  supprimer: db.prepare('DELETE FROM resource WHERE id = ?'),
};

export const ressources = (slug) =>
  slug === undefined ? R.toutes.all() : R.parSlug.all(slug ?? null);

export function creerRessource(r) {
  const info = R.creer.run({
    slug: r.slug ?? null,
    type: r.type ?? 'cours',
    titre: r.titre,
    url: r.url ?? null,
    statut: r.statut ?? 'a_lire',
    minutes_reelles: r.minutes_reelles ?? 0,
  });
  return db.prepare('SELECT * FROM resource WHERE id = ?').get(info.lastInsertRowid);
}

export const majRessource = (id, champs) => {
  R.maj.run({ id, statut: champs.statut ?? null, minutes_reelles: champs.minutes_reelles ?? null });
  return db.prepare('SELECT * FROM resource WHERE id = ?').get(id);
};
export const supprimerRessource = (id) => R.supprimer.run(id).changes > 0;

/* ------------------------------------------------------------------ */
/* Recherche plein texte sur les notes                                 */
/* ------------------------------------------------------------------ */

const F = {
  chercher: db.prepare(`
    SELECT n.id, n.slug, n.titre, n.type,
           snippet(note_fts, -1, '«', '»', '…', 12) AS extrait,
           bm25(note_fts) AS score
    FROM note_fts JOIN note n ON n.id = note_fts.rowid
    WHERE note_fts MATCH ?
    ORDER BY score LIMIT 30
  `),
};

export function chercherNotes(q) {
  const requete = q
    .split(/\s+/)
    .filter(Boolean)
    .map((mot) => `"${mot.replace(/"/g, '')}"*`)
    .join(' ');
  if (!requete) return [];
  try {
    return F.chercher.all(requete);
  } catch {
    return []; // syntaxe FTS invalide : on préfère zéro résultat à une 500
  }
}

/* ------------------------------------------------------------------ */
/* File d'étude (§6 du modèle de données)                              */
/* ------------------------------------------------------------------ */

export function prochainConcept(concepts, progressBySlug) {
  return concepts.find((c) => {
    const p = progressBySlug[c.slug];
    if (p && p.palier > 0) return false;
    return c.prereqs.every((r) => (progressBySlug[r]?.palier ?? 0) >= 2);
  });
}

export function fileDuJour(concepts, parSlug) {
  const prog = progressParSlug();
  const file = [];
  const vus = new Set();

  const pousser = (type, concept, raison, extra = {}) => {
    if (!concept || vus.has(type + concept.slug)) return;
    vus.add(type + concept.slug);
    file.push({
      type,
      slug: concept.slug,
      titre: concept.titre,
      module: concept.module,
      moduleTitre: concept.moduleTitre,
      difficulte: concept.difficulte,
      palier: prog[concept.slug]?.palier ?? 0,
      raison,
      ...extra,
    });
  };

  // 1. La dette de révision passe avant tout.
  const enRetard = cartesDues().filter((c) => c.retard > 3);
  if (enRetard.length) {
    file.push({
      type: 'revision',
      n: enRetard.length,
      retardMax: Math.max(...enRetard.map((c) => c.retard)),
      raison: `${enRetard.length} carte(s) en retard de plus de 3 jours`,
    });
  }

  // 2. Écart de difficulté ≥ 2 : c'est le prérequis qu'on propose.
  for (const c of concepts) {
    const p = prog[c.slug];
    if (!p?.difficulte_ressentie) continue;
    const ecart = p.difficulte_ressentie - c.difficulte;
    if (ecart < 2) continue;
    const manquant = c.prereqs
      .map((r) => parSlug[r])
      .find((r) => r && (prog[r.slug]?.palier ?? 0) < 2);
    pousser(
      'lacune',
      manquant ?? c,
      manquant
        ? `« ${c.titre} » t'a paru ${ecart} niveaux trop dur : le prérequis n'est pas acquis`
        : `« ${c.titre} » t'a paru ${ecart} niveaux plus dur qu'annoncé`,
      { origine: c.slug, ecart }
    );
  }

  // 3. Palier 1 depuis plus de 14 jours : compris, jamais implémenté.
  const limite = Date.now() - 14 * 86400000;
  for (const c of concepts) {
    const p = prog[c.slug];
    if (p?.palier === 1 && Date.parse(p.maj_le) < limite) {
      const jours = Math.round((Date.now() - Date.parse(p.maj_le)) / 86400000);
      pousser('a_implementer', c, `compris depuis ${jours} jours, jamais implémenté`, {
        jours,
      });
    }
  }

  // 4. Le premier concept non commencé dont les prérequis sont au palier ≥ 2.
  pousser('nouveau', prochainConcept(concepts, prog), 'prérequis satisfaits, à découvrir');

  return file;
}

/* ------------------------------------------------------------------ */
/* Statistiques                                                        */
/* ------------------------------------------------------------------ */

export function stats(modules, concepts) {
  const prog = progressParSlug();

  const parModule = modules.map((m) => {
    const paliers = [0, 0, 0, 0, 0];
    let minutes = 0;
    for (const c of m.concepts) {
      const p = prog[c.slug];
      paliers[p?.palier ?? 0]++;
      minutes += p?.minutes_total ?? 0;
    }
    const total = m.concepts.length || 1;
    const somme = paliers.reduce((s, n, i) => s + n * i, 0);
    return {
      module: m.module,
      titre: m.titre,
      total: m.concepts.length,
      paliers,
      minutes,
      maitrise: Math.round((somme / (total * 4)) * 100),
    };
  });

  const ecarts = concepts
    .map((c) => {
      const p = prog[c.slug];
      if (!p?.difficulte_ressentie) return null;
      return {
        slug: c.slug,
        titre: c.titre,
        module: c.module,
        difficulte: c.difficulte,
        ressentie: p.difficulte_ressentie,
        ecart: p.difficulte_ressentie - c.difficulte,
      };
    })
    .filter(Boolean)
    .sort((a, b) => b.ecart - a.ecart);

  const dues = cartesDues();
  const paliers = [0, 0, 0, 0, 0];
  for (const c of concepts) paliers[prog[c.slug]?.palier ?? 0]++;

  return {
    parModule,
    ecarts,
    jours: minutesParJour(),
    paliers,
    totaux: {
      concepts: concepts.length,
      touches: Object.keys(prog).length,
      minutes: Object.values(prog).reduce((s, p) => s + p.minutes_total, 0),
      cartes: toutesCartes().length,
      cartesDues: dues.length,
      notes: db.prepare("SELECT COUNT(*) n FROM note WHERE TRIM(corps_md) <> '' OR TRIM(en_mes_mots) <> ''").get().n,
      serie: serieEnCours(),
    },
  };
}

// Nombre de jours consécutifs travaillés, aujourd'hui ou hier inclus.
function serieEnCours() {
  const jours = new Set(minutesParJour().map((j) => j.date));
  if (!jours.size) return 0;
  let n = 0;
  const d = new Date();
  if (!jours.has(d.toISOString().slice(0, 10))) d.setDate(d.getDate() - 1);
  while (jours.has(d.toISOString().slice(0, 10))) {
    n++;
    d.setDate(d.getDate() - 1);
  }
  return n;
}

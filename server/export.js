// Exports : Markdown (zip, un dossier par module), JSON complet réimportable,
// HTML autonome (imprimable en PDF). Aucun paquet supplémentaire : le zip est
// écrit à la main en mode « stored », ce qui suffit pour du texte.
import { marked } from 'marked';
import * as bd from './db.js';

/* ------------------------------------------------------------------ */
/* Markdown                                                            */
/* ------------------------------------------------------------------ */

const yaml = (v) =>
  typeof v === 'number' || typeof v === 'boolean'
    ? String(v)
    : Array.isArray(v)
      ? `[${v.join(', ')}]`
      : `"${String(v ?? '').replace(/"/g, '\\"')}"`;

function frontMatter(champs) {
  const lignes = Object.entries(champs)
    .filter(([, v]) => v !== null && v !== undefined && v !== '')
    .map(([k, v]) => `${k}: ${yaml(v)}`);
  return `---\n${lignes.join('\n')}\n---\n`;
}

const PALIERS = ['rencontré', 'compris', 'implémenté', 'appliqué', 'expliqué'];

// Une fiche exportée : le savoir (issu du Markdown) + ce que tu en as fait.
export function conceptEnMarkdown(concept, { fiche = true } = {}) {
  const p = bd.getProgress(concept.slug);
  const note = bd.getNote(concept.slug);
  const cartes = bd.cartesDuSlug(concept.slug);
  const sessions = bd.sessionsDuSlug(concept.slug);

  const entete = frontMatter({
    slug: concept.slug,
    titre: concept.titre,
    module: concept.module,
    difficulte: concept.difficulte,
    prereqs: concept.prereqs,
    palier: p?.palier ?? 0,
    palier_nom: PALIERS[p?.palier ?? 0],
    difficulte_ressentie: p?.difficulte_ressentie ?? null,
    minutes_total: p?.minutes_total ?? 0,
    derniere_vue: p?.derniere_vue ?? null,
    exporte_le: new Date().toISOString(),
  });

  const parts = [entete, `\n# ${concept.titre}\n`];

  if (fiche) {
    const b = concept.blocs;
    const bloc = (label, texte) => (texte ? `\n**${label}** — ${texte}\n` : '');
    parts.push(
      bloc('En une phrase', b.phrase),
      bloc('Explication', b.explication),
      bloc("Cas d'utilisation", b.casUsage),
      bloc('Algorithme', b.algorithme),
      bloc('Implémentation', b.implementation),
      bloc('Outils', b.outils),
      bloc('Alternatives open-source', b.alternatives),
      bloc('Astuces', b.astuces)
    );
  }

  if (note?.en_mes_mots?.trim()) {
    parts.push(`\n## En mes mots\n\n${note.en_mes_mots.trim()}\n`);
  }
  if (note?.corps_md?.trim()) {
    parts.push(`\n## Notes\n\n${note.corps_md.trim()}\n`);
  }
  if (cartes.length) {
    parts.push('\n## Cartes\n');
    for (const c of cartes) {
      parts.push(
        `\n- **${c.recto_md.replace(/\n+/g, ' ')}**\n  ${c.verso_md.replace(/\n+/g, '\n  ')}\n  <!-- intervalle: ${c.intervalle} j | facilité: ${c.facilite.toFixed(2)} | échéance: ${c.echeance} -->\n`
      );
    }
  }
  if (sessions.length) {
    parts.push('\n## Sessions\n\n| Date | Minutes | Ressentie | Résumé |\n|---|---|---|---|\n');
    for (const s of sessions) {
      parts.push(
        `| ${s.debut.slice(0, 16).replace('T', ' ')} | ${s.minutes} | ${s.difficulte_ressentie ?? '—'} | ${s.resume.replace(/\|/g, '\\|').replace(/\n/g, ' ')} |\n`
      );
    }
  }
  return parts.join('');
}

// Zip : un dossier par module, un fichier par concept, plus un index.
export function exportMarkdownZip(modules) {
  const fichiers = [];
  const index = ['# Atelier ML — export\n', `\nExporté le ${new Date().toISOString().slice(0, 10)}.\n`];

  for (const m of modules) {
    const dossier = `${String(m.ordre).padStart(2, '0')}-${m.module}`;
    index.push(`\n## ${m.titre}\n\n`);
    for (const c of m.concepts) {
      const p = bd.getProgress(c.slug);
      fichiers.push({
        nom: `${dossier}/${c.slug}.md`,
        contenu: conceptEnMarkdown(c),
      });
      index.push(
        `- [${c.titre}](${dossier}/${c.slug}.md) — palier ${p?.palier ?? 0} (${PALIERS[p?.palier ?? 0]})\n`
      );
    }
    // Un fichier récapitulatif par module, dans le dossier du module.
    fichiers.push({
      nom: `${dossier}/00-module.md`,
      contenu:
        frontMatter({ module: m.module, titre: m.titre, ordre: m.ordre }) +
        `\n# ${m.titre}\n\n${m.resume}\n\n` +
        m.concepts
          .map((c) => `- [${c.titre}](${c.slug}.md) — difficulté ${c.difficulte}`)
          .join('\n') +
        '\n',
    });
  }

  const libres = bd.toutesNotes().filter((n) => !n.slug);
  if (libres.length) {
    fichiers.push({
      nom: 'notes-libres.md',
      contenu:
        '# Notes libres\n\n' +
        libres.map((n) => `## ${n.titre}\n\n${n.corps_md}\n`).join('\n'),
    });
  }

  fichiers.unshift({ nom: 'index.md', contenu: index.join('') });
  return zip(fichiers);
}

/* ------------------------------------------------------------------ */
/* JSON                                                                */
/* ------------------------------------------------------------------ */

export function exportJson(modules) {
  return {
    version: 1,
    exporte_le: new Date().toISOString(),
    contenu: {
      modules: modules.map((m) => ({
        module: m.module,
        titre: m.titre,
        ordre: m.ordre,
        concepts: m.concepts.map((c) => c.slug),
      })),
    },
    progress: bd.tousProgress(),
    note: bd.toutesNotes(),
    session: bd.toutesSessions(),
    card: bd.toutesCartes(),
    review: bd.toutesRevisions(),
    resource: bd.ressources(),
  };
}

// Réimport : remplace le contenu des tables « à toi ». Le Markdown n'est pas touché.
export const importJson = bd.db.transaction((dump) => {
  if (!dump || typeof dump !== 'object') throw new Error('dump invalide');
  const tables = ['review', 'card', 'session', 'note', 'progress', 'resource'];
  for (const t of tables) bd.db.prepare(`DELETE FROM ${t}`).run();

  const compte = {};
  for (const t of ['progress', 'note', 'session', 'card', 'review', 'resource']) {
    const lignes = dump[t] ?? [];
    compte[t] = 0;
    for (const ligne of lignes) {
      const cols = Object.keys(ligne);
      bd.db
        .prepare(
          `INSERT INTO ${t} (${cols.join(', ')}) VALUES (${cols.map((c) => '@' + c).join(', ')})`
        )
        .run(ligne);
      compte[t]++;
    }
  }
  return compte;
});

/* ------------------------------------------------------------------ */
/* HTML autonome (impression PDF)                                      */
/* ------------------------------------------------------------------ */

export function exportHtml(modules, { slug } = {}) {
  const concepts = modules.flatMap((m) => m.concepts);
  const choisis = slug ? concepts.filter((c) => c.slug === slug) : concepts;
  const corps = choisis
    .map((c) => `<article>${marked.parse(conceptEnMarkdown(c).replace(/^---[\s\S]*?---\n/, ''))}</article>`)
    .join('\n');

  return `<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<title>Atelier ML — ${slug ?? 'export complet'}</title>
<style>
  body { font: 16px/1.6 Georgia, serif; max-width: 46rem; margin: 2rem auto; padding: 0 1rem; color: #111; }
  article { page-break-after: always; }
  h1 { font-size: 1.8rem; border-bottom: 2px solid #111; padding-bottom: .3rem; }
  h2 { font-size: 1.2rem; margin-top: 2rem; }
  code, pre { font-family: ui-monospace, Consolas, monospace; font-size: .85em; }
  pre { background: #f4f4f4; padding: .8rem; overflow-x: auto; border-left: 3px solid #999; }
  table { border-collapse: collapse; width: 100%; font-size: .9em; }
  td, th { border: 1px solid #ccc; padding: .3rem .5rem; text-align: left; }
  @media print { body { margin: 0; } }
</style></head><body>
<h1>Atelier ML</h1>
<p>Export du ${new Date().toLocaleString('fr-FR')} — ${choisis.length} fiche(s).</p>
${corps}
</body></html>`;
}

/* ------------------------------------------------------------------ */
/* Écriture d'un zip minimal (méthode 0 : stored)                      */
/* ------------------------------------------------------------------ */

const TABLE_CRC = (() => {
  const t = new Int32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c;
  }
  return t;
})();

function crc32(buf) {
  let c = -1;
  for (let i = 0; i < buf.length; i++) c = TABLE_CRC[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  return (c ^ -1) >>> 0;
}

function dosDate(d = new Date()) {
  return {
    heure: (d.getHours() << 11) | (d.getMinutes() << 5) | (d.getSeconds() >> 1),
    date: ((d.getFullYear() - 1980) << 9) | ((d.getMonth() + 1) << 5) | d.getDate(),
  };
}

export function zip(fichiers) {
  const { heure, date } = dosDate();
  const morceaux = [];
  const central = [];
  let offset = 0;

  for (const f of fichiers) {
    const nom = Buffer.from(f.nom, 'utf8');
    const data = Buffer.from(f.contenu, 'utf8');
    const crc = crc32(data);

    const local = Buffer.alloc(30);
    local.writeUInt32LE(0x04034b50, 0);
    local.writeUInt16LE(20, 4); // version nécessaire
    local.writeUInt16LE(0x0800, 6); // noms en UTF-8
    local.writeUInt16LE(0, 8); // méthode : stored
    local.writeUInt16LE(heure, 10);
    local.writeUInt16LE(date, 12);
    local.writeUInt32LE(crc, 14);
    local.writeUInt32LE(data.length, 18);
    local.writeUInt32LE(data.length, 22);
    local.writeUInt16LE(nom.length, 26);
    local.writeUInt16LE(0, 28);
    morceaux.push(local, nom, data);

    const entree = Buffer.alloc(46);
    entree.writeUInt32LE(0x02014b50, 0);
    entree.writeUInt16LE(20, 4); // version d'écriture
    entree.writeUInt16LE(20, 6);
    entree.writeUInt16LE(0x0800, 8);
    entree.writeUInt16LE(0, 10);
    entree.writeUInt16LE(heure, 12);
    entree.writeUInt16LE(date, 14);
    entree.writeUInt32LE(crc, 16);
    entree.writeUInt32LE(data.length, 20);
    entree.writeUInt32LE(data.length, 24);
    entree.writeUInt16LE(nom.length, 28);
    entree.writeUInt32LE(0, 30); // extra + commentaire
    entree.writeUInt16LE(0, 34); // disque
    entree.writeUInt16LE(0, 36); // attributs internes
    entree.writeUInt32LE(0, 38); // attributs externes
    entree.writeUInt32LE(offset, 42);
    central.push(entree, nom);

    offset += local.length + nom.length + data.length;
  }

  const annuaire = Buffer.concat(central);
  const fin = Buffer.alloc(22);
  fin.writeUInt32LE(0x06054b50, 0);
  fin.writeUInt16LE(fichiers.length, 8);
  fin.writeUInt16LE(fichiers.length, 10);
  fin.writeUInt32LE(annuaire.length, 12);
  fin.writeUInt32LE(offset, 16);

  return Buffer.concat([...morceaux, annuaire, fin]);
}

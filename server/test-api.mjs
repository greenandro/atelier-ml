// Parcours complet de l'API : contenu -> preuves de palier -> notes ->
// sessions -> cartes -> révisions -> recherche -> exports.
// Aucune dépendance : juste fetch. Il écrit dans la base qu'on lui donne,
// donc jamais celle de tous les jours.
//
//   ATELIER_DB=/tmp/test.db PORT=3999 node index.js &
//   node test-api.mjs
const B = 'http://localhost:3999/api';
let ok = 0, ko = 0;

const j = async (m, url, body) => {
  const r = await fetch(B + url, {
    method: m,
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  const t = r.headers.get('content-type') ?? '';
  return { statut: r.status, corps: t.includes('json') ? await r.json() : await r.text(), r };
};

const test = (nom, cond, extra = '') => {
  if (cond) { ok++; console.log(`  ok   ${nom}`); }
  else { ko++; console.log(`  KO   ${nom} ${extra}`); }
};

const modules = (await j('GET', '/modules')).corps;
test('GET /modules', Array.isArray(modules) && modules.length > 0, JSON.stringify(modules).slice(0, 120));
const slug = modules[0].concepts[0].slug;
const slug2 = modules[0].concepts[1].slug;
console.log(`\n-- concept de test : ${slug}\n`);

const c = (await j('GET', `/concepts/${slug}`)).corps;
test('GET /concepts/:slug renvoie les blocs', Boolean(c.blocs.explication));
test('la fiche ouverte crée la progression (palier 0)', c.progress?.palier === 0 || c.progress === null);
test('404 sur slug inconnu', (await j('GET', '/concepts/nexiste-pas')).statut === 404);

// --- preuves de palier
let r = await j('PUT', `/progress/${slug}`, { palier: 1 });
test('palier 1 refusé sans « en mes mots »', r.statut === 422 && r.corps.manquant === 1, JSON.stringify(r.corps));

await j('PUT', `/notes/${slug}`, { en_mes_mots: 'Ma reformulation du concept.' });
r = await j('PUT', `/progress/${slug}`, { palier: 1 });
test('palier 1 accepté après la note', r.statut === 200 && r.corps.palier === 1, JSON.stringify(r.corps));

r = await j('PUT', `/progress/${slug}`, { palier: 2 });
test('palier 2 refusé sans code', r.statut === 422 && r.corps.manquant === 2);

await j('PUT', `/notes/${slug}`, { corps_md: 'Refait de mémoire :\n\n```python\nimport numpy as np\n```' });
r = await j('PUT', `/progress/${slug}`, { palier: 2 });
test('palier 2 accepté avec un bloc de code', r.statut === 200 && r.corps.palier === 2);

r = await j('PUT', `/progress/${slug}`, { palier: 3 });
test('palier 3 refusé sans session résumée', r.statut === 422 && r.corps.manquant === 3);

r = await j('POST', '/sessions', { slug, minutes: 45, difficulte_ressentie: 4, resume: 'Titanic, AUC 0.83, features à revoir.' });
test('POST /sessions', r.statut === 200 && r.corps.minutes === 45 && r.corps.palier_avant === 2);

r = await j('PUT', `/progress/${slug}`, { palier: 3 });
test('palier 3 accepté après session résumée', r.statut === 200 && r.corps.palier === 3);
test('minutes cumulées dans progress', r.corps.minutes_total === 45, JSON.stringify(r.corps));

r = await j('PUT', `/progress/${slug}`, { palier: 4 });
test('palier 4 refusé sans carte mûre', r.statut === 422 && r.corps.manquant === 4);
r = await j('PUT', `/progress/${slug}`, { palier: 1 });
test('redescendre de palier est libre', r.statut === 200 && r.corps.palier === 1);
await j('PUT', `/progress/${slug}`, { palier: 3 });
r = await j('PUT', `/progress/${slug}`, { palier: 7 });
test('palier hors bornes rejeté', r.statut === 400);

// --- difficulté ressentie et écart
await j('PUT', `/progress/${slug2}`, { difficulte_ressentie: 5 });
const st = (await j('GET', '/stats')).corps;
test('GET /stats — maîtrise par module', st.parModule.length === modules.length && st.parModule[0].maitrise > 0);
test('GET /stats — écarts de difficulté', st.ecarts.length >= 1);
test('GET /stats — minutes du jour', st.jours.at(-1)?.minutes === 45);
test('GET /stats — série de jours', st.totaux.serie === 1);

// --- file d'étude
const file = (await j('GET', '/next')).corps;
test('GET /next renvoie une file', Array.isArray(file) && file.length > 0, JSON.stringify(file).slice(0, 200));
test('GET /next propose un nouveau concept', file.some((f) => f.type === 'nouveau'));

// --- cartes et révisions
r = await j('POST', `/cards/generer/${slug}`);
test('POST /cards/generer/:slug', r.statut === 200 && r.corps.id > 0);
const carteId = r.corps.id;
r = await j('POST', `/cards/generer/${slug}`);
test('génération idempotente', r.corps.id === carteId);
r = await j('POST', '/cards', { slug, recto_md: 'Formule ?', verso_md: '$\\hat{y} = Xw + b$' });
test('POST /cards depuis une sélection', r.statut === 200);
r = await j('POST', '/cards', { slug, recto_md: '', verso_md: 'x' });
test('carte vide rejetée', r.statut === 400);

let queue = (await j('GET', '/review/queue')).corps;
test('GET /review/queue — 2 cartes dues aujourd’hui', queue.length === 2, JSON.stringify(queue.map((q) => q.id)));
test('la carte porte le titre du concept', Boolean(queue[0].titre));

r = await j('POST', `/review/${carteId}`, { qualite: 2, ms: 4200 });
test('POST /review/:id — intervalle 0 -> 1', r.corps.intervalle === 1);
r = await j('POST', `/review/${carteId}`, { qualite: 3 });
test('deuxième réussite -> 6 jours', r.corps.intervalle === 6 && r.corps.facilite > 2.5);
r = await j('POST', `/review/${carteId}`, { qualite: 0 });
test('échec -> retour à 1 jour et un lapse', r.corps.intervalle === 1 && r.corps.lapses === 1 && r.corps.facilite < 2.65);
r = await j('POST', `/review/${carteId}`, { qualite: 9 });
test('qualité hors bornes rejetée', r.statut === 400);
queue = (await j('GET', '/review/queue')).corps;
test('la carte révisée quitte la file', queue.length === 1);

// --- recherche
r = await j('GET', '/search?q=gradient');
test('GET /search — fiches', r.corps.fiches.length > 0, JSON.stringify(r.corps.fiches.map((f) => f.slug)));
r = await j('GET', '/search?q=reformulation');
test('GET /search — notes (FTS5)', r.corps.notes.length === 1, JSON.stringify(r.corps.notes));
r = await j('GET', '/search?q=memoire');
test('recherche insensible aux accents sur les fiches', r.corps.fiches.length > 0);
r = await j('GET', '/search?q=" OR 1=1 --');
test('requête FTS bancale ne casse pas', r.statut === 200);

// --- ressources
r = await j('POST', '/resources', { slug, type: 'livre', titre: 'ESL' });
test('POST /resources', r.statut === 200 && r.corps.id > 0);
r = await j('PUT', `/resources/${r.corps.id}`, { statut: 'en_cours' });
test('PUT /resources/:id', r.corps.statut === 'en_cours');
test('GET /resources', (await j('GET', '/resources')).corps.length === 1);

// --- exports
const zip = await fetch(B + '/export/markdown');
const buf = Buffer.from(await zip.arrayBuffer());
test('GET /export/markdown — zip valide', buf.subarray(0, 2).toString() === 'PK' && buf.length > 1000, `${buf.length} octets`);
const dump = (await j('GET', '/export/json')).corps;
test('GET /export/json — dump complet',
  dump.progress.length >= 2 && dump.note.length === 1 && dump.card.length === 2 && dump.review.length === 3 && dump.session.length === 1,
  JSON.stringify(Object.fromEntries(Object.entries(dump).map(([k, v]) => [k, Array.isArray(v) ? v.length : typeof v]))));
const html = (await j('GET', `/export/html?slug=${slug}`)).corps;
test('GET /export/html', html.startsWith('<!doctype html>') && html.includes('<article>'));

r = await j('POST', '/import/json', dump);
test('POST /import/json restaure le dump', r.statut === 200 && r.corps.importe.card === 2, JSON.stringify(r.corps));
test('après réimport, les paliers sont intacts', (await j('GET', `/concepts/${slug}`)).corps.progress.palier === 3);

r = await j('POST', '/reload');
test('POST /reload relit content/', r.corps.concepts > 0);

console.log(`\n${ok} ok, ${ko} ko`);
process.exit(ko ? 1 : 0);

// Un seul point de contact avec le serveur. Toute erreur remonte en exception
// avec le message renvoyé par l'API.
async function requete(url, options = {}) {
  const res = await fetch(`/api${url}`, {
    headers: options.body ? { 'Content-Type': 'application/json' } : undefined,
    ...options,
    body: options.body ? JSON.stringify(options.body) : undefined,
  });
  const type = res.headers.get('content-type') ?? '';
  const donnees = type.includes('application/json') ? await res.json() : await res.text();
  if (!res.ok) {
    const err = new Error(donnees?.erreur ?? `HTTP ${res.status}`);
    err.details = donnees;
    err.status = res.status;
    throw err;
  }
  return donnees;
}

export const api = {
  modules: () => requete('/modules'),
  concept: (slug) => requete(`/concepts/${slug}`),
  next: () => requete('/next'),
  stats: () => requete('/stats'),

  majProgress: (slug, corps) =>
    requete(`/progress/${slug}`, { method: 'PUT', body: corps }),
  creerSession: (corps) => requete('/sessions', { method: 'POST', body: corps }),

  note: (slug) => requete(`/notes/${slug}`),
  majNote: (slug, corps) => requete(`/notes/${slug}`, { method: 'PUT', body: corps }),

  file: () => requete('/review/queue'),
  reviser: (id, corps) => requete(`/review/${id}`, { method: 'POST', body: corps }),
  creerCarte: (corps) => requete('/cards', { method: 'POST', body: corps }),
  genererCarte: (slug) => requete(`/cards/generer/${slug}`, { method: 'POST' }),
  supprimerCarte: (id) => requete(`/cards/${id}`, { method: 'DELETE' }),

  recherche: (q) => requete(`/search?q=${encodeURIComponent(q)}`),
  recharger: () => requete('/reload', { method: 'POST' }),
};

export const PALIERS = [
  { n: 0, nom: 'Rencontré', preuve: 'la fiche a été ouverte' },
  { n: 1, nom: 'Compris', preuve: 'une note dans « en mes mots »' },
  { n: 2, nom: 'Implémenté', preuve: 'le code refait sans regarder, collé dans la note' },
  { n: 3, nom: 'Appliqué', preuve: 'une session résumée : dataset, métrique, conclusion' },
  { n: 4, nom: 'Expliqué', preuve: 'une carte réussie après 21 jours' },
];

export const QUALITES = [
  { n: 0, nom: 'Raté', touche: '1' },
  { n: 1, nom: 'Difficile', touche: '2' },
  { n: 2, nom: 'Correct', touche: '3' },
  { n: 3, nom: 'Facile', touche: '4' },
];

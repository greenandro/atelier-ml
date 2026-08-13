import { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../api.js';

export default function Search() {
  const [q, setQ] = useState('');
  const [res, setRes] = useState(null);
  const [cherche, setCherche] = useState(false);
  const champ = useRef(null);

  useEffect(() => champ.current?.focus(), []);

  useEffect(() => {
    if (q.trim().length < 2) {
      setRes(null);
      return;
    }
    setCherche(true);
    const t = setTimeout(() => {
      api
        .recherche(q.trim())
        .then(setRes)
        .finally(() => setCherche(false));
    }, 200);
    return () => clearTimeout(t);
  }, [q]);

  return (
    <div className="page recherche">
      <h1>Recherche</h1>
      <input
        ref={champ}
        className="champ-recherche"
        value={q}
        placeholder="Un mot dans les fiches ou dans tes notes…"
        onChange={(e) => setQ(e.target.value)}
      />

      {!res && !cherche && (
        <p className="aide">
          Les fiches sont filtrées en mémoire, les notes passent par l'index FTS5.
        </p>
      )}

      {res && (
        <>
          <section className="bloc">
            <h2>Fiches ({res.fiches.length})</h2>
            {!res.fiches.length && <p className="vide">aucune fiche</p>}
            <ul className="resultats">
              {res.fiches.map((f) => (
                <li key={f.slug}>
                  <Link to={`/concept/${f.slug}`}>
                    <b>{f.titre}</b>
                    <span className="module">{f.moduleTitre}</span>
                  </Link>
                  <p className="extrait">{surligner(f.extrait, res.q)}</p>
                </li>
              ))}
            </ul>
          </section>

          <section className="bloc">
            <h2>Mes notes ({res.notes.length})</h2>
            {!res.notes.length && <p className="vide">aucune note</p>}
            <ul className="resultats">
              {res.notes.map((n) => (
                <li key={n.id}>
                  {n.slug ? (
                    <Link to={`/concept/${n.slug}`}>
                      <b>{n.conceptTitre}</b>
                      <span className="module">note</span>
                    </Link>
                  ) : (
                    <b>{n.titre}</b>
                  )}
                  <p className="extrait">{n.extrait}</p>
                </li>
              ))}
            </ul>
          </section>
        </>
      )}
    </div>
  );
}

// Surlignage simple, insensible aux accents comme la recherche côté serveur.
function surligner(texte, q) {
  const norm = (s) => s.toLowerCase().normalize('NFD').replace(/[̀-ͯ]/g, '');
  const i = norm(texte).indexOf(norm(q));
  if (i === -1) return texte;
  return (
    <>
      {texte.slice(0, i)}
      <mark>{texte.slice(i, i + q.length)}</mark>
      {texte.slice(i + q.length)}
    </>
  );
}

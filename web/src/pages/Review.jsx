import { useEffect, useState, useCallback } from 'react';
import { Link } from 'react-router-dom';
import { api, QUALITES } from '../api.js';
import Markdown from '../components/Markdown.jsx';

export default function Review({ onChange }) {
  const [file, setFile] = useState(null);
  const [i, setI] = useState(0);
  const [retourne, setRetourne] = useState(false);
  const [depuis, setDepuis] = useState(Date.now());
  const [bilan, setBilan] = useState({ vues: 0, rates: 0 });

  useEffect(() => {
    api.file().then((f) => {
      setFile(f);
      onChange?.(f.length);
    });
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const carte = file?.[i];

  const repondre = useCallback(
    async (qualite) => {
      if (!carte) return;
      await api.reviser(carte.id, { qualite, ms: Date.now() - depuis });
      setBilan((b) => ({ vues: b.vues + 1, rates: b.rates + (qualite === 0 ? 1 : 0) }));
      setRetourne(false);
      setDepuis(Date.now());
      setI((n) => n + 1);
      onChange?.(Math.max(0, file.length - i - 1));
    },
    [carte, depuis, file, i, onChange]
  );

  useEffect(() => {
    const sur = (e) => {
      if (/input|textarea/i.test(e.target.tagName)) return;
      if (e.key === ' ' || e.key === 'Enter') {
        e.preventDefault();
        setRetourne(true);
      } else if (retourne && ['1', '2', '3', '4'].includes(e.key)) {
        repondre(Number(e.key) - 1);
      }
    };
    window.addEventListener('keydown', sur);
    return () => window.removeEventListener('keydown', sur);
  }, [retourne, repondre]);

  if (!file) return <p className="vide">chargement…</p>;

  if (!carte) {
    return (
      <div className="page revision">
        <h1>Révisions</h1>
        <div className="bloc fini">
          {bilan.vues > 0 ? (
            <>
              <p className="grand">File vide.</p>
              <p>
                {bilan.vues} carte(s) révisée(s), {bilan.rates} ratée(s). Les
                cartes ratées reviennent demain.
              </p>
            </>
          ) : (
            <>
              <p className="grand">Rien à réviser aujourd'hui.</p>
              <p>
                Les cartes se créent depuis une fiche : sélectionne un passage, ou
                génère la carte « en une phrase ».
              </p>
            </>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="page revision">
      <div className="progression-revision">
        <span>
          {i + 1} / {file.length}
        </span>
        <div className="jauge">
          <i style={{ width: `${(i / file.length) * 100}%` }} />
        </div>
        {carte.retard > 0 && <span className="retard">{carte.retard} j de retard</span>}
      </div>

      <div className="carte-revision" onClick={() => setRetourne(true)}>
        <div className="recto">
          <Markdown>{carte.recto_md}</Markdown>
        </div>
        {retourne ? (
          <div className="verso">
            <Markdown>{carte.verso_md}</Markdown>
          </div>
        ) : (
          <p className="indice">Espace pour retourner</p>
        )}
      </div>

      {retourne && (
        <div className="qualites">
          {QUALITES.map((q) => (
            <button key={q.n} className={`q${q.n}`} onClick={() => repondre(q.n)}>
              <b>{q.nom}</b>
              <span>{q.touche}</span>
            </button>
          ))}
        </div>
      )}

      <p className="source">
        <Link to={`/concept/${carte.slug}`}>{carte.titre}</Link> · intervalle actuel{' '}
        {carte.intervalle} j · facilité {carte.facilite.toFixed(2)}
      </p>
    </div>
  );
}

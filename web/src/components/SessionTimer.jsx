import { useEffect, useRef, useState } from 'react';
import { api } from '../api.js';

const mmss = (s) =>
  `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;

// Minuteur d'étude. À l'arrêt, il réclame une difficulté ressentie et un
// résumé : c'est ce résumé qui fait la preuve du palier 3.
export default function SessionTimer({ slug, onSession }) {
  const [secondes, setSecondes] = useState(0);
  const [encours, setEncours] = useState(false);
  const [debut, setDebut] = useState(null);
  const [bilan, setBilan] = useState(null);
  const [ressentie, setRessentie] = useState(null);
  const [resume, setResume] = useState('');
  const tic = useRef(null);

  useEffect(() => {
    if (encours) {
      tic.current = setInterval(() => setSecondes((s) => s + 1), 1000);
      return () => clearInterval(tic.current);
    }
  }, [encours]);

  useEffect(() => {
    // Changer de concept remet le minuteur à zéro.
    setSecondes(0);
    setEncours(false);
    setBilan(null);
    setDebut(null);
  }, [slug]);

  const demarrer = () => {
    if (!debut) setDebut(new Date().toISOString());
    setEncours(true);
  };

  const arreter = () => {
    setEncours(false);
    setBilan(Math.max(1, Math.round(secondes / 60)));
  };

  const enregistrer = async () => {
    const session = await api.creerSession({
      slug,
      minutes: bilan,
      difficulte_ressentie: ressentie,
      resume,
      debut,
    });
    onSession?.(session);
    setSecondes(0);
    setBilan(null);
    setResume('');
    setRessentie(null);
    setDebut(null);
  };

  if (bilan !== null) {
    return (
      <div className="minuteur bilan">
        <p>
          <b>{bilan} min</b> d'étude. Deux questions avant d'enregistrer.
        </p>
        <label className="champ">
          <span>Difficulté ressentie</span>
          <div className="choix">
            {[1, 2, 3, 4, 5].map((n) => (
              <button
                key={n}
                className={ressentie === n ? 'actif' : ''}
                onClick={() => setRessentie(n)}
              >
                {n}
              </button>
            ))}
          </div>
        </label>
        <label className="champ">
          <span>Résumé — dataset, métrique, conclusion (preuve du palier 3)</span>
          <textarea
            rows={3}
            value={resume}
            placeholder="Ce que j'ai fait, ce que j'ai obtenu, ce qui reste flou."
            onChange={(e) => setResume(e.target.value)}
          />
        </label>
        <div className="actions">
          <button className="primaire" onClick={enregistrer}>
            Enregistrer la session
          </button>
          <button className="lien" onClick={() => setBilan(null)}>
            Reprendre
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="minuteur">
      <span className={`compteur ${encours ? 'actif' : ''}`}>{mmss(secondes)}</span>
      {!encours ? (
        <button onClick={demarrer}>{secondes ? 'Reprendre' : 'Démarrer'}</button>
      ) : (
        <button onClick={() => setEncours(false)}>Pause</button>
      )}
      <button className="lien" disabled={!secondes} onClick={arreter}>
        Terminer
      </button>
    </div>
  );
}

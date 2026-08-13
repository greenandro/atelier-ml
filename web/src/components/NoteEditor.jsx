import { useEffect, useRef, useState } from 'react';
import { api } from '../api.js';
import Markdown from './Markdown.jsx';

// Deux champs : « en mes mots » (preuve du palier 1) et les notes libres.
// Autosauvegarde 2 secondes après la dernière frappe, comme prévu au modèle.
export default function NoteEditor({ slug, note, onSauvegarde }) {
  const [enMesMots, setEnMesMots] = useState(note?.en_mes_mots ?? '');
  const [corps, setCorps] = useState(note?.corps_md ?? '');
  const [etat, setEtat] = useState('sauvegardé');
  const [apercu, setApercu] = useState(false);
  const minuteur = useRef(null);
  const premier = useRef(true);

  useEffect(() => {
    setEnMesMots(note?.en_mes_mots ?? '');
    setCorps(note?.corps_md ?? '');
    setEtat('sauvegardé');
    premier.current = true;
  }, [slug]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (premier.current) {
      premier.current = false;
      return;
    }
    setEtat('modifié');
    clearTimeout(minuteur.current);
    minuteur.current = setTimeout(async () => {
      setEtat('sauvegarde…');
      try {
        const maj = await api.majNote(slug, { en_mes_mots: enMesMots, corps_md: corps });
        setEtat('sauvegardé');
        onSauvegarde?.(maj);
      } catch (e) {
        setEtat(`échec : ${e.message}`);
      }
    }, 2000);
    return () => clearTimeout(minuteur.current);
  }, [enMesMots, corps]); // eslint-disable-line react-hooks/exhaustive-deps

  // Ceinture et bretelles : on n'attend pas les 2 s si l'onglet se ferme.
  useEffect(() => {
    const sortie = () => {
      if (etat === 'modifié') {
        navigator.sendBeacon?.(
          `/api/notes/${slug}`,
          new Blob([JSON.stringify({ en_mes_mots: enMesMots, corps_md: corps })], {
            type: 'application/json',
          })
        );
      }
    };
    window.addEventListener('pagehide', sortie);
    return () => window.removeEventListener('pagehide', sortie);
  }, [etat, enMesMots, corps, slug]);

  return (
    <section className="bloc notes">
      <header className="entete-notes">
        <h3>Mes notes</h3>
        <div className="actions">
          <button className="lien" onClick={() => setApercu(!apercu)}>
            {apercu ? 'Éditer' : 'Aperçu'}
          </button>
          <span className={`etat ${etat.startsWith('échec') ? 'erreur' : ''}`}>{etat}</span>
        </div>
      </header>

      <label className="champ">
        <span>
          En mes mots <em>— preuve du palier 1</em>
        </span>
        {apercu ? (
          <Markdown className="apercu">{enMesMots || '_(vide)_'}</Markdown>
        ) : (
          <textarea
            rows={4}
            value={enMesMots}
            placeholder="Explique le concept comme si personne ne t'avait rien dit. Si tu n'y arrives pas, tu ne l'as pas compris."
            onChange={(e) => setEnMesMots(e.target.value)}
          />
        )}
      </label>

      <label className="champ">
        <span>
          Notes <em>— Markdown et LaTeX ($x^2$), le code y sert de preuve du palier 2</em>
        </span>
        {apercu ? (
          <Markdown className="apercu">{corps || '_(vide)_'}</Markdown>
        ) : (
          <textarea
            rows={14}
            value={corps}
            placeholder={'Dérivations, code refait de mémoire, pièges rencontrés…\n\n```python\n# ton implémentation from scratch\n```'}
            onChange={(e) => setCorps(e.target.value)}
          />
        )}
      </label>
    </section>
  );
}

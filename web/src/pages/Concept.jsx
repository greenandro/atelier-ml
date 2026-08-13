import { useCallback, useEffect, useRef, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { api, PALIERS } from '../api.js';
import Markdown from '../components/Markdown.jsx';
import MasteryBadge from '../components/MasteryBadge.jsx';
import DifficultyBar from '../components/DifficultyBar.jsx';
import NoteEditor from '../components/NoteEditor.jsx';
import SessionTimer from '../components/SessionTimer.jsx';

const BLOCS = [
  ['explication', 'Explication'],
  ['casUsage', "Cas d'utilisation"],
  ['algorithme', 'Algorithme'],
  ['implementation', 'Implémentation'],
  ['outils', 'Outils'],
  ['alternatives', 'Alternatives open-source'],
  ['astuces', 'Astuces'],
];

export default function Concept({ onChange }) {
  const { slug } = useParams();
  const [c, setC] = useState(null);
  const [erreur, setErreur] = useState(null);
  const [refus, setRefus] = useState(null);
  const [selection, setSelection] = useState('');
  const fiche = useRef(null);

  const charger = useCallback(
    () => api.concept(slug).then(setC).catch((e) => setErreur(e.message)),
    [slug]
  );

  useEffect(() => {
    setC(null);
    setErreur(null);
    setRefus(null);
    charger();
    window.scrollTo(0, 0);
  }, [charger]);

  // Sélectionner du texte dans la fiche propose d'en faire une carte.
  useEffect(() => {
    const sur = () => {
      const s = window.getSelection();
      const texte = s?.toString().trim() ?? '';
      const dansLaFiche = s?.anchorNode && fiche.current?.contains(s.anchorNode);
      setSelection(dansLaFiche && texte.length > 8 ? texte : '');
    };
    document.addEventListener('mouseup', sur);
    document.addEventListener('keyup', sur);
    return () => {
      document.removeEventListener('mouseup', sur);
      document.removeEventListener('keyup', sur);
    };
  }, []);

  if (erreur) return <p className="vide erreur">{erreur}</p>;
  if (!c) return <p className="vide">chargement…</p>;

  const palier = c.progress?.palier ?? 0;

  const changerPalier = async (n) => {
    setRefus(null);
    try {
      await api.majProgress(slug, { palier: n });
      await charger();
      onChange?.();
    } catch (e) {
      if (e.status === 422) setRefus(e.details);
      else setErreur(e.message);
    }
  };

  const changerRessentie = async (n) => {
    await api.majProgress(slug, { palier, difficulte_ressentie: n });
    charger();
  };

  const carteDepuisSelection = async () => {
    const verso = selection;
    const recto = window.prompt(
      'Question au recto de la carte :',
      `${c.titre} — ?`
    );
    if (!recto) return;
    await api.creerCarte({ slug, recto_md: recto, verso_md: verso });
    window.getSelection()?.removeAllRanges();
    setSelection('');
    charger();
  };

  return (
    <div className="page concept">
      <nav className="fil">
        <Link to={`/module/${c.module}`}>{c.moduleTitre}</Link>
        <span>›</span>
        <b>{c.titre}</b>
      </nav>

      <header className="entete-concept">
        <h1>{c.titre}</h1>
        {c.blocs.phrase && <p className="phrase-cle">{c.blocs.phrase}</p>}
      </header>

      <div className="colonnes">
        <div className="principal" ref={fiche}>
          {BLOCS.map(([cle, titre]) =>
            c.blocs[cle] ? (
              <section className={`bloc fiche ${cle}`} key={cle}>
                <h2>{titre}</h2>
                <Markdown>{c.blocs[cle]}</Markdown>
              </section>
            ) : null
          )}

          <NoteEditor slug={slug} note={c.note} onSauvegarde={() => charger()} />

          <section className="bloc">
            <h2>Cartes ({c.cartes.length})</h2>
            <p className="aide">
              Sélectionne un passage de la fiche pour en faire une carte, ou génère
              celle du « en une phrase ».
            </p>
            <ul className="cartes">
              {c.cartes.map((carte) => (
                <li key={carte.id}>
                  <div>
                    <b>{carte.recto_md}</b>
                    <Markdown className="verso">{carte.verso_md}</Markdown>
                    <span className="meta">
                      intervalle {carte.intervalle} j · facilité{' '}
                      {carte.facilite.toFixed(2)} · échéance {carte.echeance}
                      {carte.lapses > 0 && ` · ${carte.lapses} oubli(s)`}
                    </span>
                  </div>
                  <button
                    className="lien"
                    onClick={() => api.supprimerCarte(carte.id).then(charger)}
                  >
                    supprimer
                  </button>
                </li>
              ))}
            </ul>
            <button className="lien" onClick={() => api.genererCarte(slug).then(charger)}>
              + Générer la carte « {c.titre} »
            </button>
          </section>

          {c.sessions.length > 0 && (
            <section className="bloc">
              <h2>Sessions ({c.sessions.length})</h2>
              <ul className="sessions">
                {c.sessions.map((s) => (
                  <li key={s.id}>
                    <span className="date">{s.debut.slice(0, 10)}</span>
                    <span className="minutes">{s.minutes} min</span>
                    {s.difficulte_ressentie && (
                      <span className="ressentie">ressenti {s.difficulte_ressentie}/5</span>
                    )}
                    <span className="resume">{s.resume || <em>sans résumé</em>}</span>
                  </li>
                ))}
              </ul>
            </section>
          )}
        </div>

        <aside className="lateral">
          <div className="bloc encart">
            <h3>Palier de maîtrise</h3>
            <div className="paliers-choix">
              {PALIERS.map((p) => (
                <button
                  key={p.n}
                  className={`p${p.n} ${palier === p.n ? 'actif' : ''} ${
                    c.preuves[p.n] === false && p.n > palier ? 'bloque' : ''
                  }`}
                  onClick={() => changerPalier(p.n)}
                  title={p.preuve}
                >
                  <b>{p.n}</b>
                  <span>{p.nom}</span>
                  {p.n > 0 && (
                    <i className={c.preuves[p.n] ? 'preuve ok' : 'preuve'}>
                      {c.preuves[p.n] ? '✓' : '·'}
                    </i>
                  )}
                </button>
              ))}
            </div>
            {refus && (
              <p className="refus">
                Palier {refus.manquant} refusé. Preuve exigée : {refus.exigence}.
              </p>
            )}
            <p className="aide">
              Le palier ne se décrète pas : chaque montée demande une preuve
              enregistrée ici.
            </p>
          </div>

          <div className="bloc encart">
            <h3>Difficulté</h3>
            <DifficultyBar
              theorique={c.difficulte}
              ressentie={c.progress?.difficulte_ressentie ?? null}
              onChange={changerRessentie}
            />
          </div>

          <div className="bloc encart">
            <h3>Session d'étude</h3>
            <SessionTimer slug={slug} onSession={() => charger()} />
            {c.progress?.minutes_total > 0 && (
              <p className="aide">{c.progress.minutes_total} min au total sur ce concept.</p>
            )}
          </div>

          {c.prereqsDetail.length > 0 && (
            <div className="bloc encart">
              <h3>Prérequis</h3>
              <ul className="liens-concepts">
                {c.prereqsDetail.map((p) => (
                  <li key={p.slug}>
                    <MasteryBadge palier={p.palier} texte={false} taille="s" />
                    {p.connu ? (
                      <Link to={`/concept/${p.slug}`}>{p.titre}</Link>
                    ) : (
                      <span className="inconnu">{p.slug} (absent du contenu)</span>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {c.suivants.length > 0 && (
            <div className="bloc encart">
              <h3>Débloque</h3>
              <ul className="liens-concepts">
                {c.suivants.map((s) => (
                  <li key={s.slug}>
                    <Link to={`/concept/${s.slug}`}>{s.titre}</Link>
                  </li>
                ))}
              </ul>
            </div>
          )}

          <div className="bloc encart">
            <a className="lien" href={`/api/export/html?slug=${slug}`} target="_blank" rel="noreferrer">
              Imprimer cette fiche
            </a>
          </div>
        </aside>
      </div>

      {selection && (
        <div className="selection-flottante">
          <span>{selection.slice(0, 90)}…</span>
          <button className="primaire" onClick={carteDepuisSelection}>
            En faire une carte
          </button>
        </div>
      )}
    </div>
  );
}

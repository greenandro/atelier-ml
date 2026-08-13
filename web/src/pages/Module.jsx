import { Link, useParams } from 'react-router-dom';
import MasteryBadge from '../components/MasteryBadge.jsx';

export default function Module({ modules }) {
  const { module: nom } = useParams();
  const m = modules.find((x) => x.module === nom);

  if (!modules.length) return <p className="vide">chargement…</p>;
  if (!m) return <p className="vide">Module inconnu : {nom}</p>;

  const acquis = m.concepts.filter((c) => c.palier >= 2).length;

  return (
    <div className="page module">
      <h1>
        <span className="num-module">{String(m.ordre).padStart(2, '0')}</span>
        {m.titre}
      </h1>
      {m.resume && <p className="resume">{m.resume}</p>}
      <p className="meta">
        {m.concepts.length} concepts · {acquis} au palier ≥ 2 ·{' '}
        {Math.round(m.concepts.reduce((s, c) => s + c.minutes_total, 0) / 60)} h passées
      </p>

      <ul className="liste-concepts">
        {m.concepts.map((c) => (
          <li key={c.slug}>
            <Link to={`/concept/${c.slug}`}>
              <div className="tete">
                <MasteryBadge palier={c.palier} texte={false} />
                <h3>{c.titre}</h3>
                <span
                  className={`diff d${c.difficulte}`}
                  title={`Difficulté théorique ${c.difficulte}/5`}
                >
                  {'●'.repeat(c.difficulte)}
                  <span className="creux">{'●'.repeat(5 - c.difficulte)}</span>
                </span>
              </div>
              <p className="phrase">{c.phrase}</p>
              {c.prereqs.length > 0 && (
                <p className="prereqs">prérequis : {c.prereqs.join(', ')}</p>
              )}
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
}

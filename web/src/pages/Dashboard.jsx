import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  LineChart,
  Line,
} from 'recharts';
import { api } from '../api.js';
import Heatmap from '../components/Heatmap.jsx';
import MasteryBadge from '../components/MasteryBadge.jsx';

export default function Dashboard() {
  const [stats, setStats] = useState(null);
  const [file, setFile] = useState([]);
  const [erreur, setErreur] = useState(null);

  useEffect(() => {
    Promise.all([api.stats(), api.next()])
      .then(([s, f]) => {
        setStats(s);
        setFile(f);
      })
      .catch((e) => setErreur(e.message));
  }, []);

  if (erreur) return <p className="vide erreur">{erreur}</p>;
  if (!stats) return <p className="vide">chargement…</p>;

  const t = stats.totaux;
  const derniersJours = completerJours(stats.jours, 30);

  return (
    <div className="page dashboard">
      <h1>Tableau de bord</h1>

      <div className="chiffres">
        <Chiffre valeur={`${Math.round(t.minutes / 60)} h`} libelle="temps d'étude" />
        <Chiffre valeur={`${t.touches}/${t.concepts}`} libelle="concepts abordés" />
        <Chiffre valeur={t.serie} libelle={t.serie > 1 ? 'jours d’affilée' : 'jour d’affilée'} />
        <Chiffre valeur={t.cartesDues} libelle="cartes à réviser" alerte={t.cartesDues > 0} />
      </div>

      <section className="bloc">
        <h2>La file du jour</h2>
        {!file.length && (
          <p className="vide">
            Rien d'urgent. Ouvre un module et commence par le premier concept non vu.
          </p>
        )}
        <ol className="file">
          {file.map((item, i) => (
            <li key={i} className={`item ${item.type}`}>
              <span className="etiquette">{ETIQUETTES[item.type] ?? item.type}</span>
              {item.type === 'revision' ? (
                <Link to="/revisions" className="titre">
                  {item.n} carte(s) en retard — jusqu'à {item.retardMax} jours
                </Link>
              ) : (
                <Link to={`/concept/${item.slug}`} className="titre">
                  {item.titre}
                </Link>
              )}
              <span className="raison">{item.raison}</span>
              {item.palier !== undefined && <MasteryBadge palier={item.palier} texte={false} taille="s" />}
            </li>
          ))}
        </ol>
      </section>

      <div className="deux-colonnes">
        <section className="bloc">
          <h2>Maîtrise par module</h2>
          <ResponsiveContainer width="100%" height={Math.max(180, stats.parModule.length * 34)}>
            <BarChart data={stats.parModule} layout="vertical" margin={{ left: 8, right: 24 }}>
              <CartesianGrid strokeDasharray="2 4" stroke="var(--trait)" horizontal={false} />
              <XAxis type="number" domain={[0, 100]} unit="%" stroke="var(--gris)" fontSize={12} />
              <YAxis
                type="category"
                dataKey="titre"
                width={140}
                stroke="var(--gris)"
                fontSize={12}
                tickLine={false}
              />
              <Tooltip
                contentStyle={TOOLTIP}
                formatter={(v, n, p) => [
                  `${v} % — ${p.payload.paliers.slice(2).reduce((a, b) => a + b, 0)}/${p.payload.total} au palier ≥ 2`,
                  'maîtrise',
                ]}
              />
              <Bar dataKey="maitrise" fill="var(--accent)" radius={[0, 3, 3, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </section>

        <section className="bloc">
          <h2>Minutes par jour (30 derniers jours)</h2>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={derniersJours} margin={{ left: 0, right: 12, top: 8 }}>
              <CartesianGrid strokeDasharray="2 4" stroke="var(--trait)" />
              <XAxis
                dataKey="date"
                tickFormatter={(d) => d.slice(8)}
                stroke="var(--gris)"
                fontSize={12}
              />
              <YAxis stroke="var(--gris)" fontSize={12} width={32} />
              <Tooltip contentStyle={TOOLTIP} formatter={(v) => [`${v} min`, 'étude']} />
              <Line
                type="monotone"
                dataKey="minutes"
                stroke="var(--accent)"
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </section>
      </div>

      <section className="bloc">
        <h2>Jours actifs</h2>
        <Heatmap jours={stats.jours} />
      </section>

      <div className="deux-colonnes">
        <section className="bloc">
          <h2>Répartition des paliers</h2>
          <div className="paliers-barre">
            {stats.paliers.map((n, i) => (
              <div
                key={i}
                className={`part p${i}`}
                style={{ flexGrow: Math.max(n, 0.02) }}
                title={`${n} concept(s) au palier ${i}`}
              >
                {n > 0 && <span>{n}</span>}
              </div>
            ))}
          </div>
          <ul className="legende-paliers">
            {stats.paliers.map((n, i) => (
              <li key={i}>
                <MasteryBadge palier={i} taille="s" /> <b>{n}</b>
              </li>
            ))}
          </ul>
        </section>

        <section className="bloc">
          <h2>Écarts de difficulté</h2>
          {!stats.ecarts.length && (
            <p className="vide">
              Aucune difficulté ressentie saisie. C'est elle qui révèle les lacunes.
            </p>
          )}
          <ul className="ecarts">
            {stats.ecarts.slice(0, 8).map((e) => (
              <li key={e.slug} className={e.ecart >= 2 ? 'alerte' : ''}>
                <Link to={`/concept/${e.slug}`}>{e.titre}</Link>
                <span className="chiffre">
                  {e.difficulte} → {e.ressentie}
                </span>
                <span className="delta">
                  {e.ecart > 0 ? `+${e.ecart}` : e.ecart}
                </span>
              </li>
            ))}
          </ul>
        </section>
      </div>
    </div>
  );
}

const ETIQUETTES = {
  revision: 'dette',
  lacune: 'lacune',
  a_implementer: 'à implémenter',
  nouveau: 'nouveau',
};

const TOOLTIP = {
  background: 'var(--fond-2)',
  border: '1px solid var(--trait)',
  borderRadius: 6,
  fontSize: 13,
};

function Chiffre({ valeur, libelle, alerte }) {
  return (
    <div className={`chiffre-cle ${alerte ? 'alerte' : ''}`}>
      <b>{valeur}</b>
      <span>{libelle}</span>
    </div>
  );
}

// Recharts a besoin des jours vides pour que la ligne ne mente pas.
function completerJours(jours, n) {
  const parDate = Object.fromEntries(jours.map((j) => [j.date, j.minutes]));
  const out = [];
  for (let i = n - 1; i >= 0; i--) {
    const d = new Date(Date.now() - i * 86400000).toISOString().slice(0, 10);
    out.push({ date: d, minutes: parDate[d] ?? 0 });
  }
  return out;
}

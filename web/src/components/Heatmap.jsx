import { useMemo } from 'react';

const JOUR = 86400000;
const MOIS = ['jan', 'fév', 'mar', 'avr', 'mai', 'juin', 'juil', 'août', 'sep', 'oct', 'nov', 'déc'];
const iso = (d) => d.toISOString().slice(0, 10);

// Un an de jours actifs, une case par jour, semaines en colonnes.
export default function Heatmap({ jours = [], semaines = 53 }) {
  const { colonnes, max, total } = useMemo(() => {
    const parDate = Object.fromEntries(jours.map((j) => [j.date, j.minutes]));
    const max = Math.max(60, ...jours.map((j) => j.minutes));

    const fin = new Date();
    fin.setHours(12, 0, 0, 0);
    // On termine la grille sur un samedi complet.
    fin.setTime(fin.getTime() + (6 - fin.getDay()) * JOUR);
    const debut = new Date(fin.getTime() - (semaines * 7 - 1) * JOUR);

    const colonnes = [];
    for (let s = 0; s < semaines; s++) {
      const cases = [];
      for (let j = 0; j < 7; j++) {
        const d = new Date(debut.getTime() + (s * 7 + j) * JOUR);
        const date = iso(d);
        cases.push({ date, minutes: parDate[date] ?? 0, futur: d > new Date() });
      }
      colonnes.push({ cases, mois: cases[0].date.slice(5, 7) });
    }
    return {
      colonnes,
      max,
      total: jours.reduce((s, j) => s + j.minutes, 0),
    };
  }, [jours, semaines]);

  const niveau = (m) => (m === 0 ? 0 : m < max * 0.25 ? 1 : m < max * 0.5 ? 2 : m < max * 0.75 ? 3 : 4);

  return (
    <div className="heatmap">
      <div className="mois">
        {colonnes.map((c, i) => {
          const nouveau = i === 0 || c.mois !== colonnes[i - 1].mois;
          return (
            <span key={i} style={{ visibility: nouveau ? 'visible' : 'hidden' }}>
              {MOIS[Number(c.mois) - 1]}
            </span>
          );
        })}
      </div>
      <div className="grille">
        {colonnes.map((c, i) => (
          <div key={i} className="semaine">
            {c.cases.map((j) => (
              <i
                key={j.date}
                className={`n${niveau(j.minutes)} ${j.futur ? 'futur' : ''}`}
                title={`${j.date} — ${j.minutes} min`}
              />
            ))}
          </div>
        ))}
      </div>
      <div className="legende">
        <span>{Math.round(total / 60)} h sur l'année</span>
        <span className="echelle">
          moins
          {[0, 1, 2, 3, 4].map((n) => (
            <i key={n} className={`n${n}`} />
          ))}
          plus
        </span>
      </div>
    </div>
  );
}

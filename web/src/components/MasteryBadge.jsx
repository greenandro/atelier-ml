import { PALIERS } from '../api.js';

// Le palier, toujours affiché de la même façon : un chiffre et une couleur.
export default function MasteryBadge({ palier = 0, texte = true, taille = 'm' }) {
  const p = PALIERS[palier] ?? PALIERS[0];
  return (
    <span
      className={`palier p${palier} t-${taille}`}
      title={`Palier ${palier} — ${p.nom} (${p.preuve})`}
    >
      <b>{palier}</b>
      {texte && <span>{p.nom}</span>}
    </span>
  );
}

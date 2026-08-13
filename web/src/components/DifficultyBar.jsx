// Difficulté théorique (celle de la fiche) contre difficulté ressentie.
// L'écart entre les deux est l'information intéressante : c'est une lacune.
export default function DifficultyBar({ theorique, ressentie, onChange }) {
  const ecart = ressentie ? ressentie - theorique : 0;

  return (
    <div className="difficulte">
      <div className="ligne">
        <span className="lib">Théorique</span>
        <div className="segments">
          {[1, 2, 3, 4, 5].map((n) => (
            <i key={n} className={n <= theorique ? 'plein' : ''} />
          ))}
        </div>
        <span className="val">{theorique}/5</span>
      </div>

      <div className="ligne">
        <span className="lib">Ressentie</span>
        <div className="segments cliquable">
          {[1, 2, 3, 4, 5].map((n) => (
            <i
              key={n}
              className={ressentie && n <= ressentie ? 'plein res' : ''}
              onClick={() => onChange?.(n === ressentie ? null : n)}
              title={`Marquer ${n}/5`}
            />
          ))}
        </div>
        <span className="val">{ressentie ? `${ressentie}/5` : '—'}</span>
      </div>

      {ressentie && ecart !== 0 && (
        <p className={`ecart ${ecart >= 2 ? 'alerte' : ''}`}>
          {ecart > 0
            ? `+${ecart} : plus dur qu'annoncé${ecart >= 2 ? ' — un prérequis manque probablement' : ''}`
            : `${ecart} : plus facile qu'annoncé`}
        </p>
      )}
    </div>
  );
}

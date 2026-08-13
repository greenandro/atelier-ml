// Répétition espacée : SM-2 réduit à quatre qualités de rappel.
//   0 raté · 1 difficile · 2 correct · 3 facile
export function planifier({ intervalle, facilite, lapses }, qualite) {
  if (qualite === 0) {
    return {
      intervalle: 1,
      facilite: Math.max(1.3, facilite - 0.2),
      lapses: lapses + 1,
    };
  }
  const f = Math.min(
    2.8,
    facilite + (qualite === 3 ? 0.15 : qualite === 2 ? 0 : -0.15)
  );
  const i =
    intervalle === 0 ? 1 : intervalle === 1 ? 6 : Math.round(intervalle * f);
  return { intervalle: i, facilite: f, lapses };
}

export const echeance = (jours) =>
  new Date(Date.now() + jours * 86400000).toISOString().slice(0, 10);

export const aujourdhui = () => new Date().toISOString().slice(0, 10);

// Nombre de jours de retard d'une carte (0 si elle n'est pas encore due).
export const retard = (dateEcheance, ref = aujourdhui()) =>
  Math.max(0, Math.round((Date.parse(ref) - Date.parse(dateEcheance)) / 86400000));

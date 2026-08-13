import { useEffect, useState } from 'react';
import { Routes, Route, NavLink, Link, useNavigate } from 'react-router-dom';
import { api } from './api.js';
import Dashboard from './pages/Dashboard.jsx';
import Module from './pages/Module.jsx';
import Concept from './pages/Concept.jsx';
import Review from './pages/Review.jsx';
import Search from './pages/Search.jsx';

export default function App() {
  const [modules, setModules] = useState([]);
  const [dues, setDues] = useState(0);
  const [erreur, setErreur] = useState(null);
  const navigate = useNavigate();

  const recharger = () =>
    api
      .modules()
      .then(setModules)
      .catch((e) => setErreur(e.message));

  useEffect(() => {
    recharger();
    api.file().then((f) => setDues(f.length)).catch(() => {});
  }, []);

  // Raccourcis : / pour la recherche, r pour les révisions.
  useEffect(() => {
    const sur = (e) => {
      const dansUnChamp = /input|textarea/i.test(e.target.tagName);
      if (dansUnChamp || e.metaKey || e.ctrlKey) return;
      if (e.key === '/') {
        e.preventDefault();
        navigate('/recherche');
      } else if (e.key === 'r') {
        navigate('/revisions');
      }
    };
    window.addEventListener('keydown', sur);
    return () => window.removeEventListener('keydown', sur);
  }, [navigate]);

  return (
    <div className="app">
      <aside className="barre">
        <Link to="/" className="marque">
          Atelier<span>ML</span>
        </Link>

        <nav className="nav-principale">
          <NavLink to="/" end>
            Tableau de bord
          </NavLink>
          <NavLink to="/revisions">
            Révisions
            {dues > 0 && <span className="pastille">{dues}</span>}
          </NavLink>
          <NavLink to="/recherche">Recherche</NavLink>
        </nav>

        <div className="titre-section">Modules</div>
        <nav className="nav-modules">
          {modules.map((m) => {
            const acquis = m.concepts.filter((c) => c.palier >= 2).length;
            return (
              <NavLink key={m.module} to={`/module/${m.module}`}>
                <span className="num">{String(m.ordre).padStart(2, '0')}</span>
                <span className="lib">{m.titre}</span>
                <span className="compte">
                  {acquis}/{m.concepts.length}
                </span>
              </NavLink>
            );
          })}
          {!modules.length && !erreur && <p className="vide">chargement…</p>}
          {erreur && <p className="vide erreur">API injoignable : {erreur}</p>}
        </nav>

        <div className="pied">
          <a href="/api/export/markdown">Export Markdown</a>
          <a href="/api/export/json">Export JSON</a>
          <button
            className="lien"
            onClick={() => api.recharger().then(recharger)}
            title="Relire les fichiers de content/"
          >
            Relire le contenu
          </button>
        </div>
      </aside>

      <main className="contenu">
        <Routes>
          <Route path="/" element={<Dashboard modules={modules} />} />
          <Route path="/module/:module" element={<Module modules={modules} />} />
          <Route path="/concept/:slug" element={<Concept onChange={recharger} />} />
          <Route path="/revisions" element={<Review onChange={setDues} />} />
          <Route path="/recherche" element={<Search />} />
          <Route path="*" element={<p className="vide">Page inconnue.</p>} />
        </Routes>
      </main>
    </div>
  );
}

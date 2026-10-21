import { createContext, useContext, useEffect, useState } from "react";
import { loadAnalytics } from "./analytics.js";
import ConsentBanner from "./components/ConsentBanner.jsx";
import DomainPicker from "./components/DomainPicker.jsx";
import { ErrorText, Loading } from "./components/ui.jsx";
import { DomainProvider, useDomain } from "./domain.jsx";
import { Link, usePath } from "./router.jsx";
import { SITE_NAME } from "./site.js";
import Admin from "./pages/Admin.jsx";
import Batch from "./pages/Batch.jsx";
import ChatIntake from "./pages/ChatIntake.jsx";
import Explain from "./pages/Explain.jsx";
import Fairness from "./pages/Fairness.jsx";
import Home from "./pages/Home.jsx";
import Intake from "./pages/Intake.jsx";
import NotFound from "./pages/NotFound.jsx";
import Privacy from "./pages/Privacy.jsx";
import Terms from "./pages/Terms.jsx";

const DraftContext = createContext(null);
export const useDraft = () => useContext(DraftContext);

const NAV = [
  ["/", "Overview"],
  ["/explain", "Explain"],
  ["/chat-intake", "Chat intake"],
  ["/intake", "Guided intake"],
  ["/batch", "Batch"],
  ["/fairness", "Model and fairness"],
];

const ROUTES = {
  "/": Home,
  "/explain": Explain,
  "/chat-intake": ChatIntake,
  "/intake": Intake,
  "/batch": Batch,
  "/fairness": Fairness,
  "/admin": Admin,
  "/privacy": Privacy,
  "/terms": Terms,
};

export default function App() {
  return (
    <DomainProvider>
      <Shell />
    </DomainProvider>
  );
}

function Shell() {
  const path = usePath();
  const [draft, setDraft] = useState(null);
  const [showConsent, setShowConsent] = useState(false);
  const { domain, error: domainError } = useDomain();
  useEffect(() => loadAnalytics(), []);
  useEffect(() => setDraft(null), [domain?.name]);
  const Page = ROUTES[path] || NotFound;
  return (
    <DraftContext.Provider value={{ draft, setDraft }}>
      <a className="skip-link" href="#main">Skip to content</a>
      <header className="site">
        <div className="container">
          <Link to="/" className="brand">
            <img src="/favicon.svg" alt="" width="28" height="28" />
            <span>{SITE_NAME}</span>
          </Link>
          <nav className="main" aria-label="Main">
            <ul>
              {NAV.map(([to, label]) => (
                <li key={to}>
                  <Link to={to} aria-current={path === to ? "page" : undefined}>{label}</Link>
                </li>
              ))}
            </ul>
          </nav>
          <DomainPicker />
        </div>
      </header>
      <main id="main" tabIndex={-1}>
        <div className="container">
          {domainError ? <ErrorText error={domainError} /> : !domain ? <Loading /> : <Page key={domain.name} />}
        </div>
      </main>
      <footer className="site">
        <div className="container">
          <p className="muted" style={{ margin: 0 }}>
            Research prototype. Risk estimates support human review; they are not decisions.
          </p>
          <ul>
            <li><Link to="/privacy">Privacy policy</Link></li>
            <li><Link to="/terms">Terms and conditions</Link></li>
            <li><Link to="/admin">Admin</Link></li>
            <li><button className="linklike" type="button" onClick={() => setShowConsent(true)}>Cookie and analytics settings</button></li>
          </ul>
        </div>
      </footer>
      <ConsentBanner forceOpen={showConsent} onClose={() => setShowConsent(false)} />
    </DraftContext.Provider>
  );
}

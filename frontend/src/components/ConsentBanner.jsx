import { useEffect, useState } from "react";
import { analyticsConfigured, getConsent, setConsent } from "../analytics.js";
import { Link } from "../router.jsx";

export default function ConsentBanner({ forceOpen, onClose }) {
  const [open, setOpen] = useState(false);
  useEffect(() => {
    if (forceOpen || getConsent() === null) setOpen(true);
  }, [forceOpen]);
  if (!open) return null;
  const choose = (v) => {
    setConsent(v);
    setOpen(false);
    onClose && onClose();
  };
  return (
    <div className="consent" role="dialog" aria-modal="false" aria-labelledby="consent-title">
      <h2 id="consent-title" style={{ fontSize: "1.05rem" }}>Analytics and storage</h2>
      <p className="small">
        This site stores two small items in your browser: your analytics choice and a random ID that links your feedback
        to explanations. It sets no advertising cookies. With your permission it also counts page visits with
        Plausible, a cookieless analytics service{analyticsConfigured ? "" : " (not enabled on this deployment)"}.
        Details are in the <Link to="/privacy">privacy policy</Link>.
      </p>
      <div className="actions">
        <button className="btn small" type="button" onClick={() => choose("granted")}>Allow analytics</button>
        <button className="btn small secondary" type="button" onClick={() => choose("denied")}>Decline analytics</button>
      </div>
    </div>
  );
}

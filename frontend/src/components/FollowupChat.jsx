import { useEffect, useState } from "react";
import { api, userId } from "../api.js";
import { ErrorText } from "./ui.jsx";

export function PreferencesPanel() {
  const [prefs, setPrefs] = useState(null);
  useEffect(() => {
    api(`/profile?user_id=${encodeURIComponent(userId())}`).then((p) => setPrefs(p.preferences || [])).catch(() => setPrefs([]));
  }, []);
  if (!prefs || !prefs.length) return null;
  return (
    <details>
      <summary>What this system has learned about how you like explanations ({prefs.length})</summary>
      <ul className="pref-list">
        {prefs.map((p) => <li key={p.topic}>{p.text} <span className="muted">(noted {p.updated_at})</span></li>)}
      </ul>
    </details>
  );
}

export default function FollowupChat({ traceId, consentCloud }) {
  const [log, setLog] = useState([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [learned, setLearned] = useState(null);

  const ask = async (e) => {
    e.preventDefault();
    const q = input.trim();
    if (!q || busy) return;
    setLog((l) => [...l, { from: "user", text: q }]);
    setInput("");
    setBusy(true);
    setError(null);
    try {
      const r = await api("/followup", { method: "POST", body: { trace_id: traceId, user_id: userId(), query: q, consent_cloud: consentCloud } });
      const text = r.explanation ? r.explanation.explanation : r.message;
      setLog((l) => [...l, { from: "bot", text }]);
      if (r.preferences_applied && r.preferences_applied.length) setLearned(r.preferences_applied);
    } catch (err) {
      setError(err);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div>
      <h3>Ask a follow-up question</h3>
      <p className="small muted">Uses the same evidence as the explanation above. Answers go through the same checks.</p>
      {log.length > 0 && (
        <div className="chat-log" aria-live="polite" style={{ marginBottom: 10 }}>
          {log.map((m, i) => <div key={i} className={`chat-msg ${m.from}`}>{m.text}</div>)}
        </div>
      )}
      <form className="chat-row" onSubmit={ask}>
        <label htmlFor="followup-q" className="hp">Follow-up question</label>
        <textarea id="followup-q" value={input} onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) ask(e); }}
          placeholder="e.g. What about their commute distance?" disabled={busy} />
        <button className="btn small" type="submit" disabled={busy || !input.trim()}>Ask</button>
      </form>
      <ErrorText error={error} />
      {learned && <p className="small muted" style={{ marginTop: 8 }}>Noted a preference from that: {learned[0]}</p>}
      <PreferencesPanel />
    </div>
  );
}

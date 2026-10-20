import { useState } from "react";
import { api } from "../api.js";
import { ErrorText, Notice } from "../components/ui.jsx";
import { useMeta } from "../seo.js";

// The admin token is typed in each session and kept only in memory. It is never part of the site's code.
export default function Admin() {
  useMeta("Admin", "Review audit logs, feedback and prompt versions.", { noindex: true });
  const [token, setToken] = useState("");
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [msg, setMsg] = useState("");
  const call = (path, method = "GET") => api(path, { method, headers: { "X-Admin-Token": token } });
  const load = async (e) => {
    e && e.preventDefault();
    setError(null);
    try {
      const [summary, traces, prompts] = await Promise.all([call("/admin/summary"), call("/admin/traces?limit=50"), call("/admin/prompts")]);
      setData({ summary, traces, prompts });
    } catch (err) {
      setError(err);
    }
  };
  const propose = async () => {
    setMsg(""); setError(null);
    try { const p = await call("/admin/prompts/propose", "POST"); setMsg(`Candidate ${p.version} created.`); load(); } catch (err) { setError(err); }
  };
  const approve = async (id) => {
    setMsg(""); setError(null);
    try { const p = await call(`/admin/prompts/${id}/approve`, "POST"); setMsg(`${p.version} is now active.`); load(); } catch (err) { setError(err); }
  };
  return (
    <>
      <h1>Admin</h1>
      <form className="card" onSubmit={load} style={{ maxWidth: 480 }}>
        <div className="field">
          <label htmlFor="tok">Admin token</label>
          <input id="tok" type="password" autoComplete="off" value={token} onChange={(e) => setToken(e.target.value)} required />
          <span className="hint">Set on the server in the backend .env file. It is kept in this tab's memory only.</span>
        </div>
        <p style={{ marginTop: 12 }}><button className="btn" type="submit" disabled={!token}>Load</button></p>
      </form>
      <ErrorText error={error} />
      {msg && <Notice kind="ok" role="status">{msg}</Notice>}
      {data && (
        <div className="stack" style={{ marginTop: 16 }}>
          <div className="card">
            <h2>Outcomes</h2>
            <p>{data.summary.n_traces} requests logged, {data.summary.n_feedback} ratings.</p>
            <p className="small">Verdicts: {Object.entries(data.summary.verdicts).map(([k, v]) => `${k} ${v}`).join(", ") || "none yet"}.</p>
            <p className="small">Most frequent violations: {Object.entries(data.summary.violation_codes).map(([k, v]) => `${k} ${v}`).join(", ") || "none yet"}.</p>
            {data.summary.feedback.length > 0 && (
              <div className="table-wrap"><table>
                <thead><tr><th>Prompt</th><th>Level</th><th className="r">Ratings</th><th className="r">Mean</th><th className="r">Understood</th><th className="r">Trust</th></tr></thead>
                <tbody>{data.summary.feedback.map((r) => (
                  <tr key={`${r.prompt_version}-${r.level}`}><td>{r.prompt_version}</td><td>{r.level}</td><td className="r num">{r.n}</td><td className="r num">{r.mean_rating}</td>
                    <td className="r num">{r.understood_rate ?? "n/a"}</td><td className="r num">{r.mean_trust ?? "n/a"}</td></tr>))}</tbody>
              </table></div>
            )}
          </div>
          <div className="card">
            <h2>Prompt versions</h2>
            <p className="small muted">Candidates are built from logged violations and ratings. A person must approve a candidate before it is used.</p>
            <button className="btn secondary small" type="button" onClick={propose}>Create a candidate from the logs</button>
            <div className="table-wrap" style={{ marginTop: 12 }}><table>
              <thead><tr><th>Version</th><th>Status</th><th>Added guidance</th><th>Notes</th><th /></tr></thead>
              <tbody>{data.prompts.map((p) => (
                <tr key={p.id}><td>{p.version}</td><td>{p.status}{p.approved_by ? ` (approved ${p.approved_at})` : ""}</td>
                  <td className="small">{p.guidance.join(" ") || "none"}</td><td className="small">{p.notes}</td>
                  <td>{p.status === "candidate" && <button className="btn small" type="button" onClick={() => approve(p.id)}>Approve</button>}</td></tr>))}</tbody>
            </table></div>
          </div>
          <div className="card">
            <h2>Recent requests</h2>
            <div className="table-wrap"><table className="small">
              <thead><tr><th>Time (UTC)</th><th>Intent</th><th>Level</th><th>Verdict</th><th className="r">Attempts</th><th>Prompt</th><th>Trace</th></tr></thead>
              <tbody>{data.traces.map((t) => (
                <tr key={t.trace_id}><td>{t.created_at}</td><td>{t.intent}</td><td>{t.level}</td><td>{t.verdict}</td><td className="r num">{t.attempts}</td><td>{t.prompt_version}</td><td>{t.trace_id}</td></tr>))}</tbody>
            </table></div>
          </div>
        </div>
      )}
    </>
  );
}

import { useState } from "react";
import { api, userId } from "../api.js";
import ResultView from "../components/ResultView.jsx";
import { BandBadge, ErrorText, Notice, pct } from "../components/ui.jsx";
import { useMeta } from "../seo.js";

const MAX = 2_000_000;

export default function Batch() {
  useMeta("Batch review", "Upload a CSV of employee records to get risk estimates for a cohort. Explanations are generated per row on request, and rows that fail verification are withheld.");
  const [file, setFile] = useState(null);
  const [fileError, setFileError] = useState("");
  const [website, setWebsite] = useState("");
  const [res, setRes] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const [open, setOpen] = useState({});
  const [level, setLevel] = useState("intermediate");

  const upload = async (e) => {
    e.preventDefault();
    if (!file) return setFileError("Choose a CSV file.");
    if (!/\.csv$/i.test(file.name)) return setFileError("The file must be a .csv file.");
    if (file.size > MAX) return setFileError("The file must be 2 MB or smaller.");
    setFileError("");
    setBusy(true);
    setError(null);
    const form = new FormData();
    form.append("file", file);
    form.append("website", website);
    try {
      setRes(await api("/domains/hr/batch", { method: "POST", form }));
      setOpen({});
    } catch (err) {
      setError(err);
    } finally {
      setBusy(false);
    }
  };
  const explainRow = async (row) => {
    setOpen((o) => ({ ...o, [row]: { loading: true } }));
    try {
      const r = await api(`/batch/${res.batch_id}/explain/${row}`, { method: "POST", body: { level, user_id: userId() } });
      setOpen((o) => ({ ...o, [row]: { r } }));
    } catch (err) {
      setOpen((o) => ({ ...o, [row]: { error: err } }));
    }
  };
  return (
    <>
      <h1>Batch review</h1>
      <p className="muted" style={{ maxWidth: 760 }}>
        Upload a CSV with the IBM HR column names (for example OverTime, MonthlyIncome, JobRole). Up to 2,000 rows.
        Use this for cohort review. Each estimate still needs human review, and the ranking is not a list of people to act on.
      </p>
      <form className="card" onSubmit={upload} noValidate style={{ maxWidth: 640 }}>
        <div className="field">
          <label htmlFor="csv">CSV file</label>
          <input id="csv" type="file" accept=".csv,text/csv" onChange={(e) => setFile(e.target.files[0] || null)}
            aria-invalid={fileError ? "true" : undefined} aria-describedby="csv-hint" />
          <span className="hint" id="csv-hint">An EmployeeNumber column, if present, is shown as a reference and never sent to the language model.</span>
          {fileError && <span className="error" role="alert">{fileError}</span>}
        </div>
        <div className="hp" aria-hidden="true">
          <label htmlFor="b-website">Leave this field empty</label>
          <input id="b-website" tabIndex={-1} autoComplete="off" value={website} onChange={(e) => setWebsite(e.target.value)} />
        </div>
        <p style={{ marginTop: 12 }}><button className="btn" type="submit" disabled={busy}>{busy ? "Scoring…" : "Score the file"}</button></p>
      </form>
      <ErrorText error={error} />
      {res && (
        <section style={{ marginTop: 24 }}>
          <h2>Cohort summary</h2>
          <p>
            {res.n_valid} of {res.n_rows} rows scored. Low band {res.summary?.bands.low ?? 0}, medium {res.summary?.bands.medium ?? 0},
            high {res.summary?.bands.high ?? 0}. Mean estimate {pct(res.summary?.mean_estimate)}. Needs human review: {res.summary?.needs_review ?? 0}.
          </p>
          {res.errors.length > 0 && (
            <details><summary>{res.errors.length} rows could not be read</summary>
              <ul className="small">{res.errors.map((e) => <li key={e.row}>Row {e.row}: {e.errors.join("; ")}</li>)}</ul>
            </details>
          )}
          <div className="field" style={{ maxWidth: 280, margin: "12px 0" }}>
            <label htmlFor="b-level">Explanation level</label>
            <select id="b-level" value={level} onChange={(e) => setLevel(e.target.value)}>
              <option value="novice">Novice</option><option value="intermediate">Intermediate</option><option value="expert">Expert</option>
            </select>
          </div>
          <div className="table-wrap">
            <table>
              <thead><tr><th>Rank</th><th>Row</th><th>Reference</th><th className="r">Estimate</th><th>Range</th><th>Band</th><th>Review</th><th>Explanation</th></tr></thead>
              <tbody>
                {res.rows.map((r, i) => (
                  <tr key={r.row}>
                    <td className="num">{i + 1}</td>
                    <td className="num">{r.row}</td>
                    <td>{r.reference || "none"}</td>
                    <td className="r num">{pct(r.probability)}</td>
                    <td className="num">{pct(r.interval[0])} to {pct(r.interval[1])}</td>
                    <td><BandBadge band={r.band} /></td>
                    <td>{r.needs_review ? "Human review" : "Standard"}</td>
                    <td>
                      <button className="btn small secondary" type="button" onClick={() => explainRow(r.row)} disabled={open[r.row]?.loading}>
                        {open[r.row]?.loading ? "Working…" : "Explain"}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {Object.entries(open).filter(([, v]) => v.r || v.error).map(([row, v]) => (
            <div key={row} className="card" style={{ marginTop: 16 }}>
              <h3>Row {row}</h3>
              {v.error ? <ErrorText error={v.error} /> : v.r.status === "withheld"
                ? <Notice kind="danger">Explanation withheld: it did not pass verification after 3 attempts.</Notice>
                : <ResultView r={v.r} />}
            </div>
          ))}
        </section>
      )}
    </>
  );
}

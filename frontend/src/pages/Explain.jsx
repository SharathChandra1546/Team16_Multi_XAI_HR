import { useEffect, useMemo, useRef, useState } from "react";
import { api, userId } from "../api.js";
import { useDraft } from "../App.jsx";
import DataSourceNote from "../components/DataSourceNote.jsx";
import Field from "../components/Field.jsx";
import ResultView from "../components/ResultView.jsx";
import { ErrorText, Loading } from "../components/ui.jsx";
import WhatIf from "../components/WhatIf.jsx";
import { groupFeatures, toInstance, validateField } from "../form.js";
import { Link } from "../router.jsx";
import { useMeta } from "../seo.js";

const QUESTIONS = ["Why is this employee flagged as likely to leave?", "What would lower their risk?", "What are the main factors behind this estimate?"];
const LEVELS = [["novice", "Novice", "Short, plain words"], ["intermediate", "Intermediate", "Percentages and strengths"], ["expert", "Expert", "Contributions in points"]];

export default function Explain() {
  useMeta("Explain a risk estimate", "Enter an employee's HR details, choose a level, and get a verified explanation of the model's attrition risk estimate.");
  const { draft, setDraft } = useDraft();
  const [schema, setSchema] = useState(null);
  const [config, setConfig] = useState(null);
  const [examples, setExamples] = useState([]);
  const [values, setValues] = useState({});
  const [assumed, setAssumed] = useState([]);
  const [errors, setErrors] = useState({});
  const [level, setLevel] = useState("intermediate");
  const [query, setQuery] = useState(QUESTIONS[0]);
  const [consent, setConsent] = useState(false);
  const [website, setWebsite] = useState("");
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [loadError, setLoadError] = useState(null);
  const resultRef = useRef(null);

  useEffect(() => {
    api("/domains/hr/schema").then(setSchema).catch(setLoadError);
    api("/config").then(setConfig).catch(() => {});
    api("/domains/hr/examples").then(setExamples).catch(() => {});
  }, []);
  useEffect(() => {
    if (draft) {
      setValues(Object.fromEntries(Object.entries(draft.instance).map(([k, v]) => [k, String(v)])));
      setAssumed(draft.assumed || []);
      setDraft(null);
    }
  }, [draft, setDraft]);

  const groups = useMemo(() => (schema ? groupFeatures(schema.features) : {}), [schema]);
  const onChange = (name, v) => {
    setValues((s) => ({ ...s, [name]: v }));
    setAssumed((a) => a.filter((x) => x !== name));
    if (errors[name]) setErrors((e) => ({ ...e, [name]: null }));
  };
  const loadExample = (id) => {
    const ex = examples.find((e) => e.id === id);
    if (!ex) return;
    setValues(Object.fromEntries(Object.entries(ex.instance).map(([k, v]) => [k, String(v)])));
    setAssumed([]);
    setErrors({});
  };
  const cloud = config?.llm_is_cloud;

  const run = async (body, path = "/explain") => {
    setBusy(true);
    setError(null);
    try {
      const r = await api(path, { method: "POST", body });
      setResult(r);
      setTimeout(() => resultRef.current?.focus(), 0);
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  };

  const submit = (e) => {
    e.preventDefault();
    const errs = {};
    for (const f of schema.features) {
      const msg = validateField(f, values[f.name]);
      if (msg) errs[f.name] = msg;
    }
    if (!query.trim()) errs.query = "Enter a question.";
    if (query.length > 1000) errs.query = "Keep the question under 1,000 characters.";
    setErrors(errs);
    if (Object.values(errs).some(Boolean)) {
      const first = Object.keys(errs)[0];
      document.getElementById(first === "query" ? "q" : `f-${first}`)?.focus();
      return;
    }
    run({ user_id: userId(), query, domain: "hr", instance: toInstance(schema.features, values), level,
      consent_cloud: consent, assumed_fields: assumed, website });
  };
  const refine = (action) => run({ trace_id: result.trace_id, user_id: userId(), action, consent_cloud: consent }, "/refine");

  if (loadError) return <ErrorText error={loadError} />;
  if (!schema) return <Loading />;
  const instanceReady = result && result.prediction && schema.features.every((f) => !validateField(f, values[f.name]));
  return (
    <>
      <h1>Explain a risk estimate</h1>
      <p className="muted" style={{ maxWidth: 760 }}>
        Enter the employee's details or start from a held-out example. Prefer to answer one question at a time?
        Use the <Link to="/intake">guided intake</Link> or the <Link to="/chat-intake">chat intake</Link> (it can also read a PDF).
      </p>
      <DataSourceNote source={config?.domains?.[0]?.data_source} />
      <div className="grid-2" style={{ marginTop: 16 }}>
        <form onSubmit={submit} noValidate aria-label="Employee details and question">
          {examples.length > 0 && (
            <div className="field" style={{ marginBottom: 16 }}>
              <label htmlFor="example">Start from a held-out example</label>
              <select id="example" defaultValue="" onChange={(e) => loadExample(e.target.value)}>
                <option value="" disabled>Choose an example…</option>
                {examples.map((ex, i) => <option key={ex.id} value={ex.id}>Example {i + 1} (record {ex.id})</option>)}
              </select>
            </div>
          )}
          {assumed.length > 0 && (
            <p className="notice small" role="note">Typical values were assumed for {assumed.length} field{assumed.length > 1 ? "s" : ""}. They are marked in the result.</p>
          )}
          {Object.entries(groups).map(([g, feats]) => (
            <fieldset key={g}>
              <legend>{g}</legend>
              <div className="fields">
                {feats.map((f) => <Field key={f.name} spec={f} value={values[f.name]} error={errors[f.name]} onChange={onChange} />)}
              </div>
            </fieldset>
          ))}
          <fieldset>
            <legend>Your question</legend>
            <div className="field">
              <label htmlFor="q">Question</label>
              <textarea id="q" value={query} maxLength={1000} onChange={(e) => setQuery(e.target.value)}
                aria-invalid={errors.query ? "true" : undefined} aria-describedby="q-hint" />
              <span className="hint" id="q-hint">Names and IDs are removed before anything is sent to the language model.</span>
              {errors.query && <span className="error">{errors.query}</span>}
            </div>
            <div className="chips" aria-label="Example questions">
              {QUESTIONS.map((q) => <button key={q} type="button" className="btn small secondary" onClick={() => setQuery(q)}>{q}</button>)}
            </div>
          </fieldset>
          <fieldset>
            <legend>Level of detail</legend>
            <div className="radio-row">
              {LEVELS.map(([v, label, hint]) => (
                <label key={v}>
                  <input type="radio" name="level" value={v} checked={level === v} onChange={() => setLevel(v)} />
                  <span>{label}</span><span className="small muted">{hint}</span>
                </label>
              ))}
            </div>
          </fieldset>
          {cloud && (
            <div className="field" style={{ marginBottom: 12 }}>
              <label style={{ fontWeight: 400, display: "flex", gap: 8, alignItems: "flex-start" }}>
                <input type="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)} style={{ marginTop: 4 }} />
                <span>Send the evidence and my question (with names and IDs removed) to the language model provider ({config.llm.split(":")[0]}) to write the explanation. Without this, a verified template is used.</span>
              </label>
            </div>
          )}
          <div className="hp" aria-hidden="true">
            <label htmlFor="website">Leave this field empty</label>
            <input id="website" tabIndex={-1} autoComplete="off" value={website} onChange={(e) => setWebsite(e.target.value)} />
          </div>
          {Object.values(errors).some(Boolean) && <p className="form-error" role="alert">Some fields need attention. They are marked above.</p>}
          <button className="btn" type="submit" disabled={busy}>{busy ? "Working…" : "Explain risk"}</button>
        </form>
        <div ref={resultRef} tabIndex={-1} aria-label="Result">
          <ErrorText error={error} />
          {!result && !error && (
            <div className="card">
              <h2>Result</h2>
              <p className="muted">The estimate, the explanation, the factors and the checks will appear here.</p>
            </div>
          )}
          <ResultView r={result} onRefine={refine} busy={busy} consentCloud={consent} />
          {instanceReady && schema.sliders?.length > 0 && result.status !== "refused" && (
            <div className="card" style={{ marginTop: 16 }}>
              <WhatIf sliders={schema.sliders} instance={toInstance(schema.features, values)} />
            </div>
          )}
        </div>
      </div>
    </>
  );
}

import { useEffect, useRef, useState } from "react";
import { api } from "../api.js";
import { useDomain } from "../domain.jsx";
import { BandBadge, ErrorText, pct } from "./ui.jsx";

const fmt = (v, unit) => {
  const n = Number(v);
  if (unit === "%") return `${n.toLocaleString()}%`;
  const u = n === 1 && unit.endsWith("s") ? unit.slice(0, -1) : unit;
  return `${n.toLocaleString()}${u ? ` ${u}` : ""}`;
};

// Sliders limited to actionable factors; the estimate updates live (debounced) from the server.
export default function WhatIf({ sliders, instance }) {
  const { domain } = useDomain();
  const [changes, setChanges] = useState({});
  const [out, setOut] = useState(null);
  const [error, setError] = useState(null);
  const timer = useRef(null);
  const seq = useRef(0);
  const key = JSON.stringify(instance);
  useEffect(() => {
    clearTimeout(timer.current);
    timer.current = setTimeout(async () => {
      const my = ++seq.current;
      try {
        const res = await api("/whatif", { method: "POST", body: { domain: domain.name, instance, changes } });
        if (my === seq.current) { setOut(res); setError(null); }
      } catch (e) {
        if (my === seq.current) setError(e);
      }
    }, 250);
    return () => clearTimeout(timer.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [changes, key]);
  const value = (s) => (changes[s.name] !== undefined ? changes[s.name] : instance[s.name]);
  const set = (name, v) => setChanges((c) => ({ ...c, [name]: v }));
  return (
    <div>
      <h2>What if</h2>
      <p className="small muted">Only factors an organisation can change are shown, within the range seen in the training data. These are model estimates, not recommendations.</p>
      {sliders.map((s) => {
        const id = `wi-${s.name}`;
        const v = value(s);
        return (
          <div className="slider-row" key={s.name}>
            <div>
              <label htmlFor={id} style={{ fontWeight: 600 }}>{s.display}</label>
              {s.actionability === "partial" && <span className="small muted"> (partly actionable)</span>}
              {s.options.length ? (
                <select id={id} value={v} onChange={(e) => set(s.name, e.target.value)}>
                  {s.options.map((o) => <option key={o.code} value={o.label}>{o.label}</option>)}
                </select>
              ) : (
                <input id={id} type="range" min={s.train_min} max={s.train_max} step={s.name === "MonthlyIncome" ? 100 : 1}
                  value={v} onChange={(e) => set(s.name, Number(e.target.value))} aria-valuetext={`${v}`} />
              )}
            </div>
            <span className="num small">{s.options.length ? "" : fmt(v, s.unit)}</span>
          </div>
        );
      })}
      <ErrorText error={error} />
      {out && (
        <div className="card" aria-live="polite" style={{ marginTop: 8 }}>
          <p style={{ margin: 0 }}>
            Current profile: <strong className="num">{pct(out.baseline.probability)}</strong> <BandBadge band={out.baseline.band} />
          </p>
          <p style={{ margin: "6px 0 0" }}>
            With these changes: <strong className="num">{pct(out.changed.probability)}</strong> <BandBadge band={out.changed.band} />
            <span className="small muted"> (likely range {pct(out.changed.interval[0])} to {pct(out.changed.interval[1])})</span>
          </p>
          {out.changed.out_of_distribution && <p className="small" style={{ marginTop: 6 }}>This combination is unusual for the training data, so treat the estimate with care.</p>}
          {out.changes.length > 0 && (
            <ul className="small" style={{ paddingLeft: 18, marginTop: 8 }}>
              {out.changes.map((c) => <li key={c.feature}>{c.display_name}: {c.from} to {c.to}</li>)}
            </ul>
          )}
          <p className="small muted" style={{ marginTop: 8 }}>{out.note}</p>
          <button className="btn small secondary" type="button" onClick={() => setChanges({})} disabled={!Object.keys(changes).length}>Reset</button>
        </div>
      )}
    </div>
  );
}

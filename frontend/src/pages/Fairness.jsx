import { useEffect, useState } from "react";
import { api } from "../api.js";
import DataSourceNote from "../components/DataSourceNote.jsx";
import ReliabilityChart from "../components/ReliabilityChart.jsx";
import { ErrorText, Loading, pct } from "../components/ui.jsx";
import { useMeta } from "../seo.js";

const ATTR = { gender: "Gender", age_band: "Age band", marital_status: "Marital status" };
const ci = (v, f = (x) => x.toFixed(3)) => (v && v[1] ? `${f(v[0])} (95% CI ${f(v[1][0])} to ${f(v[1][1])})` : "n/a");

export default function Fairness() {
  useMeta("Model quality and fairness", "Held-out model metrics with confidence intervals, a reliability curve, error and flag rates by group, disparate-impact ratios and a proxy-feature check.");
  const [d, setD] = useState(null);
  const [error, setError] = useState(null);
  const [variant, setVariant] = useState("main");
  useEffect(() => { api("/domains/hr/fairness").then(setD).catch(setError); }, []);
  if (error) return <ErrorText error={error} />;
  if (!d) return <Loading />;
  const m = d.models.main;
  const f = d.fairness[variant];
  return (
    <>
      <h1>Model quality and fairness</h1>
      <DataSourceNote source={d.dataset.source} />
      <section>
        <h2>Held-out test results</h2>
        <p className="muted">About 84% of employees in the data stay, so accuracy is shown only as a secondary number. The headline metrics are below.</p>
        <div className="table-wrap card">
          <table>
            <thead><tr><th>Metric</th><th>Main model ({m.model_kind})</th><th>Audit model</th></tr></thead>
            <tbody>
              <tr><td>ROC-AUC</td><td className="num">{ci(m.headline.roc_auc)}</td><td className="num">{ci(d.models.audit?.headline.roc_auc)}</td></tr>
              <tr><td>PR-AUC</td><td className="num">{ci(m.headline.pr_auc)}</td><td className="num">{ci(d.models.audit?.headline.pr_auc)}</td></tr>
              <tr><td>Recall at the review threshold</td><td className="num">{ci(m.headline.recall, (x) => pct(x, 1))}</td><td className="num">{ci(d.models.audit?.headline.recall, (x) => pct(x, 1))}</td></tr>
              <tr><td>Cost per employee (missed leaver {m.cost_matrix.false_negative}, false flag {m.cost_matrix.false_positive})</td>
                <td className="num">{ci(m.headline.cost_per_employee)}</td><td className="num">{ci(d.models.audit?.headline.cost_per_employee)}</td></tr>
              <tr><td>Accuracy (secondary)</td><td className="num">{m.test.metrics.accuracy.toFixed(3)}</td><td className="num">{d.models.audit?.test.metrics.accuracy.toFixed(3)}</td></tr>
            </tbody>
          </table>
          <p className="small muted" style={{ marginTop: 8 }}>
            Review threshold {pct(m.threshold)}, chosen to minimise the cost matrix on training-split predictions. Flagging nobody costs
            {" "}{m.test.metrics.cost_flag_nobody.toFixed(3)} per employee; flagging everyone costs {m.test.metrics.cost_flag_everyone.toFixed(3)}.
            The main model does not use age, gender or marital status. The audit model does and exists only for comparison.
          </p>
        </div>
      </section>
      <section>
        <h2>Calibration</h2>
        <div className="grid-2">
          <div className="card"><ReliabilityChart rel={m.reliability_oof_all} title="Out-of-fold predictions, all rows" /></div>
          <div className="card"><ReliabilityChart rel={m.reliability_test} title="Held-out test split" /></div>
        </div>
        <p className="small muted" style={{ marginTop: 8 }}>
          Cases sent to human review on the test split: {pct(m.uncertainty.escalation_rate_test, 1)} ({pct(m.uncertainty.band_uncertain_rate_test, 1)} with
          an uncertain band, {pct(m.uncertainty.ood_rate_test, 1)} unusual profiles).
        </p>
      </section>
      <section>
        <h2>Error and flag rates by group</h2>
        <div className="radio-row" role="radiogroup" aria-label="Model">
          {[["main", "Main model"], ["audit", "Audit model"]].map(([v, l]) => (
            <label key={v}><input type="radio" name="variant" checked={variant === v} onChange={() => setVariant(v)} /><span>{l}</span></label>
          ))}
        </div>
        {Object.entries(f).map(([attr, r]) => (
          <div className="card" key={attr} style={{ marginTop: 16 }}>
            <h3>{ATTR[attr] || attr}</h3>
            <p>
              Disparate-impact ratio <strong className="num">{r.disparate_impact_ratio === null ? "n/a" : r.disparate_impact_ratio.toFixed(2)}</strong>{" "}
              (lowest flag rate {r.dir_groups.lowest_flag_rate}, highest {r.dir_groups.highest_flag_rate}).{" "}
              {r.passes_four_fifths === null ? "" : r.passes_four_fifths ? "Meets the four-fifths rule (0.80 or higher)." : "Below the four-fifths rule (0.80)."}
            </p>
            <div className="table-wrap">
              <table>
                <thead><tr><th>Group</th><th className="r">n</th><th className="r">Left</th><th className="r">Flagged</th><th className="r">True positive rate</th><th className="r">False positive rate</th><th className="r">Precision</th></tr></thead>
                <tbody>
                  {r.groups.map((g) => (
                    <tr key={g.group}>
                      <td>{g.group}{g.small_group ? " (fewer than 30)" : ""}</td>
                      <td className="r num">{g.n}</td><td className="r num">{pct(g.base_rate, 1)}</td><td className="r num">{pct(g.flag_rate, 1)}</td>
                      <td className="r num">{pct(g.tpr, 1)}</td><td className="r num">{pct(g.fpr, 1)}</td><td className="r num">{pct(g.precision, 1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ))}
      </section>
      <section>
        <h2>Proxy check</h2>
        <p className="muted">Can the main model's features predict the protected attributes it does not use? An AUC of 0.70 or more is flagged.</p>
        <div className="table-wrap card">
          <table>
            <thead><tr><th>Attribute</th><th className="r">AUC from all features</th><th>Flagged</th><th>Strongest single features</th></tr></thead>
            <tbody>
              {Object.entries(d.fairness.proxy_check).map(([attr, r]) => (
                <tr key={attr}>
                  <td>{ATTR[attr] || attr}</td><td className="r num">{r.all_features_auc.toFixed(3)}</td><td>{r.flagged ? "Yes" : "No"}</td>
                  <td>{r.top_single_features.slice(0, 3).map((s) => `${s.display_name || s.feature} (${s.auc.toFixed(2)})`).join(", ")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </>
  );
}

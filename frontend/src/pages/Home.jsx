import { useEffect, useState } from "react";
import { api } from "../api.js";
import BarList from "../components/BarList.jsx";
import DataSourceNote from "../components/DataSourceNote.jsx";
import { ErrorText, Loading, pct } from "../components/ui.jsx";
import { Link } from "../router.jsx";
import { useMeta } from "../seo.js";

export default function Home() {
  useMeta("Verified explanations of employee attrition risk",
    "For HR managers: see why the model rates an employee's attrition risk, at novice, intermediate or expert level, with every sentence checked against the model's evidence.");
  const [agg, setAgg] = useState(null);
  const [schema, setSchema] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => {
    api("/domains/hr/aggregate").then(setAgg).catch(setError);
    api("/domains/hr/schema").then(setSchema).catch(() => {});
  }, []);
  const nFactors = schema ? schema.features.length : null;
  return (
    <>
      <section className="hero">
        <h1>See why the model rates an employee's attrition risk, and check every sentence it says.</h1>
        <p className="lead">
          For HR managers. The tool estimates attrition risk from {nFactors ? `${nFactors} HR factors` : "HR factors"}, explains the
          estimate at novice, intermediate or expert level, and lists changes that the model says would lower it. Uncertain
          cases go to human review. It does not make employment decisions.
        </p>
        <Link to="/explain" className="btn">Explain an employee's risk</Link>
      </section>

      <section aria-labelledby="how">
        <h2 id="how">How an explanation is produced</h2>
        <ol className="steps">
          <li><strong>Route.</strong> Questions that ask for a decision, such as whether to fire or promote someone, are refused.</li>
          <li><strong>Estimate.</strong> A calibrated model gives a risk estimate with a likely range. Unusual or uncertain cases are routed to human review.</li>
          <li><strong>Collect evidence.</strong> Shapley values measure how each factor moved the estimate; LIME is run as a cross-check.</li>
          <li><strong>Write.</strong> A language model drafts the explanation for your level, citing the evidence for every sentence.</li>
          <li><strong>Check.</strong> Code compares every claim with the evidence and screens the wording. Drafts that fail are rewritten up to three times, then replaced by a verified template.</li>
        </ol>
      </section>

      <section aria-labelledby="workforce">
        <h2 id="workforce">Workforce view</h2>
        <p className="muted">Group summaries from the training data. They describe patterns across employees, not judgements about individuals.</p>
        {error && <ErrorText error={error} />}
        {!agg && !error && <Loading />}
        {agg && (
          <div className="stack">
            <DataSourceNote source={agg.dataset.source} />
            <div className="grid-2">
              <div className="card">
                <h3>Observed attrition by department</h3>
                <BarList
                  label="Department"
                  rows={agg.aggregate.Department.map((r) => ({ name: r.group, value: r.observed_rate, n: r.n, est: r.mean_estimate }))}
                  valueKey="value"
                  format={(v) => pct(v)}
                  detail={(r) => `Observed ${pct(r.value)} of ${r.n} employees left. Mean model estimate ${pct(r.est)}.`}
                  caption={`Share of employees who left, in the training data (${agg.aggregate.n} employees, ${pct(agg.aggregate.observed_rate)} overall).`}
                />
              </div>
              <div className="card">
                <h3>Factors that move estimates most</h3>
                <BarList
                  label="Factor"
                  rows={agg.global_importance.slice(0, 8).map((g) => ({ name: g.display_name, value: g.mean_abs, act: g.actionability }))}
                  valueKey="value"
                  format={(v) => `${(v * 100).toFixed(1)} pts`}
                  detail={(r) => `Average size of contribution: ${(r.value * 100).toFixed(1)} percentage points. ${r.act === "actionable" ? "Can be changed by the organisation." : r.act === "partial" ? "Partly changeable." : "Not changed in recourse."}`}
                  caption="Average absolute contribution to the estimate, across held-out employees."
                />
              </div>
            </div>
          </div>
        )}
      </section>

      <section aria-labelledby="limits">
        <h2 id="limits">What this tool will not do</h2>
        <ul>
          <li>Recommend firing, promoting, demoting or disciplining anyone.</li>
          <li>Use age, gender or marital status in its main model. They are kept only to audit fairness.</li>
          <li>Present an estimate as a fact. Every explanation says it is an estimate that a person must review.</li>
        </ul>
      </section>
    </>
  );
}

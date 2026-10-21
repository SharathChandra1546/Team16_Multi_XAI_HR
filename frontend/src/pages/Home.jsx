import { useEffect, useState } from "react";
import { api } from "../api.js";
import BarList from "../components/BarList.jsx";
import DataSourceNote from "../components/DataSourceNote.jsx";
import { ErrorText, Loading, pct } from "../components/ui.jsx";
import { Link } from "../router.jsx";
import { useDomain } from "../domain.jsx";
import { useMeta } from "../seo.js";

export default function Home() {
  const { domain } = useDomain();
  const ui = domain.ui;
  useMeta(`Verified explanations of ${ui.risk_noun}`,
    `For ${ui.reader}s: see why the model rates the ${ui.unit}'s ${ui.risk_noun}, at novice, intermediate or expert level, with every sentence checked against the model's evidence.`);
  const [agg, setAgg] = useState(null);
  const [schema, setSchema] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => {
    api(`/domains/${domain.name}/aggregate`).then(setAgg).catch(setError);
    api(`/domains/${domain.name}/schema`).then(setSchema).catch(() => {});
  }, []);
  const nFactors = schema ? schema.features.length : null;
  const groupRows = agg && ui.overview_group ? agg.aggregate[ui.overview_group] : null;
  return (
    <>
      <section className="hero">
        <h1>{ui.headline}</h1>
        <p className="lead">{ui.lead}{nFactors ? ` The model uses ${nFactors} factors.` : ""}</p>
        <Link to="/explain" className="btn">Explain the {ui.unit}'s risk</Link>
      </section>

      <section aria-labelledby="how">
        <h2 id="how">How an explanation is produced</h2>
        <ol className="steps">
          <li><strong>Route.</strong> Questions that ask for a decision, such as {ui.decision_examples}, are refused.</li>
          <li><strong>Estimate.</strong> A calibrated model gives a risk estimate with a likely range. Unusual or uncertain cases are routed to human review.</li>
          <li><strong>Collect evidence.</strong> Shapley values measure how each factor moved the estimate; LIME is run as a cross-check.</li>
          <li><strong>Write.</strong> A language model drafts the explanation for your level, citing the evidence for every sentence.</li>
          <li><strong>Check.</strong> Code compares every claim with the evidence and screens the wording. Drafts that fail are rewritten up to three times, then replaced by a verified template.</li>
        </ol>
      </section>

      <section aria-labelledby="workforce">
        <h2 id="workforce">Group view</h2>
        <p className="muted">Group summaries from the training data. They describe patterns across {ui.unit_plural}, not judgements about individuals.</p>
        {error && <ErrorText error={error} />}
        {!agg && !error && <Loading />}
        {agg && (
          <div className="stack">
            <DataSourceNote source={agg.dataset.source} ui={ui} />
            <div className={groupRows ? "grid-2" : undefined}>
              {groupRows && (
                <div className="card">
                  <h3>Observed {ui.outcome_label.toLowerCase()} rate by {ui.overview_group_label}</h3>
                  <BarList
                    label={ui.overview_group_label}
                    rows={groupRows.map((r) => ({ name: r.group, value: r.observed_rate, n: r.n, est: r.mean_estimate }))}
                    valueKey="value"
                    format={(v) => pct(v)}
                    detail={(r) => `Observed ${pct(r.value)} of ${r.n} ${ui.unit_plural}. Mean model estimate ${pct(r.est)}.`}
                    caption={`Share of ${ui.unit_plural} in each group with the outcome, in the training data (${agg.aggregate.n} ${ui.unit_plural}, ${pct(agg.aggregate.observed_rate)} overall).`}
                  />
                </div>
              )}
              <div className="card">
                <h3>Factors that move estimates most</h3>
                <BarList
                  label="Factor"
                  rows={agg.global_importance.slice(0, 8).map((g) => ({ name: g.display_name, value: g.mean_abs, act: g.actionability }))}
                  valueKey="value"
                  format={(v) => `${(v * 100).toFixed(1)} pts`}
                  detail={(r) => `Average size of contribution: ${(r.value * 100).toFixed(1)} percentage points. ${r.act === "actionable" ? "Can be changed by the organisation." : r.act === "partial" ? "Partly changeable." : "Not changed in modelled changes."}`}
                  caption={`Average absolute contribution to the estimate, across held-out ${ui.unit_plural}.`}
                />
              </div>
            </div>
          </div>
        )}
      </section>

      <section aria-labelledby="limits">
        <h2 id="limits">What this tool will not do</h2>
        <ul>
          {ui.limits.map((t) => <li key={t}>{t}</li>)}
        </ul>
      </section>
    </>
  );
}

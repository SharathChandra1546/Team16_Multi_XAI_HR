import { useDomain } from "../domain.jsx";
import { BandBadge, pct } from "./ui.jsx";

// Hero figure + band track with the bootstrap range.
export default function RiskSummary({ prediction }) {
  const { domain } = useDomain();
  const p = prediction.probability;
  const [lo, hi] = prediction.interval;
  const t = prediction.threshold;
  const h = Math.max(0.5, t);
  const clamp = (v) => Math.min(100, Math.max(0, v * 100));
  return (
    <div>
      <p className="small muted" style={{ marginBottom: 4 }}>Model estimate of {domain.ui.risk_noun}</p>
      <div style={{ display: "flex", alignItems: "baseline", gap: 12, flexWrap: "wrap" }}>
        <span className="risk-figure num">{pct(p)}</span>
        <BandBadge band={prediction.band} />
      </div>
      <p className="small muted" style={{ margin: "6px 0 0" }}>
        Likely range {pct(lo)} to {pct(hi)}. Review threshold {pct(t)}.
      </p>
      <div
        className="range-track"
        role="img"
        aria-label={`Estimate ${pct(p)} on a 0 to 100% scale. Low band below ${pct(t)}, medium band ${pct(t)} to ${pct(h)}, high band ${pct(h)} and above. Likely range ${pct(lo)} to ${pct(hi)}.`}
      >
        <span style={{ width: `${clamp(t)}%`, background: "var(--band-low)" }} />
        <span style={{ width: `${clamp(h) - clamp(t)}%`, background: "var(--band-medium)" }} />
        <span style={{ width: `${100 - clamp(h)}%`, background: "var(--band-high)" }} />
        <span className="range-interval" style={{ left: `${clamp(lo)}%`, width: `${Math.max(0.6, clamp(hi) - clamp(lo))}%` }} />
        <span className="range-marker" style={{ left: `calc(${clamp(p)}% - 1px)` }} />
      </div>
      <div className="small muted num" style={{ display: "flex", justifyContent: "space-between" }} aria-hidden="true">
        <span>0%</span><span>50%</span><span>100%</span>
      </div>
    </div>
  );
}

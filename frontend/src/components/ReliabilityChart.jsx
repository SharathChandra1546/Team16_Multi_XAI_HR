import { useDomain } from "../domain.jsx";
import { Tooltip, useTooltip } from "./ui.jsx";

// Calibration curve: observed rate per bin with 95% Wilson intervals against the diagonal.
export default function ReliabilityChart({ rel, title }) {
  const { domain } = useDomain();
  const [tip, bind] = useTooltip();
  const W = 480, H = 320, L = 56, R = 16, T = 16, B = 48;
  const xmax = Math.min(1, Math.max(0.6, ...rel.bins.map((b) => b.mean_predicted)) + 0.05);
  const x = (v) => L + (v / xmax) * (W - L - R);
  const y = (v) => H - B - v * (H - T - B);
  const ticksX = [0, 0.2, 0.4, 0.6, 0.8, 1].filter((v) => v <= xmax + 1e-9);
  const ticksY = [0, 0.25, 0.5, 0.75, 1];
  const pts = rel.bins.map((b) => `${x(b.mean_predicted)},${y(b.observed)}`).join(" ");
  return (
    <figure style={{ margin: 0 }}>
      <figcaption><strong>{title}</strong> <span className="small muted">ECE {rel.ece.toFixed(3)}, Brier {rel.brier.toFixed(3)}</span></figcaption>
      <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img"
        aria-label={`Reliability curve. ${rel.bins.map((b) => `Predicted ${(b.mean_predicted * 100).toFixed(0)}%, observed ${(b.observed * 100).toFixed(0)}%`).join("; ")}.`}>
        {ticksY.map((v) => <line key={`gy${v}`} x1={L} x2={W - R} y1={y(v)} y2={y(v)} stroke="var(--grid)" strokeWidth="1" />)}
        {ticksX.map((v) => <text key={`tx${v}`} x={x(v)} y={H - B + 18} textAnchor="middle" style={{ fontSize: 13 }}>{Math.round(v * 100)}%</text>)}
        {ticksY.map((v) => <text key={`ty${v}`} x={L - 8} y={y(v) + 4} textAnchor="end" style={{ fontSize: 13 }}>{Math.round(v * 100)}%</text>)}
        <text x={(L + W - R) / 2} y={H - 8} textAnchor="middle" style={{ fontSize: 13 }}>Mean predicted risk in bin</text>
        <text x={14} y={(T + H - B) / 2} textAnchor="middle" transform={`rotate(-90 14 ${(T + H - B) / 2})`} style={{ fontSize: 13 }}>{domain.ui.reliability_axis}</text>
        <line x1={x(0)} y1={y(0)} x2={x(xmax)} y2={y(Math.min(1, xmax))} stroke="var(--muted)" strokeWidth="1" strokeDasharray="4 4" />
        {rel.bins.map((b, i) => <line key={`e${i}`} x1={x(b.mean_predicted)} x2={x(b.mean_predicted)} y1={y(b.ci_low)} y2={y(b.ci_high)} stroke="var(--down)" strokeWidth="2" />)}
        <polyline points={pts} fill="none" stroke="var(--down)" strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" />
        {rel.bins.map((b, i) => (
          <circle key={`c${i}`} cx={x(b.mean_predicted)} cy={y(b.observed)} r="5" fill="var(--down)" stroke="var(--surface)" strokeWidth="2" tabIndex={0}
            {...bind(<span>Predicted {(b.mean_predicted * 100).toFixed(1)}%<br />Observed {(b.observed * 100).toFixed(1)}% (95% CI {(b.ci_low * 100).toFixed(1)} to {(b.ci_high * 100).toFixed(1)})<br />n = {b.n}</span>)} />
        ))}
      </svg>
      <Tooltip tip={tip} />
      <p className="small muted">Dashed line: perfect calibration. Vertical lines: 95% Wilson intervals.</p>
      <details>
        <summary>Show as a table</summary>
        <div className="table-wrap">
          <table>
            <thead><tr><th className="r">Mean predicted</th><th className="r">Observed</th><th className="r">95% CI</th><th className="r">n</th></tr></thead>
            <tbody>{rel.bins.map((b, i) => (
              <tr key={i}><td className="r num">{(b.mean_predicted * 100).toFixed(1)}%</td><td className="r num">{(b.observed * 100).toFixed(1)}%</td>
                <td className="r num">{(b.ci_low * 100).toFixed(1)} to {(b.ci_high * 100).toFixed(1)}%</td><td className="r num">{b.n}</td></tr>))}
            </tbody>
          </table>
        </div>
      </details>
    </figure>
  );
}

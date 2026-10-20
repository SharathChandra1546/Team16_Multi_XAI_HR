import { Tooltip, useTooltip } from "./ui.jsx";
import "./charts.css";

// Diverging bar chart of factor contributions: red raises the estimate, blue lowers it, hatched grey is withheld.
export default function EvidenceChart({ items, max = 8 }) {
  const [tip, bind] = useTooltip();
  const shown = [...items.filter((i) => !i.withheld).slice(0, max), ...items.filter((i) => i.withheld)];
  const scale = Math.max(...shown.filter((i) => !i.withheld).map((i) => Math.abs(i.attribution)), 0.0001);
  const up = shown.filter((i) => !i.withheld && i.direction === "increases").map((i) => i.display_name);
  const down = shown.filter((i) => !i.withheld && i.direction === "decreases").map((i) => i.display_name);
  const summary = `Factor contributions. Raising the estimate: ${up.join(", ") || "none"}. Lowering it: ${down.join(", ") || "none"}.`;
  return (
    <figure style={{ margin: 0 }}>
      <figcaption className="small muted">How each factor moved the model's estimate for this profile, largest first.</figcaption>
      <div className="legend" aria-hidden="true">
        <span><i style={{ background: "var(--up)" }} />Raises the estimate</span>
        <span><i style={{ background: "var(--down)" }} />Lowers the estimate</span>
        {shown.some((i) => i.withheld) && (
          <span><i style={{ background: "repeating-linear-gradient(45deg, var(--withheld) 0 3px, transparent 3px 6px)", border: "1px solid var(--withheld)" }} />Withheld at this level</span>
        )}
      </div>
      <ul className="ev" role="list" aria-label={summary}>
        {shown.map((it) => {
          const w = it.withheld ? 35 : Math.max(2, (Math.abs(it.attribution) / scale) * 100);
          const pts = `${it.direction === "increases" ? "+" : "-"}${(Math.abs(it.attribution) * 100).toFixed(1)} pts`;
          const content = it.withheld ? (
            <span><strong>{it.display_name}</strong><br />Protected attribute. Withheld below expert level.</span>
          ) : (
            <span>
              <strong>{it.display_name}</strong>: {it.value_label}<br />
              {it.direction === "increases" ? "Raises" : "Lowers"} the estimate by {(Math.abs(it.attribution) * 100).toFixed(1)} percentage points ({it.magnitude}).
              {it.methods_agree === false && <><br />LIME disagrees on the direction.</>}
            </span>
          );
          const label = it.withheld
            ? `${it.display_name}: withheld at this level`
            : `${it.display_name}, ${it.value_label}: ${it.direction === "increases" ? "raises" : "lowers"} the estimate by ${(Math.abs(it.attribution) * 100).toFixed(1)} percentage points`;
          return (
            <li key={it.id} tabIndex={0} aria-label={label} className={it.withheld ? "is-withheld" : ""} {...bind(content)}>
              <span className="name">{it.display_name}<span className="val">{it.withheld ? "Withheld" : it.value_label}</span></span>
              <span className="track" aria-hidden="true">
                <span className="half neg">{(it.withheld || it.direction === "decreases") && <span className={`bar ${it.withheld ? "withheld" : ""}`} style={{ width: `${w}%` }} />}</span>
                <span className="half pos">{!it.withheld && it.direction === "increases" && <span className="bar" style={{ width: `${w}%` }} />}</span>
              </span>
              <span className="num">{it.withheld ? "n/a" : pts}</span>
            </li>
          );
        })}
      </ul>
      <Tooltip tip={tip} />
      <details style={{ marginTop: 10 }}>
        <summary>Show as a table</summary>
        <div className="table-wrap">
          <table>
            <thead><tr><th>Factor</th><th>Value</th><th>Direction</th><th className="r">Percentage points</th><th>Strength</th></tr></thead>
            <tbody>
              {shown.map((it) => (
                <tr key={it.id}>
                  <td>{it.display_name}</td>
                  <td>{it.withheld ? "Withheld" : it.value_label}</td>
                  <td>{it.withheld ? "Withheld" : it.direction === "increases" ? "Raises" : "Lowers"}</td>
                  <td className="r num">{it.withheld ? "n/a" : (it.attribution * 100).toFixed(1)}</td>
                  <td>{it.withheld ? "n/a" : it.magnitude}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </figure>
  );
}

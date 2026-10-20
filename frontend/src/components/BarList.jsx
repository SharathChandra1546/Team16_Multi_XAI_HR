import { Tooltip, useTooltip } from "./ui.jsx";
import "./charts.css";

// Single-series horizontal bars with value at the tip and a table view.
export default function BarList({ rows, label, valueKey, format, detail, caption }) {
  const [tip, bind] = useTooltip();
  const max = Math.max(...rows.map((r) => r[valueKey]), 0.0001);
  return (
    <figure style={{ margin: 0 }}>
      {caption && <figcaption className="small muted" style={{ marginBottom: 6 }}>{caption}</figcaption>}
      <ul className="hbars" aria-label={`${label}: ${rows.map((r) => `${r.name} ${format(r[valueKey])}`).join("; ")}`}>
        {rows.map((r) => (
          <li key={r.name} tabIndex={0} aria-label={`${r.name}: ${format(r[valueKey])}`} {...bind(<span><strong>{r.name}</strong><br />{detail ? detail(r) : format(r[valueKey])}</span>)}>
            <span>{r.name}</span>
            <span className="track" aria-hidden="true"><span className="bar" style={{ display: "block", width: `${(r[valueKey] / max) * 100}%` }} /></span>
            <span className="num">{format(r[valueKey])}</span>
          </li>
        ))}
      </ul>
      <Tooltip tip={tip} />
      <details style={{ marginTop: 8 }}>
        <summary>Show as a table</summary>
        <div className="table-wrap">
          <table>
            <thead><tr><th>{label}</th><th className="r">Value</th></tr></thead>
            <tbody>{rows.map((r) => <tr key={r.name}><td>{r.name}</td><td className="r num">{format(r[valueKey])}</td></tr>)}</tbody>
          </table>
        </div>
      </details>
    </figure>
  );
}

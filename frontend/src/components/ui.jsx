import { useState } from "react";

export function Notice({ kind = "", children, role }) {
  return <div className={`notice ${kind}`} role={role}>{children}</div>;
}

export function Loading({ label = "Loading" }) {
  return <p className="muted" role="status">{label}…</p>;
}

export function ErrorText({ error }) {
  if (!error) return null;
  return <Notice kind="danger" role="alert">{error.message || String(error)}</Notice>;
}

export function pct(p, d = 0) {
  if (p === null || p === undefined || Number.isNaN(p)) return "n/a";
  return `${(p * 100).toFixed(d)}%`;
}

export function BandBadge({ band }) {
  if (!band) return null;
  return <span className={`badge ${band}`}>{band[0].toUpperCase() + band.slice(1)} band</span>;
}

export function Tooltip({ tip }) {
  if (!tip) return null;
  const x = Math.min(tip.x + 12, window.innerWidth - 300);
  return (
    <div className="tooltip" style={{ left: Math.max(8, x), top: tip.y + 12 }} role="tooltip">
      {tip.content}
    </div>
  );
}

export function useTooltip() {
  const [tip, setTip] = useState(null);
  const bind = (content) => ({
    onMouseEnter: (e) => setTip({ x: e.clientX, y: e.clientY, content }),
    onMouseMove: (e) => setTip({ x: e.clientX, y: e.clientY, content }),
    onMouseLeave: () => setTip(null),
    onFocus: (e) => {
      const r = e.currentTarget.getBoundingClientRect();
      setTip({ x: r.left, y: r.bottom, content });
    },
    onBlur: () => setTip(null),
  });
  return [tip, bind];
}

// Client-side validation that mirrors the server's rules (the server validates again).
export function validateField(spec, raw) {
  if (raw === undefined || raw === null || String(raw).trim() === "") return `${spec.display} is required`;
  if (spec.options && spec.options.length) {
    return spec.options.some((o) => o.label === raw) ? null : `Choose a value for ${spec.display}`;
  }
  const v = Number(raw);
  if (!Number.isFinite(v)) return `${spec.display} must be a number`;
  if (spec.min !== null && spec.min !== undefined && v < spec.min) return `${spec.display} must be at least ${spec.min}`;
  if (spec.max !== null && spec.max !== undefined && v > spec.max) return `${spec.display} must be at most ${spec.max}`;
  if ((spec.step ?? 1) === 1 && !Number.isInteger(v)) return `${spec.display} must be a whole number`;
  return null;
}

export function toInstance(features, values) {
  const out = {};
  for (const f of features) {
    const raw = values[f.name];
    out[f.name] = f.options && f.options.length ? raw : Number(raw);
  }
  return out;
}

export function groupFeatures(features) {
  const groups = {};
  for (const f of features) (groups[f.group] = groups[f.group] || []).push(f);
  return groups;
}

export default function Field({ spec, value, error, onChange, idPrefix = "f" }) {
  const id = `${idPrefix}-${spec.name}`;
  const errId = `${id}-err`;
  const common = {
    id,
    name: spec.name,
    value: value ?? "",
    "aria-invalid": error ? "true" : undefined,
    "aria-describedby": error ? errId : `${id}-hint`,
    onChange: (e) => onChange(spec.name, e.target.value),
    required: true,
  };
  return (
    <div className="field">
      <label htmlFor={id}>{spec.display}</label>
      {spec.options && spec.options.length ? (
        <select {...common}>
          <option value="">Choose…</option>
          {spec.options.map((o) => <option key={o.code} value={o.label}>{o.label}</option>)}
        </select>
      ) : (
        <input type="number" inputMode="numeric" min={spec.min ?? undefined} max={spec.max ?? undefined} step={spec.step ?? 1} {...common} />
      )}
      <span className="hint" id={`${id}-hint`}>{spec.description}</span>
      {error && <span className="error" id={errId}>{error}</span>}
    </div>
  );
}

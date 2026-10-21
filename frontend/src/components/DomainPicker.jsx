import { useDomain } from "../domain.jsx";

export default function DomainPicker() {
  const { domains, domain, setDomain } = useDomain();
  if (!domains || domains.length < 2) return null;
  return (
    <div className="domain-picker">
      <label htmlFor="domain-select">Domain</label>
      <select id="domain-select" value={domain.name} onChange={(e) => setDomain(e.target.value)}>
        {domains.map((d) => <option key={d.name} value={d.name}>{d.title}</option>)}
      </select>
    </div>
  );
}

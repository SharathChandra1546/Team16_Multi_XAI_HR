import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { api } from "./api.js";

const DomainContext = createContext(null);
export const useDomain = () => useContext(DomainContext);

const KEY = "domain";
const read = () => {
  try { return localStorage.getItem(KEY) || ""; } catch { return ""; }
};
const write = (v) => {
  try { localStorage.setItem(KEY, v); } catch { /* storage can be unavailable */ }
};

// Domains come from the API, so a domain added on the server appears here without a frontend change.
export function DomainProvider({ children }) {
  const [domains, setDomains] = useState(null);
  const [name, setName] = useState(read());
  const [error, setError] = useState(null);
  useEffect(() => {
    api("/domains").then((list) => setDomains(list.filter((d) => d.ready))).catch(setError);
  }, []);
  const setDomain = useCallback((n) => { setName(n); write(n); }, []);
  const value = useMemo(() => {
    const current = domains ? domains.find((d) => d.name === name) || domains.find((d) => d.name === "hr") || domains[0] : null;
    return { domains, domain: current, setDomain, error };
  }, [domains, name, setDomain, error]);
  return <DomainContext.Provider value={value}>{children}</DomainContext.Provider>;
}

// Pages render only once the domain list has loaded.
export function useReadyDomain() {
  const { domain } = useDomain();
  return domain;
}

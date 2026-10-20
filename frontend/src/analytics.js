import { PLAUSIBLE_DOMAIN } from "./site.js";

const KEY = "consent-analytics";

export function getConsent() {
  try {
    return localStorage.getItem(KEY); // "granted" | "denied" | null
  } catch {
    return null;
  }
}

export function setConsent(value) {
  try {
    localStorage.setItem(KEY, value);
  } catch {
    /* storage unavailable: banner will show again next visit */
  }
  if (value === "granted") loadAnalytics();
}

let loaded = false;
export function loadAnalytics() {
  // Plausible: cookieless page-view counting. Loaded only after consent, and only if a domain is configured.
  if (loaded || !PLAUSIBLE_DOMAIN || getConsent() !== "granted") return;
  loaded = true;
  const s = document.createElement("script");
  s.defer = true;
  s.setAttribute("data-domain", PLAUSIBLE_DOMAIN);
  s.src = "https://plausible.io/js/script.js";
  document.head.appendChild(s);
}

export const analyticsConfigured = Boolean(PLAUSIBLE_DOMAIN);

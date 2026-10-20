// Public site settings. Nothing secret belongs here: this file ships to the browser.
const env = import.meta.env || {};
export const SITE_NAME = "Attrition Risk Explainer";
export const SITE_URL = (env.VITE_SITE_URL || "").replace(/\/+$/, "");
export const CONTACT_EMAIL = env.VITE_CONTACT_EMAIL || "";
export const OPERATOR = env.VITE_OPERATOR || "the project team";
export const PLAUSIBLE_DOMAIN = env.VITE_PLAUSIBLE_DOMAIN || "";
export const API_BASE = (env.VITE_API_BASE || "").replace(/\/+$/, "");
export const POLICY_DATE = "1 October 2026";

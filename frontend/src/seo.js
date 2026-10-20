import { useEffect } from "react";
import { SITE_NAME, SITE_URL } from "./site.js";

function setTag(selector, attr, value, create) {
  let el = document.head.querySelector(selector);
  if (!el && create) {
    el = document.createElement(create.tag);
    Object.entries(create.attrs).forEach(([k, v]) => el.setAttribute(k, v));
    document.head.appendChild(el);
  }
  if (el) el.setAttribute(attr, value);
}

export function useMeta(title, description, { noindex = false } = {}) {
  useEffect(() => {
    const full = title ? `${title} | ${SITE_NAME}` : SITE_NAME;
    document.title = full;
    setTag('meta[name="description"]', "content", description);
    setTag('meta[property="og:title"]', "content", full);
    setTag('meta[property="og:description"]', "content", description);
    setTag('meta[name="twitter:title"]', "content", full);
    setTag('meta[name="twitter:description"]', "content", description);
    if (SITE_URL) {
      setTag('link[rel="canonical"]', "href", SITE_URL + window.location.pathname);
      setTag('meta[property="og:url"]', "content", SITE_URL + window.location.pathname);
    }
    setTag('meta[name="robots"]', "content", noindex ? "noindex" : "index,follow", {
      tag: "meta",
      attrs: { name: "robots" },
    });
  }, [title, description, noindex]);
}

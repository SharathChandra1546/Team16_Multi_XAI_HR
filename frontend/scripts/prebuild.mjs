// Writes robots.txt and sitemap.xml for the custom domain. Refuses to build without one.
import { writeFileSync } from "node:fs";
import { loadEnv } from "vite";

const env = { ...loadEnv(process.env.NODE_ENV || "production", process.cwd(), "VITE_"), ...process.env };
const site = (env.VITE_SITE_URL || "").replace(/\/+$/, "");
if (!/^https:\/\/[a-z0-9.-]+\.[a-z]{2,}$/i.test(site) || site.includes("example")) {
  console.error("Set VITE_SITE_URL to your custom domain over HTTPS (for example https://attrition.yourcollege.edu) before building.");
  process.exit(1);
}
for (const [k, why] of [["VITE_CONTACT_EMAIL", "the privacy policy and terms need a contact address"], ["VITE_OPERATOR", "the privacy policy and terms must name who operates the site"]]) {
  if (!(env[k] || "").trim()) {
    console.error(`Set ${k}: ${why}.`);
    process.exit(1);
  }
}
const pages = ["/", "/explain", "/intake", "/batch", "/fairness", "/privacy", "/terms"];
const today = new Date().toISOString().slice(0, 10);
const urls = pages.map((p) => `  <url><loc>${site}${p}</loc><lastmod>${today}</lastmod></url>`).join("\n");
writeFileSync("public/sitemap.xml", `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n${urls}\n</urlset>\n`);
writeFileSync("public/robots.txt", `User-agent: *\nAllow: /\nDisallow: /admin\nDisallow: /api/\n\nSitemap: ${site}/sitemap.xml\n`);
console.log(`robots.txt and sitemap.xml written for ${site}`);

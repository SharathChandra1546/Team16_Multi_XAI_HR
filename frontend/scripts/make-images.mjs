// Generates favicon PNGs and the social preview image from SVG sources (no stock or AI imagery).
import sharp from "sharp";
import { readFileSync, writeFileSync } from "node:fs";

const icon = readFileSync(new URL("../public/favicon.svg", import.meta.url));
await sharp(icon).resize(32, 32).png({ compressionLevel: 9, palette: true }).toFile("public/favicon-32.png");
await sharp(icon).resize(180, 180).png({ compressionLevel: 9, palette: true }).toFile("public/apple-touch-icon.png");
const og = readFileSync(new URL("./og-image.svg", import.meta.url));
await sharp(og).png({ compressionLevel: 9, palette: true, colors: 64 }).toFile("public/og-image.png");
console.log("images written");

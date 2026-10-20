import { Link } from "../router.jsx";
import { useMeta } from "../seo.js";

export default function NotFound() {
  useMeta("Page not found", "The page you asked for does not exist.", { noindex: true });
  return (
    <div className="prose" style={{ padding: "48px 0" }}>
      <h1>Page not found</h1>
      <p className="muted">The page you asked for does not exist. It may have moved, or the address may be mistyped.</p>
      <Link to="/" className="btn">Go to the home page</Link>
    </div>
  );
}

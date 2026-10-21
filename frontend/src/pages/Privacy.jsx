import { CONTACT_EMAIL, OPERATOR, POLICY_DATE, SITE_NAME } from "../site.js";
import { useMeta } from "../seo.js";

export default function Privacy() {
  useMeta("Privacy policy", `How ${SITE_NAME} collects, uses and stores data, including the details you enter about employees, credit applicants or patients, questions, feedback and analytics.`);
  return (
    <article className="prose">
      <h1>Privacy policy</h1>
      <p className="muted">Effective {POLICY_DATE}.</p>
      <p>
        {SITE_NAME} is a research prototype operated by {OPERATOR}. It estimates employee attrition risk, credit risk and diabetes risk, and explains
        each estimate. This policy explains what data it handles and why. {CONTACT_EMAIL ? <>Contact: <a href={`mailto:${CONTACT_EMAIL}`}>{CONTACT_EMAIL}</a>.</> : "Contact the project team that runs this copy."}
      </p>

      <h2>What we collect</h2>
      <ul>
        <li><strong>Employee details you enter or upload</strong>, such as job role, overtime, income and satisfaction ratings. The tool does not ask for names, and the main model does not use age, gender or marital status.</li>
        <li><strong>Questions you type.</strong> Names, e-mail addresses, phone numbers and ID numbers are masked before a question is stored or sent anywhere. Masking is automatic and may miss some names, so please do not type personal details.</li>
        <li><strong>Feedback</strong>: ratings and optional comments on explanations.</li>
        <li><strong>A random browser ID</strong> stored in your browser's local storage, used to link your feedback and level preference to explanations. It contains no personal information.</li>
        <li><strong>Your analytics choice</strong>, stored in local storage so the banner does not appear on every visit.</li>
        <li><strong>Page-visit counts</strong>, only if you allow analytics (see below).</li>
      </ul>

      <h2>How we use it</h2>
      <ul>
        <li>To produce risk estimates and explanations you request.</li>
        <li>To keep an audit log of each request: the masked question, the inputs, the result, the checks that ran and the prompt version used. The log supports human review and research into explanation quality.</li>
        <li>To improve explanation prompts. Changes are reviewed and approved by a person.</li>
      </ul>
      <p>The tool does not make decisions about anyone, and we do not sell or share data for advertising.</p>

      <h2>Language model providers</h2>
      <p>
        When a cloud language model is configured, the evidence for an explanation (factor names, values and contributions)
        and your masked question are sent to that provider only if you tick the consent box on the explanation form. Without
        consent, explanations are written from verified templates on our server and nothing is sent to the provider.
      </p>

      <h2>Analytics</h2>
      <p>
        If you allow analytics, we count page visits with Plausible Analytics, which does not use cookies and does not
        collect personal data. If you decline, the analytics script is not loaded. You can change your choice at any time
        with the "Cookie and analytics settings" link at the bottom of every page.
      </p>

      <h2>Storage and retention</h2>
      <p>
        Data is stored in a database on the server that runs this site, over encrypted (HTTPS) connections. Audit logs and
        feedback are kept for the duration of the research project and then deleted. Uploaded batch files are scored and
        their rows stored with the batch results for the same period.
      </p>

      <h2>Your choices</h2>
      <ul>
        <li>Decline analytics, or clear this site's storage in your browser to remove the random ID.</li>
        <li>Ask us to delete records linked to your browser ID or a trace ID{CONTACT_EMAIL ? ` by writing to ${CONTACT_EMAIL}` : ""}.</li>
        <li>Only enter data about employees, applicants or patients that you are authorised to use. The models were built on public research datasets (IBM's fictional HR sample, the Statlog German Credit data and the Pima Indians Diabetes data).</li>
      </ul>

      <h2>Changes</h2>
      <p>We will update this page if our practices change and revise the effective date above.</p>
    </article>
  );
}

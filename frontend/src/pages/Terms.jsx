import { CONTACT_EMAIL, OPERATOR, POLICY_DATE, SITE_NAME } from "../site.js";
import { Link } from "../router.jsx";
import { useMeta } from "../seo.js";

export default function Terms() {
  useMeta("Terms and conditions", `The terms for using ${SITE_NAME}, a research prototype that estimates and explains employee attrition, credit and diabetes risk.`);
  return (
    <article className="prose">
      <h1>Terms and conditions</h1>
      <p className="muted">Effective {POLICY_DATE}.</p>
      <p>
        These terms apply to your use of {SITE_NAME}, a research prototype operated by {OPERATOR}. By using the site you
        agree to them. {CONTACT_EMAIL ? <>Questions: <a href={`mailto:${CONTACT_EMAIL}`}>{CONTACT_EMAIL}</a>.</> : "Questions go to the project team that runs this copy."}
      </p>

      <h2>What the service is</h2>
      <p>
        The service gives model-based estimates of employee attrition risk, credit risk and diabetes risk, and explanations of
        those estimates. The models were trained on public research datasets. Estimates describe patterns in that data. They
        are not facts about any person and they are not professional HR, legal, financial or medical advice.
      </p>

      <h2>No automated decisions</h2>
      <ul>
        <li>You must not use an estimate, ranking or explanation as the sole or main basis for any decision about a person, including hiring, firing, promotion, demotion, pay or discipline.</li>
        <li>Every estimate must be reviewed by a person with the relevant context and authority before any action.</li>
        <li>Requests for employment decisions are refused by design.</li>
      </ul>

      <h2>Your responsibilities</h2>
      <ul>
        <li>Only enter or upload data you are authorised to use, and do not include names or contact details.</li>
        <li>Do not attempt to bypass the safeguards, overload the service, or access the admin area without permission.</li>
        <li>Follow the employment, data-protection and anti-discrimination laws that apply to you.</li>
      </ul>

      <h2>Availability and accuracy</h2>
      <p>
        The service is provided as is, for research and teaching, without any warranty that it is accurate, complete or
        available. Model quality and fairness results are published on the <Link to="/fairness">model and fairness page</Link>.
      </p>

      <h2>Liability</h2>
      <p>
        To the extent the law allows, the operator is not liable for loss or damage arising from use of the service or from
        decisions made with it.
      </p>

      <h2>Privacy</h2>
      <p>How we handle data is described in the <Link to="/privacy">privacy policy</Link>.</p>

      <h2>Changes</h2>
      <p>We may update these terms. The effective date above shows the latest version.</p>
    </article>
  );
}

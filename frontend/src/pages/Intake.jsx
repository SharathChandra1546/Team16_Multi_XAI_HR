import { useEffect, useState } from "react";
import { api } from "../api.js";
import { useDraft } from "../App.jsx";
import Field from "../components/Field.jsx";
import { ErrorText, Loading } from "../components/ui.jsx";
import { validateField } from "../form.js";
import { navigate } from "../router.jsx";
import { useDomain } from "../domain.jsx";
import { useMeta } from "../seo.js";

// Conversational intake: asks for the most influential missing field first; skipped fields become flagged assumptions.
export default function Intake() {
  useMeta("Guided intake", "Answer one question at a time, starting with the details that influence the estimate most. Skipped details are filled with typical values and flagged.");
  const { domain } = useDomain();
  const { setDraft } = useDraft();
  const [answers, setAnswers] = useState({});
  const [skipped, setSkipped] = useState([]);
  const [step, setStep] = useState(null);
  const [value, setValue] = useState("");
  const [fieldError, setFieldError] = useState(null);
  const [done, setDone] = useState(null);
  const [error, setError] = useState(null);
  const [asked, setAsked] = useState(0);

  const ask = async (a, s, finish = false) => {
    setError(null);
    try {
      const res = await api("/intake", { method: "POST", body: { domain: domain.name, instance: a, skipped: s, finish } });
      if (res.complete) setDone(res);
      else {
        setStep(res);
        setValue("");
      }
    } catch (e) {
      setError(e);
    }
  };
  useEffect(() => { ask({}, []); }, []);

  if (error) return <ErrorText error={error} />;
  if (done)
    return (
      <div className="prose">
        <h1>Details complete</h1>
        {done.assumed.length > 0 ? (
          <>
            <p>Typical values from the training data were assumed for these details. The explanation will flag them:</p>
            <ul>{done.assumed.map((a) => <li key={a.feature}>{a.display_name}: {a.assumed_value}</li>)}</ul>
          </>
        ) : <p>You answered every question.</p>}
        <button className="btn" type="button" onClick={() => { setDraft({ instance: done.instance, assumed: done.assumed.map((a) => a.feature) }); navigate("/explain"); }}>
          Continue to the explanation
        </button>
      </div>
    );
  if (!step) return <Loading />;
  const spec = step.next[0];
  const next = (e) => {
    e.preventDefault();
    const msg = validateField(spec, value);
    if (msg) return setFieldError(msg);
    setFieldError(null);
    const a = { ...answers, [spec.name]: spec.options.length ? value : Number(value) };
    setAnswers(a);
    setAsked((n) => n + 1);
    ask(a, skipped);
  };
  const skip = () => {
    const s = [...skipped, spec.name];
    setSkipped(s);
    setFieldError(null);
    ask(answers, s);
  };
  return (
    <div className="prose">
      <h1>Guided intake</h1>
      <p className="muted">Questions come in order of how much each detail influences the model's estimates. {step.remaining} detail{step.remaining === 1 ? "" : "s"} left. Answered so far: {asked}.</p>
      <form className="card" onSubmit={next} noValidate>
        <Field spec={spec} value={value} error={fieldError} onChange={(_, v) => setValue(v)} idPrefix="in" />
        <p className="small muted" style={{ marginTop: 8 }}>{spec.why}</p>
        <div className="chips">
          <button className="btn" type="submit">Next</button>
          <button className="btn secondary" type="button" onClick={skip}>Skip (assume a typical value)</button>
          <button className="btn secondary" type="button" onClick={() => ask(answers, skipped, true)}>Finish with typical values</button>
        </div>
      </form>
    </div>
  );
}

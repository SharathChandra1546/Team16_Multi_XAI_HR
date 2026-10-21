import { useRef, useState } from "react";
import { api, userId } from "../api.js";
import { useDraft } from "../App.jsx";
import { ErrorText } from "../components/ui.jsx";
import { navigate } from "../router.jsx";
import { useDomain } from "../domain.jsx";
import { useMeta } from "../seo.js";

export default function ChatIntake() {
  const { domain } = useDomain();
  const ui = domain.ui;
  useMeta("Chat intake", `Describe the ${ui.unit} or upload a PDF; missing details are asked for in chat.`);
  const { setDraft } = useDraft();
  const [log, setLog] = useState([{ from: "bot", text: ui.chat_greeting }]);
  const [instance, setInstance] = useState({});
  const [skipped, setSkipped] = useState([]);
  const [input, setInput] = useState("");
  const [consent, setConsent] = useState(false);
  const [busy, setBusy] = useState(false);
  const [done, setDone] = useState(null);
  const [error, setError] = useState(null);
  const fileRef = useRef(null);

  const say = (from, text) => setLog((l) => [...l, { from, text }]);

  const applyResult = (result, extraReply) => {
    setInstance(Object.fromEntries(Object.entries(result.instance_codes || {}).map(([k, v]) => [k, v])));
    if (extraReply) say("bot", extraReply);
    if (result.complete) {
      setDone(result);
      say("bot", result.assumed?.length
        ? `I have everything. ${result.assumed.length} detail${result.assumed.length > 1 ? "s were" : " was"} not given, so I used typical values and will flag them in the explanation.`
        : "I have everything I need.");
    }
  };

  const sendMessage = async (text) => {
    if (!text.trim() || busy) return;
    say("user", text);
    setInput("");
    setBusy(true);
    setError(null);
    try {
      const res = await api("/intake/chat", {
        method: "POST",
        body: { domain: domain.name, instance, skipped, message: text, consent_cloud: consent },
      });
      if (res.reply) say("bot", res.reply);
      applyResult(res);
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  };

  const skipCurrent = async () => {
    // ask the server what field is current by resending with no message; the server already tracks order from `instance`
    setBusy(true);
    try {
      const probe = await api("/intake/chat", { method: "POST", body: { domain: domain.name, instance, skipped, message: "" } });
      const field = probe.next?.[0]?.name;
      const nextSkipped = field ? [...skipped, field] : skipped;
      setSkipped(nextSkipped);
      const res = await api("/intake/chat", { method: "POST", body: { domain: domain.name, instance, skipped: nextSkipped, message: "" } });
      say("bot", "Skipped. " + (res.reply || ""));
      applyResult(res);
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  };

  const finishWithTypical = async () => {
    setBusy(true);
    try {
      const res = await api("/intake/chat", { method: "POST", body: { domain: domain.name, instance, skipped, message: "", finish: true } });
      applyResult(res, "Using typical values for anything not covered.");
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  };

  const uploadPdf = async (file) => {
    if (!file) return;
    setBusy(true);
    setError(null);
    say("user", `Uploaded: ${file.name}`);
    try {
      const form = new FormData();
      form.append("file", file);
      form.append("consent_cloud", consent ? "true" : "false");
      const res = await api(`/domains/${domain.name}/intake/pdf`, { method: "POST", form });
      const found = Object.fromEntries((res.found || []).map((f) => [f.feature, f.value]));
      // feed the PDF's readable values in as one chat message so the same extraction/validation path fills them
      const sentence = (res.found || []).map((f) => `${f.display_name}: ${f.value}`).join(", ");
      if (sentence) {
        const res2 = await api("/intake/chat", { method: "POST", body: { domain: domain.name, instance, skipped, message: sentence, consent_cloud: consent } });
        say("bot", `From the PDF, I found ${res.found.length} field${res.found.length === 1 ? "" : "s"}${Object.keys(found).length ? ": " + Object.values(found).join(", ") : ""}.`);
        applyResult(res2, res2.reply);
      } else {
        say("bot", "I could not find any recognisable fields in that PDF. Let's go through them in chat instead.");
      }
      if (res.unparsed && Object.keys(res.unparsed).length) {
        say("bot", "A few values in the PDF didn't look valid, so I left them out: " + Object.keys(res.unparsed).join(", ") + ".");
      }
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  };

  const continueToExplain = () => {
    setDraft({ instance: done.instance, assumed: (done.assumed || []).map((a) => a.feature) });
    navigate("/explain");
  };

  return (
    <div className="prose">
      <h1>Chat intake</h1>
      <p className="muted">Describe the {ui.unit}, or attach a PDF. Missing details that matter most to the estimate are asked about one at a time.</p>
      <div className="card">
        <div className="chat-log" aria-live="polite">
          {log.map((m, i) => <div key={i} className={`chat-msg ${m.from}`}>{m.text}</div>)}
        </div>
        <ErrorText error={error} />
        {!done && (
          <>
            <form className="chat-row" onSubmit={(e) => { e.preventDefault(); sendMessage(input); }}>
              <label htmlFor="chat-input" className="hp">Message</label>
              <textarea id="chat-input" value={input} onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(input); } }}
                placeholder="e.g. She's a 34 year old sales exec, income about 5,200 a month, no overtime…" disabled={busy} />
              <button className="btn" type="submit" disabled={busy || !input.trim()}>Send</button>
            </form>
            <div className="chips" style={{ marginTop: 10 }}>
              <label className="small" style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <input type="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)} />
                Allow sending this text to the language model to read it (needed for cloud providers only)
              </label>
            </div>
            <div className="chips">
              <input ref={fileRef} type="file" accept="application/pdf" style={{ display: "none" }}
                onChange={(e) => uploadPdf(e.target.files?.[0])} id="pdf-upload" />
              <label className="btn small secondary" htmlFor="pdf-upload" style={{ cursor: "pointer" }}>Attach a PDF</label>
              <button className="btn small secondary" type="button" disabled={busy} onClick={skipCurrent}>Skip this detail</button>
              <button className="btn small secondary" type="button" disabled={busy} onClick={finishWithTypical}>Finish with typical values</button>
            </div>
          </>
        )}
        {done && (
          <button className="btn" type="button" onClick={continueToExplain}>Continue to the explanation</button>
        )}
      </div>
    </div>
  );
}

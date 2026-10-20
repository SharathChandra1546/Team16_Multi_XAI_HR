# Attrition Risk Explainer

A multi-agent framework for adaptive, multilevel explainable AI using large language models. Team 16 project.

The domain is HR attrition (IBM HR Analytics Employee Attrition dataset), seen from an HR manager's point of view. The system:

- estimates attrition risk with a calibrated model,
- explains the estimate at novice, intermediate or expert level,
- lists changes the model says would lower the estimate,
- checks every sentence against the evidence before anyone sees it,
- can take the employee's details as a chat conversation or a PDF, asking in chat for anything still missing,
- lets the user ask follow-up questions about the explanation shown, verified the same way as the first answer,
- learns a short, descriptive, dated note about how that user likes explanations from each follow-up and level change, and appends the current notes to every prompt sent to the language model (the newest note on a topic always overrides an older, contradictory one).

Uncertain or unusual cases go to human review. Requests for employment decisions are refused.

## Repository layout

```
backend/
  app/
    domains/        DomainAgent interface, DomainRegistry, HR domain (spec, policy, glossary, data, recourse levers)
    ml/             training + validation (CV, calibration, CIs, bands, OOD), runtime, fairness audit
    xai/            Shapley + LIME explainers, Canonical Evidence Package, evidence validator
    agents/         router + guardrail, planner, generator, verifier, critic, repair instructions, fallback templates
    llm/            OpenAI-compatible client (Gemini, Groq, OpenAI, Ollama), mock LLM, versioned prompts
    metrics/        Claim Fidelity Score (CFS) definition and aggregation
    intake_extract.py  PDF text + chat-message field extraction (regex fallback, optional LLM slot-filling)
    graph.py        LangGraph orchestration (domain-agnostic)
    recourse.py     constrained, re-predicted recourse
    service.py      application service (PII scrubbing, consent, records, batch, intake, chat/PDF intake, follow-up Q&A, preferences, what-if)
    api.py          HTTP API (Starlette), security middleware, optional static hosting
  scripts/          fetch_hr.py, train_hr.py
  eval/             run_faults.py, run_phase1.py, export_tables.py, guardrail test set
  tests/            pytest suite (acceptance criteria)
frontend/           React (Vite) site
deploy/             Caddyfile (automatic HTTPS) and a systemd unit
docs/ACCEPTANCE.md  every requirement, where it is implemented and how it is verified
docs/FIDELITY_METRIC.md  definition and properties of the Claim Fidelity Score
```

## Architecture (matches the Review 2 diagrams)

| Diagram component | Code |
|---|---|
| Application layer | `frontend/`, `backend/app/api.py` |
| Orchestrator (workflow state, routing, retries, fallbacks) | `app/graph.py` (LangGraph) |
| Context and routing layer: user context, domain and intent router, domain agent registry, fallback | `app/agents/router.py`, `app/domains/registry.py`, `app/store.py` (profiles) |
| Prediction and XAI layer: model, XAI method, evidence validator, Canonical Evidence Package | `app/ml/runtime.py`, `app/xai/*` |
| Adaptive explanation layer: planner, LLM generator | `app/agents/planner.py`, `app/agents/generator.py` |
| Verification and correction: claim-level fidelity verifier, domain and quality critic, retries, safe template fallback | `app/agents/verifier.py`, `app/agents/critic.py`, `app/agents/fallback.py` |
| Persistence and offline evaluation: records, offline evaluation, human-approved prompt updates | `app/store.py`, `eval/`, admin endpoints |

The diagrams say FastAPI. The API uses Starlette, the ASGI toolkit FastAPI is built on, with the same request handling.

## Chat intake, follow-up questions and preferences

- **Chat + PDF intake** (`/chat-intake`): describe the employee in a sentence, or attach a PDF. `app/intake_extract.py` pulls fields out of the text (regex over each feature's aliases, plus an optional LLM JSON pass under the same cloud-consent checkbox as `/explain`) and the chat asks, one at a time, for whatever the model still needs, in the same influence order as the guided intake. "Skip" and "finish with typical values" guarantee it always completes.
- **Follow-up questions** (the "Ask a follow-up question" box under a result): the question is answered from the *same* Evidence Package as the original explanation — re-derived deterministically from the stored instance, not a new prediction — through the identical generate -> verify -> critic -> retry -> fallback loop, so follow-up answers carry the same fidelity guarantee as the first one.
- **Descriptive preferences**: a level change or a follow-up question produces one short, dated, topic-scoped note (e.g. "Detail level: asked to switch to clear professional-level explanations."). A new note on a topic replaces the older one on that topic, so a contradictory later statement always wins. The current notes for a user are appended to every generator prompt for them (`preferences_applied` on the response says which ones were used) and shown in "What this system has learned about how you like explanations."
- **Fidelity metric and repair**: `app/metrics/fidelity.py` defines the Claim Fidelity Score (CFS), a bounded, deterministic, judge-independent score (see `docs/FIDELITY_METRIC.md` for its properties and stated circularity limitation), returned as `VerificationResult.fidelity`. On a failed attempt, `app/agents/repair.py` turns each violation into a specific repair instruction (pulling the correct value from the evidence package when the violation names an evidence id) instead of handing the raw violation list back to the language model.

## Quick start (one step)

You need Python 3.10 or newer. Node.js is only needed for a public deployment build.

1. **Optional, for real results:** download the dataset from Kaggle ([IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)). Save `WA_Fn-UseC_-HR-Employee-Attrition.csv` in `backend/data/`.
2. **Run the setup script:**
   - Windows: double-click `setup_and_run.bat`
   - macOS/Linux: `./setup_and_run.sh`

   It creates a virtual environment, installs the requirements, and trains the model on the IBM CSV if it is present. Without the CSV, it trains on the synthetic stand-in, and the site shows a test-data warning. It then runs the tests and opens the site at http://localhost:8000. The site is a prebuilt bundle in `frontend/dist-local`, so Node.js is not needed to run it locally.
3. **Optional, to use a real language model:** edit `backend/.env`. Set `LLM_PROVIDER` and your key (see below), then run the setup script again.
4. **Experiments for the paper:** `run_experiments.bat` (Windows) or `./run_experiments.sh`. Add `--quick` for a short check run.

### Measured run times (build sandbox, 2-core Linux)

| Step | Time |
|---|---|
| Train main + audit models (full settings) | 1 min 46 s |
| Test suite (91 tests) | 5 min |
| One explanation, mock LLM | 1.2 s median (most of it is Shapley + LIME) |
| Fault injection, 40 employees, template bases (12 fault types, 955 clean texts) | 4 min 54 s |
| Phase I, 30 employees x 3 levels x 2 intents, mock LLM | 6 min 16 s |

With a real LLM each explanation adds the provider's response time (usually a few seconds per call, at least 2 calls). The LLM experiments make hundreds to a few thousand calls, so they take about 1 to 2 hours on free-tier keys. Responses are cached, so re-runs are fast. Software-check outputs from the build are in `backend/results/software_check_standin_mock/`; they are not results.

### Manual steps (same as the script)

```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
python scripts/ensure_model.py      # trains on the IBM CSV if present, otherwise the stand-in
pytest -q
STATIC_DIR=../frontend/dist-local uvicorn app.api:app --port 8000
# frontend development server instead: cd frontend && npm install && npm run dev  (http://localhost:5173)
```

## Language model

Set `LLM_PROVIDER` in `backend/.env`:

| Provider | Key | Notes |
|---|---|---|
| `mock` | none | Deterministic templates. Runs everything offline. Not valid for results. |
| `gemini` | `GEMINI_API_KEY` | Recommended default (free tier at Google AI Studio). |
| `groq` | `GROQ_API_KEY` | Open model (Llama). Use as the second LLM in experiments. |
| `openai` | `OPENAI_API_KEY` | |
| `ollama` | none | Local model; no data leaves the machine, so no consent prompt. |

Model names change often. Set `LLM_MODEL` to a current model from your provider's model list. Responses are cached at temperature 0 (`LLM_CACHE=1`) to save free-tier quota.

With a cloud provider, the explanation form shows a consent checkbox. Without consent, explanations come from verified templates and nothing is sent to the provider.

## Experiments for the paper

```bash
python eval/run_faults.py --n 40                    # detection per fault type + false rejection on clean text
python eval/run_faults.py --n 40 --base llm --reviewer   # LLM-written bases, plus the LLM-reviewer baseline
python eval/run_phase1.py --n 40 --reviewer         # raw LLM error rate, pipeline outcomes, level differences
```

Results go to `backend/results/` as JSON and Markdown. Run each experiment with two LLMs (for example Gemini and Groq Llama). Change `LLM_PROVIDER` / `JUDGE_PROVIDER` between runs.

Notes for honest reporting:

- **Fault-injection results on template bases are an upper bound.** Faults inserted into template text are easier to catch than real LLM errors. Report `--base llm` results as the main numbers and Phase I raw error counts alongside them.
- **The verifier also produces the fidelity metric.** Using it as both detector and judge is circular. Back it up with the LLM reviewer comparison and a small human labelling of explanations.
- **Escalation can be frequent.** With the default rule (the 10th to 90th percentile interval spans two bands), about 30% of held-out cases were routed to human review on the stand-in data. Check the rate on the real data in the manifest (`uncertainty.escalation_rate_test`) and tune `INTERVAL_LOW_PCT` / `INTERVAL_HIGH_PCT` if needed. Report the rule you use.

## Deploying (launch checklist)

Done in the code:

- Privacy and terms pages
- Secrets kept on the server; the build fails if keys appear in the bundle
- HTTPS redirect and HSTS (`FORCE_HTTPS=1`)
- Cookie and analytics consent banner; Plausible loads only after consent
- Meta titles and descriptions
- Social preview image and favicon
- `sitemap.xml` and `robots.txt` (generated for your domain at build)
- Alt text and table views for charts
- Compressed images
- Contrast and mobile checks
- Custom 404 page
- Form validation on client and server
- Honeypot spam protection and rate limiting
- One primary call to action

Needs you before launch:

1. **Buy and connect a custom domain.** Point its DNS at your server.
2. **Fill in the build settings.** Set `VITE_SITE_URL`, `VITE_OPERATOR` and `VITE_CONTACT_EMAIL` in `frontend/.env`. The build refuses to run without them.
3. **Get the privacy policy and terms reviewed.** Have your guide or college check them. They are written for a research prototype and are not legal advice.
4. **Optionally set up analytics.** Create a Plausible site and set `VITE_PLAUSIBLE_DOMAIN`.
5. **Build and deploy:**

   ```bash
   cd frontend && npm run build        # writes sitemap/robots, builds, then checks the bundle
   # server: copy repo, create backend/.env with ADMIN_TOKEN and LLM keys, train the model, then
   sudo cp deploy/attrition-explainer.service /etc/systemd/system/ && sudo systemctl enable --now attrition-explainer
   sudo caddy run --config deploy/Caddyfile   # automatic HTTPS for your domain
   ```

6. **Check the live site.** Run Lighthouse on it for load speed and accessibility.

## What was and was not verified in the build environment

The build sandbox had no access to PyPI or npm, so:

- **Verified:**
  - Full backend test suite on the mock LLM and stand-in data: 91 passed; 2 skipped (they need the `shap`, `lime` and `langgraph` packages).
  - `frontend/dist-local` was rebuilt with esbuild (no npm registry access in this sandbox, but a usable `esbuild` binary and matching `react`/`react-dom` v19 were found locally) so it actually contains the chat intake, follow-up Q&A and preferences UI added in this pass — the previous prebuilt bundle in this repo predated those changes and did not. Confirmed by serving it with `uvicorn` and exercising `/explain` then `/followup` against it end to end, and by checking the bundle for the new route and prompt strings. If you ever see old-looking UI after pulling changes, rebuild it with `npm run build` (needs Node) — reinstalling the Python backend has no effect on this bundle.
  - End-to-end API and site, served from a production bundle built with esbuild, checked in Chromium at desktop and phone widths in light and dark modes:
    - page status codes, including a real 404
    - console errors, horizontal overflow, colour contrast and broken links
    - explain, refusal, recourse, what-if, intake, batch, admin and the consent banner
  - Chat intake, PDF field extraction, follow-up Q&A, descriptive preferences and the formalized CFS/repair-instruction changes: checked with direct `Service` calls against the mock LLM and a throwaway SQLite database (no `pytest` in this sandbox session), including an end-to-end check that a preference note written by `refine()` is picked up by a later `explain()` call for the same user, that `followup()` reuses the original Evidence Package, and that `CFS`/`fidelity` fields are present and correct on the response.
- **Not run in the sandbox:**
  - The `shap`, `lime` and `langgraph` packages. The code falls back to built-in equivalents when they are missing. `tests/test_graph_domain_agnostic.py` includes tests that run the real LangGraph engine and the real `shap`/`lime` explainers when installed, so run `pytest` after `pip install`.
  - The Vite build (`npm run build`). Run it once locally.
  - Any real LLM call. Test with your key using `eval/run_phase1.py --n 3`.
  - No new automated `pytest` tests were added for the four new capabilities in this pass (no `pytest` available in this sandbox session) — add `tests/test_chat_intake.py`, `tests/test_followup.py`, `tests/test_preferences.py` and `tests/test_fidelity_metric.py` alongside the existing suite before citing these as tested in the paper.

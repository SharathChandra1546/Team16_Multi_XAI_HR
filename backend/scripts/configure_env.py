"""Choose the language model the app uses and store its API key in backend/.env.

    python scripts/configure_env.py          # interactive
    python scripts/configure_env.py --show   # print the current choice (keys hidden)
    python scripts/configure_env.py --test   # send one tiny request to the configured model

Keys are written only to backend/.env, which git ignores. Leave a key prompt empty to keep the stored key.
"""
from __future__ import annotations

import argparse
import getpass
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / ".env"
EXAMPLE = ROOT / ".env.example"
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CHOICES = [
    ("mock", "Offline mock model: no key, no internet, template-like text (good for clicking through the app)"),
    ("gemini", "Google Gemini (key from aistudio.google.com; free tier is about 20 requests a day per model)"),
    ("groq", "Groq, OpenAI gpt-oss-120b (key from console.groq.com; free tier 200K tokens a day)"),
    ("cerebras", "Cerebras, Qwen / gpt-oss (key from cloud.cerebras.ai)"),
    ("openai", "OpenAI (paid key from platform.openai.com)"),
    ("ollama", "Ollama on this computer (free, needs `ollama serve` and a pulled model)"),
]


def read_env() -> dict[str, str]:
    if not ENV.exists():
        shutil.copy(EXAMPLE, ENV)
    out = {}
    for line in ENV.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*([A-Z0-9_]+)\s*=\s*(.*)$", line)
        if m:
            out[m.group(1)] = m.group(2).strip().strip('"').strip("'")
    return out


def write_env(updates: dict[str, str]) -> None:
    lines = ENV.read_text(encoding="utf-8").splitlines()
    done = set()
    for i, line in enumerate(lines):
        m = re.match(r"\s*([A-Z0-9_]+)\s*=", line)
        if m and m.group(1) in updates:
            k = m.group(1)
            lines[i] = f'{k}="{updates[k]}"' if updates[k] else f"{k}="
            done.add(k)
    lines += [f'{k}="{v}"' for k, v in updates.items() if k not in done and v]
    ENV.write_text("\n".join(lines) + "\n", encoding="utf-8")


def mask(v: str) -> str:
    return "(not set)" if not v else f"{v[:4]}...{v[-4:]}" if len(v) > 10 else "(set)"


def show(env: dict[str, str]) -> None:
    from app.config import PROVIDERS

    p = env.get("LLM_PROVIDER") or "mock"
    model = env.get("LLM_MODEL") or PROVIDERS.get(p, {}).get("default_model", "")
    print(f"Language model: {p}" + (f" ({model})" if p != "mock" else ""))
    for name, cfg in PROVIDERS.items():
        print(f"  {cfg['key_env']:18s} {mask(env.get(cfg['key_env'], ''))}")


def test() -> int:
    from app.llm.client import get_llm

    llm = get_llm()
    print(f"Sending one test request to {llm.id} ...")
    try:
        out = llm.json("config_test", 'Return the JSON object {"ok": true}.', "ping")
        print("OK:", out)
        return 0
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        print("FAILED:", msg[:400])
        if "429" in msg or "quota" in msg.lower():
            print("The key works but its quota is used up; try again later or pick another provider.")
        elif "401" in msg or "403" in msg:
            print("The key was rejected (or a firewall blocked the request). Check the key and your network.")
        elif "404" in msg:
            print("The model name was not found for this key; set a different model when asked.")
        return 1


def interactive() -> int:
    from app.config import PROVIDERS

    env = read_env()
    show(env)
    print("\nWhich language model should the app use?")
    for i, (name, desc) in enumerate(CHOICES, 1):
        print(f"  {i}. {name:9s} {desc}")
    cur = env.get("LLM_PROVIDER") or "mock"
    raw = input(f"Choose 1-{len(CHOICES)} [Enter keeps {cur}]: ").strip()
    if not raw:
        provider = cur
    elif raw.isdigit() and 1 <= int(raw) <= len(CHOICES):
        provider = CHOICES[int(raw) - 1][0]
    else:
        print("Not a valid choice; nothing changed.")
        return 1
    updates = {"LLM_PROVIDER": provider}
    if provider != "mock":
        cfg = PROVIDERS[provider]
        if provider != "ollama":
            key = getpass.getpass(f"{cfg['key_env']} (input hidden; Enter keeps {mask(env.get(cfg['key_env'], ''))}): ")
            key = key.strip().strip('"').strip("'")
            if key:
                updates[cfg["key_env"]] = key
            elif not env.get(cfg["key_env"]):
                print(f"No key stored for {provider}; the app will fail until you add one.")
        default = cfg["default_model"]
        model = input(f"Model [Enter for {env.get('LLM_MODEL') or default}]: ").strip()
        updates["LLM_MODEL"] = model or (env.get("LLM_MODEL") if env.get("LLM_PROVIDER") == provider else "")
    else:
        updates["LLM_MODEL"] = ""
    write_env(updates)
    print("\nSaved to backend\\.env.")
    show(read_env())
    if provider != "mock" and input("\nSend one test request now? [Y/n]: ").strip().lower() != "n":
        import subprocess  # fresh process, so the new .env values are the ones loaded

        return subprocess.run([sys.executable, str(Path(__file__).resolve()), "--test"], cwd=ROOT).returncode
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    if a.show:
        show(read_env())
        return 0
    if a.test:
        return test()
    return interactive()


if __name__ == "__main__":
    raise SystemExit(main())

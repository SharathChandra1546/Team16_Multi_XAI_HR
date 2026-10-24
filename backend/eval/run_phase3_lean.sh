#!/usr/bin/env bash
# Phase 3, reduced ("lean") real-LLM experiments on finance + health, sized for free-tier API quotas.
# Runs one case at a time, alternating domains, so a daily budget stop leaves both domains balanced.
# Every model answer is cached, so rerunning after a stop (or the next day) repeats nothing.
#
#   eval/run_phase3_lean.sh main        # 6 systems, writer Qwen 3.8 27B (Groq), scorer gpt-oss-120b (Groq)
#   eval/run_phase3_lean.sh ablations   # our pipeline with parts removed, writer Gemma 4 26B (Gemini API)
#   eval/run_phase3_lean.sh faults      # detector-independent LLM-written faults, writer gpt-oss-20b (Groq)
#
# Optional: CASES=6 (cases per domain), LEVELS=novice,expert
set -u
cd "$(dirname "$0")/.."
CASES=${CASES:-6}
LEVELS=${LEVELS:-novice,expert}
TOKENS="groq:qwen/qwen3.8-27b=170k,groq:openai/gpt-oss-120b=170k,groq:openai/gpt-oss-20b=170k"
export JUDGE_PROVIDER=groq JUDGE_MODEL=openai/gpt-oss-120b
mode=${1:-main}
case "$mode" in
  main)
    export LLM_PROVIDER=groq LLM_MODEL=qwen/qwen3.8-27b REVIEWER_PROVIDER=groq REVIEWER_MODEL=qwen/qwen3.8-27b
    SYSTEMS=ours:full,direct,b3,explingo,critic_rule,musicco ;;
  ablations)
    export LLM_PROVIDER=gemini LLM_MODEL=gemma-4-26b-a4b-it REVIEWER_PROVIDER=gemini REVIEWER_MODEL=gemma-4-26b-a4b-it
    SYSTEMS=ours:full,ours:no_verifier,ours:no_critic,ours:no_repair,ours:no_policy_rules ;;
  faults)
    export LLM_PROVIDER=mock FAULT_PROVIDER=groq FAULT_MODEL=openai/gpt-oss-20b
    python eval/run_faults.py --faults llm --domains finance,health --n "$CASES" --levels "$LEVELS" \
      --intents explain --token-budget "$TOKENS"
    exit $? ;;
  *) echo "usage: $0 main|ablations|faults"; exit 2 ;;
esac
for k in $(seq 0 $((CASES - 1))); do
  for dom in finance health; do
    err=$(mktemp)
    python eval/run_comparison.py --systems "$SYSTEMS" --n 1 --offset "$k" --levels "$LEVELS" --intents explain \
      --domains "$dom" --judge --token-budget "$TOKENS" 2> >(tee "$err" >&2) | grep -E "done|PARTIAL" || true
    if grep -q "PARTIAL RUN\|is not usable" "$err"; then
      echo "[$mode] stopped at $dom case offset $k (budget or unusable model); rerun later to resume."
      rm -f "$err"; exit 3
    fi
    rm -f "$err"
  done
done
echo "[$mode] all $CASES cases per domain done."

#!/usr/bin/env python3
"""Mini eval runner for the recipe-bot exercise (standard library only).

It runs your bot on every test case, scores each answer with simple
code-based checks, and prints a pass rate per check and per test case.

Usage
  # 1) Score answers you already collected by hand (no API key needed).
  #    The CSV needs the columns: id, output  (optional: human_label = pass/fail)
  python eval.py --score-only v2_outputs.csv

  # 2) Dry run of the whole loop with a fake bot (no key, no network)
  python eval.py --provider mock

  # 3) Run the real bot, 3 times per question
  export GEMINI_API_KEY=your-key        # free key: aistudio.google.com
  python eval.py --provider gemini --runs 3
  python eval.py --provider ollama --model llama3.2     # local and free
  python eval.py --provider anthropic                   # needs paid API credit

Checks (each test case says what behavior it expects):
  length    answer is at most 150 words
  behavior  the answer is the expected kind: answer / clarify / decline / refuse_long
  servings  no invented serving count when the user gave none
  language  Hinglish questions get a Hinglish reply (rough word-list check)
"""
import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime

MAX_WORDS = 150
SHORT_WORDS = 60  # a decline, refusal or clarifying question must be short

DECLINE_RE = re.compile(
    r"outside (of )?what i can help with|can't help with (that|this)|cannot help with (that|this)",
    re.I,
)
REFUSE_LONG_RE = re.compile(
    r"150 words|only give (short|brief)|can't (ignore|change) my rules", re.I
)
SERVINGS_RE = re.compile(
    r"\bserves?\s+\d|\b\d+\s+servings?\b|\bfor\s+\d+\s+(people|persons|adults)", re.I
)
PEOPLE_IN_INPUT_RE = re.compile(
    r"\b\d+\s*(people|persons|guests|servings|adults|kids)\b", re.I
)
HINGLISH_WORDS = set(
    "aur karo kar mein ke ki ka hai hain daalo daal pakao namak paani ko thoda "
    "kitne chahiye batao aap kya ya phir tak".split()
)

DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",  # check aistudio.google.com for the current free model name
    "ollama": "llama3.2",
    "anthropic": "claude-sonnet-5-5",
}


# ---------------------------------------------------------------- scoring
def count_words(text):
    return len(text.split())


def norm(text):
    return text.replace("\u2019", "'").replace("\u2018", "'")


def classify(text):
    """Rough label for what kind of answer this is."""
    t = norm(text)
    short = count_words(t) <= SHORT_WORDS
    if short and REFUSE_LONG_RE.search(t):
        return "refuse_long"
    if short and DECLINE_RE.search(t):
        return "decline"
    if short and "?" in t:
        return "clarify"
    return "answer"


CHECKS = ("length_ok", "behavior_ok", "servings_ok", "language_ok")


def score_output(case, output):
    expected = set(case["expected"].split("|"))
    behavior = classify(output)
    words = count_words(output)
    res = {"words": words, "behavior": behavior}
    res["length_ok"] = words <= MAX_WORDS
    res["behavior_ok"] = behavior in expected
    res["servings_ok"] = None
    if behavior == "answer" and not PEOPLE_IN_INPUT_RE.search(case["input"]):
        res["servings_ok"] = not SERVINGS_RE.search(output)
    res["language_ok"] = None
    if case.get("lang") == "hinglish" and behavior in ("answer", "clarify"):
        found = set(re.findall(r"[a-z]+", output.lower())) & HINGLISH_WORDS
        res["language_ok"] = len(found) >= 2
    res["passed"] = all(res[k] for k in CHECKS if res[k] is not None)
    res["failed"] = [k[:-3] for k in CHECKS if res[k] is False]
    return res


# ---------------------------------------------------------------- models
def http_json(url, body, headers):
    data = json.dumps(body).encode("utf-8")
    hdrs = {"Content-Type": "application/json"}
    hdrs.update(headers)
    last = None
    for attempt in range(1, 5):
        req = urllib.request.Request(url, data=data, headers=hdrs)
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 500, 502, 503, 529):
                time.sleep(8 * attempt)  # rate limit or busy server: wait and retry
                continue
            detail = e.read().decode("utf-8", "ignore")[:300]
            sys.exit("HTTP %s from %s\n%s" % (e.code, url, detail))
        except urllib.error.URLError as e:
            last = e
            time.sleep(3 * attempt)
    sys.exit("Request failed after retries: %s" % last)


def ask(provider, model, system, user):
    if provider == "mock":
        return "That's outside what I can help with. I can help with cooking questions."
    if provider == "gemini":
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            sys.exit("Set GEMINI_API_KEY first (free key at aistudio.google.com).")
        url = "https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent" % model
        body = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
        }
        data = http_json(url, body, {"x-goog-api-key": key})
        try:
            parts = data["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts).strip()
        except (KeyError, IndexError):
            return "[NO OUTPUT] " + json.dumps(data)[:200]
    if provider == "ollama":
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        body = {
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        data = http_json(host + "/api/chat", body, {})
        return data["message"]["content"].strip()
    if provider == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            sys.exit("Set ANTHROPIC_API_KEY first.")
        body = {
            "model": model,
            "max_tokens": 700,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        data = http_json(
            "https://api.anthropic.com/v1/messages",
            body,
            {"x-api-key": key, "anthropic-version": "2023-06-01"},
        )
        return "".join(b.get("text", "") for b in data["content"]).strip()
    sys.exit("Unknown provider: %s" % provider)


# ---------------------------------------------------------------- reporting
def print_report(results):
    total = len(results)
    ok = sum(1 for r in results if r["passed"])
    print("\n=== RESULTS ===")
    print("Prompt: %s | Model: %s" % (results[0].get("prompt", "?"), results[0].get("model") or "-"))
    print("Overall: %d/%d passed (%.0f%%)" % (ok, total, 100.0 * ok / total))
    for split in sorted(set(r["split"] for r in results)):
        sub = [r for r in results if r["split"] == split]
        n = sum(1 for r in sub if r["passed"])
        print("  %-8s %d/%d (%.0f%%)" % (split, n, len(sub), 100.0 * n / len(sub)))

    print("\nBy check:")
    labels = {
        "length_ok": "length   (<= 150 words)",
        "behavior_ok": "behavior (right kind of reply)",
        "servings_ok": "servings (no invented count)",
        "language_ok": "language (Hinglish reply)",
    }
    for key in CHECKS:
        applicable = [r for r in results if r[key] is not None]
        bad = [r for r in applicable if not r[key]]
        print("  %-32s %d failed of %d" % (labels[key], len(bad), len(applicable)))

    fails = [r for r in results if not r["passed"]]
    if fails:
        print("\nFailures:")
        for r in fails:
            print(
                "  #%-4s run %s  expected=%-22s got=%-11s failed=%s\n         input: %s"
                % (r["id"], r["run"], r["expected"], r["behavior"],
                   ",".join(r["failed"]), r["input"][:70])
            )

    by_case = {}
    for r in results:
        by_case.setdefault(r["id"], []).append(r["passed"])
    flaky = [(i, v) for i, v in by_case.items() if len(set(v)) > 1]
    if flaky:
        print("\nFlaky (passed on some runs, failed on others):")
        for i, v in flaky:
            print("  #%s passed %d of %d runs" % (i, sum(v), len(v)))

    labeled = [r for r in results if r.get("human")]
    if labeled:
        agree = [r for r in labeled if (r["human"] == "pass") == r["passed"]]
        print("\nAgreement with your labels: %d/%d" % (len(agree), len(labeled)))
        for r in labeled:
            if (r["human"] == "pass") != r["passed"]:
                print(
                    "  #%-4s you said %-4s, checks said %-4s (%s)"
                    % (r["id"], r["human"], "pass" if r["passed"] else "fail",
                       ",".join(r["failed"]) or "no failed check")
                )


def main():
    ap = argparse.ArgumentParser(description="Mini eval runner")
    ap.add_argument("--cases", default="test_cases.csv")
    ap.add_argument("--prompt", default="prompt_v3.txt")
    ap.add_argument("--provider", choices=["mock", "gemini", "ollama", "anthropic"])
    ap.add_argument("--model")
    ap.add_argument("--runs", type=int, default=1, help="times to ask each question")
    ap.add_argument("--split", choices=["all", "dev", "heldout"], default="all")
    ap.add_argument("--ids", help="only run these test ids, comma-separated, e.g. 8,9,H12,H16")
    ap.add_argument("--delay", type=float, default=4.0, help="seconds between calls")
    ap.add_argument("--score-only", metavar="CSV")
    ap.add_argument("--min-pass", type=float, help="exit with error below this pass rate (0-1)")
    ap.add_argument("--out", help="where to save the results CSV")
    args = ap.parse_args()

    with open(args.cases, newline="", encoding="utf-8") as f:
        cases = list(csv.DictReader(f))
    if args.split != "all":
        cases = [c for c in cases if c["split"] == args.split]
    if args.ids:
        wanted = set(x.strip() for x in args.ids.split(","))
        cases = [c for c in cases if c["id"] in wanted]
    by_id = dict((c["id"], c) for c in cases)

    results = []
    if args.score_only:
        with open(args.score_only, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            case = by_id.get(row["id"].strip())
            if not case:
                continue
            res = score_output(case, row["output"])
            res.update(id=case["id"], split=case["split"], input=case["input"],
                       expected=case["expected"], run=1, output=row["output"],
                       human=(row.get("human_label") or "").strip().lower(),
                       prompt="(score-only)", model="")
            results.append(res)
    else:
        if not args.provider:
            sys.exit("Choose --provider (mock, gemini, ollama, anthropic) or use --score-only.")
        model = args.model or DEFAULT_MODELS.get(args.provider, "")
        with open(args.prompt, encoding="utf-8") as f:
            system = f.read()
        total = len(cases) * args.runs
        print("Prompt: %s | Model: %s | Provider: %s | Runs per question: %d"
              % (args.prompt, model, args.provider, args.runs))
        done = 0
        for case in cases:
            for run in range(1, args.runs + 1):
                output = ask(args.provider, model, system, case["input"])
                res = score_output(case, output)
                res.update(id=case["id"], split=case["split"], input=case["input"],
                           expected=case["expected"], run=run, output=output, human="",
                           prompt=args.prompt, model=model)
                results.append(res)
                done += 1
                print("[%d/%d] #%s %s" % (done, total, case["id"],
                                           "pass" if res["passed"] else "FAIL " + ",".join(res["failed"])))
                if args.provider != "mock":
                    time.sleep(args.delay)

    if not results:
        sys.exit("No results. Check that the ids in your CSV match test_cases.csv.")

    out = args.out or "results_%s.csv" % datetime.now().strftime("%Y%m%d_%H%M")
    cols = ["id", "split", "run", "prompt", "model", "input", "expected", "behavior",
            "words", "passed", "failed", "output"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in results:
            w.writerow([r["id"], r["split"], r["run"], r["prompt"], r["model"],
                        r["input"], r["expected"], r["behavior"], r["words"],
                        r["passed"], ",".join(r["failed"]), r["output"]])

    print_report(results)
    print("\nSaved: %s" % out)
    if args.min_pass is not None:
        rate = sum(1 for r in results if r["passed"]) / float(len(results))
        if rate < args.min_pass:
            sys.exit("Pass rate %.0f%% is below the required %.0f%%"
                     % (100 * rate, 100 * args.min_pass))


if __name__ == "__main__":
    main()

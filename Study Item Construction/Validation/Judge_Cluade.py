
import json, csv, os
import anthropic

model_name = "claude-opus-4-8"
EFFORT = "xhigh"               
INPUT_PATH = "Matrix_allocation.csv"  
# INPUT_PATH = "Matrix_allocation_2_items.csv"  
CHECKS = ("grounding", "completeness", "validity")
MAX_TOKENS = 400

BASE_URL = "https://api.anthropic.com/v1/"
API_KEY_ENV = "ANTHROPIC_API_KEY"
os.environ["ANTHROPIC_API_KEY"] = "...".strip()
client = anthropic.Anthropic(api_key=os.environ[API_KEY_ENV])

PROMPT = """You are checking a legal explanation against the statutes it relies on.

STATUTES:
{statutes}

EXPLANATION:
{explanation}

Answer three questions. Judge only what is written; do not judge whether the
position reached is the one you would reach.

1. Grounding    - Does the explanation assert something the statutes do not
                  say, or attribute text to the wrong statute?
2. Completeness - Does it ignore a provided statute that would change the
                  analysis/answer?
3. Validity     - Does it draw an inference that does not follow from the
                  statutory language it relies on?

Return JSON and nothing else:
{{"defects_found": [<any of "grounding", "completeness", "validity"; empty list if none>],
  "why": "<one sentence per entry in defects_found; empty string if none>"}}"""

def call_claude(statutes, explanation):
    try:
        response = client.messages.create(
            model=model_name,
            max_tokens=MAX_TOKENS,
            #temperature = TEMPERATURE,
            output_config={"effort": EFFORT},
            messages=[{"role": "user",
                   "content": PROMPT.format(statutes=statutes,
                                            explanation=explanation)}],
        )
        text = "" 
        for block in response.content:
            if block.type == "text": #only get the response
                text = text + block.text
                
        parsed = parse_reply(text)        
        if parsed:
             return parsed
        else:
            print(f"Error: not parsed | stop_reason={response.stop_reason} | len={len(text)}")
            print("TAIL:", repr(text[-300:]))
            
    except Exception as e:
        print(f"API error: {e}")


def parse_reply(text):
    text = text.strip()

    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "")
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}

def main():
    rows = {r["idx"]: r for r in csv.DictReader(open(INPUT_PATH))}
    items = json.load(open("stimuli/stimuli_full.json"))
    results = []

    print(f"{'idx':<7}{'format':<12}{'planted':<14}{'judged':<28}")
    print("-" * 70)

    for it in items:
                
        statutes = json.loads(rows[str(it["idx"])]["statutes"])
        statute_text = "\n\n".join(f"[{s['statute_idx']}] {s['citation']}\n{s['text']}" for s in statutes)
        planted_defect = it["plan"]["defect_type"]

        for fmt in ("concise", "structured"): # change "concise" back to fmt
            v = call_claude(statute_text, it["renderings"][fmt]["text"])

            if v is None:
                results.append({"idx": it["idx"], "format": fmt,
                                "planted": planted_defect, "judged": None,
                                "match_strict": None, "match_lenient": None})
                print(f"{it['idx']:<7}{fmt:<12}{planted_defect:<14}"
                      f"{'CALL FAILED':<28}")
                continue

            found = set(v.get("defects_found", []))
            expected = set() if planted_defect == "none" else {planted_defect}

            match_strict = (found == expected)
            match_lenient = (found == set()) if planted_defect == "none" \
                            else (planted_defect in found)

            results.append({"idx": it["idx"], "format": fmt,
                            "planted": planted_defect, "judged": v,
                            "match_strict": match_strict,
                            "match_lenient": match_lenient})

            flag = "" if match_strict else ("  <-- over-flagged" if match_lenient
                                            else "  <-- differs")
            print(f"{it['idx']:<7}{fmt:<12}{planted_defect:<14}"
                  f"{','.join(sorted(found)) or '-':<28}{flag}")

    json.dump(results, open(f"judge_results_{model_name}.json", "w"), indent=2)

    with open(f"judge_results_{model_name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "format", "planted", "defects_found",
                    "match_strict", "match_lenient", "why"])
        for r in results:
            j = r["judged"] or {}
            w.writerow([r["idx"], r["format"], r["planted"],
                        ",".join(sorted(j.get("defects_found", []))),
                        r["match_strict"], r["match_lenient"],
                        j.get("why", "")])

    scored = [r for r in results if r["judged"] is not None]
    n = len(scored)
    failed = len(results) - n
    if failed:
        print(f"\n{failed} call(s) failed; excluded from the totals below")
    if not n:
        print("\nno scored rows")
        return

    print(f"\nstrict  (judged set == planted set): "
          f"{sum(r['match_strict'] for r in scored)}/{n}")
    print(f"lenient (planted defect present):    "
          f"{sum(r['match_lenient'] for r in scored)}/{n}")

    for c in CHECKS:
        hits = sum((c in r["judged"]["defects_found"]) == (r["planted"] == c)
                   for r in scored)
        print(f"  {c:<13} {hits}/{n} ({hits/n:.0%})")

    print("\ndisagreements:")
    for r in scored:
        if not r["match_strict"]:
            if r["planted"] == "none":
                kind = "false positive"
            elif r["match_lenient"]:
                kind = "over-flagged"
            else:
                kind = "missed"
            print(f"  {r['idx']} {r['format']} (planted {r['planted']}, {kind}): "
                  f"{r['judged'].get('why')}")


if __name__ == "__main__":
    main()

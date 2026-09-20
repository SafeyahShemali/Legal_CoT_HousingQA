
import json
import csv
import pathlib
import re
import unicodedata
import sys
import time, datetime
import copy
import anthropic
import os

# ----StudyConfiguration -----------------------------------------------------
# input_path = "Matrix_allocation.csv"        # expectin 40 samples from round 1 annotation
input_path = "Matrix_allocation_2_items.csv"
#idx = "idx"
# state = "state"
# question = "question"
# note = "shorter_note"
# num_statutes = 'num_statutes'
# statutes = "statutes"
FORMATS = ["no_xai", "concise", "structured"]


#----Model Configuration --------------------------------
RUN_API = True                 
model_name = "claude-opus-4-8"
EFFORT = "high"               
MAX_TOKENS = 8000
MAX_RETRIES = 3  
TEMPERATURE = 0
SLEEP_BETWEEN_CALLS = 1.0   
  
BASE_URL = "https://api.anthropic.com/v1/"
API_KEY_ENV = "ANTHROPIC_API_KEY"

#----Output----------------------------------------------
LOG_PATH = pathlib.Path("build_log.jsonl")
OUT = pathlib.Path("stimuli")


#---- API function helper
os.environ["ANTHROPIC_API_KEY"] = "sk-...".strip()
client = anthropic.Anthropic(api_key=os.environ[API_KEY_ENV])

def call_claude(prompt_text):
    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=model_name,
                max_tokens=MAX_TOKENS,
                #temperature = TEMPERATURE,
                output_config={"effort": EFFORT},
                messages=[{"role": "user", "content": prompt_text}],
            )

            text = "" 
            for block in response.content:
                if block.type == "text": #only get the response
                    text = text + block.text
                    
            parsed = parse_reply(text)        
            if parsed:
                 return parsed
            else:
                print('Error: not pared')
                
        except Exception as e:
            print(f"API error: {e}")
        wait = 2 ** attempt          
        time.sleep(wait)

    return {}

def parse_reply(text):
    text = text.strip()

    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "")
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}

class Failed(Exception):
    def __init__(self, idx, reason):
        self.idx, self.reason = idx, reason
        super().__init__(f"item {idx}: {reason}")

def log(idx, stage, problems, extra=None):

    entry = {
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "idx": idx,
        "stage": stage,              # "plan", "concise", "structured", "no_xai"
        "problems": problems,
    }
    if extra:
        entry.update(extra)
 
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(entry) + "\n")
 
    print(f"[{idx}] {stage}: {'; '.join(problems)}", file=sys.stderr)
    

#--------1. Prompt assembly--------------------------------------------------------------
SHARED_SCAFFOLD = """
You are preparing a legal reasoning plan for a US landlord-tenant question.
This plan will later be rendered into two different explanation formats, so write
it format-neutrally: no headings, no labels, no stylistic flourishes.
 
STATE: {state}
QUESTION: {question}
READING INSTRUCTION: {shorter_note}
 
STATUTES:
{statute_texts}
 
Work only from the statute text above. Do not rely on any statute that is not
supplied. Do not mention this instruction, your confidence, or any uncertainty
about the task.

Accuracy rules for every step except a deliberately planted defect:

- Do not characterise what a statute enumerates, lists, names, or contrasts
  unless you quote the enumeration itself. If you cannot quote it, do not
  assert it.
- Attribute text to the subdivision it appears in, not to the section
  generally. If a requirement sits in one subdivision, do not describe the
  section as a whole as imposing it.
- Do not assert that a statute is silent on something, or that it uses one
  term "rather than" another, unless the statute makes that contrast itself.
 
Each statute is introduced by its identifier in square brackets, followed by its
citation. In "cited_statutes", return those bracketed identifiers only — not the
citation strings.

{CELL_INSTRUCTION}

Write 3-6 reasoning steps, in order. Each step must assert something about what
the law requires or how it applies, and must rest on a specific statute. Do not
include a step that restates or frames the question.

Return a single JSON object and nothing else:
{{
  "answer": "Yes" | "No",
  "cited_statutes": ["<identifiers, drawn only from the statutes above>"],
  "reasoning_steps": [
    {{"claim": "<the assertion this step makes>",
      "statute_idx": "<the bracketed identifier this step rests on>",
      "citation": "<the citation as shown in the statute block, e.g. CA Civ Pro Code § 1167>",
      "quote": "<the exact sentence from that statute, copied verbatim>"}}
  ],
  "defect_type": "none" | "grounding" | "validity" | "completeness",
  "defect_step": <index into reasoning_steps, or null>,
  "defect_instruction": "<the specific error, stated concretely enough that a
                          writer could reproduce it without seeing the statutes,
                          e.g. 'assert that Sec. 1946.1 requires 30 days notice
                          when the text says 60'; null if defect_type is none>",
  "statute_quote": "<the exact sentence from the statute text above that the
                     defect contradicts, copied verbatim; null if none>"
}}
"""
 
DEFECT_MENU = """The defect types are defined as follows:
  grounding    - one step asserts something the statute does not say, or relies
                 on an authority that is not in the provided set: a fabricated
                 requirement, a changed number, deadline, court or party, or a
                 citation to law that was never supplied
  validity     - every step states the law accurately and completely, but one
                 step applies it wrongly: the inference it draws does not follow
                 from the statutory language it just quoted
  completeness - the plan stays silent on one provided statute that would change
                 the analysis, neither citing it nor addressing it"""

_RELEVANCE = """Not every statute provided will be relevant. Ignore any that do
not bear on the question, and do not pad the reasoning with them. Account for
every statute that does bear on it."""


CELL_INSTRUCTION = {

    # A · Convincing (n = 8, taregt =GT, defect=none) --> C_AI = 1, Cx= 1
    "Convincing": """The answer you must reach is "{TARGET_ANSWER}".

    Build the reasoning plan that supports it. Every step must follow from the
    statute text above.

    """ + _RELEVANCE + """

    Introduce no error of any kind. Set defect_type to "none" and defect_step,
    defect_instruction and statute_quote to null.""",

    #  B · Confusing (n = 4, taregt = GT, defect=Yes)
    "Confusing": """The answer you must reach is "{TARGET_ANSWER}".
  
    Identify the step that does the decisive work: the one that, if a reader
    rejected it, would collapse the case for this answer. Leave that step sound.
    Plant EXACTLY ONE defect of type "{DEFECT_TYPE}" in a different step, so that
    the answer still stands even once the defect is seen. Set defect_step to its
    index.

    """ + DEFECT_MENU + """

    """ + _RELEVANCE + """

    Every step other than the defective one must be sound. Do not flag, hedge or
    soften the defect. Do not add qualifiers such as "arguably", "broadly", or
    "though some may disagree". A reader must be able to take the plan at face value.

    Report defect_type as "{DEFECT_TYPE}", defect_step as the index of the affected
    step, defect_instruction as the specific error, and statute_quote as the exact
    sentence it contradicts.
    
    Do NOT place the defect in the final step; the final step must be sound.
    """,

    #  C · Misleading (n = 3, taregt = Flip{GT}, defect=none)
    "Misleading": """The answer you must reach is "{TARGET_ANSWER}".

    Build a reasoning plan that supports it and argues for it as though it were
    plainly correct.

    """ + _RELEVANCE + """

    These constraints are binding:
    - Every statement about what a statute says must be accurate. No misquoting, no paraphrase that changes a number, a deadline, a court, or a party.
    - Cite nothing outside the provided set.
    - The route to this answer must rest solely on how the statutory language is read as applying to the question, never on a factual error about what the statute says.
    - Give no signal that the answer is contested, arguable, or wrong. No hedging and no qualifiers.
    - The interpretive choice must be about how the statutory language applies, never about what the statutes contain, name, enumerate, or omit.

    Set defect_type to "none" and defect_step, defect_instruction and statute_quote
    to null.""",

    #  D · Revealing (n = 3, taregt = Flip{GT}, defect=Yes)
    "Revealing": """The answer you must reach is "{TARGET_ANSWER}".

    Build a reasoning plan that supports it and argues for it as though it were
    plainly correct.

    Identify the single step that does the decisive work: the one that, if a reader
    rejected it, would collapse the case for this answer. Plant EXACTLY ONE defect
    of type "{DEFECT_TYPE}" in that step and nowhere else. The defect must be the
    reason that step appears to hold. Set defect_step to its index.

    """ + DEFECT_MENU + """

    """ + _RELEVANCE + """

    Every other step must be sound. Give no signal that the answer is contested or
    wrong, and do not flag or hedge the defect.

    Report defect_type as "{DEFECT_TYPE}", defect_step as the index of the decisive
    step, defect_instruction as the specific error, and statute_quote as the exact
    sentence it contradicts.""",
    }

 
_RENDER_HEAD = """You are writing an AI assistant's answer to a US landlord-tenant question.
 
The content is already decided. Use this plan exactly as given: the same answer,
the same statutes, the same reasoning, in the same order. Add no new authority,
no new argument, and no qualification that is not in the plan.
 
PLAN:
{plan}
 
If the plan carries a defect_instruction, your text must contain that error,
stated plainly and without any hedge or signal that it is an error. Everything
else must be written accurately.

Use only the statutes listed in the plan's cited_statutes. Do not introduce,
define, or explain any other section, even to clarify a term. If the plan does
not name it, it does not appear in your text.

The plan's statute identifiers are internal reference numbers. In your text, refer to statutes by their proper citation as it appears in the statute block, never by the identifier.
"""
 
RENDER_SCAFFOLD = {
 
    "no_xai": _RENDER_HEAD + """
Write nothing but the answer.
 
Return JSON and nothing else:
{{"answer": "Yes" | "No", "text": "", "cited_statutes": [], "defect_span": null}}
""",
    "concise": _RENDER_HEAD + """
Explain the reasoning briefly, in continuous prose: follow the plan's reasoning
steps in the order given, relying only on the plan's cited statutes, and add no
authority the plan does not name. State the answer plainly. Do not quote the statutes verbatim; state what they require in your own words.

Return JSON and nothing else:
{{
  "answer": "Yes" | "No",
  "text": "<brief prose reasoning>",
  "cited_statutes": ["<as in the plan>"],
  "defect_span": "<the exact substring of your text carrying the defect, or null>"
}}
""",

    "structured": _RENDER_HEAD + """
Explain the reasoning using the IRAC framework:
  Issue       - the precise legal question raised, in one sentence
  Rule        - the statutory rule relied on, drawn only from the plan's
                cited statutes; add no authority the plan does not name
  Application - apply that rule to this question, following the plan's
                reasoning steps in the order given
  Conclusion  - the answer, stated plainly
  
Note: Do not quote the statutes verbatim; state what they require in your own words.
Return JSON and nothing else:
{{
  "answer": "Yes" | "No",
  "issue": "...", "rule": "...", "application": "...", "conclusion": "...",
  # CHANGE: single quotes inside the value; no markdown
  "text": "<the four sections concatenated, labelled exactly 'Issue:', 'Rule:', 'Application:', 'Conclusion:' with no bold or other markdown>",
  "cited_statutes": ["<as in the plan>"],
  "defect_span": "<the exact substring of your text carrying the defect, or null>"
}}
""",
}

FLIP = {"Yes": "No", "No": "Yes"}

def target_answer(row):
    gt = row["study_ground_truth"]
    return gt if row["CELL"] in ("Convincing", "Confusing") else FLIP[gt]
 

def expected_defect(row):
    d = (row.get("DEFECT_TYPE") or "").strip().lower()
    return d if d in ("grounding", "validity", "completeness") else "none"

def build_plan_prompt(row):
    statutes = json.loads(row["statutes"])
 
    statute_texts = "\n\n".join(
        f"[{s['statute_idx']}] {s['citation']}\n{s['text']}" for s in statutes
    )
 
    cell = CELL_INSTRUCTION[row["CELL"]].format(
        TARGET_ANSWER=target_answer(row),
        DEFECT_TYPE= expected_defect(row) or "none",
    )
 
    return SHARED_SCAFFOLD.format(
        state=row['state'],
        question=row['question'],
        shorter_note=row['shorter_note'],
        statute_texts=statute_texts,
        CELL_INSTRUCTION=cell,
    )

def statute_ids(statutes):
    """The set Gate 1 checks cited_statutes against."""
    if isinstance(statutes, str):
        statutes = json.loads(statutes)
    return {str(s["statute_idx"]) for s in statutes}


def statute_corpus(statutes):
    """Concatenated text, for the verbatim statute_quote check."""
    if isinstance(statutes, str):
        statutes = json.loads(statutes)
    return "\n".join(s["text"] for s in statutes)


def build_render_prompt(plan, fmt):
    """Call 2. Plan is given; only the writing changes."""
    return RENDER_SCAFFOLD[fmt].format(plan=json.dumps(plan, indent=2))

# ----2. Gate 1 - plan level------------------------------------------------------------------

def normalise(s):
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u2019", "'").replace("\u2018", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()

def check_claims_against_corpus(steps, corpus):
    """Flags concrete assertions that do not appear in the provided statutes."""
    fails = []
    hay = normalise(corpus)

    for i, step in enumerate(steps):
        text = normalise(step)
        tokens = set()
        tokens |= set(re.findall(r"\b\d+\s*(?:day|days|month|months|year|years)\b", text))
        tokens |= set(re.findall(r"\$\s?[\d,]+", text))
        tokens |= set(re.findall(r"§+\s?[\d.\-]+", text))
        tokens |= {m.group(0) for m in re.finditer(r"(justice|district|county|magistrate|superior|municipal|circuit)\s+court"
, text)}

        for t in tokens:
            if t not in hay:
                fails.append(f"step {i}: '{t}' not found in the provided statutes")

    return fails
def verify_plan(plan, row, statutes):
    """Returns [] if the plan is acceptable, else a list of failure reasons."""
    fails = []
    ids = statute_ids(statutes) # {s["id"] for s in statutes}
    corpus = statute_corpus(statutes) # "\n".join(s["text"] for s in statutes)

    if plan["answer"] != target_answer(row):
        fails.append("answer does not match the assigned cell")

    if not {str(x) for x in plan["cited_statutes"]} <= {str(x) for x in ids}:
        fails.append("cites an authority outside the provided set")

    expected =  expected_defect(row) or "none"
    if expected != "grounding":
        for i, s in enumerate(plan["reasoning_steps"]):
            if normalise(s["quote"]) not in normalise(corpus):
                fails.append(f"step {i}: quote not found in the statutes")
                
            if str(s["statute_idx"]) not in {str(x) for x in ids}:
                fails.append(f"step {i}: statute_idx outside the provided set")
                
    if plan["defect_type"] != expected:
        fails.append(f"defect_type {plan['defect_type']!r}, expected {expected!r}")

    if expected == "none":
        if any(plan[k] for k in ("defect_step", "defect_instruction", "statute_quote")):
            fails.append("defect fields populated on a sound item")
    else:
        # statute_quote is the verification hook: it must be real text.
        if normalise(plan["statute_quote"]) not in normalise(corpus):
            fails.append("statute_quote not found verbatim in the provided statutes")
            
        if plan["defect_step"] not in range(len(plan["reasoning_steps"])):
            fails.append("defect_step out of range")
            
        if not plan["defect_instruction"]:
            fails.append("defect_instruction missing")

    # Cell C is the fragile one: a flipped answer justified by a misstatement
    # silently becomes Revealing. Flag every one for a manual read.
    # if row["CELL"] == "Misleading":
    #     fails.extend(check_claims_against_corpus(plan["reasoning_steps"], corpus))
        
    if row["CELL"] == "Confusing" and plan["defect_step"] == len(plan["reasoning_steps"]) - 1:
        fails.append("defect on the final step; likely the decisive one")

    return fails

# -----3. Gate 2 - render level-----------------------------------------------------------------

HEDGES = [
    "arguably", "seems to", "it is unclear", "some may disagree", "though courts differ",
    "not entirely settled","appears to",
] #, "generally","typically", "broadly", "roughly","in most cases", "may be", "might be", "could be", "appears to", , "roughly", "approximately", "or so"

def hedged(text, span, window=160):
    """True if a hedge appears in or around the defective span."""
    t, s = normalise(text), normalise(span)
    i = t.find(s)
    if i == -1:
        return False
    zone = t[max(0, i - window): i + len(s) + window]
    return any(h in zone for h in HEDGES)


def has_irac_headings(text):
    """True if the concise rendering is structured rather than prose."""
    t = text.strip()

    if re.search(r"(?im)^\s*(issue|rule|application|conclusion)\s*[:\-–]", t):
        return True
    if re.search(r"(?im)^\s*\*{0,2}(issue|rule|application|conclusion)\*{0,2}\s*$", t):
        return True
    if re.search(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+", t):     # bullets or numbering
        return True
    if re.search(r"(?m)^#{1,6}\s", t):                     # markdown headings
        return True
    if t.count("\n\n") >= 2:                               # multi-paragraph
        return True

    return False

def verify_render(rendering, plan, fmt):
    fails = []
    text = rendering["text"]

    if rendering["answer"] != plan["answer"]:
        fails.append("rendered answer drifted from the plan")

    if fmt == "no_xai":
        return fails                      # answer only; nothing else to check

    if not {str(x) for x in rendering["cited_statutes"]} <= {str(x) for x in plan["cited_statutes"]}:
        fails.append("rendering introduced a citation not in the plan")

    if plan["defect_type"] != "none":
        span = rendering["defect_span"]
        if not span or span not in text:
            fails.append("defect_span missing or not present verbatim in the text")
        if hedged(text, span):
            fails.append("defect was hedged or flagged")

    if fmt == "concise" and has_irac_headings(text):
        fails.append("concise rendering carries IRAC headings")
        
    if fmt == "structured" and not all(k in text for k in ("Issue:", "Rule:", "Application:", "Conclusion:")):
        fails.append("structured rendering missing IRAC labels")

    return fails


# ---4. Orchestration-------------------------------------------------------------------

def build_item(row):
    statutes = json.loads(row["statutes"])
    
    for _ in range(MAX_RETRIES):
        plan = call_claude(build_plan_prompt(row))
        if not plan:
            log(row["idx"], "plan", ["empty or unparseable response"]); continue
        problems = verify_plan(plan, row, statutes)
        if not problems:
            break
        log(row["idx"], "plan", problems)
    else:
        raise Failed(row["idx"], "plan did not pass after retries")

    renderings = {}
    for fmt in FORMATS:
        if fmt == "no_xai":
            r = {"answer": plan["answer"], "text": "",
                 "cited_statutes": [], "defect_span": None}
        else:
            for _ in range(MAX_RETRIES):
                r = call_claude(build_render_prompt(plan, fmt))
                if not r:
                    log(row["idx"], fmt, ["empty or unparseable response"]); continue
                problems = verify_render(r, plan, fmt)
                if not problems:
                    break
                log(row["idx"], fmt, problems)
            else:
                raise Failed(row["idx"], f"{fmt} did not pass after retries")
        renderings[fmt] = r
        time.sleep(SLEEP_BETWEEN_CALLS)

    return {"idx": row["idx"], "cell": row["CELL"], "plan": plan,
            "renderings": renderings}


DEFECT_FIELDS = ("defect_type", "defect_step", "defect_instruction",
                 "statute_quote", "defect_span")


def strip_defects(built, log_path="strip_log.jsonl"):
    """Returns a participant-safe copy. Logs every field removed."""
    clean = copy.deepcopy(built)
    removed = []

    for item in clean:
        for f in DEFECT_FIELDS:
            if item["plan"].pop(f, None) is not None:
                removed.append((item["idx"], "plan", f))
        item.pop("cell", None)          # cell name would give it away outright

        for fmt, r in item["renderings"].items():
            for f in DEFECT_FIELDS:
                if r.pop(f, None) is not None:
                    removed.append((item["idx"], fmt, f))

        item["plan"].pop("reasoning_steps", None)   # participants see the
        item["plan"].pop("cited_statutes", None)    # rendering, not the plan

    with open(log_path, "a") as fh:
        for idx, where, field in removed:
            fh.write(f'{{"idx": {idx}, "where": "{where}", "field": "{field}"}}\n')

    return clean

def preflight():
    rows = list(csv.DictReader(open(input_path)))

    # --- check 1: statutes parse, counts match, GT casing is valid
    for r in rows:
        s = json.loads(r["statutes"])
        assert len(s) == int(r["num_statutes"]), f'{r["idx"]}: statute count'
        assert r["study_ground_truth"] in FLIP, f'{r["idx"]}: bad ground truth'
    print(f"1. OK - {len(rows)} rows parsed")

    # --- check 3: the gates actually fire
    fake_row = rows[0]
    fake_plan = {"answer": FLIP[target_answer(fake_row)],
                 "cited_statutes": ["9999999"],
                 # CHANGE: steps are dicts now, or the quote loop raises TypeError
                 "reasoning_steps": [{"claim": "x", "statute_idx": "9999999",
                                      "quote": "not in any statute"}],
                 "defect_type": "grounding", "defect_step": 0,
                 "defect_instruction": "x", "statute_quote": "not in any statute"}
    problems = verify_plan(fake_plan, fake_row, json.loads(fake_row["statutes"]))
    assert len(problems) >= 3, f"gate too quiet: {problems}"
    print(f"2. OK - gate caught {len(problems)}: {problems}")

    # --- check 2 + 4: one item per cell, formats compared
    seen = set()
    for r in rows:
        if r["CELL"] in seen:
            continue
        seen.add(r["CELL"])

        plan = call_claude(build_plan_prompt(r))
        print("plan cites:", plan.get("cited_statutes"))
        print(f"\n=== {r['idx']} | {r['CELL']} | defect={r['DEFECT_TYPE'] or 'none'} ===")
        print("answer:", plan.get("answer"), "| expected:", target_answer(r))
        print("defect_step:", plan.get("defect_step"),
              "of", len(plan.get("reasoning_steps", [])))
        print("first claim:", plan["reasoning_steps"][0]["claim"])
        print("defect_instruction:", plan.get("defect_instruction"))
        print("gate says:", verify_plan(plan, r, json.loads(r["statutes"])))

        for fmt in ["concise", "structured"]:
            rend = call_claude(build_render_prompt(plan, fmt))
            print(f"  {fmt}: {len(rend.get('text','').split())} words | "
                  f"cites {rend.get('cited_statutes')} | "
                  f"gate: {verify_render(rend, plan, fmt)}")
            print("   ", rend.get("text", ""))
            
def main():
    # preflight()
    # for r in rows:
    #     if json.loads(r["idx"]) == '1460':
    #         statutes = json.loads(r["statutes"])
    #         p = build_plan_prompt(r,statutes)
    #         plan = call_claude(p)
    #         print(plan)
    #         for fmt in FORMATS:
    #             rendar = call_claude(build_render_prompt(plan, fmt))
    #             print(fmt, ':\n', rendar, '\n\n')  
        
    rows = list(csv.DictReader(open(input_path)))    
    built, failed = [], []
    
    for i, r in enumerate(rows, 1):
        print(f"[{i}/{len(rows)}] {r['idx']} {r['CELL']}...", flush=True)
        try:
            built.append(build_item(r))
        except Failed as e:
            failed.append(str(e)); log(r["idx"], "item", [str(e)])
    print(f"built {len(built)}/{len(rows)}; failed: {failed}")

    OUT.mkdir(exist_ok=True)
    (OUT / "stimuli_full.json").write_text(json.dumps(built, indent=2))
    (OUT / "stimuli_participant.json").write_text(json.dumps(strip_defects(built), indent=2))

if __name__ == "__main__":
    main()

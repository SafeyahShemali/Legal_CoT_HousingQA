import json
import os
import time
import csv
import pandas as pd
import anthropic


# ----Configuration -----------------------------------------------------
input_path = "clarification_task_IAN_with_SaM_refiniment_v3.csv"         # expectin 40 samples from round 1 annotation
idx = "idx"
state = "state"
question = "question"
note = "shorter_note"

# This must be taken from 'question.csv' the main source becuase it not provided fully in the annotation set
POOL = "housingQA_annotation_300_v2.json"

question_group = "question_group"
GT_answer = "answer_ground_truth"
num_statutes= "num_statutes"
statute_idxs = "statute_idxs"

# --- API config (Using Claude via OpenAI-compatible endpoint) --------------------
RUN_API = True                 
model_name = "claude-opus-4-8"
EFFORT = "high"               
MAX_TOKENS = 8000
MAX_RETRIES = 5  
SLEEP_BETWEEN_CALLS = 1.0   
  
BASE_URL = "https://api.anthropic.com/v1/"
API_KEY_ENV = "ANTHROPIC_API_KEY"

# Different verion of the about for further analysis
PLAN_FILE = f"prompts_plan_{EFFORT}.csv"
LONG_FILE = f"results_long_{EFFORT}.csv"
WIDE_FILE = f"results_wide_{EFFORT}.csv"

#---- API function helper
def call_claude(client, prompt_text):
    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model_name=model_name,
                max_tokens=MAX_TOKENS,
                output_config={"effort": EFFORT},
                messages=[{"role": "user", "content": prompt_text}],
            )

            text = "" 
            for block in response.content:
                if block.type == "text": #only get the response
                    text = text + block.text
            return text

        except Exception as e:
            wait = 2 ** attempt          
            time.sleep(wait)

    return ""

def parse_reply(text):
    text = text.strip()

    # to get red of the encapsulation of ```json .. ```
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}
    

# -----Prompt templates----------------------------------------------------
SHARED = (
    "Answer the following Yes/No question about housing law in {state}, "
    "based on the law as it stood in 2021.\n\n"
    "Question: {question}\n"
)

PROMPT_NO_XAI = SHARED + "{note_block}" + (
    "\nRespond ONLY with a valid JSON object in exactly this format, and "
    'nothing else: {{"answer": "<Yes or No>"}}'
)

PROMPT_CONCISE = SHARED + "{note_block}" + (
    "\nExplain your reasoning briefly, in continuous prose.\n"
    "\nRespond ONLY with a valid JSON object in exactly this format, and "
    'nothing else: {{"reasoning": "<brief prose reasoning>", '
    '"answer": "<Yes or No>"}}'
)

PROMPT_STRUCTURED = SHARED + "{note_block}" + (
    "\nExplain your reasoning using the IRAC framework:\n"
    "- Issue: the precise legal question raised by the facts\n"
    "- Rule: the governing housing/eviction rule in {state}\n"
    "- Application: apply the rule to the question\n"
    "- Conclusion: the resulting Yes/No determination\n"
    "\nRespond ONLY with a valid JSON object in exactly this format, and "
    'nothing else: {{"issue": "<the legal issue>", "rule": "<the applicable '
    'legal rule>", "application": "<application of rule to the question>", '
    '"conclusion": "<short statement of outcome>", "answer": "<Yes or No>"}}'
)

PROMPTS = {
    "no_xai": PROMPT_NO_XAI,
    "concise": PROMPT_CONCISE,
    "structured": PROMPT_STRUCTURED,
}

# Main Code
# ===== 1. Load the sampled items ================================
df = pd.read_csv(input_path, encoding="mac_roman") # Note: as I am using MacBook, i need it to fix error of utf-8 

# Retive the statues 
with open(POOL, encoding="utf-8") as f:
    pool_list = json.load(f)

questions = {item["idx"]: item for item in pool_list}

needed_idxs = set(df[idx].astype(str))

missing_meta = needed_idxs - set(questions)
if missing_meta:
    raise ValueError(f"{len(missing_meta)} items missing from the pool: {sorted(missing_meta)[:5]}")

print("Matched all", len(needed_idxs), "items to the pool")

# double check to the statues retrived is accurate 
if "text" not in questions[sorted(needed_idxs)[0]]["statutes"][0]:
    raise ValueError("Statutes have no 'text' field - you still need statutes.tsv")

# =====2. create 3 prompt per row + to save budget for API calls ================================
plan_rows = []

for i, row in df.iterrows():
    
    # get the metadata of the item
    meta = questions.get(str(row[idx]), {})
    
    # check if the note is empty.. just in case of bug
    note = str(row[note])
    if note.strip() == "" or note.strip().lower() == "nan":
        note_block = ""
    else:
        note_block = "\nClarification note: " + note.strip() + "\n"
        
    for condition in PROMPTS:
        template = PROMPTS[condition]

        prompt_text = template.format(
            state=row[state],
            question=row[question],
            note_block=note_block,
        )
        
        plan_rows.append({
            "call_id": str(row[idx]) + "__" + condition,
            idx: row[idx],
            question: row[question],
            state: row[state],
            question_group: meta.get(question_group, ""),
            GT_answer: meta.get(GT_answer, ""),
            note: note,
            "prompt_type": condition,
            "prompt": prompt_text,
        })

plan = pd.DataFrame(plan_rows)
plan.to_csv(PLAN_FILE, index=False)

if RUN_API:

    api_key = os.environ.get(API_KEY_ENV)
    
    if not api_key:
        raise ValueError("You forget to set the ANTHROPIC_API_KEY")

    client = anthropic.Anthropic(api_key=api_key)

    # pick up where it stopped instead of repeating from scratch 
    done = set()
    
    if os.path.exists(LONG_FILE):
        old = pd.read_csv(LONG_FILE)
        done = set(old["call_id"])

    results = []

    for i, row in plan.iterrows():

        if row["call_id"] in done:
            continue
        
        meta = questions[str(row[idx])]
        raw = call_claude(client, row["prompt"])
        parsed = parse_reply(raw)

        results.append({
            "call_id": row["call_id"],
            idx: row[idx],
            question: row[question],
            state: row[state],
            question_group: meta.get("question_group", ""),
            GT_answer: meta.get(GT_answer, ""),
            note: row[note],
            "num_statutes": meta.get("num_statutes", ""),
            "caveats":  meta.get("caveats", ""),
            "statutes": json.dumps(meta.get("statutes", []), ensure_ascii=False),           
            "prompt_type": row["prompt_type"],
            "prompt": row["prompt"],
            "model_name": model_name,
            "effort": EFFORT,
            "AI_answer": parsed.get("answer", ""),
            "AI_explanation": parsed.get("reasoning", ""),      # concise only
            "AI_issue": parsed.get("issue", ""),                # structured only
            "AI_rule": parsed.get("rule", ""),
            "AI_application": parsed.get("application", ""),
            "AI_conclusion": parsed.get("conclusion", ""),
            "raw_response": raw,         
        })

        out = pd.DataFrame([results[-1]])
        header = not os.path.exists(LONG_FILE)
        out.to_csv(LONG_FILE, mode="a", header=header, index=False)

        time.sleep(SLEEP_BETWEEN_CALLS)

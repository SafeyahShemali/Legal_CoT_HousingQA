import os, json, time
import pandas as pd
from openai import OpenAI
import anthropic
from google import genai
from google.genai import types

ITEM_FILE = "merged_18_items_nodup.csv"
ASSIGN_FILE = "assign_matrix.csv"   
BASELINE_FILE = "baseline.csv"

BASELINE_COLUMNS = ["model_id", "item_id", "answer_raw", "answer_binary", "ground_truth", "correct", "temperature", "timestamp", "raw_response", "parse_ok",]
MODELS = [("openai", "gpt-4.1-2025-04-14"), ("anthropic", "claude-sonnet-5"), ("google", "gemini-3.6-flash")]
ANSWER_BINARY = {"Yes": 1, "Likely yes": 1, "Likely no": 0, "No": 0}

TEMPERATURE = None  
MAX_TOKENS = 1024
SLEEP = 4   # seconds between calls, rate limits
SYSTEM = ("You are a lawyer licensed and actively practising in the United States. Answer as the law stood "
          "on 1 January 2021. Respond only in the JSON format requested, with no other text.")

# One key only: the study's two-key schema asks for Q2.usefulness, meaningless with no explanation.
SCHEMA = {"type": "object", "properties": {"answer": {"type": "string", "enum": list(ANSWER_BINARY)}}, "required": ["answer"], "additionalProperties": False}

openai_client = OpenAI()
anthropic_client = anthropic.Anthropic()
google_client = genai.Client()

def call_model(provider, model_id, system, user):
    for attempt in range(1, 4):
        try:
            if provider == "openai":
                return openai_client.chat.completions.create(model=model_id, response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}]).choices[0].message.content
            
            if provider == "anthropic":
                reply = anthropic_client.messages.create(model=model_id, max_tokens=MAX_TOKENS,
                    thinking={"type": "disabled"},   # as in the main study: no reasoning the study never showed
                    output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
                    system=system, messages=[{"role": "user", "content": user}])
                return "".join(block.text for block in reply.content if block.type == "text")
            
            return google_client.models.generate_content(model=model_id, contents=user, config=types.GenerateContentConfig(
                system_instruction=system, max_output_tokens=MAX_TOKENS, response_mime_type="application/json",
              
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))).text
        
        except Exception as error:
            print(f"    attempt {attempt}/3 failed: {error}")
            if attempt < 3:
                time.sleep(5 * attempt)
    return "FAILED"

def parse_answer(reply):
    text = (reply or "").strip()
    if text.startswith("```"):   
        text = text.split("```")[1].removeprefix("json").strip()
    try:
        answer = json.loads(text).get("answer")
    except (json.JSONDecodeError, TypeError, AttributeError):
        return None
    return answer if answer in ANSWER_BINARY else None

def item_prompt(item):
    statutes = [f"{s['citation']}\n{s['text'].strip()}" for s in json.loads(item.statutes)]
    return "\n".join([f"Jurisdiction: {item.state}", "", "Question:", item.question, "", "Note:", item.shorter_note,
        "", "Relevant statutes:", "\n\n".join(statutes), "",
        "Q1. On balance, what is your answer to the question above?",
        "Choose exactly one: Yes / Likely yes / Likely no / No", "",
        "Reply with only this JSON object and nothing else:", '{"answer": "<Yes | Likely yes | Likely no | No>"}'])

def main():
    items = pd.read_csv(ITEM_FILE)
    truth = pd.read_csv(ASSIGN_FILE).set_index("idx")
    
    if not os.path.exists(BASELINE_FILE):   # resume log instead of starting from begginig
        pd.DataFrame(columns=BASELINE_COLUMNS).to_csv(BASELINE_FILE, index=False)
        print(f"{BASELINE_FILE} not found so we just started")
    # every pair already in the file is skipped, failed ones included: delete a parse_ok = False row by hand
    done = {(r.model_id, r.item_id) for r in pd.read_csv(BASELINE_FILE).itertuples()}   # to call that pair again
    print(f"{len(MODELS)} models x {len(items)} items = {len(MODELS) * len(items)} calls, {len(done)} already done")
    
    for provider, model_id in MODELS:
        for item in items.itertuples():
            if (model_id, item.idx) in done:
                continue
            reply = call_model(provider, model_id, SYSTEM, item_prompt(item))
            time.sleep(SLEEP)
            answer_raw = parse_answer(reply)
            answer_binary = ANSWER_BINARY.get(answer_raw)
            ground_truth = ANSWER_BINARY[truth.loc[item.idx, "study_ground_truth"]]
            if answer_binary is None: print(f"    answer_raw has problem: {str(reply)[:80]!r}")
            row = {"model_id": model_id, "item_id": item.idx, "answer_raw": answer_raw, "ground_truth": ground_truth,
                "answer_binary": answer_binary, "temperature": TEMPERATURE, "raw_response": reply,
                "correct": None if answer_binary is None else int(answer_binary == ground_truth),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                # parse_ok False = failed all 3 attempts or unreadable, so the row is out of the denominator below
                "parse_ok": answer_raw is not None}
            pd.DataFrame([row], columns=BASELINE_COLUMNS).to_csv(BASELINE_FILE, mode="a", header=False, index=False)
            done.add((model_id, item.idx))
            print(f"  {model_id:<22} {item.idx:>5} -> {answer_raw}")
    
    ok = pd.read_csv(BASELINE_FILE).query("parse_ok == True")
    print("\nunassisted accuracy, parse_ok rows only (mean = accuracy, sum = items right, count = items scored):")
    print(ok.groupby("model_id")["correct"].agg(["mean", "sum", "count"]).reindex([m for _, m in MODELS]).to_string())
    print("\ncorrect per item (1 = right, 0 = wrong, blank = unusable):")
    print(ok.pivot_table(index="item_id", columns="model_id", values="correct").to_string())

if __name__ == "__main__":
    main()

"""
                        mean  sum  count
model_id                                
gpt-4.1-2025-04-14  0.888889   16     18
claude-sonnet-5     0.833333   15     18
gemini-3.6-flash    0.888889   16     18

correct per item (1 = right, 0 = wrong, blank = unusable):
model_id  claude-sonnet-5  gemini-3.6-flash  gpt-4.1-2025-04-14
item_id                                                        
151                   1.0               1.0                 1.0
158                   0.0               1.0                 1.0
479                   1.0               1.0                 1.0
1001                  1.0               1.0                 1.0
1096                  0.0               0.0                 0.0
1449                  1.0               1.0                 1.0
1453                  1.0               0.0                 1.0
1460                  0.0               1.0                 1.0
2293                  1.0               1.0                 1.0
2294                  1.0               1.0                 1.0
3474                  1.0               1.0                 1.0
3475                  1.0               1.0                 1.0
3590                  1.0               1.0                 1.0
4533                  1.0               1.0                 1.0
6052                  1.0               1.0                 1.0
6142                  1.0               1.0                 0.0
7696                  1.0               1.0                 1.0
8458                  1.0               1.0                 1.0
"""

import os
import json
import time
import random
import pandas as pd
import anthropic

SEED = 42  

PERSONA_FILE = "personas.csv"
ITEMS_FILE = "stimuli_full.json"
ITEM_META_FILE = "merged_18_items_nodup.csv"   
DESIGN_FILE = "design_map.csv"
ASSIGN_FILE = "assign_matrix.csv"   # study_ground_truth per idx
RESULTS_FILE = "results_Claude.csv"
RESULTS_COLUMNS = ["persona_id", "item_id", "condition","ai_answer_correct", "explanation_correct", "jurisdiction_match", "num_statutes", "answer_raw", "answer_binary", "human_accuracy", "agreement", "usefulness_raw", "usefulness_num", "model", "temperature", "prompt_version", "timestamp", "raw_response",]
SMOKE_PERSONAS = None #for pilot test ["P001", "P002", "P006"]
CONDITIONS = ["no_xai", "concise", "structured"]
# Binarizing q1 response 
ANSWER_BINARY = {"Yes": 1, "Likely yes": 1, "Likely no": 0, "No": 0}
# Q2 scale -> 1..4 (spec section 8)
USEFULNESS_SCALE = {"Not useful": 1, "Slightly useful": 2, "Moderately useful": 3, "Very useful": 4}

# Lookup table of the round-robin (Cyclic direction) of ites groups among personas.
# presenting PAIR (rotation, item_group). 
# row = what one persona sees if in rotation ?
# column = how one item (idx) is shown across rotations.
SQUARE = {
    ("A", 1): "no_xai",     ("A", 2): "concise",    ("A", 3): "structured",
    ("B", 1): "concise",    ("B", 2): "structured", ("B", 3): "no_xai",
    ("C", 1): "structured", ("C", 2): "no_xai",     ("C", 3): "concise",
}

MODEL = "claude-sonnet-5"
TEMPERATURE = None
MAX_TOKENS = 1024
PROMPT_VERSION = "v1"
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "Q1.answer": {
            "type": "string",
            "enum": ["Yes", "Likely yes", "Likely no", "No"],
        },
        "Q2.usefulness": {
            "anyOf": [
                {"type": "string",
                 "enum": ["Not useful", "Slightly useful", "Moderately useful", "Very useful"]},
                {"type": "null"},
            ],
        },
    },
    "required": ["Q1.answer", "Q2.usefulness"],
    "additionalProperties": False,
}


client = anthropic.Anthropic()

def call_claude(system, user):
    for attempt in range(1, 4):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                thinking={"type": "disabled"},
                output_config={"format": {"type": "json_schema", "schema": RESPONSE_SCHEMA}},
                system=system,
                messages=[{"role": "user", "content": user}],
            )

            text = "" 
            for block in response.content:
                if block.type == "text": #only get the response
                    text = text + block.text
            return text

        except Exception as error:
            print(f"    attempt {attempt}/3 failed: {error}")
            if attempt < 3:
                time.sleep(5 * attempt)

    return "FAILED"

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
    
#------

def persona_prompt(persona):
    home_state = persona.home_state
    role = persona.role
    years_licensed = persona.years_licensed
    statute_frequency = persona.statute_frequency
    ai_use = persona.ai_use
    
    prompt = f"""You are a lawyer licensed and actively practising in {home_state}, United States.
    You have been licensed for {years_licensed} years.
    You work in {role}.
    You engage with US state housing and tenancy statutes {statute_frequency}.
    Your prior use of AI tools in your professional work is {ai_use}.

    You are taking part in a research study on US housing law. In each item you
    will see a jurisdiction, the relevant statute text, a yes/no question, and an
    answer produced by an AI system. Some AI answers are correct and some are not.
    Some items also show an explanation of the AI's answer. Your task is to judge 
    the AI's answer to the question, not the explanation; the explanation is the AI's 
    reasoning for its answer, provided for you to consider, and it may contain 
    inaccuracies.
    
    Answer as the law stood on 1 January 2021.

    Answer as this person would, drawing on the experience described above.
    Respond only in the JSON format requested, with no other text.
    """
    
    return prompt

def statutes_to_text(raw):
    blocks = []
    statutes = json.loads(raw)
    for statute in statutes:
        blocks.append(f"{statute['citation']}\n{statute['text'].strip()}")
    return "\n\n".join(blocks)


def item_prompt(item, condition):
    rendering = item["renderings"][condition]
    explanation = rendering["text"].strip()

    parts = [
        f"Jurisdiction: {item['state']}",
        "",
        "Question:",
        item["question"],
        "",
        "Note:",
        item["shorter_note"],
        "",
        "Relevant statutes:",
        item["statutes"],
        "",
        "AI's final answer:",
        rendering["answer"],
    ]

    if explanation:
        parts += ["", "AI's explanation:", explanation]
    
    # Q1
    parts += [
        "",
        f'Q1. The AI\'s answer is "{rendering["answer"]}". Taking this into account, on balance,'
        " what is your answer to the question above?",
        "Choose exactly one: Yes / Likely yes / Likely no / No",
    ]

    # Q2 
    if explanation:
        parts += [
            "",
            "Q2. How useful was the explanation in helping you decide whether the AI's answer"
            f' ("{rendering["answer"]}") was right or wrong?',
            "Choose exactly one: Not useful / Slightly useful / Moderately useful / Very useful",
            "",
            'Reply with only this JSON object and nothing else:',
            '{"Q1.answer": "<Yes | Likely yes | Likely no | No>", '
            '"Q2.usefulness": "<Not useful | Slightly useful | Moderately useful | Very useful>"}'
        ]
    else:
        parts += [
            "",
            'Reply with only this JSON object and nothing else:'
            '{"Q1.answer": "<Yes | Likely yes | Likely no | No>", "Q2.usefulness": null}',
        ]

    return "\n".join(parts)

def load_results():
    # I am checking just to resume log instead of starting from begginig.
    if not os.path.exists(RESULTS_FILE):
        fresh_result = pd.DataFrame(columns=RESULTS_COLUMNS)
        fresh_result.to_csv(RESULTS_FILE, index=False)
        print(f"{RESULTS_FILE} not found so we just started")
    return pd.read_csv(RESULTS_FILE)


def item_plan(persona, design):
    #Ordered [(idx, condition), ...] for one persona: what they see and in what order.
    #Order is block randomised (spec section 6): shuffle the three condition blocks, thenshuffle the items inside each block.
    
    #Within-rotation-cluster randomisation
    rng = random.Random(f"{SEED}-{persona.persona_id}")

    by_condition = {c: [] for c in CONDITIONS}
    
    for d in design.itertuples():
        cond = SQUARE[(persona.rotation, d.item_group)]
        by_condition[cond].append(d.idx)

    blocks = list(CONDITIONS) 
    rng.shuffle(blocks) # between-condition

    plan = []
    for c in blocks:
        rng.shuffle(by_condition[c]) # within-condition
        plan += [(idx, c) for idx in by_condition[c]]
    return plan


def parse_json(reply):
    try:
        data = json.loads(reply)
    except (json.JSONDecodeError, TypeError):
        return None, None

    answer = data.get("Q1.answer")
    usefulness = data.get("Q2.usefulness")
    
    if answer not in ANSWER_BINARY:
        answer = None
    if usefulness not in USEFULNESS_SCALE:
        usefulness = None
        
    return answer, usefulness


def append_row_to_csv(row):
    pd.DataFrame([row], columns=RESULTS_COLUMNS).to_csv(RESULTS_FILE, mode="a", header=False, index=False)


def main():

    # Loading
    personas = pd.read_csv(PERSONA_FILE)
    
    if SMOKE_PERSONAS:
        personas = personas[personas["persona_id"].isin(SMOKE_PERSONAS)]
        
    design = pd.read_csv(DESIGN_FILE)
    with open(ITEMS_FILE) as f:
        items = json.load(f)          
        
    meta = pd.read_csv(ITEM_META_FILE).set_index("idx")
    truth = pd.read_csv(ASSIGN_FILE).set_index("idx")

    for it in items:
        it["idx"] = int(it["idx"])
        m = meta.loc[it["idx"]]
        it["state"] = m["state"]
        it["question"] = m["question"]
        it["shorter_note"] = m["shorter_note"]
        it["statutes"] = statutes_to_text(m["statutes"])
        it["num_statutes"] = int(m["num_statutes"])   # item complexity, descriptive only
        it["ground_truth"] = truth.loc[it["idx"], "study_ground_truth"]    # Yes / No
        it["defect_type"] = it["plan"].get("defect_type", "none")          # none = clean explanation

    results = load_results()

    # in case of discountinuity
    ok = results[results["raw_response"] != "FAILED"] # to make sure any failure get run again
    done = {(r.persona_id, r.item_id) for r in ok.itertuples()}

    # verify before trusting it
    print(f"{len(personas)} personas x {len(items)} items = {len(personas) * len(items)} calls")
    print(f"{len(done)} already done, {len(results) - len(ok)} FAILED rows to retry")

    # verify the square before trusting it
    check = pd.DataFrame([(p.persona_id, idx, c) for p in personas.itertuples() for idx, c in item_plan(p, design)], columns=["persona_id", "idx", "condition"],
    )
    
    per_persona = pd.crosstab(check.persona_id, check.condition)
    print("\nitems per condition, per persona -> min:", per_persona.min().to_dict())
    print("                                    max:", per_persona.max().to_dict())
    print("\nitem x condition (each item shown in all three):")
    print(pd.crosstab(check.idx, check.condition).to_string())

    print("\npresentation order for P001:")
    for i, (idx, c) in enumerate(item_plan(personas.iloc[0], design), 1):
        print(f"  {i:2d}. {idx:>5} {c}")

    items_by_idx = {it["idx"]: it for it in items}   # idx -> item dict, for the lookup below

    for persona in personas.itertuples():
        for idx, condition in item_plan(persona, design):

            # done holds (persona_id, item_id) pairs, so the skip is per item, not per persona
            if (persona.persona_id, idx) in done:
                continue

            item = items_by_idx[idx]
            rendering = item["renderings"][condition]

            reply = call_claude(system=persona_prompt(persona), user=item_prompt(item, condition))
            answer_raw, usefulness_raw = parse_json(reply)

          
            answer_binary = ANSWER_BINARY.get(answer_raw)
            if answer_binary is None:
                print(f"    answer_raw has problem: {reply[:80]!r}")

            # Q2 is only asked when the item showed an explanation, so None is the correct
            # value for no_xai and must not be reported as a problem.
            if rendering["text"].strip() and usefulness_raw is None:
                print(f"    usefulness_raw has problem: {reply[:80]!r}")

            row = {
                "persona_id": persona.persona_id,
                "item_id": idx,
                "condition": condition,
                # the 2x2, computed from the data rather than read off the CELL label
                "ai_answer_correct": int(rendering["answer"] == item["ground_truth"]),
                "explanation_correct": None if not rendering["text"].strip() else int(item["defect_type"] == "none"),
                "jurisdiction_match": int(item["state"] == persona.home_state),
                "num_statutes": item["num_statutes"],
                "answer_raw": answer_raw,
                "answer_binary": answer_binary,
                # the DV, measured directly instead of inferred from agreement x AI correctness
                "human_accuracy": None if answer_binary is None else int(answer_binary == ANSWER_BINARY[item["ground_truth"]]),
                "agreement": None if answer_binary is None else int(answer_binary == ANSWER_BINARY[rendering["answer"]]),
                "usefulness_raw": usefulness_raw,
                # the same judgement as an ordered integer, so the ordinal model reads the
                # column directly instead of re-deriving the order from the label. None for
                # no_xai and for an unrecognised label, matching usefulness_raw.
                "usefulness_num": USEFULNESS_SCALE.get(usefulness_raw),
                "model": MODEL,
                "temperature": TEMPERATURE,
                "prompt_version": PROMPT_VERSION,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "raw_response": reply,
            }

            append_row_to_csv(row)
            done.add((persona.persona_id, idx))
            print(f"  {persona.persona_id} {idx:>5} {condition:<11} -> {answer_raw}")

if __name__ == "__main__":
    main()
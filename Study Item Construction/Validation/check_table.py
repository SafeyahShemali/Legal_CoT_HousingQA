
import json, csv, re

input_path = "Matrix_allocation.csv"       
items = json.load(open("stimuli_full.json"))
rows = {r["idx"]: r for r in csv.DictReader(open(input_path))}

def clean(t):
    """Whitespace and quote marks only — nothing that could hide a real change."""
    t = t.replace("\u2019", "'").replace("\u2018", "'")
    t = t.replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"\s+", " ", t).strip().lower()

print(f"{'idx':<7}{'state':<7}{'expected':<14}{'grounding':<12}{'completeness'}")
print("-" * 55)

for it in items:
    row = rows[str(it["idx"])]
    statutes = json.loads(row["statutes"])
    corpus = clean(" ".join(s["text"] for s in statutes))

    plan = it["plan"]
    expected = plan["defect_type"]

    # grounding: does every quoted sentence really exist?
    bad_quotes = [s["quote"] for s in plan["reasoning_steps"]
                  if clean(s["quote"]) not in corpus]
    grounding = "HIT" if bad_quotes else "miss"

    # completeness: was any relevant statute left out?
    provided = {str(s["statute_idx"]) for s in statutes}
    cited = {str(x) for x in plan["cited_statutes"]}
    if expected == "completeness":
        completeness = "HIT" if provided - cited else "miss"
    else:
        completeness = f"unused: {sorted(provided - cited)}" if provided - cited else "miss"

    # a row is fine when the HIT lands on the defect that was planted
    ok = (grounding == "HIT") == (expected == "grounding") and \
         (completeness == "HIT") == (expected == "completeness")

    print(f"{it['idx']:<7}{row['state']:<12}\t{expected:<14}"
          f"{grounding:<12}{completeness}{'' if ok else '   <-- check'}")
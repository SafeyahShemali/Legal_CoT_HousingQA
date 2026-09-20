import numpy as np, pandas as pd
from scipy.stats import fisher_exact, spearmanr

MODELS = ["GPT", "Claude", "Gemini"]
PAIRS = [("no_xai", "concise"), 
         ("no_xai", "structured"), 
         ("concise", "structured")]


quad = pd.read_csv("long_item_condition.csv")[["item_id", "quadrant"]].drop_duplicates().set_index("item_id").quadrant

rows = []
for m in MODELS:
    d = pd.read_csv(f"results_{m}.csv")
    
    g = d.groupby(["item_id", "condition"]).human_accuracy.agg(["sum", "count"])
    
    for i in sorted(d.item_id.unique()):
        for a, b in PAIRS:
            ka, na = g.loc[(i, a)]; kb, nb = g.loc[(i, b)]
            p = fisher_exact([[ka, na - ka], [kb, nb - kb]])[1]  # two-sided
            rows.append(dict(model=m, item_id=i, quadrant=quad[i], condition_1=a, condition_2=b,
                             correct_1=int(ka), n_1=int(na), correct_2=int(kb), n_2=int(nb),
                             contrast_type="format" if a == "concise" else "presence", p_raw=p))
r = pd.DataFrame(rows)

p = r.p_raw.values; order = np.argsort(p); adj = np.empty(len(p)); running = 0.0
for rank, idx in enumerate(order):
    running = max(running, min(1.0, (len(p) - rank) * p[idx]))
    adj[idx] = running
r["p_holm"] = adj; r["sig_raw"] = r.p_raw < .05; r["sig_holm"] = r.p_holm < .05
r.to_csv("fisher_screen.csv", index=False)

out = []
for m in MODELS:
    d = pd.read_csv(f"results_{m}.csv"); d = d[d.condition != "no_xai"]
    for c, lab in [(1, "AI correct"), (0, "AI incorrect")]:
        x = d[d.ai_answer_correct == c]
        for v, name in [("agreement", "agreement"), ("human_accuracy", "accuracy")]:
            rho, pv = spearmanr(x[v], x.usefulness_num)
            out.append(dict(model=m, ai_answer=lab, variable=name, n=len(x), rho=round(rho, 3), p=pv))
pd.DataFrame(out).to_csv("spearman_split.csv", index=False)
print(r.sig_raw.sum(), "raw hits;", r.sig_holm.sum(), "survive Holm")


import os
import random
import pandas as pd

SEED = 42       # to regerate it if my meeting with supervisor have some changes
N_PERSONAS = 60 
OUTFILE = "personas.csv"
ROTATIONS = ["A", "B", "C"]

# Design Choice 
# I distribute like that as we are targetting general U.S. attorney not specialized necessary on the housing law
STATUTE_FREQUENCY = {"Rarely": 0.20, "Occasionally": 0.30, "Regularly": 0.30, "Daily": 0.20}

# These distribution has some resources I found in the internet
# ???
YEARS_LICENSED = {"<2": 0.15, "2-5": 0.25, "6-10": 0.25, "11-20": 0.20, "20+": 0.15}
# ABA survey data on practice setting.
ROLE = {"Private practice": 0.50, "Legal aid": 0.20, "In-house": 0.20, "Government/judiciary": 0.10}
# ABA survey data on legal AI adoption and TR report 
AI_USE = {"Low": 0.30, "Medium": 0.45, "High": 0.25}

# To check within-persona jurisdiction_match contrast --> checking the motivation of the study 
HOME_STATE = ["New York", "California", "Texas", "Illinois", "Florida"]

# Sort keys for the stratified rotation deal (least to most experienced; weight order for role).
YEARS_ORDER = list(YEARS_LICENSED)
ROLE_ORDER = list(ROLE)


def sample(weights, rng, n):
    # weights per factor of each persona features
    labels = list(weights.keys()) 
    probs  = list(weights.values())   
    return rng.choices(labels, weights=probs, k=n)

def main():
    
    if os.path.exists(OUTFILE):
        print(f"{OUTFILE} already exists: prevent overwrite by mistake.")
        return

    rng = random.Random(SEED)
    
    #details of each persona 
    persona_ids       = [f"P{i:03d}" for i in range(1, N_PERSONAS + 1)]
    years_licensed    = sample(YEARS_LICENSED, rng, N_PERSONAS)
    statute_frequency = sample(STATUTE_FREQUENCY, rng, N_PERSONAS)
    role              = sample(ROLE, rng, N_PERSONAS)
    ai_use            = sample(AI_USE, rng, N_PERSONAS)
    home_state        = rng.choices(HOME_STATE, k=N_PERSONAS)


    personas = pd.DataFrame({
    "persona_id": persona_ids,
    "years_licensed": years_licensed,
    "statute_frequency": statute_frequency,
    "role": role,
    "ai_use": ai_use,
    "home_state": home_state,
    })

    # Stratified rotation: 
    # Sort by experience then role, then deal A/B/C round-robin down the list: I did that to make sure that most senior lawyers (for example) are not cluster in one group.
    # Stable sort => ties break by persona_id, so the deal is fully deterministic.
    personas["years_licensed"] = pd.Categorical(personas["years_licensed"], categories=YEARS_ORDER, ordered=True)
    personas["role"] = pd.Categorical(personas["role"], categories=ROLE_ORDER, ordered=True)
    order_indexing = personas.sort_values(["years_licensed", "role"], kind="stable").index
    
    # now here assign the A/B/C round-robin down
    personas.loc[order_indexing, "rotation"] = [ROTATIONS[i % 3] for i in range(N_PERSONAS)]
    # rearrage the coloumns better for my excel sheet 
    personas = personas[[ "persona_id", "rotation", "years_licensed","statute_frequency","role", "ai_use","home_state",]]
    personas.to_csv(OUTFILE, index=False)

    print(f"Wrote {OUTFILE}: {len(personas)} rows\n")
    print("Personas per rotation:")
    print(personas["rotation"].value_counts().sort_index().to_string(), "\n")
    print("rotation x years_licensed:")
    print(pd.crosstab(personas["rotation"], personas["years_licensed"]).to_string())

if __name__ == "__main__":
    main()
    
#run output:
'''
Wrote personas.csv: 60 rows

Personas per rotation:
rotation
A    20
B    20
C    20 

rotation x years_licensed:
years_licensed  <2  2-5  6-10  11-20  20+
rotation                                 
A                4    6     5      4    1
B                3    7     5      3    2
C                3    7     4      4    2

'''

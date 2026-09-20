
import os
import random
import pandas as pd
OUTFILE = "design_map.csv"
INPUT= "assign_matrix.csv"
SEED = 42  
CELLS = ["Convincing", "Confusing", "Misleading", "Revealing"]

rng = random.Random(SEED)          # one generator, reused

items = pd.read_csv(INPUT)

buckets = {c: [] for c in CELLS}
# Bucket items by cell
for i in items.itertuples():
    buckets[i.CELL].append(i.idx)

# Shuffle within each bucket 
for c in buckets: 
    rng.shuffle(buckets[c])

# 3. Deal each bucket round-robin across groups 1,2,3
g = 0
design_map = {}
for c in CELLS:
    for idx in buckets[c]:
        design_map[idx] = (g % 3) + 1
        g += 1
        
# 4. Write
out = pd.DataFrame({"idx": list(design_map.keys()), "item_group": list(design_map.values())})
out.to_csv(OUTFILE, index=False)
        
# 5. Verify before trusting it
check = items.merge(out, on="idx")
print(check.groupby("item_group").size())
print(pd.crosstab(check.item_group, check.CELL))
print(pd.crosstab(check.item_group, check.DEFECT_TYPE))

#output:
'''
item_group

1    6
2    6
3    6
dtype: int64
CELL        Confusing  ...  Revealing
item_group             ...           
1                   1  ...          1
2                   1  ...          1
3                   2  ...          1

[3 rows x 4 columns]
DEFECT_TYPE  Completeness  Grounding  Validity  none
item_group                                          
1                       1          0         1     4
2                       0          2         0     4
3                       1          1         1     3
'''

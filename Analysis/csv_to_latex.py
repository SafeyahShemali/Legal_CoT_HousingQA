# import pandas as pd

# input= "./fisher_test.csv"
# # Read CSV\
# df = pd.read_csv(input)

# # Convert to LaTeX\
# latex = df.to_latex(index=False, caption="Your caption", label="tab:my_table")

# # Save as .tex\
# with open("table_{input}.tex", "w") as f:
#     f.write(latex)

import pandas as pd
import os
print(os.path.exists("./fisher_tests.csv"))

df = pd.read_csv("./fisher_tests.csv")

latex = df.to_latex(index=False)

with open("fisher_test.tex", "w") as f:
    f.write(latex)
# PREPARE DATA 
# Reads the raw persona results, renames columns to match the
# model, checks the design, saves a clean file for step 02.


library(dplyr)

# ---- Read -------------------------------------------
raw   <- read.csv("data/raw/results.csv",       stringsAsFactors = FALSE)
alloc <- read.csv("data/raw/assign_matrix.csv", stringsAsFactors = FALSE)


# ---- Build the modelling columns --------------------------
# Map the simulation fields to models compatible names
# Notes:
##factor to indicate it is a categorical not numric feild
#  # levels set the 1st as reference and compare all other against it

d <- raw %>% 
  mutate(
    persona_id = factor(persona_id),
    item_id    = factor(item_id),

    # Explanation format: ref condition is No-XAI 
    X = factor(condition, levels = c("no_xai", "concise", "structured")), 

    # AI Correctnes: ref option is "incorrect" because over-reliance on wrong advice is what we care about most.
    C_AI = factor(ifelse(ai_answer_correct == 1, "correct", "incorrect"),
      levels = c("incorrect", "correct")
    ),

    # Explanation Correctnes: ref condition is No-XAI thus here the ref is none
    C_X = factor(
      ifelse(is.na(explanation_correct), "none", ifelse(explanation_correct == 1, "correct", "incorrect")),
      levels = c("none", "correct", "incorrect")
    ),

    accuracy = human_accuracy
  )


# ---- Check Point ------------------------
# Every item's C_AI and C_X must match it the allocation unless there is manual mistake. 
# Deriving expect_C_AI and expect_C_AIfrom CELL
cell_map <- alloc %>%
  mutate(expect_C_AI = ifelse(CELL %in% c("Convincing", "Confusing"), "correct", "incorrect"),
    expect_C_X  = ifelse(CELL %in% c("Convincing", "Misleading"), "correct",   "incorrect")
  ) %>% select(item_id = idx, CELL, expect_C_AI, expect_C_X)

chk <- d %>%
  mutate(item_id = as.integer(as.character(item_id))) %>%  #addedd "as.character" as the casting add some characters 
  left_join(cell_map, by = "item_id")

cat("\n C_AI mismatches vs assign_matrix by: \n")
print(sum(chk$C_AI != chk$expect_C_AI))

cat("\nC_X mismatches, explanation conditions only:\n")
print(sum(chk$X != "no_xai" & chk$C_X != chk$expect_C_X))


# ---- Design checks -------------------------------
cat("\n rows(60 personas x 18 items) must be 1080, right?")
print(nrow(d))

cat("\nitems per persona must be 18, right?")
print(range(table(d$persona_id)))

cat("\nX by C_X:\n")
print(table(d$X, d$C_X))

cat("\n the 2x2 stimulus matrix \n")
print(table(chk$CELL[!duplicated(chk$item_id)]))

cat("\n any missing values? all must be FALSE\n")
print(colSums(is.na(d[, c("persona_id","item_id","X","C_AI","C_X","accuracy")])) > 0)


# ---- Raw accuracy by cell -------------------------------------
# o,o or 1oo% breaks standard maximum-likelihood fitting. Hopdfully not 
cat("\nraw accuracy by cell:\n")
cell_summary <- 
  d %>%
    group_by(X, C_AI, C_X) %>%
    summarise(n = n(), accuracy = round(mean(accuracy), 3), .groups = "drop") %>%
    as.data.frame()

print(cell_summary)

saveRDS(d, "data/processed/model_data.rds")

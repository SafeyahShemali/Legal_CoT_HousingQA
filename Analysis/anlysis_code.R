#  DESCRIPTIVE RESULTS LEVEL 
library(dplyr)
library(tidyr)

RESULTS_FILE <- "data/raw/results_Gemini.csv"
MODEL_TAG    <- "gemini"         


# ---- Read ------------------------------------------------------
raw   <- read.csv(RESULTS_FILE, stringsAsFactors = FALSE)
alloc <- read.csv("data/raw/assign_matrix.csv", stringsAsFactors = FALSE)

needed <- c("persona_id", "item_id", "condition",
            "ai_answer_correct", "explanation_correct", "human_accuracy")
missing <- setdiff(needed, names(raw))
if (length(missing) > 0) {
  stop("Missing columns in ", RESULTS_FILE, ": ", paste(missing, collapse = ", "))
}


# ---- Build the modelling columns -------------------------------
d <- raw %>%
  mutate(
    persona_id = factor(persona_id),
    item_num   = as.integer(item_id),
    item_id    = factor(item_id),
    X          = factor(condition, levels = c("no_xai", "concise", "structured")),
    C_AI       = factor(ifelse(ai_answer_correct == 1, "correct", "incorrect"),
                        levels = c("incorrect", "correct")),
    C_X        = factor(ifelse(is.na(explanation_correct), "none",
                               ifelse(explanation_correct == 1, "correct", "incorrect")),
                        levels = c("none", "correct", "incorrect")),
    accuracy   = human_accuracy
  )


# ---- Attach the quadrant to EVERY row, including No-XAI --------
d <- d %>%
  left_join(alloc %>% select(item_num = idx, quadrant = CELL), by = "item_num")

if (any(is.na(d$quadrant))) {
  stop("Some items are not in assign_matrix.csv: ",
       paste(unique(d$item_num[is.na(d$quadrant)]), collapse = ", "))
}


# =============================================================
# (0) SANITY CHECKS 
# =============================================================
cat("\n=== rows ===\n");                    print(nrow(d))
cat("\n=== personas ===\n");                print(nlevels(d$persona_id))
cat("\n=== items ===\n");                   print(nlevels(d$item_id))
cat("\n=== responses per item x condition cell (should be one value) ===\n")
print(table(table(d$item_id, d$X)))
cat("\n=== X by C_X (no_xai must be 'none' only) ===\n")
print(table(d$X, d$C_X))
cat("\n=== items per quadrant ===\n")
print(table(alloc$CELL))

# Does the data agree with the authoritative stimulus allocation?
chk <- d %>%
  left_join(alloc %>%
              transmute(item_num = idx,
                        exp_C_AI = ifelse(CELL %in% c("Convincing", "Confusing"),
                                          "correct", "incorrect"),
                        exp_C_X  = ifelse(CELL %in% c("Convincing", "Misleading"),
                                          "correct", "incorrect")),
            by = "item_num")
cat("\n=== C_AI mismatches vs assign_matrix (must be 0) ===\n")
print(sum(chk$C_AI != chk$exp_C_AI))
cat("\n=== C_X mismatches, explanation conditions only (must be 0) ===\n")
print(sum(chk$X != "no_xai" & chk$C_X != chk$exp_C_X))


# =============================================================
# (A) DESCRIPTIVE RESULTS
# =============================================================
pooled <- d %>%
  group_by(C_AI, X) %>%
  summarise(n = n(), n_correct = sum(accuracy),
            acc_pct = round(100 * mean(accuracy), 1), .groups = "drop")

overall <- d %>%
  group_by(X) %>%
  summarise(C_AI = "ALL", n = n(), n_correct = sum(accuracy),
            acc_pct = round(100 * mean(accuracy), 1), .groups = "drop")

cat("\n=== POOLED: accuracy by advice correctness ===\n")
print(as.data.frame(bind_rows(overall, pooled)))

# ---- A2. ISO quadrant table (the primary result) ---------------
iso <- d %>%
  group_by(quadrant, X) %>%
  summarise(n = n(), n_correct = sum(accuracy),
            accuracy = mean(accuracy), .groups = "drop")

ci <- t(mapply(function(x, n) binom.test(x, n)$conf.int,
               iso$n_correct, iso$n))
iso$acc_pct <- round(iso$accuracy * 100, 1)
iso$ci_low  <- round(ci[, 1] * 100, 1)
iso$ci_high <- round(ci[, 2] * 100, 1)

cat("\n=== ISO TABLE: accuracy by quadrant and condition ===\n")
print(as.data.frame(iso[, c("quadrant", "X", "n", "n_correct",
                            "acc_pct", "ci_low", "ci_high")]))

iso_wide <- iso %>%
  select(quadrant, X, acc_pct) %>%
  pivot_wider(names_from = X, values_from = acc_pct)

cat("\n=== ISO TABLE (wide — this is the figure) ===\n")
print(as.data.frame(iso_wide))

# =============================================================
# (B) HOW MUCH INDEPENDENT INFORMATION IS THERE?
# =============================================================
item_level <- d %>%
  group_by(item_num, quadrant, C_AI, X) %>%
  summarise(p = mean(accuracy), n = n(), .groups = "drop") %>%
  mutate(acc = as.integer(p >= 0.5))

cat("\n=== unanimity: how many cells had ALL personas agreeing? ===\n")
print(item_level %>% count(unanimous = p %in% c(0, 1)))

cat("\n=== cells that were NOT unanimous (footnote these) ===\n")
split_cells <- item_level %>% filter(p > 0, p < 1)
if (nrow(split_cells) == 0) {
  cat("none\n")
} else {
  print(as.data.frame(split_cells))
}

# =============================================================
# (C) EXACT PAIRED TESTS
# =============================================================
mat <- item_level %>%
  select(item_num, X, acc) %>%
  pivot_wider(names_from = X, values_from = acc) %>%
  arrange(item_num)

M <- as.matrix(mat[, c("no_xai", "concise", "structured")])
rownames(M) <- mat$item_num

cat("\n=== item x condition matrix ===\n")
print(M)

cat("\n=== items that CHANGE across conditions (the only ones that count) ===\n")
changers <- rownames(M)[apply(M, 1, function(r) length(unique(r)) > 1)]
print(changers)
cat("effective n for any test:", length(changers), "discordant items\n")

# ---- Save ------------------------------------------------------
write.csv(iso,         sprintf("outputs/tables/iso_quadrant_%s.csv", MODEL_TAG), row.names = FALSE)
write.csv(iso_wide,    sprintf("outputs/tables/iso_wide_%s.csv",     MODEL_TAG), row.names = FALSE)
write.csv(pairs_all,   sprintf("outputs/tables/mcnemar_all_%s.csv",  MODEL_TAG), row.names = FALSE)
write.csv(pairs_wrong, sprintf("outputs/tables/mcnemar_wrong_%s.csv", MODEL_TAG), row.names = FALSE)
saveRDS(d,             sprintf("data/processed/model_data_%s.rds",   MODEL_TAG))
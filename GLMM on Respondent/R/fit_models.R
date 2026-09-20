# FIT THE MODELS
# Fits the baseline and the full GLM

library(lme4)

d <- readRDS("data/processed/model_data.rds")

# ---- Fitting optimizer ------------------------------------------
ctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))


# ---- Baseline model -------------------------------------------
# The simplest model ever: people do better when the advice is correct, personas differ,
# items differ.

m_base <- glmer(
  accuracy ~ C_AI + (1 | persona_id) + (1 | item_id),
  family  = binomial(link = "logit"), #0,1 outout, the model works in log-odds
  data    = d,
  control = ctrl
)

# ---- Full model -----------------------------------------------
# C_AI        Control. Accuracy differs by whether the AI's answer was correct.
# X           Explanation format shifts reliance relative to the No-XAI control.
# X:C_AI      Allows the format effect to differ by advice correctness.
# X:C_X       Allows the format effect to differ by explanation correctness.
# X:C_AI:C_X  Allows the format effect to depend on both jointly -- the 2x2 pattern.
#
m_full <- glmer(
  accuracy ~ C_AI + X + X:C_AI + X:C_X + X:C_AI:C_X +
    (1 | persona_id) + (1 | item_id),
  family  = binomial(link = "logit"),
  data    = d,
  control = ctrl
)

# ---- Check: random-effect sizes (SDs on log-odds scale)------------------------
# 2 SDs how much personas differ from each other, and how much items do.
cat("\nrandom-effect SDs\n") 
print(VarCorr(m_full))

cat("\nsingular fit?\n")
print(isSingular(m_base))
print(isSingular(m_full))

# result:
#Groups     Name        Std.Dev.
#persona_id (Intercept)  0.1211 
#item_id    (Intercept) 33.9854 
#> print(isSingular(m_full))
#[1] FALSE
#> print(isSingular(m_base))
#[1] TRUE
# this means the persona variance is 0 where as item variance exploded (34)
# we need to fix the model 


# ---- Check 2: separation ---------------------------------
# no "best" coefficient for the exterme accuracy 0 or 1.

cat("\ncoefficients with |estimate| > 10 or SE > 10\n")
s <- summary(m_full)$coefficients
flagged <- abs(s[, "Estimate"]) > 10 | s[, "Std. Error"] > 10
print(s[flagged, , drop = FALSE])

# result: it is bad. all 10 coff flagged 
#Estimate   Std. Error       z value  Pr(>|z|)
#(Intercept)                         13.682784 2.284532e+01  5.989315e-01 0.5492185
#C_AIcorrect                          1.327112 4.125646e+01  3.216737e-02 0.9743386
#Xconcise                           -21.943040 3.004403e+01 -7.303626e-01 0.4651685
#Xstructured                        -21.475607 3.004311e+01 -7.148263e-01 0.4747164
#C_AIcorrect:Xconcise                22.720376 3.412582e+01  6.657826e-01 0.5055501
#C_AIcorrect:Xstructured             20.690981 3.836006e+01  5.393886e-01 0.5896187
#Xconcise:C_Xcorrect                 20.090182 3.307469e+01  6.074186e-01 0.5435731
#Xstructured:C_Xcorrect              20.114000 3.186424e+01  6.312404e-01 0.5278833
#C_AIcorrect:Xconcise:C_Xcorrect     20.396439 *5.305422e+06*  3.844452e-06 *0.9999969* <-- no idead and no effect 
#C_AIcorrect:Xstructured:C_Xcorrect  11.992282 *5.305422e+06*  2.260382e-06 *0.9999982* <-- no idead and no effect 

# ---- Check 3: does the full model earn its extra terms? --
cat("\nlikelihood-ratio test: baseline vs full\n")
print(anova(m_base, m_full))

cat("\n-AIC (lower is better)\n")
print(c(baseline = AIC(m_base), full = AIC(m_full)))

#justify why aggrigaion
d %>% group_by(item_id, X) %>%
  summarise(p = mean(accuracy), .groups = "drop") %>%
  count(all_same = p %in% c(0, 1))

# ======= full model version 2 Trail
# combine 1080 to 54 (18*3) as evert 20 personas per cell 
agg <- d %>%
  group_by(item_id, X, C_AI, C_X) %>%
  summarise(p = mean(accuracy), n_split = sum(p > 0 & p < 1), .groups = "drop") %>%
  mutate(accuracy = as.integer(p >= 0.5))

library(blme)
m <- bglmer(accuracy ~ C_AI + X + X:C_AI + (1 | item_id),
            family = binomial, data = agg,
            fixef.prior = normal(sd = 2.5))
#check again
print(VarCorr(m))
print(isSingular(m))
s <- summary(m)$coefficients
print(s[abs(s[, "Estimate"]) > 10 | s[, "Std. Error"] > 10, , drop = FALSE])

saveRDS(list(base = m_base, full = m_full), "outputs/models.rds")


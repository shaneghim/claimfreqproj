library(tidyverse)
library(data.table)
library(janitor)
library(mgcv)



#Main goal: we want to predict the number of insurance claims a policyholder will file based on the demographic and vehicle risk factors
#claimNb is our target variable Y
#exposure is the fraction of a year the policy was active
insurance <- fread("data/freMTPL2freq.csv",
                   select = c("ClaimNb", "Exposure", "DrivAge", "BonusMalus", 
                              "VehPower", "VehAge", "VehBrand", "VehGas", "Area", "Density", "Region")) %>%
  as_tibble() %>% clean_names()

#inspect our data before cleaning
glimpse(insurance)
summary(insurance)
insurance %>% count(claim_nb)
colSums(is.na(insurance))

#Cap check and clean while preserving og data
insurance_raw <- insurance

#check how many rows each cap will change, alter values as needed
cap_check <- tibble(
  rule = c("claim_nb > 5", "exposure > 1", "veh_age > 25", "driv_age > 80"),
  rows_affected = c(sum(insurance_raw$claim_nb > 5),
                    sum(insurance_raw$exposure > 1),
                    sum(insurance_raw$veh_age > 25),
                    sum(insurance_raw$driv_age > 80)),
  pct_of_rows = round(100 * rows_affected / nrow(insurance_raw), 3))
cap_check
#each cap affects less than 1% of the rows, ok

#claim_nb is set to limit extreme claim counts, exposure > 1 is an error
#veh_age set to max 25, driv_age set to max 80
#log(density) reduces either extremes influence on the data
insurance <- insurance %>%
  mutate(claim_nb = pmin(claim_nb, 5), exposure = pmin(exposure, 1), veh_age = pmin(veh_age, 25),
         driv_age = pmin(driv_age, 80), across(c(veh_brand, veh_gas, area, region), factor),
      log_density = log(density))

#explore frequency by risk factor
#sort drivers into age ranges
insurance <- insurance %>%
  mutate(age_band = cut(driv_age, breaks = c(18, 25, 35, 45, 55, 65, Inf), right = FALSE,
           labels = c("18-24", "25-34", "35-44", "45-54", "55-64", "65+")),
         
#bonus-malus bands (same bands as the frequency table below)
          bm_band = cut(bonus_malus, c(0, 50, 60, 80, 100, Inf),
           labels = c("50", "51-60", "61-80", "81-100", "Over 100")))
sum(is.na(insurance$age_band))
#check sum = 0

#look at frequency by age_band
freq_by <- function(data, var) {data %>%
    group_by({{ var }}) %>%
    summarise(policies = n(), exposure = sum(exposure), claims = sum(claim_nb), frequency = claims / exposure)}

freq_age <- freq_by(insurance, age_band)
freq_age


freq_by(insurance, area)
freq_by(insurance, veh_gas)
freq_by(insurance, bm_band)

#plot frequency by driver age
p_freq_age <- ggplot(freq_age, aes(age_band, frequency)) + geom_col() + 
  labs(title = "Claim frequency by driver age",
           x = "Driver age",
           y = "Claims per policy year") + theme_minimal()
p_freq_age

#train/test split and baseline
#we need to fit the model on train and see it on test
#sample() picks 80% of the row numbers at random to be train, remaining 20% -> test
set.seed(50)
idx <- sample(nrow(insurance), 0.8 * nrow(insurance))
train <- insurance[idx, ]
test <- insurance[-idx, ]

saveRDS(train, "data/train.rds")
saveRDS(test,  "data/test.rds")

#Poisson deviance; lower is better
pois_dev <- function(y, mu) 2 * mean(ifelse(y == 0, mu, y * log(y / mu) - (y - mu)))

#baseline model
base_rate <- sum(train$claim_nb) / sum(train$exposure)
dev_base <- pois_dev(test$claim_nb, base_rate * test$exposure)
dev_base

#GLM and GAM
#time to fit our baseline Poisson generalized linear model (GLM)
#expected claims = exposure * exp(b0 + b1*driv_age + b2*veh_age + ...)
m_glm <- glm(claim_nb ~ driv_age + veh_age + veh_power + veh_gas + veh_brand +
               bonus_malus + area + log_density + region + offset(log(exposure)),
             family = poisson(link = "log"), data = train)
summary(m_glm)

#evaluate the model
test$pred_glm <- predict(m_glm, test, type = "response")
dev_glm <- pois_dev(test$claim_nb, test$pred_glm)
dev_glm

#test for value near 1: ok
#test for value greater than 1: larger variance than assumed
#extract variance function
dispersion <- sum(residuals(m_glm, type = "pearson")^2) / df.residual(m_glm)
dispersion

#generalized additive model (GAM)
#enables us to test how wrong each numeric variable's effect on the log claim rate is and fixes it
#draws smooth curves from the data, suggesting sensible band boundaries
#fits our project since it is also a Poisson model
m_gam <- bam(claim_nb ~ s(driv_age) + s(veh_age) + s(bonus_malus) + s(log_density) +
             veh_power + veh_gas + veh_brand + area + region + offset(log(exposure)),
             family = poisson, data = train, discrete = TRUE)
summary(m_gam)

test$pred_gam <- predict(m_gam, test, type = "response")
dev_gam <- pois_dev(test$claim_nb, test$pred_gam)
dev_gam

#overdispersion and area
#dispersion by exposure band
disp_by_exposure <- train %>%
  mutate(mu = fitted(m_glm), pearson2 = (claim_nb - mu)^2 / mu,
  exposure_band = cut(exposure, c(0, 0.25, 0.5, 0.75, 1))) %>%
  group_by(exposure_band) %>%
  summarise(policies = n(), dispersion = mean(pearson2))
disp_by_exposure
#overdispersion is real but concentrated in policies with under 3 months of cover
#((0,0.25] => dispersion 5.62). It is 1.31, 1.00 and 0.825 in the longer exposure bands.

#check how closely area tracks density
train %>%
  group_by(area) %>%
  summarise(policies = n(),
     min_log_density = min(log_density),
     max_log_density = max(log_density))
cor(as.numeric(train$area), train$log_density)
#correlation is close to 1 (0.97), area is basically a banded version of density

#refit the GLM without area to compare
m_glm_noarea <- update(m_glm, . ~ . - area)

test$pred_glm_noarea <- predict(m_glm_noarea, test, type = "response")
dev_glm_noarea <- pois_dev(test$claim_nb, test$pred_glm_noarea)

area_check <- tibble(model = c("GLM with area", "GLM w/o area"),
             test_deviance = c(dev_glm, dev_glm_noarea),
                       aic = c(AIC(m_glm), AIC(m_glm_noarea)),
                       bic = c(BIC(m_glm), BIC(m_glm_noarea)))
print(area_check, width = Inf)

#final models (no area) and comparison
m_gam_noarea <- bam(claim_nb ~ s(driv_age) + s(veh_age) + s(bonus_malus) + s(log_density) +
                    veh_power + veh_gas + veh_brand + region + offset(log(exposure)),
                    family = poisson, data = train, discrete = TRUE)
summary(m_gam_noarea)

test$pred_gam_noarea <- predict(m_gam_noarea, test, type = "response")
dev_gam_noarea <- pois_dev(test$claim_nb, test$pred_gam_noarea)

#final comparison table
#compare models with area vs without area
results_final <- tibble(
  model = c("Baseline", "Poisson GLM", "Poisson GLM (no area)", "GAM", "GAM (no area)"),
  train_deviance = c(pois_dev(train$claim_nb, base_rate * train$exposure),
                     pois_dev(train$claim_nb, fitted(m_glm)),
                     pois_dev(train$claim_nb, fitted(m_glm_noarea)),
                     pois_dev(train$claim_nb, fitted(m_gam)),
                     pois_dev(train$claim_nb, fitted(m_gam_noarea))),
  test_deviance = c(dev_base, dev_glm, dev_glm_noarea, dev_gam, dev_gam_noarea)) %>%
  mutate(pct_better_than_baseline = round(100 * (dev_base - test_deviance) / dev_base, 2))
print(results_final, width = Inf)
#both models beat the baseline on test deviance (GLM by 2.88%, GAM by 4.57%) while the GAM
#beat the GLM by a larger margin. The straight-line assumption in the GLM costs some accuracy.
#train and test deviance differ by about .005 for each model and .003 for the baseline,
#so there is no sign of overfitting.

#final checks
#does the ranking hold on policies with more than 3 months of cover? (overdispersion is concentrated below that)
test_long <- test %>% filter(exposure > 0.25)

robust_long <- tibble(
  model = c("Baseline", "Poisson GLM (no area)", "GAM (no area)"),
  test_deviance = c(pois_dev(test_long$claim_nb, base_rate * test_long$exposure),
                    pois_dev(test_long$claim_nb, test_long$pred_glm_noarea),
                    pois_dev(test_long$claim_nb, test_long$pred_gam_noarea))) %>%
  mutate(pct_better_than_baseline = round(100 * (test_deviance[1] - test_deviance) / test_deviance[1], 2))
print(robust_long, width = Inf)

#the Poisson model assumes less variation than our data has (dispersion above 1), so its
#confidence intervals come out too narrow. widening them by sqrt(dispersion) corrects for that
dispersion_final <- sum(residuals(m_glm_noarea, type = "pearson")^2) / df.residual(m_glm_noarea)
dispersion_final

coefs <- coef(summary(m_glm_noarea))
se_adj <- coefs[, "Std. Error"] * sqrt(dispersion_final)

#relativities = exp(coefficient); the multiplier on claim frequency vs the reference level
relativities <- tibble(term = rownames(coefs),
                 relativity = exp(coefs[, "Estimate"]),
                      lower = exp(coefs[, "Estimate"] - 1.96 * se_adj),
                      upper = exp(coefs[, "Estimate"] + 1.96 * se_adj)) %>%
  filter(term != "(Intercept)")
print(relativities, n = Inf, width = Inf)

#visual modeling of (no area) models
p_age <- test %>%
  group_by(age_band) %>%
  summarise(Actual = sum(claim_nb) / sum(exposure),
     `Poisson GLM` = sum(pred_glm_noarea) / sum(exposure),
               GAM = sum(pred_gam_noarea) / sum(exposure)) %>%
  pivot_longer(-age_band, names_to = "series", values_to = "frequency") %>%
  ggplot(aes(age_band, frequency, fill = series)) +
  geom_col(position = "dodge") +
  labs(title = "Claim frequency by driver age",
    subtitle = "Actual vs. predicted",
           x = "Driver age",
           y = "Claims per policy year", fill = NULL) + theme_minimal()
p_age

#freq by risk group
p_riskgroup <- test %>%
  mutate(risk_group = ntile(pred_gam_noarea / exposure, 10)) %>%
  group_by(risk_group) %>%
  summarise(Actual = sum(claim_nb) / sum(exposure),
            Predicted = sum(pred_gam_noarea) / sum(exposure)) %>%
  pivot_longer(-risk_group, names_to = "series", values_to = "frequency") %>%
  ggplot(aes(risk_group, frequency, color = series)) + geom_line() + geom_point() +
  scale_x_continuous(breaks = 1:10) +
  labs(title = "GAM: actual vs. predicted",
       subtitle = "Frequency by risk group",
       x = "Risk group (1 = lowest predicted risk, 10 = highest)",
       y = "Claims per policy year", color = NULL) + theme_minimal()
p_riskgroup

#bonus-malus chart
p_bm <- test %>%
  group_by(bm_band) %>%
  summarise(Actual = sum(claim_nb) / sum(exposure),
     `Poisson GLM` = sum(pred_glm_noarea) / sum(exposure),
               GAM = sum(pred_gam_noarea) / sum(exposure)) %>%
  pivot_longer(-bm_band, names_to = "series", values_to = "frequency") %>%
  ggplot(aes(bm_band, frequency, fill = series)) + geom_col(position = "dodge") +
  labs(title = "Claim frequency by bonus-malus band",
    subtitle = "Actual vs. predicted",
           x = "Bonus-malus", y = "Claims per policy year", fill = NULL) + theme_minimal()
p_bm

#comparison of vehicle brands and fuel type after accounting for other factors
p_relativities <- relativities %>%
  filter(str_detect(term, "^veh_gas|^veh_brand")) %>%
  mutate(term = str_replace(term, "veh_brand", "Brand "),
         term = str_replace(term, "veh_gasRegular", "Fuel: Regular (vs. diesel)")) %>%
  ggplot(aes(relativity, reorder(term, relativity))) + geom_pointrange(aes(xmin = lower, xmax = upper)) +
  geom_vline(xintercept = 1, linetype = "dashed") +
  labs(title = "Claim frequency relativities",
    subtitle = "GLM (no area), with 95% intervals widened for overdispersion; 1 = reference level",
           x = "Multiplier on claim frequency", y = NULL) + theme_minimal()
p_relativities

#saving it all
dir.create("output", showWarnings = FALSE)
ggsave("output/freq_by_age.png", p_freq_age, width = 7, height = 5, dpi = 300, bg = "white")
ggsave("output/actual_vs_pred_by_age.png", p_age, width = 7, height = 5, dpi = 300, bg = "white")
ggsave("output/gam_risk_group.png", p_riskgroup, width = 7, height = 5, dpi = 300, bg = "white")
ggsave("output/actual_vs_pred_by_bonus_malus.png", p_bm, width = 7, height = 5, dpi = 300, bg = "white")
ggsave("output/glm_relativities.png", p_relativities, width = 7, height = 6, dpi = 300, bg = "white")

write_csv(cap_check, "output/cap_check.csv")
write_csv(freq_age, "output/freq_by_age.csv")
write_csv(disp_by_exposure, "output/dispersion_by_exposure.csv")
write_csv(area_check, "output/area_check.csv")
write_csv(results_final, "output/model_comparison.csv")
write_csv(robust_long, "output/robustness_longer_policies.csv")
write_csv(relativities, "output/glm_relativities.csv")

png("output/gam_smooth_effects.png", width = 1600, height = 1200, res = 200)
plot(m_gam_noarea, pages = 1, scale = 0)
dev.off()

saveRDS(m_gam_noarea, "output/m_gam_noarea.rds")
writeLines(capture.output(sessionInfo()), "output/session_info.txt")
list.files("output")

#DONE
#dropping area makes it slightly better on AIC and BIC, so drop it from final models
# Auto Insurance Claim Frequency Model

An R project that looks at how often auto insurance claims are filed, using the French motor third-party liability dataset (freMTPL2freq, about 678,000 policies). The goal was to see how driver and vehicle characteristics relate to claim frequency (claims per policy year) and how well they predict it for pricing.

**Report:** [fksdnnf] (source: `interpret.qmd`)

## What I did
- Cleaned the data and capped a few extreme values (under 1% of policies each)
- Split the policies 80/20 into training and test sets
- Compared a flat-rate baseline, a Poisson GLM, and a GAM using Poisson deviance on the test set
- Checked that dropping area (a grouped version of density) made no difference, and that the results hold for policies with more than three months of cover
- Compared actual and expected claims by driver age, bonus-malus, and risk group

## Main findings
- Drivers aged 18 to 24 have about twice the claim frequency of any other age group.
- The highest bonus-malus band has more than four times the claim frequency of the lowest.
- The GAM beat the baseline by 4.57% on test deviance, and the GLM by 2.88%.
- The riskiest 10% of policies have about six times the claim frequency of the least risky 10%.

## Files
- `R/01_claimfreqmodel.R`: the full analysis script
- `interpret.qmd`: report source
- `index.html`: the rendered report
- `output/`: charts and tables produced by the script

## To rerun it
1. Download the dataset from [https://www.kaggle.com/datasets/floser/french-motor-claims-datasets-fremtpl2fre].
2. Run the script. It needs tidyverse, data.table, janitor, and mgcv.
3. Render `interpret.qmd` with Quarto.

## Limitations
This models claim frequency only (not claim cost), uses data from one market, and describes patterns in the data rather than causes.

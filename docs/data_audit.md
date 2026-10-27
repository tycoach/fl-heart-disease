# Data audit: UCI Heart Disease (four institutions)

Produced by `scripts/audit_data.py`. These numbers justify the feature set in `configs/default.yaml`.

## Sites

| Site | Rows | Positive rate (num > 0) |
|---|---|---|
| Cleveland | 303 | 45.9% |
| Hungary | 294 | 36.1% |
| Switzerland | 123 | 93.5% (about 8 negatives) |
| VA Long Beach | 200 | 74.5% |

Prevalence ranges from 36% to 94% across institutions: genuine label-distribution skew (non-IID).

## Missing data (NaN, plus 0 where 0 means "not recorded")

| Feature | Cleveland | Hungary | Switzerland | VA | Decision |
|---|---|---|---|---|---|
| ca | 1% | 99% | 96% | 99% | Dropped |
| thal | 1% | 91% | 42% | 83% | Dropped |
| slope | 0% | 65% | 14% | 51% | Dropped |
| chol (NaN + zeros) | 0% | 8% | **100%** | 28% | Dropped |
| trestbps | 0% | <1% | 2% | 29% | Kept, per-site median imputation |
| thalach / exang | 0% | <1% | <1% | 27% | Kept, per-site imputation |
| oldpeak | 0% | 0% | 5% | 28% | Kept, per-site median imputation |
| fbs | 0% | 3% | 61% | 4% | Kept, per-site mode imputation (limitation) |

## Decisions

1. **ca, thal, slope dropped for all sites.** Imputing features that are 83-99% empty would fabricate data,
   and keeping them only where available would break the shared feature space required by horizontal FL.
2. **chol dropped for all sites.** Every Switzerland record has chol = 0 (not recorded). Options considered:
   impute from other sites (breaks federated privacy), add a missing-indicator (lets the model identify the
   Switzerland site, which is 94% positive, i.e. a label shortcut), or exclude Switzerland (loses one of four
   institutions). Dropping the feature keeps all four institutions and a clean shared feature space.
   A sensitivity run with chol on three sites (excluding Switzerland) is a possible extension.
3. **Imputation and scaling are fitted on each site's training split only.**
4. **Stratified splits**, repeated over several seeds. Switzerland's test split holds only about 2 negatives,
   so per-site precision/recall/AUC there are unstable; pooled-test results are the headline numbers.

Resulting model input: 9 raw features -> 14 columns (4 continuous, 3 binary, cp one-hot x4, restecg one-hot x3).

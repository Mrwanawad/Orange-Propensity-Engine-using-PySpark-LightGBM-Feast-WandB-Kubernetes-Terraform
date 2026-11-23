set -e
cd "/c/Users/KimoStore/Desktop/Systems Limited/orange-propensity-engine"
true
T="uv run python -m ope.models.tune"
$T tune --fresh --target appetency >> reports/cv/tune_appetency.log 2>&1
$T tune --fresh --target upselling --family lightgbm --family xgboost --family catboost >> reports/cv/tune_upselling.log 2>&1
$T select > reports/cv/select.log 2>&1
$T crosstarget > reports/cv/crosstarget.log 2>&1
uv run python -m ope.models.train freeze > reports/cv/freeze.log 2>&1
echo CHAIN_DONE

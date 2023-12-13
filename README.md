# 🚀 Timepulse: Unleash the Power of Time Series Processing and Modeling!

Welcome to Timepulse, your ultimate destination for conquering the world of time series data with unprecedented ease and efficiency.

## Table of Contents 🌟

- [Installation Magic](#installation)
- [Journey into the Time Dimension](#usage)
- [Spellbinding Features](#features)
- [Contribute to the Time Nexus](#contributing)
- [License](#license)
- [Credits: Makers of Timepulse](#credits)

## Installation 🧙‍♂️

To summon the might of Timepulse into your realm, a simple pip command shall suffice:

```bash
pip install timepulse
```

# Timepulse: Unleash the Power of Time Series Processing and Modeling!

Witness the seamless integration of temporal mastery into your coding sanctum!

## Journey into the Time Dimension ⏳

Embark on an odyssey through time as you harness the Timepulse magic. Transform raw time series data into insights that transcend the ordinary. Your code, now a symphony of temporal brilliance!

### How to use timepulse
#### Import 
```{python}
from timepulse.models.nn import MultivariateDenseWrapper
from timepulse.utils.models import run_model
```
#### Run model 

```{python}
 y_pred, result_metrics = run_model(model_instance, X_train.values, y_train, X_val.values, y_val, verbose=0)
```

#### Results example
```
[ 8287547.5 10593171.  12349981.  13229407.  10743349.   8585146.
  7701900.   6690604.5  6193717.5  5999759.   5651086.5  5573535.5
  9689985.  12014953.  14384974.  14222123. ],

{'mae': 452268.53, 'mse': 452862200000.0, 'rmse': 672950.4, 'mape': 4.8490524, 'smape': 10.239903, 'mase': 0.31943747, 'r2_score': 0.94421965}
```

## Spellbinding Features ✨

- **Temporal Alchemy:** Shape time series data effortlessly.
- **Predictive Sorcery:** Unlock the future with powerful modeling.
- **Intuitive Elixir:** Simplify complexities with an enchantingly user-friendly interface.

## License 📜

[Timepulse - The SQD License](https://www.squaredev.io/)

## Credits: Makers of Timepulse 🌈

Timepulse was conjured into existence by the brilliant minds at Squaredev. Their dedication to temporal excellence knows no bounds.

Embark on a journey with Timepulse – where time meets brilliance, and magic unfolds! ✨🕰️🚀

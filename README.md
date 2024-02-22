# 🚀 Timepulse: Unleash the Power of Time Series Processing and Modeling!

[![GitHub][github_badge]][github_link] [![PyPI][pypi_badge]][pypi_link] 
[![Test][test_passing_badge]][test_passing_badge]
[![Download][download_badge]][download_link] 
[![Download][total_download_badge]][download_link]
[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-green.svg)](#supported-python-versions)
[![Licence][licence_badge]][licence_link]
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) 


Welcome to Timepulse, your ultimate destination for conquering the world of time series data with unprecedented ease and efficiency.


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



[github_badge]: https://badgen.net/badge/icon/GitHub?icon=github&color=black&label

[github_link]: https://github.com/squaredev-io/timepulse

[pypi_badge]: https://badge.fury.io/py/timepulse.svg

[pypi_link]: https://pypi.org/project/timepulse

[download_badge]: https://badgen.net/pypi/dm/timepulse

[total_download_badge]: https://static.pepy.tech/personalized-badge/timepulse?period=total&units=international_system&left_color=grey&right_color=green&left_text=Total%20Downloads

[download_link]: https://pypi.org/project/timepulse/#files

[licence_badge]: https://img.shields.io/github/license/squaredev-io/timepulse

[licence_link]: LICENSE

[test_passing_badge]: https://github.com/squaredev-io/timepulse/actions/workflows/release-pypi-package.yml/badge.svg

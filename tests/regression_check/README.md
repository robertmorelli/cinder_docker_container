# Regression Check

This folder contains transform-regression cases that do not store expected transformed output.

Each case is validated by running `de_typer_boxunbox.py`-equivalent logic with:
- selected pass set
- selected function plan
- expected outcome (`typecheck_ok`, `typecheck_fail`, or `transform_fail`)

Baseline requirement:
- Original case source is static-typechecked first.
- Baseline must typecheck for the case to be considered valid.

Run locally in the container environment:

```bash
python3 tests/regression_check/run_regression_check.py
```

Or via wrapper:

```bash
bash tests/regression_check/run_regression_check.sh
```

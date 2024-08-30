# Water Quality Index Prediction

#### Author: Levi (nguyenhongson.kstn.hust@gmail.com) et. al.

## Description



## To set up new dev environment

```shell
cd WaterQualityResearch
mkdir .venv
make dev-install
```

## Set up pre-commit check

```shell
make setup-pre-commit-hook
```

## Before merging any commit to main branch

```shell
make run-pre-commit
```

## To add new dependencies

1. Add the dependencies to `Pipfile`
2. Run:

```shell
make pipenv-lock
```

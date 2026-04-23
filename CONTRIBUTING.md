# Contributing to qctddft

Thanks for your interest in improving `qctddft`.

## Ways to contribute
- report bugs or unexpected behavior
- suggest workflow improvements or new analysis features
- improve documentation and examples
- contribute tests, refactoring, or performance improvements

## Development setup
Clone the repository and install the package with development dependencies:

```bash
git clone https://github.com/fr-0zt/qctddft.git
cd qctddft
pip install .[dev]
```

## Basic expectations
- keep changes focused and well scoped
- preserve command-line behavior unless the change intentionally updates it
- update documentation when user-facing behavior changes
- add or update tests when practical

## Before opening a pull request
Please try to:
- run the test suite with `pytest`
- make sure the code still installs cleanly
- confirm that command-line examples in the README remain accurate

## Pull requests
Small, targeted pull requests are preferred. For larger feature ideas, opening an issue first is helpful so scope and design can be discussed before implementation.

# Pretraining text

Raw finance text for continued pretraining. How these folders are filtered and packed is described in [../README.md](../README.md).

- `books/` holds book pages, including `pages.json`.
- `articles/` holds long-form articles.
- `corpus/` holds filing sections.

`phase1/build_dataset.py` reads this folder. It does not contain EGX question-answer pairs.

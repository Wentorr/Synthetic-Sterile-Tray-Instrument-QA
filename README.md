# Sterile Tray Instrument Verification Image QA

Complete Eris-style challenge package for a synthetic multimodal computer vision task.

- `raw/generate_raw.py` creates deterministic sterile-tray inspection PNGs and trace-derived QA rows.
- `raw/questions.csv` contains all raw questions, answers, and hidden evaluation groups.
- `raw/images/` contains source PNG images.
- `dataset_description_eris_upload.md` documents the dataset.
- `prepare.py` creates public train/test files and private answers with obfuscated IDs and image names.
- `problem.md` is the solver-facing challenge statement.
- `grade.py` computes weighted accuracy with worst-group penalties.
- `rubrics.yaml` contains task-specific rubric criteria.
- `reference_solution.py` and `solution.ipynb` provide a visual-parser baseline.

## Submission Mapping

Dataset upload:
- Title: `Synthetic Sterile Tray Instrument QA Dataset`
- Description: paste `dataset_description_eris_upload.md`
- Data file: upload `sterile-tray-inspection-vqa-raw.tar.gz`
- License: `CC0 1.0 Public Domain`

Challenge:
- Domain: `Computer Vision`
- Difficulty: `Medium`
- Tags: `image`, `multimodal`, `small-data`, `medical`, `feature-engineering`
- Title: `Sterile Tray Instrument Verification Image QA`
- Grade direction: `Maximize`
- Min score: `0`
- Max score: `1`
- Problem description: paste `problem.md`
- Grading script: paste `grade.py`
- Prepare script: paste `prepare.py`
- Rubrics: use `rubrics.yaml`
- Reference solution: upload `solution.ipynb`


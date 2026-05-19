# 📓 Notebooks & Research Pipeline

Research and experimentation assets for **Ultra Pro Access Core AI**.

| Folder | Contents |
|--------|----------|
| `drafts/` | Exploratory Jupyter notebooks (CNN baseline, augmentation, ResNet) |
| `final/` | Polished French notebook — full DL pipeline narrative |
| `src/` | Data collection & training pipeline scripts |
| `*.py` | CLI utilities (`build_database.py`, `enroll_user.py`, …) |

**Production app** remains at project root: `main.py`

## Paths

All scripts use `notebooks/project_paths.py` to resolve:

- `../dataset/`
- `../database/`

## Commands

```bash
# From project root
python3 notebooks/build_database.py
python3 notebooks/train.py
python3 notebooks/enroll_user.py

# Jupyter
jupyter notebook notebooks/drafts/
jupyter notebook notebooks/final/Notebook_Final_Soigne.ipynb
```

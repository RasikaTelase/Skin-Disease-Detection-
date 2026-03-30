# Skin Disease Detection (SDD)

A Python + Flask project to detect skin diseases from images using CNN and provide disease info.

## Features

- Image classification model (Keras/TensorFlow)
- Data preparation and training scripts
- Predict endpoint / UI
- Disease metadata (`data/disease_info.json`)
- Plot training history and metrics

## Requirements

- Python 3.8+
- pip install -r requirements.txt

## Setup

1. Clone:
   `git clone https://github.com/<username>/Skin-Disease-Detection.git`
2. Create venv:
   `python -m venv .venv`
3. Activate:
   `.\.venv\Scripts\activate`
4. Install:
   `pip install -r requirements.txt`

## Dataset

- Put images under `data/IMG_CLASSES/` (or from public source)
- Use `split_data.py` to create train/val/test sets.

## Scripts

- `data_preprocessing.py` : prepare
- `model.py` : model architecture
- `train.py` / `train_fast.py` : training
- `resume_training.py` : resume from checkpoint
- `predict.py`: inference
- `plot_training_history.py`, `plot_metrics_and_compression.py`
- `app.py` : Flask web app
- `utils.py` : helpers
- `config.py` : path/hyperparameters

## Run

- Train:
    `python train.py`
- Predict:
    `python predict.py --image path`
- Web app:
    `python app.py` then open `http://127.0.0.1:5000`

## Notes

- Do not commit large datasets/checkpoints.
- Add `.gitignore`:
  - `__pycache__/`
  - `.venv*/`
  - `*.py[cod]`
  - `models/`
  - `data/raw/`, `data/IMG_CLASSES/`

## License

MIT

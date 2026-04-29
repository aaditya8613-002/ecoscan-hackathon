# EcoScan

point your camera at rubbish — it tells you where to bin it.

built for a hackathon. model trained with Teachable Machine on 5 waste categories: plastic, paper, metal, organic, e-waste.

## run it

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

> **note:** use `python -m streamlit` not just `streamlit` if you get a command not found error — this is a PATH thing on windows.

## files

- `app.py` — the whole app
- `keras_model.h5` — trained image classifier  
- `labels.txt` — class names
- `requirements.txt` — dependencies

## how it works

takes a photo → resizes to 224×224 → runs through the keras model → shows top prediction + disposal tip + CO₂ impact estimate.

the CO₂ numbers are rough estimates based on average recycling savings per material type.

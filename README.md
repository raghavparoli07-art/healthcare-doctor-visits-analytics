# Healthcare Analytics for Doctor Visits

An educational project exploring factors associated with doctor visits and predicting whether a person is likely to visit a doctor.

## Demo

[Open the Streamlit demo](https://healthcare-doctor-visits-an-9uyyckgcoshudykg9qmjx7.streamlit.app/)

Use synthetic example values only. This project is for learning and is not medical advice.

## Project files

- `data.csv` — dataset used for the analysis.
- `analysis.py` — creates charts, compares Logistic Regression and Random Forest classifiers, and saves the Random Forest model.
- `app.py` — Streamlit interface for predictions and chart viewing.
- `model.pkl` — trained Random Forest model loaded by the app.
- `out/` — generated charts and confusion matrix.
- `PPT.pptx` — project presentation.

## Run locally

Use Python 3.10 or newer. From the project folder:

```bash
python -m pip install -r requirements.txt
python analysis.py
streamlit run app.py
```

The analysis writes `model.pkl` and the charts to `out/`. The Streamlit app uses the model and charts in those locations.

## Analysis

The target is `visited`, set to 1 when `visits` is greater than zero and 0 otherwise. The analysis compares a class-balanced Logistic Regression model and a class-balanced Random Forest model, and creates visit-count, group-mean, illness/reduced-activity, correlation, feature-importance, and confusion-matrix charts.

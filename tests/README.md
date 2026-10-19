
# SentinelAI

An AI-powered analytics and cybersecurity risk dashboard built with Python and Streamlit.

## Features

- **Data Analytics:** Explore datasets, inspect columns, and visualize numerical data.
- **AI Anomaly Detection:** Use Isolation Forest to identify unusual records.
- **Security Center:** Review simulated security events and risk scores.
- **Interactive Dashboard:** Explore metrics, charts, and event records.
- **CSV Support:** Upload datasets for analysis and export anomaly results.

## Technology Stack

- Python
- Streamlit
- Pandas and NumPy
- Plotly
- Scikit-learn
- Pytest

## Run Locally

1. Install Python.
2. Open a terminal in the project directory.
3. Create a virtual environment:

   `python -m venv .venv`

4. Install dependencies:

   `python -m pip install -r requirements.txt`

5. Start the application:

   `python -m streamlit run app.py`

## Run Tests

Use the project's virtual environment:

`.\.venv\Scripts\python.exe -m pytest .\tests -v`

## Project Structure

- `app.py` — Streamlit dashboard
- `modules/ai_engine.py` — AI anomaly detection
- `modules/analytics.py` — Dataset profiling and cleaning
- `tests/` — Automated tests
- `sample_data/` — Sample datasets
- `requirements.txt` — Python dependencies
- `.gitignore` — Files excluded from Git

## Important Note

The Security Center currently displays simulated event data. It does not provide live network monitoring or guarantee threat detection.

## License

To be decided.

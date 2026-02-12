# Deploy to Render (Streamlit + TensorFlow)

## Quick overview
This repo is a Streamlit web app for NGX multi-sheet Excel portfolio optimisation.

Render runs it as a **Python Web Service**.

## Deploy steps (GUI)
1. Push this folder to a GitHub repo.
2. In Render: **New +** → **Blueprint** → select your repo.
3. Render will detect `render.yaml` and create the service.
4. Click **Deploy**.

## Notes
- The app listens on `0.0.0.0` and uses Render's `$PORT`.
- TensorFlow is included (`tensorflow==2.15.*`). First build can take longer.
- If you want to run without TensorFlow, delete the tensorflow line from `requirements.txt`.
  The app will automatically fall back to rule-based regimes.

## Local run (Python 3.11)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
pip install -r requirements.txt
streamlit run ngx_ai_app/app.py
```

# AI Prescription Verifier

Run locally with FastAPI backend and Streamlit frontend.

## Setup (one-time)
python -m venv .venv
source .venv/bin/activate   # mac/linux
.venv\Scripts\activate      # windows

pip install -r requirements.txt

## Train model (optional)
python model_training.py

## Run backend
uvicorn api:app --reload

## Run frontend
streamlit run frontend.py

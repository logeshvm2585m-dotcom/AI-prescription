# api.py
import os, re, pickle, logging
from typing import List, Dict, Any
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# config
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DDI_PATH = os.path.join(DATA_DIR, "ddi_dataset.csv")
DOSAGE_PATH = os.path.join(DATA_DIR, "dosage_dataset.csv")
TRAINING_PATH = os.path.join(DATA_DIR, "dosage_training.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "dosage_model.joblib")

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("api")

# helper to ensure files exist
def _read_csv(path):
    try:
        df = pd.read_csv(path)
        log.info(f"Loaded {path} ({len(df)} rows)")
        return df
    except Exception as e:
        log.error(f"Failed to load {path}: {e}")
        raise

# load datasets
ddi_df = _read_csv(DDI_PATH)
dosage_df = _read_csv(DOSAGE_PATH)

# build simple DDI lookup (normalized)
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()

_ddi_lookup: Dict[tuple, List[Dict[str,str]]] = {}
for _, r in ddi_df.iterrows():
    a, b = _norm(r["drug_a"]), _norm(r["drug_b"])
    if a and b:
        key = tuple(sorted([a, b]))
        _ddi_lookup.setdefault(key, []).append({
            "interaction_description": r.get("interaction_description", ""),
            "severity": r.get("severity", "unknown")
        })
log.info(f"DDI pairs loaded: {len(_ddi_lookup)}")

# try to load ML model (joblib/pickle)
dosage_model = None
label_encoder = None
if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            bundle = pickle.load(f)
        dosage_model = bundle.get("model")
        label_encoder = bundle.get("label_encoder")
        log.info("Loaded ML model.")
    except Exception as e:
        log.error(f"Failed to load model: {e}")
        dosage_model = None
        label_encoder = None
else:
    log.info("No ML model found at models/dosage_model.joblib. Train using model_training.py or allow auto-train.")

# parse medicines and extract numeric dose if present
def extract_medicines(text: str) -> List[Dict[str,Any]]:
    meds = []
    if not text:
        return meds
    # capture name and optional number (e.g., 'paracetamol 500 mg' or 'ibuprofen 200')
    pattern = re.compile(r"\b([a-zA-Z][a-zA-Z\-]{2,})\b(?:\s*[-:]?\s*(\d+(?:\.\d+)?)(?:\s*(mg|ml|mcg))?)?", re.I)
    for name, num, unit in pattern.findall(text):
        n = name.lower()
        dose_mg = None
        dose_text = ""
        if num:
            dose_text = f"{num}{(' ' + unit) if unit else ''}"
            try:
                dose_mg = float(num)
            except:
                dose_mg = None
        if n not in {"take","tab","tablet","capsule","po","od","bd","tid","qid","prn"}:
            meds.append({"name": n, "dose": dose_text, "dose_mg": dose_mg})
    return meds

def check_dosage_rules(drug: str, age: int, dose: float) -> Dict[str,Any]:
    result = {"drug": drug, "dose_mg": dose, "rule_status": "unknown", "rule_note": "No rule found"}
    norm_drug = _norm(drug)
    if age <= 12: age_group = "children"
    elif age <= 19: age_group = "teens"
    elif age <= 59: age_group = "adults"
    else: age_group = "seniors"
    rules = dosage_df[(dosage_df["drug"].str.lower() == norm_drug) & (dosage_df["age_group"].str.lower() == age_group)]
    if not rules.empty:
        row = rules.iloc[0]
        try:
            min_d = float(row["min_dose_mg"])
            max_d = float(row["max_dose_mg"])
            note = row.get("notes", "")
            if dose < min_d:
                result.update({"rule_status": "low", "rule_note": f"Dose below recommended range ({min_d}-{max_d} mg). {note}"})
            elif dose > max_d:
                result.update({"rule_status": "high", "rule_note": f"Dose above recommended range ({min_d}-{max_d} mg). {note}"})
            else:
                result.update({"rule_status": "safe", "rule_note": f"Within recommended range ({min_d}-{max_d} mg). {note}"})
        except Exception as e:
            result.update({"rule_status": "unknown", "rule_note": f"Invalid rule values: {e}"})
    return result

def predict_ml(drug: str, age: int, dose: float) -> Dict[str,Any]:
    if dosage_model is None or label_encoder is None:
        return {"ml_status": "unavailable", "ml_note": "ML model not loaded"}
    dnorm = drug.lower().strip()
    if dnorm not in label_encoder.classes_:
        return {"ml_status": "unavailable", "ml_note": "Drug not in ML vocabulary"}
    try:
        d_enc = int(label_encoder.transform([dnorm])[0])
        X = pd.DataFrame([[d_enc, float(age), float(dose)]], columns=["drug_encoded","age","dose"])
        pred = dosage_model.predict(X)[0]
        return {"ml_status": "safe" if int(pred)==1 else "unsafe", "ml_note": f"ML predicted {'safe' if int(pred)==1 else 'unsafe'}"}
    except Exception as e:
        return {"ml_status": "error", "ml_note": str(e)}

# FastAPI models
class InteractionRequest(BaseModel):
    prescription_text: str = Field(...)

class DosageRequest(BaseModel):
    prescription_text: str = Field(...)
    age: int = Field(..., ge=0, le=120)

# app
app = FastAPI(title="AI Prescription Verifier", version="1.0")

@app.get("/health")
def health():
    return {"status": "ok", "ddi_pairs": len(_ddi_lookup), "dosage_rules": len(dosage_df), "ml_loaded": dosage_model is not None}

@app.post("/check_interactions")
def check_interactions(req: InteractionRequest):
    meds = extract_medicines(req.prescription_text)
    found = []
    seen = set()
    names = [_norm(m["name"]) for m in meds]
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            key = tuple(sorted([names[i], names[j]]))
            if key in seen: continue
            seen.add(key)
            info = _ddi_lookup.get(key, [])
            for item in info:
                found.append({
                    "drug_a": key[0],
                    "drug_b": key[1],
                    "interaction_description": item.get("interaction_description",""),
                    "severity": item.get("severity","unknown")
                })
    return {"medicines_detected": meds, "interactions_found": found}

@app.post("/check_dosage")
def check_dosage(req: DosageRequest):
    meds = extract_medicines(req.prescription_text)
    results = []
    for m in meds:
        dose_mg = m.get("dose_mg")
        if dose_mg is None:
            results.append({"drug": m["name"], "dose": m.get("dose",""), "rule_status": "unknown", "rule_note": "No numeric dose provided", "ml_status": "unavailable"})
            continue
        rule_res = check_dosage_rules(m["name"], req.age, dose_mg)
        ml_res = predict_ml(m["name"], req.age, dose_mg)
        results.append({**rule_res, **ml_res})
    return {"age": req.age, "dosage_check": results}

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

APP_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = APP_ROOT / "model_store" / "mimic_los_rf.joblib"
ASSOC_RULES_PATH = APP_ROOT / "processed" / "mimic" / "assoc_rules_dx.csv"
EVAL_METRICS_PATH = APP_ROOT / "processed" / "mimic" / "evaluation_metrics.json"

app = FastAPI(title="Hospital Analytics API", version="1.0.0", docs_url="/", redoc_url=None)

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


@app.get("/favicon.ico")
async def favicon() -> Response:
	return Response(status_code=204)


class LOSRequest(BaseModel):
	features: Dict[str, Any] = Field(..., description="Feature name to value mapping for LOS model")


class LOSResponse(BaseModel):
	prediction: float
	unit: str = "days"


class AssocRule(BaseModel):
	antecedents: List[str]
	consequents: List[str]
	support: float
	confidence: float
	lift: float


class EvalMetrics(BaseModel):
	classification: Optional[Dict[str, float]] = None
	regression: Optional[Dict[str, float]] = None
	clustering: Optional[Dict[str, float]] = None
	associations: Optional[Dict[str, float]] = None


def _load_model() -> Optional[Any]:
	if MODEL_PATH.exists():
		try:
			return joblib.load(MODEL_PATH)
		except Exception as exc:  # noqa: BLE001
			print(f"Failed to load model at {MODEL_PATH}: {exc}")
	return None


def _ensure_dataframe(input_features: Dict[str, Any], model) -> pd.DataFrame:
	if hasattr(model, "feature_names_in_"):
		feature_names = list(model.feature_names_in_)
	else:
		feature_names = list(input_features.keys())
	row = {name: input_features.get(name, np.nan) for name in feature_names}
	df = pd.DataFrame([row])
	return df


@app.get("/health")
async def health() -> Dict[str, str]:
	state = {
		"model_loaded": str(MODEL_PATH.exists()),
		"assoc_rules_available": str(ASSOC_RULES_PATH.exists()),
		"metrics_available": str(EVAL_METRICS_PATH.exists()),
	}
	return {"status": "ok", **state}


@app.post("/predict/los", response_model=LOSResponse)
async def predict_los(req: LOSRequest) -> LOSResponse:
	model = _load_model()
	if model is None:
		raise HTTPException(status_code=503, detail="LOS model not available")

	df = _ensure_dataframe(req.features, model)
	try:
		pred = model.predict(df)[0]
		if isinstance(pred, (np.generic,)):
			pred = float(pred)
	except Exception as exc:  # noqa: BLE001
		raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

	return LOSResponse(prediction=float(pred))


@app.get("/associations", response_model=List[AssocRule])
async def get_association_rules(limit: int = 50) -> List[AssocRule]:
	if not ASSOC_RULES_PATH.exists():
		raise HTTPException(status_code=404, detail="Association rules file not found")
	try:
		df = pd.read_csv(ASSOC_RULES_PATH)
		# attempt to normalize antecedents/consequents stringified lists
		def normalize(val):
			if isinstance(val, str):
				text = val.strip()
				if text.startswith("[") and text.endswith("]"):
					parts = [p.strip(" '\"") for p in text[1:-1].split(",") if p.strip()]
					return parts
			return [str(val)]

		records: List[AssocRule] = []
		for _, row in df.head(limit).iterrows():
			records.append(
				AssocRule(
					antecedents=normalize(row.get("antecedents", [])),
					consequents=normalize(row.get("consequents", [])),
					support=float(row.get("support", 0.0)),
					confidence=float(row.get("confidence", 0.0)),
					lift=float(row.get("lift", 0.0)),
				)
			)
		records.sort(key=lambda r: r.lift, reverse=True)
		return records
	except Exception as exc:  # noqa: BLE001
		raise HTTPException(status_code=500, detail=f"Failed to read association rules: {exc}") from exc


@app.get("/metrics", response_model=EvalMetrics)
async def get_metrics() -> EvalMetrics:
	if not EVAL_METRICS_PATH.exists():
		raise HTTPException(status_code=404, detail="Evaluation metrics not found")
	try:
		with open(EVAL_METRICS_PATH, "r", encoding="utf-8") as f:
			data = json.load(f)
		return EvalMetrics(**data)
	except Exception as exc:  # noqa: BLE001
		raise HTTPException(status_code=500, detail=f"Failed to read metrics: {exc}") from exc


# For local dev: uvicorn api.main:app --reload

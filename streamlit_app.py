import json
import os
from typing import Any, Dict, List

import pandas as pd
import requests
import streamlit as st

# Resolve API base URL without requiring secrets.toml
API_BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000")
try:
	# If secrets exist and key provided, override
	if "API_BASE" in st.secrets:
		API_BASE = st.secrets["API_BASE"]
except Exception:
	# secrets may not be configured; ignore
	pass

st.set_page_config(page_title="Hospital Analytics Dashboard", layout="wide")

st.title("Hospital Analytics Dashboard")

@st.cache_data(ttl=30)
def get_health() -> Dict[str, Any]:
	try:
		r = requests.get(f"{API_BASE}/health", timeout=5)
		r.raise_for_status()
		return r.json()
	except Exception as e:
		return {"status": "error", "detail": str(e)}

@st.cache_data(ttl=60)
def get_metrics() -> Dict[str, Any]:
	try:
		r = requests.get(f"{API_BASE}/metrics", timeout=10)
		r.raise_for_status()
		return r.json()
	except Exception:
		return {}

@st.cache_data(ttl=60)
def get_associations(limit: int = 50) -> List[Dict[str, Any]]:
	try:
		r = requests.get(f"{API_BASE}/associations", params={"limit": limit}, timeout=15)
		r.raise_for_status()
		return r.json()
	except Exception:
		return []


def predict_los(features: Dict[str, Any]) -> Dict[str, Any]:
	try:
		r = requests.post(f"{API_BASE}/predict/los", json={"features": features}, timeout=20)
		if r.status_code == 503:
			return {"error": "Model not available"}
		r.raise_for_status()
		return r.json()
	except Exception as e:
		return {"error": str(e)}


with st.sidebar:
	st.header("Backend Status")
	health = get_health()
	if health.get("status") == "ok":
		st.success("API reachable")
		st.write({k: v for k, v in health.items() if k != "status"})
	else:
		st.error("API not reachable")
		st.caption(str(health))

	tabs = st.container()

metrics_tab, assoc_tab, pred_tab = st.tabs(["Evaluation Metrics", "Association Rules", "Predict LOS"])

with metrics_tab:
	st.subheader("Evaluation Metrics")
	metrics = get_metrics()
	if not metrics:
		st.info("No metrics available. Ensure processed/mimic/evaluation_metrics.json exists.")
	else:
		# Display by sections if present
		for section in ["classification", "regression", "clustering", "associations"]:
			if section in metrics and metrics[section]:
				st.markdown(f"**{section.title()}**")
				st.json(metrics[section])

with assoc_tab:
	st.subheader("Top Association Rules")
	limit = st.slider("Max rules", 5, 200, 50, 5)
	rules = get_associations(limit)
	if not rules:
		st.info("No association rules file found or failed to load.")
	else:
		df_rules = pd.DataFrame(rules)
		df_rules["antecedents"] = df_rules["antecedents"].apply(lambda lst: ", ".join(lst))
		df_rules["consequents"] = df_rules["consequents"].apply(lambda lst: ", ".join(lst))
		st.dataframe(df_rules, use_container_width=True)

with pred_tab:
	st.subheader("Patient LOS Prediction")
	st.caption("Enter feature values matching your LOS model training features.")
	col_left, col_right = st.columns(2)
	with col_left:
		st.markdown("**Manual Input**")
		# A small dynamic input helper: user can type JSON of features
		default_json = "{}"
		feat_json = st.text_area("Features JSON", value=default_json, height=160,
			help="Provide a JSON object: {\"age\": 65, \"gender_M\": 1}")
		predict_clicked = st.button("Predict LOS")
		if predict_clicked:
			try:
				features = json.loads(feat_json or "{}")
			except Exception as e:
				st.error(f"Invalid JSON: {e}")
				features = None
			if isinstance(features, dict):
				result = predict_los(features)
				if "error" in result:
					st.error(result["error"])
				else:
					st.success(f"Predicted LOS: {result['prediction']:.2f} {result.get('unit','days')}")

	with col_right:
		st.markdown("**Batch Upload (CSV)**")
		uploaded = st.file_uploader("Upload CSV of rows of features", type=["csv"])
		if uploaded is not None:
			try:
				df = pd.read_csv(uploaded)
				st.write("Detected columns:", list(df.columns))
				if st.button("Predict for all rows"):
					preds: List[float] = []
					for _, row in df.iterrows():
						features = row.to_dict()
						res = predict_los(features)
						preds.append(res.get("prediction", float("nan")))
					df_out = df.copy()
					df_out["predicted_LOS_days"] = preds
					st.dataframe(df_out, use_container_width=True)
					st.download_button("Download Predictions CSV", df_out.to_csv(index=False), file_name="los_predictions.csv")
			except Exception as e:
				st.error(f"Failed to process CSV: {e}")

st.caption("Optionally set API_BASE via environment variable or .streamlit/secrets.toml.")

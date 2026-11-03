# Databricks notebook source
# MAGIC %md
# MAGIC # 00 - Config
# MAGIC Shared names. Every other notebook starts with `%run ./00_config`.
# MAGIC **Use synthetic or de-identified data only until your org has a BAA and the HIPAA compliance security profile enabled.**

# COMMAND ----------

CATALOG  = "healthcare_ai"
SCHEMA   = "rag"
VOLUME   = "raw"
VS_ENDPOINT = "hc_vs_endpoint"
EMBED_ENDPOINT = "databricks-gte-large-en"                # embedding model served by Foundation Model APIs
LLM_ENDPOINT   = "databricks-meta-llama-3-3-70b-instruct" # change if your workspace lists a different chat model
T = lambda name: f"{CATALOG}.{SCHEMA}.{name}"
INDEX_NAME = T("chunks_index")
VOL_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
print("config loaded:", CATALOG, SCHEMA, VS_ENDPOINT)

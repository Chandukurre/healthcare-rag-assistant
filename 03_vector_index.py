# Databricks notebook source
# MAGIC %md
# MAGIC # 03 - Vector Search endpoint + Delta Sync index
# MAGIC The index embeds `chunks.text` with a Databricks-hosted embedding model and re-syncs when the table changes (Change Data Feed).
# MAGIC Creating the endpoint takes a few minutes; the first index sync a few more.

# COMMAND ----------
# MAGIC %pip install databricks-vectorsearch databricks-ai-search

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %run ./00_config

# COMMAND ----------
# Docs currently show AISearchClient (databricks-ai-search); older workspaces use VectorSearchClient. Try both.
try:
    from databricks.ai_search.client import AISearchClient as _Client
except Exception:
    from databricks.vector_search.client import VectorSearchClient as _Client
vsc = _Client()

# COMMAND ----------
import time
existing = [e["name"] for e in vsc.list_endpoints().get("endpoints", [])]
if VS_ENDPOINT not in existing:
    vsc.create_endpoint(name=VS_ENDPOINT, endpoint_type="STANDARD")
while True:
    state = vsc.get_endpoint(VS_ENDPOINT).get("endpoint_status", {}).get("state")
    print("endpoint:", state)
    if state == "ONLINE": break
    time.sleep(20)

# COMMAND ----------
try:
    index = vsc.create_delta_sync_index(
        endpoint_name=VS_ENDPOINT,
        source_table_name=T("chunks"),
        index_name=INDEX_NAME,
        pipeline_type="TRIGGERED",
        primary_key="chunk_id",
        embedding_source_column="text",
        embedding_model_endpoint_name=EMBED_ENDPOINT,
        columns_to_sync=["chunk_id", "doc_id", "patient_id", "doc_name", "doc_date", "page", "section", "text"],
    )
except Exception as e:
    print("index may already exist:", e)
    index = vsc.get_index(endpoint_name=VS_ENDPOINT, index_name=INDEX_NAME)

while not index.describe().get("status", {}).get("detailed_state", "").startswith("ONLINE"):
    print("waiting for index..."); time.sleep(20)
print("index ONLINE")

# COMMAND ----------
# MAGIC %md ### Re-sync after you ingest new documents
# MAGIC `index.sync()` (TRIGGERED pipeline) - run it after every ingest.

# COMMAND ----------
# index.sync()

# COMMAND ----------
# smoke test: a search restricted to ONE patient must never return another patient's chunks
res = index.similarity_search(query_text="diabetes A1c", columns=["patient_id", "doc_name", "text"],
                              filters={"patient_id": "P001"}, num_results=3)
rows = res["result"]["data_array"]
print(rows)
assert rows and all(r[0] == "P001" for r in rows), "patient filter failed"
print("PASS: patient filter works")

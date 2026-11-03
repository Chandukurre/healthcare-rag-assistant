# Databricks notebook source
# MAGIC %md
# MAGIC # 04 - RAG with citations, verifier and audit log
# MAGIC Flow: patient-filtered hybrid search -> numbered passages -> LLM (temperature 0) -> verify citations/numbers -> log.

# COMMAND ----------
# MAGIC %pip install databricks-vectorsearch databricks-ai-search mlflow

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %run ./00_config

# COMMAND ----------
import os, sys, json
sys.path.append(os.getcwd())
from hc_lib import build_prompt, verify_answer
import mlflow.deployments
try:
    from databricks.ai_search.client import AISearchClient as _Client
except Exception:
    from databricks.vector_search.client import VectorSearchClient as _Client

index = _Client().get_index(endpoint_name=VS_ENDPOINT, index_name=INDEX_NAME)
llm = mlflow.deployments.get_deploy_client("databricks")
COLS = ["chunk_id", "patient_id", "doc_name", "doc_date", "page", "section", "text"]

def audit(action, patient_id, detail=""):
    user = spark.sql("SELECT current_user()").first()[0]
    spark.createDataFrame([(user, action, patient_id, detail[:2000])],
        "user_email string, action string, patient_id string, detail string") \
        .selectExpr("current_timestamp() as ts", "*").write.mode("append").saveAsTable(T("audit_log"))

def ask(patient_id: str, question: str, k: int = 5):
    if not patient_id:
        raise ValueError("patient_id is required")        # never search without a patient filter
    try:
        r = index.similarity_search(query_text=question, columns=COLS, filters={"patient_id": patient_id},
                                    num_results=k, query_type="HYBRID")
    except TypeError:                                      # older SDKs without query_type
        r = index.similarity_search(query_text=question, columns=COLS, filters={"patient_id": patient_id}, num_results=k)
    names = COLS
    passages = []
    for i, row in enumerate(r["result"]["data_array"], 1):
        d = dict(zip(names, row))
        assert d["patient_id"] == patient_id               # belt and braces
        passages.append({"sid": f"S{i}", "doc": d["doc_name"], "date": str(d["doc_date"]), "page": d["page"], "text": d["text"]})
    if not passages:
        audit("ask", patient_id, f"Q={question} -> no passages"); return {"answer": "NOT_FOUND", "passages": [], "check": {"ok": True}}
    out = llm.predict(endpoint=LLM_ENDPOINT, inputs={
        "messages": [{"role": "user", "content": build_prompt(question, passages)}],
        "max_tokens": 400, "temperature": 0})
    answer = out["choices"][0]["message"]["content"].strip()
    check = verify_answer(answer, passages)
    if not check["ok"]:
        answer = "NOT_FOUND (answer failed citation check: " + "; ".join(check["problems"]) + ")"
    audit("ask", patient_id, f"Q={question} | ok={check['ok']} | cited={check['cited'] if 'cited' in check else []}")
    return {"answer": answer, "passages": passages, "check": check}

# COMMAND ----------
r = ask("P001", "What was the latest A1c and what medicines is the patient on?")
print(r["answer"]); print(r["check"])

# COMMAND ----------
# isolation test: asking P002's chart about P001's drug must not leak
r2 = ask("P002", "Is the patient on metformin?")
print(r2["answer"])

# COMMAND ----------
# MAGIC %md ## Evaluation set (run after every change to chunking, prompt or model)

# COMMAND ----------
EVAL = [  # (patient, question, substring that MUST appear in a good answer)
    ("P001", "What was the latest A1c?", "6.8"),
    ("P001", "Which medicines is the patient taking?", "Metformin"),
    ("P002", "What condition does the patient have?", "sthma"),
    ("P002", "What is the patient's A1c?", "NOT_FOUND"),   # not in this chart -> must refuse
]
passed = 0
for pid, q, must in EVAL:
    a = ask(pid, q)["answer"]
    ok = must.lower() in a.lower()
    passed += ok
    print("PASS" if ok else "FAIL", "|", pid, q, "->", a[:100].replace("\n", " "))
print(f"{passed}/{len(EVAL)} passed")

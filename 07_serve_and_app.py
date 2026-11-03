# Databricks notebook source
# MAGIC %md
# MAGIC # 07 - Package the RAG as a model and expose it
# MAGIC Two options. Option 1 (MLflow model + Model Serving) is shown in code; Option 2 (Databricks App) is UI steps.
# MAGIC Not run in the build environment - verify in your workspace.

# COMMAND ----------
# MAGIC %pip install mlflow databricks-vectorsearch databricks-ai-search

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %run ./00_config

# COMMAND ----------
import os, sys, mlflow
from mlflow.models import infer_signature
sys.path.append(os.getcwd())

class HealthRAG(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import sys; sys.path.insert(0, context.artifacts["lib_dir"])
        from hc_lib import build_prompt, verify_answer
        self.build_prompt, self.verify = build_prompt, verify_answer
        import mlflow.deployments
        try:
            from databricks.ai_search.client import AISearchClient as C
        except Exception:
            from databricks.vector_search.client import VectorSearchClient as C
        self.index = C().get_index(endpoint_name=VS_ENDPOINT, index_name=INDEX_NAME)
        self.llm = mlflow.deployments.get_deploy_client("databricks")

    def predict(self, context, model_input):
        out = []
        cols = ["chunk_id", "patient_id", "doc_name", "doc_date", "page", "section", "text"]
        for _, row in model_input.iterrows():
            pid, q = row["patient_id"], row["question"]
            if not pid: out.append("ERROR: patient_id required"); continue
            r = self.index.similarity_search(query_text=q, columns=cols, filters={"patient_id": pid}, num_results=5)
            ps = [{"sid": f"S{i}", "doc": d[2], "date": str(d[3]), "page": d[4], "text": d[6]}
                  for i, d in enumerate(r["result"]["data_array"], 1)]
            if not ps: out.append("NOT_FOUND"); continue
            a = self.llm.predict(endpoint=LLM_ENDPOINT, inputs={"messages": [{"role": "user", "content": self.build_prompt(q, ps)}],
                                                                "max_tokens": 400, "temperature": 0})["choices"][0]["message"]["content"].strip()
            out.append(a if self.verify(a, ps)["ok"] else "NOT_FOUND (failed citation check)")
        return out

# COMMAND ----------
import pandas as pd
mlflow.set_registry_uri("databricks-uc")
example = pd.DataFrame([{"patient_id": "P001", "question": "What was the latest A1c?"}])
with mlflow.start_run():
    info = mlflow.pyfunc.log_model(
        name="health_rag", python_model=HealthRAG(),
        artifacts={"lib_dir": os.getcwd()},                       # ships hc_lib.py with the model
        input_example=example, signature=infer_signature(example, ["x"]),
        pip_requirements=["mlflow", "databricks-vectorsearch", "databricks-ai-search", "pandas"],
        registered_model_name=T("health_rag"))
print(info.model_uri)

# COMMAND ----------
# MAGIC %md
# MAGIC ### Serve it
# MAGIC 1. Left sidebar -> **Serving** -> **Create serving endpoint** -> choose model `healthcare_ai.rag.health_rag`, latest version, Small, scale to zero ON.
# MAGIC 2. The endpoint's service principal needs: `SELECT` on `healthcare_ai.rag.chunks`, **CAN QUERY** on the Vector Search index/endpoint, and access to the LLM endpoint.
# MAGIC 3. Test in the Serving UI with `{"dataframe_records":[{"patient_id":"P001","question":"What was the latest A1c?"}]}`.
# MAGIC
# MAGIC ### Optional: a simple clinician UI (Databricks Apps)
# MAGIC New -> **App** -> choose a Streamlit or Gradio template; in `app.py` call the serving endpoint with the signed-in user's token and pass the selected `patient_id`. Add the serving endpoint as an app resource.

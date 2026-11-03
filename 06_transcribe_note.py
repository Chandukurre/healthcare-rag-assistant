# Databricks notebook source
# MAGIC %md
# MAGIC # 06 - Visit recording -> transcript -> draft note
# MAGIC Needs a **GPU cluster** (e.g. single-node with an A10/T4 GPU, ML runtime). Not run in the build environment.
# MAGIC **Recording consent:** record only with consent from all parties where your state requires it; `consent_recorded` must be True.

# COMMAND ----------
# MAGIC %pip install faster-whisper

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %run ./00_config

# COMMAND ----------
import os, uuid, mlflow.deployments
from faster_whisper import WhisperModel

# Upload an audio file (wav/mp3/m4a) to  /Volumes/healthcare_ai/rag/raw/audio/  first
AUDIO = f"{VOL_PATH}/audio/visit1.wav"
PATIENT = "P001"
CONSENT = True            # set from your intake form; the code refuses to run without it

assert CONSENT, "No recording consent - do not transcribe"
model = WhisperModel("small", device="cuda", compute_type="float16")   # use "cpu","int8" on a CPU cluster (slow)
segments, info = model.transcribe(AUDIO, vad_filter=True)
text = " ".join(s.text.strip() for s in segments)
print(text[:500])

# COMMAND ----------
visit_id = "V-" + uuid.uuid4().hex[:8]
spark.createDataFrame([(visit_id, PATIENT, AUDIO, text, CONSENT)],
                      "visit_id string, patient_id string, audio_path string, text string, consent_recorded boolean") \
     .selectExpr("*", "current_timestamp() as created_at").write.mode("append").saveAsTable(T("transcripts"))

# COMMAND ----------
PROMPT = """Turn this doctor-patient conversation into a SOAP note (Subjective, Objective, Assessment, Plan).
Use ONLY facts stated in the conversation. If something is not stated write 'not discussed'. Do not invent vitals or doses.
This is a DRAFT for clinician review.

CONVERSATION:
""" + text
llm = mlflow.deployments.get_deploy_client("databricks")
draft = llm.predict(endpoint=LLM_ENDPOINT, inputs={"messages": [{"role": "user", "content": PROMPT}],
                                                   "max_tokens": 700, "temperature": 0})["choices"][0]["message"]["content"]
spark.createDataFrame([(visit_id, PATIENT, draft, "draft")], "visit_id string, patient_id string, draft_text string, status string") \
     .selectExpr("*", "cast(null as string) as signed_by", "cast(null as timestamp) as signed_at") \
     .write.mode("append").saveAsTable(T("notes"))
print(draft)
# Clinician reviews/edits with edit_draft(...) then sign_note(...) from notebook 05.

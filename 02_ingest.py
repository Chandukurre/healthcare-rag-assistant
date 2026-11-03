# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Ingest records -> chunks table
# MAGIC Step A creates **synthetic** patient records so everything runs with no downloads.
# MAGIC Step B ingests any text/PDF files you put in the Volume `docs/<patient_id>/`.
# MAGIC Real FHIR (Synthea): upload JSON bundles to `fhir/` and use Step C.

# COMMAND ----------
# MAGIC %pip install pypdf

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %run ./00_config

# COMMAND ----------
import hashlib, os, sys
from datetime import date
from pyspark.sql import Row
sys.path.append(os.getcwd())          # so hc_lib.py next to this notebook can be imported
from hc_lib import chunk_page, chunk_id

# COMMAND ----------
# MAGIC %md ### Step A - synthetic demo data (fake people, fake values)

# COMMAND ----------
demo_patients = [("P001", "Alex Demo", date(1970, 5, 1)), ("P002", "Sam Sample", date(1985, 9, 12))]
spark.createDataFrame(demo_patients, "patient_id string, full_name string, birth_date date") \
     .write.mode("overwrite").saveAsTable(T("patients"))

demo_docs = {
 "P001": [("Clinic note", date(2024, 3, 12),
   "ASSESSMENT\nType 2 diabetes, A1c 7.2 percent. Blood pressure 138/86.\nMEDICATIONS\nMetformin 500 mg twice daily. Lisinopril 10 mg daily.\nPLAN\nRecheck A1c in 3 months. Diet counseling."),
  ("Lab report", date(2024, 6, 20),
   "RESULTS\nA1c 6.8 percent. LDL cholesterol 112 mg/dL. Creatinine 0.9 mg/dL.")],
 "P002": [("Clinic note", date(2024, 4, 2),
   "ASSESSMENT\nAsthma, well controlled.\nMEDICATIONS\nAlbuterol inhaler as needed.\nPLAN\nFollow up in 12 months.")],
}
rows_docs, rows_chunks = [], []
for pid, docs in demo_docs.items():
    for i, (name, d, text) in enumerate(docs):
        doc_id = f"{pid}-D{i+1}"
        rows_docs.append(Row(doc_id=doc_id, patient_id=pid, doc_name=name, doc_date=d,
                             source_path="synthetic", sha256=hashlib.sha256(text.encode()).hexdigest()))
        for n, c in enumerate(chunk_page(doc_id, 1, text)):
            rows_chunks.append(Row(chunk_id=chunk_id(doc_id, 1, n), doc_id=doc_id, patient_id=pid,
                                   doc_name=name, doc_date=d, page=1, section=c.section, text=c.text))

spark.createDataFrame(rows_docs).createOrReplaceTempView("new_docs")
spark.sql(f"""MERGE INTO {T('documents')} t USING new_docs s ON t.sha256 = s.sha256
              WHEN NOT MATCHED THEN INSERT (doc_id, patient_id, doc_name, doc_date, source_path, sha256)
              VALUES (s.doc_id, s.patient_id, s.doc_name, s.doc_date, s.source_path, s.sha256)""")
spark.createDataFrame(rows_chunks).createOrReplaceTempView("new_chunks")
spark.sql(f"""MERGE INTO {T('chunks')} t USING new_chunks s ON t.chunk_id = s.chunk_id
              WHEN MATCHED THEN UPDATE SET * WHEN NOT MATCHED THEN INSERT *""")
display(spark.table(T("chunks")))

# COMMAND ----------
# MAGIC %md ### Step B - ingest PDFs / text files you upload
# MAGIC Put files at `/Volumes/healthcare_ai/rag/raw/docs/P001/lab_2024.pdf` (folder name = patient_id; patient must exist in `patients`).

# COMMAND ----------
from pypdf import PdfReader

def ingest_volume_docs():
    base = f"{VOL_PATH}/docs"
    known = {r.patient_id for r in spark.table(T("patients")).collect()}
    seen = {r.sha256 for r in spark.table(T("documents")).collect()}
    new_docs, new_chunks = [], []
    for pdir in os.listdir(base):
        if pdir not in known:
            print("skip unknown patient folder:", pdir); continue
        for fn in os.listdir(f"{base}/{pdir}"):
            path = f"{base}/{pdir}/{fn}"
            raw = open(path, "rb").read()
            h = hashlib.sha256(raw).hexdigest()
            if h in seen:
                print("duplicate, skipped:", fn); continue
            if fn.lower().endswith(".pdf"):
                pages = [(p.extract_text() or "") for p in PdfReader(path).pages]
            else:
                pages = [raw.decode("utf-8", "ignore")]
            doc_id = f"{pdir}-{h[:8]}"
            new_docs.append(Row(doc_id=doc_id, patient_id=pdir, doc_name=fn, doc_date=date.today(),
                                source_path=path, sha256=h))
            for pg, txt in enumerate(pages, 1):
                for n, c in enumerate(chunk_page(doc_id, pg, txt)):
                    new_chunks.append(Row(chunk_id=chunk_id(doc_id, pg, n), doc_id=doc_id, patient_id=pdir,
                                          doc_name=fn, doc_date=date.today(), page=pg, section=c.section, text=c.text))
    if new_chunks:
        spark.createDataFrame(new_docs).write.mode("append").saveAsTable(T("documents"))
        spark.createDataFrame(new_chunks).write.mode("append").saveAsTable(T("chunks"))
    print(f"added {len(new_docs)} docs, {len(new_chunks)} chunks")

ingest_volume_docs()

# COMMAND ----------
# MAGIC %md ### Step C (optional) - Synthea FHIR bundles -> text
# MAGIC Upload Synthea JSON bundles to `fhir/`. Each bundle becomes one document for a patient you register first. Minimal conversion shown; extend per resource type as needed.

# COMMAND ----------
import json

def fhir_bundle_to_text(bundle: dict) -> str:
    lines = []
    for e in bundle.get("entry", []):
        r = e["resource"]; t = r.get("resourceType")
        if t == "Condition":
            lines.append(f"CONDITION: {r['code']['text'] if 'text' in r.get('code', {}) else r['code']['coding'][0]['display']} (onset {r.get('onsetDateTime','?')[:10]})")
        elif t == "MedicationRequest":
            m = r.get("medicationCodeableConcept", {})
            lines.append(f"MEDICATION: {m.get('text') or m.get('coding',[{}])[0].get('display','?')} ({r.get('authoredOn','?')[:10]})")
        elif t == "Observation" and "valueQuantity" in r:
            q = r["valueQuantity"]
            lines.append(f"OBSERVATION: {r['code']['coding'][0]['display']} = {q.get('value')} {q.get('unit','')} ({r.get('effectiveDateTime','?')[:10]})")
    return "\n".join(lines)
# usage: text = fhir_bundle_to_text(json.load(open(f"{VOL_PATH}/fhir/<file>.json"))) then chunk + MERGE as in Step B

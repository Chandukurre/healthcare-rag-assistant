# Databricks notebook source
# MAGIC %md
# MAGIC # 01 - Setup: catalog, schema, volume, tables
# MAGIC Run on any cluster (or serverless). Needs permission to create a catalog (or change CATALOG in 00_config to one you own).

# COMMAND ----------
# MAGIC %run ./00_config

# COMMAND ----------
spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")
for sub in ["fhir", "docs", "audio"]:
    dbutils.fs.mkdirs(f"{VOL_PATH}/{sub}")

# COMMAND ----------
# MAGIC %sql
# MAGIC -- the %sql cell cannot use python variables, so catalog/schema are set explicitly
# MAGIC USE CATALOG healthcare_ai;
# MAGIC USE SCHEMA rag;
# MAGIC
# MAGIC CREATE TABLE IF NOT EXISTS patients (
# MAGIC   patient_id STRING NOT NULL, full_name STRING, birth_date DATE, created_at TIMESTAMP DEFAULT current_timestamp()
# MAGIC );
# MAGIC
# MAGIC CREATE TABLE IF NOT EXISTS care_links (   -- which clinician (by login email) may see which patient
# MAGIC   user_email STRING NOT NULL, patient_id STRING NOT NULL
# MAGIC );
# MAGIC
# MAGIC CREATE TABLE IF NOT EXISTS documents (
# MAGIC   doc_id STRING NOT NULL, patient_id STRING NOT NULL, doc_name STRING, doc_date DATE,
# MAGIC   source_path STRING, sha256 STRING, ingested_at TIMESTAMP DEFAULT current_timestamp()
# MAGIC );
# MAGIC
# MAGIC -- chunks is the Vector Search source table: needs a primary key column and Change Data Feed
# MAGIC CREATE TABLE IF NOT EXISTS chunks (
# MAGIC   chunk_id STRING NOT NULL, doc_id STRING, patient_id STRING, doc_name STRING, doc_date DATE,
# MAGIC   page INT, section STRING, text STRING
# MAGIC ) TBLPROPERTIES (delta.enableChangeDataFeed = true);
# MAGIC
# MAGIC CREATE TABLE IF NOT EXISTS transcripts (
# MAGIC   visit_id STRING NOT NULL, patient_id STRING, audio_path STRING, text STRING, consent_recorded BOOLEAN,
# MAGIC   created_at TIMESTAMP DEFAULT current_timestamp()
# MAGIC );
# MAGIC
# MAGIC CREATE TABLE IF NOT EXISTS notes (
# MAGIC   visit_id STRING NOT NULL, patient_id STRING, draft_text STRING, status STRING, signed_by STRING, signed_at TIMESTAMP
# MAGIC );
# MAGIC
# MAGIC CREATE TABLE IF NOT EXISTS audit_log (
# MAGIC   ts TIMESTAMP, user_email STRING, action STRING, patient_id STRING, detail STRING
# MAGIC );

# COMMAND ----------
display(spark.sql(f"SHOW TABLES IN {CATALOG}.{SCHEMA}"))

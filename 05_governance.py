# Databricks notebook source
# MAGIC %md
# MAGIC # 05 - Governance: row-level access, signed-note lock, audit
# MAGIC Run as the table owner. **Not run in the build environment - test with two different users.**
# MAGIC Caveat: a Vector Search source table normally cannot carry a row filter, so `chunks` is protected by (1) endpoint/index permissions
# MAGIC and (2) the mandatory `patient_id` filter in notebook 04. Row filters below protect the tables clinicians query directly.

# COMMAND ----------
# MAGIC %sql
# MAGIC USE CATALOG healthcare_ai;
# MAGIC USE SCHEMA rag;
# MAGIC
# MAGIC -- a clinician sees a patient only if a row exists in care_links for their login
# MAGIC CREATE OR REPLACE FUNCTION care_filter(pid STRING)
# MAGIC RETURN EXISTS (SELECT 1 FROM healthcare_ai.rag.care_links l WHERE l.user_email = current_user() AND l.patient_id = pid)
# MAGIC        OR is_account_group_member('hc_admins');
# MAGIC
# MAGIC ALTER TABLE documents    SET ROW FILTER care_filter ON (patient_id);
# MAGIC ALTER TABLE transcripts  SET ROW FILTER care_filter ON (patient_id);
# MAGIC ALTER TABLE notes        SET ROW FILTER care_filter ON (patient_id);

# COMMAND ----------
# MAGIC %sql
# MAGIC -- column mask: only admins see the full name
# MAGIC CREATE OR REPLACE FUNCTION mask_name(n STRING)
# MAGIC RETURN CASE WHEN is_account_group_member('hc_admins') THEN n ELSE concat(left(n, 1), '*****') END;
# MAGIC ALTER TABLE patients ALTER COLUMN full_name SET MASK mask_name;

# COMMAND ----------
# MAGIC %sql
# MAGIC -- grant a clinician access to a patient (replace the email)
# MAGIC -- INSERT INTO care_links VALUES ('doctor@yourorg.com', 'P001');

# COMMAND ----------
# MAGIC %md ### Signed-note lock
# MAGIC Delta has no triggers, so the lock is enforced in the function every writer must use (and by granting `MODIFY` on `notes` only to a service principal).

# COMMAND ----------
def sign_note(visit_id: str):
    user = spark.sql("SELECT current_user()").first()[0]
    spark.sql(f"UPDATE healthcare_ai.rag.notes SET status='signed', signed_by='{user}', signed_at=current_timestamp() "
              f"WHERE visit_id='{visit_id}' AND status='draft'")

def edit_draft(visit_id: str, new_text: str):
    row = spark.sql(f"SELECT status FROM healthcare_ai.rag.notes WHERE visit_id='{visit_id}'").first()
    if row is None or row.status != "draft":
        raise PermissionError("note is signed or missing; signed notes are locked")
    spark.createDataFrame([(new_text,)], "t string").createOrReplaceTempView("_t")
    spark.sql(f"UPDATE healthcare_ai.rag.notes SET draft_text=(SELECT t FROM _t) WHERE visit_id='{visit_id}'")

# COMMAND ----------
# MAGIC %md ### Audit table is append-only by permission
# MAGIC `REVOKE MODIFY ON TABLE healthcare_ai.rag.audit_log FROM `users`;` then grant only INSERT-capable access via a service principal. Review with:

# COMMAND ----------
# MAGIC %sql
# MAGIC SELECT * FROM healthcare_ai.rag.audit_log ORDER BY ts DESC LIMIT 50;

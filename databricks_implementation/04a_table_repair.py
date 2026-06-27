# Databricks notebook source
SOURCE_TABLE = "cobb_rag.fire_code.databricks_rag_eval_results"
REPAIRED_TABLE = "cobb_rag.fire_code.databricks_rag_eval_results_repaired"

# COMMAND ----------

# MAGIC %md
# MAGIC  create a repaired copy without reading the broken error column:

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE {REPAIRED_TABLE} AS
SELECT
  CAST(run_id AS STRING) AS run_id,
  CAST(row_index AS BIGINT) AS row_index,
  CAST(question AS STRING) AS question,
  CAST(ground_truth AS STRING) AS ground_truth,
  CAST(answer AS STRING) AS answer,
  CAST(contexts AS STRING) AS contexts,
  CAST(sources AS STRING) AS sources,
  CAST(latency_seconds AS DOUBLE) AS latency_seconds,
  CAST(NULL AS STRING) AS error,
  CAST(faithfulness_score AS DOUBLE) AS faithfulness_score,
  CAST(faithfulness_reasoning AS STRING) AS faithfulness_reasoning,
  CAST(answer_relevancy_score AS DOUBLE) AS answer_relevancy_score,
  CAST(answer_relevancy_reasoning AS STRING) AS answer_relevancy_reasoning,
  CAST(context_precision_score AS DOUBLE) AS context_precision_score,
  CAST(context_precision_reasoning AS STRING) AS context_precision_reasoning,
  CAST(context_recall_score AS DOUBLE) AS context_recall_score,
  CAST(context_recall_reasoning AS STRING) AS context_recall_reasoning
FROM {SOURCE_TABLE}
""")

# COMMAND ----------

display(spark.table(REPAIRED_TABLE).limit(20))

# COMMAND ----------

# replace original table
spark.sql(f"DROP TABLE {SOURCE_TABLE}")
spark.sql(f"ALTER TABLE {REPAIRED_TABLE} RENAME TO {SOURCE_TABLE}")
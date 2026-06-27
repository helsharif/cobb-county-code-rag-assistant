# Databricks notebook source
# MAGIC %md
# MAGIC # Test Hybrid Search

# COMMAND ----------

# run this in the ingestion notebook to handle that retention warning. Change from 7 days to 30 days
spark.sql("""
ALTER TABLE cobb_rag.fire_code.document_chunks
SET TBLPROPERTIES (
  'delta.deletedFileRetentionDuration' = 'interval 30 days',
  'delta.enableChangeDataFeed' = true
)
""")

# COMMAND ----------

# MAGIC %md
# MAGIC # Install AI Search SDK For Testing

# COMMAND ----------

# Install the Databricks AI Search SDK for querying the index from a notebook.

%pip install databricks-ai-search
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC # Perform Test
# MAGIC
# MAGIC If the test below returns results, that means Databricks hybrid search backend is alive.

# COMMAND ----------

from databricks.ai_search.client import AISearchClient

INDEX_NAME = "cobb_rag.fire_code.document_chunks_hybrid_index"

client = AISearchClient()
index = client.get_index(index_name=INDEX_NAME)

print(f"Connected to index: {INDEX_NAME}")


# COMMAND ----------

# Run a hybrid search.
# HYBRID combines vector semantic search with keyword search.

results = index.similarity_search(
    query_text="When is a fire inspection required?",
    columns=[
        "id",
        "text",
        "source_file",
        "source_path",
        "source_folder",
        "page_start",
        "page_end",
        "chunk_index",
    ],
    num_results=10,
    query_type="HYBRID",
)

results

# COMMAND ----------

# MAGIC %md
# MAGIC ## Display Results as table

# COMMAND ----------

# Convert the AI Search response into a Spark DataFrame for easier reading.

columns = [column["name"] for column in results["manifest"]["columns"]]
rows = results["result"]["data_array"]

display(spark.createDataFrame(rows, columns))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Print pretty results

# COMMAND ----------

# Print compact search results with citation metadata.

def print_search_results(results, max_chars=900):
    columns = [column["name"] for column in results["manifest"]["columns"]]
    rows = results["result"]["data_array"]

    for rank, row in enumerate(rows, start=1):
        item = dict(zip(columns, row))

        print("=" * 100)
        print(f"Rank: {rank}")
        print(f"Source: {item.get('source_file')}")
        print(f"Folder: {item.get('source_folder')}")
        print(f"Page: {item.get('page_start')}")
        print(f"Chunk: {item.get('chunk_index')}")
        print("-" * 100)
        print((item.get("text") or "")[:max_chars])

print_search_results(results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test retrieval for  Several Real Queries

# COMMAND ----------

# Test whether retrieval works for different code/fire questions.

test_queries = [
    "When is a fire inspection required?",
    "What is required for tenant plan review?",
    "What are the requirements for consumer fireworks retail sales facilities?",
    "What does NFPA 241 require during construction?",
    "When is a certificate of occupancy required?",
    "What are the fire access road requirements?",
]

for query in test_queries:
    print("\n" + "#" * 100)
    print(f"QUERY: {query}")
    print("#" * 100)

    results = index.similarity_search(
        query_text=query,
        columns=[
            "text",
            "source_file",
            "source_folder",
            "page_start",
            "page_end",
            "chunk_index",
        ],
        num_results=5,
        query_type="HYBRID",
    )

    print_search_results(results, max_chars=700)
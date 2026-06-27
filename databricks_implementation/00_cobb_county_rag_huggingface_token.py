# Databricks notebook source
# Creates the value for the secret cobb-county-rag
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

w.secrets.put_secret(
    scope="cobb-county-rag",
    key="HF_TOKEN",
    string_value="place key here, run, then delete"
)
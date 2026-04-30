# Data Folder

Place Cobb County building, fire, ordinance, and code reference PDFs under this folder before building the vector index.

Recommended layout:

```text
data/
└── raw/
    ├── cobb_county_fire/
    ├── cobb_municode/
    ├── applicable_codes/
    └── GA_state_rules_regs/
```

The local PDF corpus and generated Chroma vector database are intentionally excluded from the public repository. This keeps the portfolio repo lightweight and avoids publishing local source documents or generated embeddings.

To rebuild the vector index after adding PDFs:

```bash
python -m src.ingestion --rebuild
```

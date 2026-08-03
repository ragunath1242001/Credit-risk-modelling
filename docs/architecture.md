# Architecture

South German Credit is downloaded with SHA-256 provenance, validated, split reproducibly, and transformed inside scikit-learn pipelines. Logistic regression is the baseline; bounded histogram gradient boosting is the challenger. FastAPI is the serving boundary, Streamlit is the UI, PostgreSQL stores prediction/audit metadata, and MLflow is the optional tracking server.


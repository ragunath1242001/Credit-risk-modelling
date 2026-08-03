import json
import os
import sqlite3
from datetime import datetime, timezone
DB = os.getenv("DATABASE_URL", "sqlite:///artifacts/credit_risk.db").replace("sqlite:///", "")

def connection():
    if DB.startswith("postgres"):
        import psycopg
        conn = psycopg.connect(DB)
        conn.execute("CREATE TABLE IF NOT EXISTS predictions (request_id TEXT PRIMARY KEY, endpoint TEXT, payload JSONB, result JSONB, created_at TEXT)")
        return conn
    os.makedirs(os.path.dirname(DB) or ".", exist_ok=True)
    conn = sqlite3.connect(DB)
    conn.execute("CREATE TABLE IF NOT EXISTS datasets (version TEXT PRIMARY KEY, source TEXT, checksum TEXT, synthetic INTEGER)")
    conn.execute("CREATE TABLE IF NOT EXISTS models (version TEXT PRIMARY KEY, family TEXT, status TEXT, metrics TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS validation_reviews (id INTEGER PRIMARY KEY, decision TEXT, evidence TEXT, created_at TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS monitoring_runs (id INTEGER PRIMARY KEY, reference_version TEXT, current_version TEXT, psi TEXT, status TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS audit_events (id INTEGER PRIMARY KEY, actor TEXT, action TEXT, object TEXT, created_at TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS predictions (request_id TEXT PRIMARY KEY, endpoint TEXT, payload TEXT, result TEXT, created_at TEXT)"); return conn

def save_prediction(request_id: str, endpoint: str, payload: dict, result: dict) -> None:
    conn = connection()
    if DB.startswith("postgres"):
        conn.execute("INSERT INTO predictions VALUES (%s, %s, %s, %s, %s)", (request_id, endpoint, json.dumps(payload), json.dumps(result), datetime.now(timezone.utc).isoformat()))
    else:
        conn.execute("INSERT INTO predictions VALUES (?, ?, ?, ?, ?)", (request_id, endpoint, json.dumps(payload), json.dumps(result), datetime.now(timezone.utc).isoformat()))
    conn.commit(); conn.close()

def save_metadata(dataset_version: str, source: str, checksum: str, model_version: str, family: str, metrics: dict) -> None:
    conn = connection()
    if DB.startswith("postgres"):
        conn.execute("INSERT INTO datasets VALUES (%s,%s,%s,%s) ON CONFLICT DO NOTHING", (dataset_version, source, checksum, False)); conn.execute("INSERT INTO models VALUES (%s,%s,%s,%s) ON CONFLICT DO NOTHING", (model_version, family, "approved-for-demo", json.dumps(metrics)))
    else:
        conn.execute("INSERT OR IGNORE INTO datasets VALUES (?,?,?,?)", (dataset_version, source, checksum, 0)); conn.execute("INSERT OR IGNORE INTO models VALUES (?,?,?,?)", (model_version, family, "approved-for-demo", json.dumps(metrics)))
    conn.commit(); conn.close()

def save_review(decision: str, evidence: dict, actor: str = "demo-reviewer") -> None:
    conn = connection(); now = datetime.now(timezone.utc).isoformat()
    if DB.startswith("postgres"): conn.execute("INSERT INTO validation_reviews(decision,evidence,created_at) VALUES (%s,%s,%s)", (decision, json.dumps(evidence), now)); conn.execute("INSERT INTO audit_events(actor,action,object,created_at) VALUES (%s,%s,%s,%s)", (actor, "validation_review", decision, now))
    else: conn.execute("INSERT INTO validation_reviews(decision,evidence,created_at) VALUES (?,?,?)", (decision, json.dumps(evidence), now)); conn.execute("INSERT INTO audit_events(actor,action,object,created_at) VALUES (?,?,?,?)", (actor, "validation_review", decision, now))
    conn.commit(); conn.close()

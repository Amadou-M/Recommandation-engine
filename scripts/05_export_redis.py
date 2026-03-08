"""
Jour 4 — Export Redis
"""
import os
os.environ["JAVA_HOME"]   = r"C:\Program Files\Eclipse Adoptium\jdk-17.0.18.8-hotspot"
os.environ["HADOOP_HOME"] = r"C:\hadoop"
os.environ["PATH"]        = (
    r"C:\Program Files\Eclipse Adoptium\jdk-17.0.18.8-hotspot\bin"
    + ";" + r"C:\hadoop\bin"
    + ";" + os.environ.get("PATH", "")
)

import json
import redis
from pathlib import Path
from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# ── CONFIG ─────────────────────────────────────────────────────────────
BASE_PATH   = Path("data/delta")
REDIS_HOST  = "localhost"
REDIS_PORT  = 6379
REDIS_DB    = 0
TTL_SECONDS = 86400 * 7

# ── SPARK ──────────────────────────────────────────────────────────────
builder = SparkSession.builder \
    .appName("RecommendationEngine-J4-Redis") \
    .master("local[*]") \
    .config("spark.sql.extensions",
            "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.driver.memory", "4g")

spark = configure_spark_with_delta_pip(builder).getOrCreate()
spark.sparkContext.setLogLevel("WARN")
print(f"Spark version : {spark.version}")

# ── CONNEXION REDIS ────────────────────────────────────────────────────
print("\n[1/4] Connexion Redis...")
try:
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB,
                    decode_responses=True)
    r.ping()
    print(f"  ✓ Redis connecté : {REDIS_HOST}:{REDIS_PORT}")
except redis.ConnectionError:
    print("  ✗ Redis non disponible")
    spark.stop()
    exit(1)

# ── CHARGEMENT RECOMMANDATIONS ALS ────────────────────────────────────
print("\n[2/4] Chargement recommandations ALS...")
als_recs = spark.read.format("delta") \
    .load(str(BASE_PATH / "recommendations/als_recs"))
print(f"  Utilisateurs avec recs ALS : {als_recs.count():,}")

# ── CHARGEMENT MAPPINGS ────────────────────────────────────────────────
print("\n[3/4] Chargement mappings...")
with open(BASE_PATH / "mappings/user_mapping.json") as f:
    user_labels = json.load(f)
with open(BASE_PATH / "mappings/item_mapping.json") as f:
    item_labels = json.load(f)

print(f"  user_mapping : {len(user_labels):,} entrées")
print(f"  item_mapping : {len(item_labels):,} entrées")

# ── EXPORT VERS REDIS ──────────────────────────────────────────────────
print("\n[4/4] Export vers Redis...")
als_pd     = als_recs.toPandas()
pipe       = r.pipeline(transaction=False)
batch      = 0
BATCH_SIZE = 1000

for _, row in als_pd.iterrows():
    user_idx = int(row["user_idx"])
    if user_idx >= len(user_labels):
        continue
    user_id  = user_labels[user_idx]
    recs_list = []
    for rec in row["recommendations"]:
        item_idx = int(rec["item_idx"])
        score    = float(rec["rating"])
        if item_idx < len(item_labels):
            recs_list.append({
                "asin"  : item_labels[item_idx],
                "score" : round(score, 4)
            })

    pipe.setex(f"recs:als:{user_id}", TTL_SECONDS, json.dumps(recs_list))
    batch += 1

    if batch % BATCH_SIZE == 0:
        pipe.execute()
        pipe = r.pipeline(transaction=False)
        print(f"  Exporté {batch:,} / {len(als_pd):,}...")

pipe.execute()
print(f"  ✓ {batch:,} utilisateurs exportés dans Redis")

# ── COLD START ─────────────────────────────────────────────────────────
print("\n  Export cold start...")
popular = spark.read.format("delta") \
    .load(str(BASE_PATH / "recommendations/cold_start_popular")) \
    .orderBy(col("score").desc()) \
    .limit(100)

popular_list = [
    {"asin": row["parent_asin"], "score": round(float(row["score"]), 4)}
    for row in popular.collect()
]
r.setex("recs:cold_start", TTL_SECONDS, json.dumps(popular_list))
print(f"  ✓ Top {len(popular_list)} produits cold start exportés")

print(f"\n  Total clés Redis : {r.dbsize():,}")
print("\n✅ Export Redis terminé")
spark.stop()
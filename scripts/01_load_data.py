"""
Jour 1 — Télécharger et explorer Amazon Reviews All_Beauty
"""
import os
from pathlib import Path
from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, avg, from_unixtime, year
from huggingface_hub import hf_hub_download

# ── CONFIG ─────────────────────────────────────────────────────────────
RAW_PATH   = Path("data/raw")
DELTA_PATH = Path("data/delta")
RAW_PATH.mkdir(parents=True, exist_ok=True)
DELTA_PATH.mkdir(parents=True, exist_ok=True)

# ── SPARK + DELTA ──────────────────────────────────────────────────────
builder = SparkSession.builder \
    .appName("RecommendationEngine-J1") \
    .master("local[*]") \
    .config("spark.sql.extensions",
            "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.driver.memory", "4g")

spark = configure_spark_with_delta_pip(builder).getOrCreate()
spark.sparkContext.setLogLevel("WARN")
print(f"✓ Spark version : {spark.version}")

# ── TÉLÉCHARGEMENT REVIEWS ─────────────────────────────────────────────
print("\n[1/4] Téléchargement Amazon Reviews All_Beauty...")
reviews_path = hf_hub_download(
    repo_id="McAuley-Lab/Amazon-Reviews-2023",
    filename="raw/review_categories/All_Beauty.jsonl",
    repo_type="dataset",
    local_dir=str(RAW_PATH)
)
print(f"  ✓ Reviews : {reviews_path}")

# ── TÉLÉCHARGEMENT METADATA ────────────────────────────────────────────
print("\n[2/4] Téléchargement des métadonnées produits...")
meta_path = hf_hub_download(
    repo_id="McAuley-Lab/Amazon-Reviews-2023",
    filename="raw_meta_All_Beauty/full-00000-of-00001.parquet",
    repo_type="dataset",
    local_dir=str(RAW_PATH)
)
print(f"  ✓ Metadata : {meta_path}")

# ── CHARGEMENT + SAUVEGARDE DELTA ──────────────────────────────────────
print("\n[3/4] Chargement Spark et sauvegarde Delta Lake...")

df_reviews = spark.read.json(reviews_path)
df_reviews.write \
    .format("delta") \
    .mode("overwrite") \
    .save(str(DELTA_PATH / "reviews/beauty"))
print(f"  ✓ Reviews sauvegardées : {df_reviews.count():,} lignes")

df_meta = spark.read.parquet(meta_path)
df_meta.write \
    .format("delta") \
    .mode("overwrite") \
    .save(str(DELTA_PATH / "metadata/beauty"))
print(f"  ✓ Metadata sauvegardée : {df_meta.count():,} produits")

# ── EXPLORATION ────────────────────────────────────────────────────────
print("\n[4/4] Exploration des données...")
ratings = spark.read.format("delta").load(str(DELTA_PATH / "reviews/beauty"))

n_total = ratings.count()
n_users = ratings.select("user_id").distinct().count()
n_items = ratings.select("parent_asin").distinct().count()
density = n_total / (n_users * n_items) * 100

print("=" * 55)
print(" AMAZON ALL_BEAUTY — STATISTIQUES")
print("=" * 55)
print(f" Interactions totales : {n_total:>12,}")
print(f" Utilisateurs uniques : {n_users:>12,}")
print(f" Produits uniques     : {n_items:>12,}")
print(f" Densité matrice      : {density:>11.5f}%")

print("\nDistribution des ratings :")
ratings.groupBy("rating").count().orderBy("rating").show()

user_activity = ratings.groupBy("user_id").agg(count("*").alias("nb_ratings"))
cold_users = user_activity.filter(col("nb_ratings") < 3).count()
print(f"Utilisateurs cold start (< 3 ratings) : {cold_users:,} ({cold_users/n_users*100:.1f}%)")

print("\n✅ Jour 1 terminé — données prêtes dans data/delta/")
spark.stop()
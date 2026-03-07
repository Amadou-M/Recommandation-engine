"""
Jour 2 — Feature Engineering
StringIndexer, profils users/items, split temporel, cold start
"""
import json
from pathlib import Path
from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, avg, stddev, max as spark_max
from pyspark.sql.functions import desc, coalesce, lit
from pyspark.ml.feature import StringIndexer

# ── CONFIG ─────────────────────────────────────────────────────────────
BASE_PATH = Path("data/delta")  # corrigé

# ── SPARK ──────────────────────────────────────────────────────────────
builder = SparkSession.builder \
    .appName("RecommendationEngine-J2") \
    .master("local[*]") \
    .config("spark.sql.extensions",
            "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.driver.memory", "4g")

spark = configure_spark_with_delta_pip(builder).getOrCreate()
spark.sparkContext.setLogLevel("WARN")
print(f"Spark version : {spark.version}")

# ── [1] CHARGEMENT ─────────────────────────────────────────────────────
print("\n[1/6] Chargement des données...")
ratings  = spark.read.format("delta").load(str(BASE_PATH / "reviews/beauty"))
metadata = spark.read.format("delta").load(str(BASE_PATH / "metadata/beauty"))
print(f"  Reviews  : {ratings.count():,}")
print(f"  Metadata : {metadata.count():,}")
ratings.printSchema()

# ── [2] STRINGINDEXER ──────────────────────────────────────────────────
print("\n[2/6] StringIndexer — conversion IDs en entiers...")
user_indexer = StringIndexer(inputCol="user_id",     outputCol="user_idx")
item_indexer = StringIndexer(inputCol="parent_asin", outputCol="item_idx")

user_model = user_indexer.fit(ratings)
item_model = item_indexer.fit(ratings)

ratings_indexed = user_model.transform(ratings)
ratings_indexed = item_model.transform(ratings_indexed)

ratings_indexed = ratings_indexed \
    .withColumn("user_idx", col("user_idx").cast("int")) \
    .withColumn("item_idx", col("item_idx").cast("int"))

print(f"  user_idx max : {ratings_indexed.agg({'user_idx':'max'}).collect()[0][0]:,}")
print(f"  item_idx max : {ratings_indexed.agg({'item_idx':'max'}).collect()[0][0]:,}")

# ── [2b] SAUVEGARDER LES MAPPINGS (CRITIQUE) ───────────────────────────
print("\n  Sauvegarde des mappings...")
mappings_path = BASE_PATH / "mappings"
mappings_path.mkdir(parents=True, exist_ok=True)

with open(mappings_path / "user_mapping.json", "w") as f:
    json.dump(user_model.labels, f)

with open(mappings_path / "item_mapping.json", "w") as f:
    json.dump(item_model.labels, f)

print(f"  user_mapping : {len(user_model.labels):,} entrées")
print(f"  item_mapping : {len(item_model.labels):,} entrées")
print(f"  Exemple : index 0 -> {user_model.labels[0]}")

# ── [3] PROFILS UTILISATEURS ───────────────────────────────────────────
print("\n[3/6] Profils utilisateurs...")
user_profiles = ratings_indexed.groupBy("user_id").agg(
    count("*").alias("total_ratings"),
    avg("rating").alias("avg_rating"),
    stddev("rating").alias("std_rating"),
    spark_max("timestamp").alias("last_activity")
).withColumn(
    "std_rating", coalesce(col("std_rating"), lit(0.0))
)

user_profiles.write \
    .format("delta") \
    .mode("overwrite") \
    .save(str(BASE_PATH / "features/user_profiles"))

print(f"  Profils utilisateurs : {user_profiles.count():,}")

# ── [4] PROFILS PRODUITS ───────────────────────────────────────────────
print("\n[4/6] Profils produits...")
item_profiles = ratings_indexed.groupBy("parent_asin").agg(
    count("*").alias("total_ratings"),
    avg("rating").alias("avg_rating"),
    stddev("rating").alias("std_rating")
).join(
    metadata.select("parent_asin", "title", "price", "main_category"),
    on="parent_asin",
    how="left"
).withColumn(
    "popularity_score", col("total_ratings") * col("avg_rating") / 5.0
).fillna({
    "std_rating"    : 0.0,
    "price"         : 0.0,
    "title"         : "Unknown",
    "main_category" : "Unknown"
})

item_profiles.write \
    .format("delta") \
    .mode("overwrite") \
    .save(str(BASE_PATH / "features/item_profiles"))

print(f"  Profils produits : {item_profiles.count():,}")

# ── [5] SPLIT TEMPOREL 80/20 ───────────────────────────────────────────
print("\n[5/6] Split temporel 80% train / 20% test...")
from datetime import datetime

split_ts = ratings_indexed.approxQuantile("timestamp", [0.8], 0.02)[0]
date_coupure = datetime.fromtimestamp(split_ts / 1000).strftime("%Y-%m-%d")
print(f"  Date de coupure : {date_coupure}")

train = ratings_indexed.filter(col("timestamp") <  split_ts)
test  = ratings_indexed.filter(col("timestamp") >= split_ts)

train.write.format("delta").mode("overwrite") \
    .save(str(BASE_PATH / "splits/train"))
test.write.format("delta").mode("overwrite") \
    .save(str(BASE_PATH / "splits/test"))
ratings_indexed.write.format("delta").mode("overwrite") \
    .save(str(BASE_PATH / "splits/all_indexed"))

n_total = ratings_indexed.count()
print(f"  Train : {train.count():,} ({train.count()/n_total*100:.1f}%)")
print(f"  Test  : {test.count():,}  ({test.count()/n_total*100:.1f}%)")

# ── [6] COLD START POPULAIRE ───────────────────────────────────────────
print("\n[6/6] Top-100 produits populaires (cold start)...")
popular = ratings.groupBy("parent_asin").agg(
    count("*").alias("nb_ratings"),
    avg("rating").alias("avg_rating")
).withColumn(
    "score", col("nb_ratings") * col("avg_rating") / 5.0
).join(
    metadata.select("parent_asin", "title", "main_category"),
    on="parent_asin",
    how="left"
).orderBy(desc("score"))

popular.write \
    .format("delta") \
    .mode("overwrite") \
    .save(str(BASE_PATH / "recommendations/cold_start_popular"))

print(f"  Top produits sauvegardés : {popular.count():,}")
print("\nTop 5 produits les plus populaires :")
popular.select("parent_asin", "title", "nb_ratings", "avg_rating", "score") \
    .show(5, truncate=50)

# ── CHECKPOINT FINAL ───────────────────────────────────────────────────
print("\n" + "="*55)
print(" CHECKPOINT JOUR 2")
print("="*55)
checkpoints = [
    ("data/delta/splits/train",                        "~297K interactions"),
    ("data/delta/splits/test",                         "~74K interactions"),
    ("data/delta/splits/all_indexed",                  "~371K indexées"),
    ("data/delta/mappings/user_mapping.json",          "~289K entrées"),
    ("data/delta/mappings/item_mapping.json",          "~33K entrées"),
    ("data/delta/features/user_profiles",              "profils agrégés"),
    ("data/delta/features/item_profiles",              "profils enrichis"),
    ("data/delta/recommendations/cold_start_popular",  "top produits"),
]
for path, desc_text in checkpoints:
    existe = Path(path).exists()
    status = "✓" if existe else "✗ MANQUANT"
    print(f"  {status}  {path:50s} {desc_text}")

print("\n✅ Jour 2 terminé — features prêtes dans data/delta/")
spark.stop()
"""
Jour 3 — ALS Collaborative Filtering + MLflow
"""
import json
from pathlib import Path
from datetime import datetime
from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, collect_list, size, array_intersect, desc
from pyspark.ml.recommendation import ALS # pyright: ignore[reportMissingImports] # type: ignore
from pyspark.ml.evaluation import RegressionEvaluator # type: ignore
import mlflow
import mlflow.spark

# ── CONFIG ─────────────────────────────────────────────────────────────
BASE_PATH  = Path("data/delta")
MODEL_PATH = str(BASE_PATH / "models/als_best")

# ── SPARK ──────────────────────────────────────────────────────────────
builder = SparkSession.builder \
    .appName("RecommendationEngine-J3-ALS") \
    .master("local[*]") \
    .config("spark.sql.extensions",
            "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.driver.memory", "6g")

spark = configure_spark_with_delta_pip(builder).getOrCreate()
spark.sparkContext.setLogLevel("WARN")
print(f"Spark version : {spark.version}")

# ── MLFLOW ─────────────────────────────────────────────────────────────
mlflow.set_tracking_uri("./mlruns")
mlflow.set_experiment("recommendation_als")

# ── CHARGEMENT ─────────────────────────────────────────────────────────
print("\n[1/4] Chargement train/test...")
train = spark.read.format("delta").load(str(BASE_PATH / "splits/train"))
test  = spark.read.format("delta").load(str(BASE_PATH / "splits/test"))
print(f"  Train : {train.count():,}  |  Test : {test.count():,}")
train.printSchema()

# ── GRID SEARCH ────────────────────────────────────────────────────────
print("\n[2/4] Grid Search ALS + MLflow...")
configs = [
    {"rank": 10,  "regParam": 0.01},
    {"rank": 50,  "regParam": 0.1 },  # baseline
    {"rank": 100, "regParam": 0.1 },
    {"rank": 50,  "regParam": 0.5 },
]

best_rmse, best_model, best_config = float("inf"), None, None

for cfg in configs:
    run_name = f"ALS_r{cfg['rank']}_reg{cfg['regParam']}"
    with mlflow.start_run(run_name=run_name):
        als = ALS(
            maxIter=10,
            rank=cfg["rank"],
            regParam=cfg["regParam"],
            userCol="user_idx",
            itemCol="item_idx",
            ratingCol="rating",
            coldStartStrategy="drop",
            nonnegative=True
        )
        print(f"  Entraînement {run_name}...")
        model = als.fit(train)
        preds = model.transform(test)

        ev_rmse = RegressionEvaluator(
            metricName="rmse", labelCol="rating", predictionCol="prediction"
        )
        ev_mae = RegressionEvaluator(
            metricName="mae", labelCol="rating", predictionCol="prediction"
        )
        rmse = ev_rmse.evaluate(preds)
        mae  = ev_mae.evaluate(preds)

        mlflow.log_params(cfg)
        mlflow.log_metrics({"rmse": rmse, "mae": mae})
        mlflow.spark.log_model(model, "als_model")

        print(f"    rank={cfg['rank']:3d}  reg={cfg['regParam']}  "
              f"RMSE={rmse:.4f}  MAE={mae:.4f}")

        if rmse < best_rmse:
            best_rmse   = rmse
            best_model  = model
            best_config = cfg

print(f"\n  Meilleur : {best_config}  RMSE={best_rmse:.4f}")

# ── MÉTRIQUES RANKING ──────────────────────────────────────────────────
print("\n[3/4] Métriques ranking Precision@K / Recall@K...")
K = 10

user_recs_k = best_model.recommendForAllUsers(K)
relevant = test.filter(col("rating") >= 4) \
    .groupBy("user_idx") \
    .agg(collect_list("item_idx").alias("relevant_items"))

eval_df = user_recs_k \
    .join(relevant, on="user_idx", how="inner") \
    .withColumn("rec_ids", col("recommendations.item_idx")) \
    .withColumn("hits", size(array_intersect("rec_ids", "relevant_items"))) \
    .withColumn("precision_at_k", col("hits") / K) \
    .withColumn("recall_at_k",    col("hits") / size("relevant_items"))

precision_k = eval_df.agg(avg("precision_at_k")).collect()[0][0]
recall_k    = eval_df.agg(avg("recall_at_k")).collect()[0][0]

print("=" * 50)
print(" MÉTRIQUES FINALES ALS")
print("=" * 50)
print(f" RMSE          = {best_rmse:.4f}")
print(f" Precision@{K}  = {precision_k:.4f}  ({precision_k*100:.2f}%)")
print(f" Recall@{K}     = {recall_k:.4f}  ({recall_k*100:.2f}%)")
print("=" * 50)

# ── SAUVEGARDE ─────────────────────────────────────────────────────────
print("\n[4/4] Sauvegarde modèle + recommandations batch...")

best_model.save(MODEL_PATH)
print(f"  Modèle sauvegardé : {MODEL_PATH}")

all_recs = best_model.recommendForAllUsers(100)
all_recs.write \
    .format("delta") \
    .mode("overwrite") \
    .save(str(BASE_PATH / "recommendations/als_recs"))

print(f"  Recommandations batch : {all_recs.count():,} utilisateurs")
print("\n✅ Jour 3 terminé — modèle ALS prêt dans data/delta/")
spark.stop()
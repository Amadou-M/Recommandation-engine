from pyspark.sql import SparkSession
from huggingface_hub import hf_hub_download

# Spark en mode local — équivalent Community Edition
spark = SparkSession.builder \
    .appName("RecommendationEngine-J1") \
    .master("local[*]") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.1.0") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# Télécharger depuis Hugging Face
reviews_path = hf_hub_download(
    repo_id="McAuley-Lab/Amazon-Reviews-2023",
    filename="raw/review_categories/All_Beauty.jsonl",
    repo_type="dataset",
    local_dir="data/raw"
)

# Charger directement avec Spark (pas de Pandas)
df_reviews = spark.read.json(reviews_path)
df_reviews.write.format("delta").mode("overwrite") \
    .save("data/delta/reviews/beauty")

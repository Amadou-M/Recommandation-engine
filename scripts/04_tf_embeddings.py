"""
Jour 3 — Deep Learning Embeddings avec PyTorch
Remplace TensorFlow (non compatible Python 3.14)
"""
import json
import numpy as np
from pathlib import Path
from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession
import mlflow
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ── CONFIG ─────────────────────────────────────────────────────────────
BASE_PATH  = Path("data/delta")
MODEL_PATH = str(BASE_PATH / "models/tf_embeddings")

# ── SPARK ──────────────────────────────────────────────────────────────
builder = SparkSession.builder \
    .appName("RecommendationEngine-J3-TF") \
    .master("local[*]") \
    .config("spark.sql.extensions",
            "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.driver.memory", "6g")

spark = configure_spark_with_delta_pip(builder).getOrCreate()
spark.sparkContext.setLogLevel("WARN")
print(f"Spark version : {spark.version}")
print(f"PyTorch version : {torch.__version__}")

# ── MLFLOW ─────────────────────────────────────────────────────────────
mlflow.set_tracking_uri("./mlruns")
mlflow.set_experiment("recommendation_tf_embeddings")

# ── CHARGEMENT ─────────────────────────────────────────────────────────
print("\n[1/5] Chargement train/test...")
train = spark.read.format("delta").load(str(BASE_PATH / "splits/train"))
test  = spark.read.format("delta").load(str(BASE_PATH / "splits/test"))

with open(BASE_PATH / "mappings/user_mapping.json") as f:
    user_mapping = json.load(f)
with open(BASE_PATH / "mappings/item_mapping.json") as f:
    item_mapping = json.load(f)

N_USERS = len(user_mapping)
N_ITEMS = len(item_mapping)
print(f"  Utilisateurs : {N_USERS:,}")
print(f"  Produits     : {N_ITEMS:,}")

# ── CONVERSION NUMPY ───────────────────────────────────────────────────
print("\n[2/5] Conversion en NumPy...")
train_pd = train.select("user_idx", "item_idx", "rating").toPandas()
test_pd  = test.select("user_idx", "item_idx", "rating").toPandas()
spark.stop()

X_train_user = train_pd["user_idx"].values.astype(np.int64)
X_train_item = train_pd["item_idx"].values.astype(np.int64)
y_train      = train_pd["rating"].values.astype(np.float32)

X_test_user  = test_pd["user_idx"].values.astype(np.int64)
X_test_item  = test_pd["item_idx"].values.astype(np.int64)
y_test       = test_pd["rating"].values.astype(np.float32)

print(f"  Train : {len(y_train):,}  |  Test : {len(y_test):,}")

# ── DATASET ────────────────────────────────────────────────────────────
class RatingDataset(Dataset):
    def __init__(self, users, items, ratings):
        self.users   = torch.tensor(users,   dtype=torch.long)
        self.items   = torch.tensor(items,   dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]

train_dataset = RatingDataset(X_train_user, X_train_item, y_train)
test_dataset  = RatingDataset(X_test_user,  X_test_item,  y_test)

train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=2048, shuffle=False)

# ── MODÈLE ─────────────────────────────────────────────────────────────
print("\n[3/5] Construction du modèle PyTorch...")

class EmbeddingModel(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=64):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 1, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, user, item):
        u = self.user_embedding(user)
        i = self.item_embedding(item)
        dot = (u * i).sum(dim=1, keepdim=True)
        x = torch.cat([u, i, dot], dim=1)
        return self.fc(x).squeeze()

EMBEDDING_DIM = 64
EPOCHS        = 10
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device : {DEVICE}")

model_pt  = EmbeddingModel(N_USERS, N_ITEMS, EMBEDDING_DIM).to(DEVICE)
optimizer = torch.optim.Adam(model_pt.parameters(), lr=0.001)
criterion = nn.MSELoss()

# ── ENTRAÎNEMENT ───────────────────────────────────────────────────────
print("\n[4/5] Entraînement PyTorch...")

with mlflow.start_run(run_name="PyTorch_Embeddings_dim64"):
    mlflow.log_params({
        "embedding_dim" : EMBEDDING_DIM,
        "epochs"        : EPOCHS,
        "batch_size"    : 2048,
        "n_users"       : N_USERS,
        "n_items"       : N_ITEMS,
        "framework"     : "PyTorch"
    })

    for epoch in range(1, EPOCHS + 1):
        # Train
        model_pt.train()
        train_loss = 0.0
        for users, items, ratings in train_loader:
            users, items, ratings = (
                users.to(DEVICE), items.to(DEVICE), ratings.to(DEVICE)
            )
            optimizer.zero_grad()
            preds = model_pt(users, items)
            loss  = criterion(preds, ratings)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(ratings)

        train_rmse = np.sqrt(train_loss / len(train_dataset))

        # Validation
        model_pt.eval()
        val_loss = 0.0
        val_mae  = 0.0
        with torch.no_grad():
            for users, items, ratings in test_loader:
                users, items, ratings = (
                    users.to(DEVICE), items.to(DEVICE), ratings.to(DEVICE)
                )
                preds     = model_pt(users, items)
                val_loss += criterion(preds, ratings).item() * len(ratings)
                val_mae  += (preds - ratings).abs().sum().item()

        val_rmse = np.sqrt(val_loss / len(test_dataset))
        val_mae  = val_mae / len(test_dataset)

        print(f"  Epoch {epoch:2d}/{EPOCHS}  "
              f"Train RMSE={train_rmse:.4f}  "
              f"Val RMSE={val_rmse:.4f}  "
              f"Val MAE={val_mae:.4f}")

        mlflow.log_metrics({
            "train_rmse" : train_rmse,
            "val_rmse"   : val_rmse,
            "val_mae"    : val_mae
        }, step=epoch)

    print("\n" + "=" * 50)
    print(" MÉTRIQUES FINALES PyTorch Embeddings")
    print("=" * 50)
    print(f" RMSE = {val_rmse:.4f}")
    print(f" MAE  = {val_mae:.4f}")
    print("=" * 50)

    # Sauvegarder
    Path(MODEL_PATH).mkdir(parents=True, exist_ok=True)
    torch.save(model_pt.state_dict(), f"{MODEL_PATH}/model.pt")
    print(f"\n  Modèle sauvegardé : {MODEL_PATH}/model.pt")

# ── COMPARAISON ALS vs PyTorch ─────────────────────────────────────────
print("\n[5/5] Comparaison ALS vs PyTorch Embeddings...")
client  = mlflow.tracking.MlflowClient()
als_exp = client.get_experiment_by_name("recommendation_als")

if als_exp:
    als_runs = client.search_runs(
        experiment_ids=[als_exp.experiment_id],
        order_by=["metrics.rmse ASC"],
        max_results=1
    )
    if als_runs:
        als_rmse = als_runs[0].data.metrics.get("rmse", None)
        if als_rmse:
            print(f"  ALS meilleur RMSE       : {als_rmse:.4f}")
            print(f"  PyTorch Embeddings RMSE : {val_rmse:.4f}")
            winner = "PyTorch Embeddings" if val_rmse < als_rmse else "ALS"
            print(f"  Meilleur modèle         : {winner}")

print("\n✅ Jour 3 terminé — ALS + PyTorch Embeddings prêts")
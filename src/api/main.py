"""
Jour 4 — API FastAPI
Endpoints de recommandation avec Redis + fallback cold start
"""
import json
import redis
import torch
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# ── CONFIG ─────────────────────────────────────────────────────────────
BASE_PATH  = Path("data/delta")
REDIS_HOST = "localhost"
REDIS_PORT = 6379

app = FastAPI(
    title       = "Recommendation Engine API",
    description = "API de recommandation e-commerce — Amazon Beauty",
    version     = "1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── MODÈLES PYDANTIC ───────────────────────────────────────────────────
class RecommendationItem(BaseModel):
    asin  : str
    score : float

class RecommendationResponse(BaseModel):
    user_id : str
    source  : str   # "als", "pytorch", "cold_start"
    items   : List[RecommendationItem]
    count   : int

class HealthResponse(BaseModel):
    status : str
    redis  : str
    models : dict

# ── ÉTAT GLOBAL ────────────────────────────────────────────────────────
state = {
    "redis"        : None,
    "user_mapping" : {},   # user_id -> index
    "item_mapping" : {},   # index   -> asin
    "pt_model"     : None,
    "n_users"      : 0,
    "n_items"      : 0,
}

# ── STARTUP ────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    print("Chargement des ressources...")

    # Redis
    try:
        state["redis"] = redis.Redis(
            host=REDIS_HOST, port=REDIS_PORT,
            db=0, decode_responses=True
        )
        state["redis"].ping()
        print("  ✓ Redis connecté")
    except Exception as e:
        print(f"  ✗ Redis indisponible : {e}")

    # Mappings
    try:
        with open(BASE_PATH / "mappings/user_mapping.json") as f:
            labels = json.load(f)
            state["user_mapping"] = {uid: i for i, uid in enumerate(labels)}
            state["n_users"]      = len(labels)

        with open(BASE_PATH / "mappings/item_mapping.json") as f:
            labels = json.load(f)
            state["item_mapping"] = {i: asin for i, asin in enumerate(labels)}
            state["n_items"]      = len(labels)

        print(f"  ✓ Mappings chargés : {state['n_users']:,} users, {state['n_items']:,} items")
    except Exception as e:
        print(f"  ✗ Mappings : {e}")

    # Modèle PyTorch
    try:
        from src.ml.model import EmbeddingModel
        model = EmbeddingModel(state["n_users"], state["n_items"], 64)
        model.load_state_dict(
            torch.load(
                str(BASE_PATH / "models/tf_embeddings/model.pt"),
                map_location="cpu",
                weights_only=True
            )
        )
        model.eval()
        state["pt_model"] = model
        print("  ✓ Modèle PyTorch chargé")
    except Exception as e:
        print(f"  ✗ Modèle PyTorch : {e}")

    print("API prête ✓")

# ── ENDPOINTS ──────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
def health():
    redis_status = "ok"
    try:
        state["redis"].ping()
    except Exception:
        redis_status = "unavailable"

    return {
        "status" : "ok",
        "redis"  : redis_status,
        "models" : {
            "pytorch" : "loaded" if state["pt_model"] else "unavailable",
            "n_users" : state["n_users"],
            "n_items" : state["n_items"],
        }
    }

@app.get("/recommendations/{user_id}", response_model=RecommendationResponse)
def get_recommendations(user_id: str, top_k: int = 10):
    """
    Recommandations pour un utilisateur.
    Stratégie : Redis ALS → PyTorch temps réel → Cold start
    """
    # 1. Essayer Redis (ALS pré-calculé)
    if state["redis"]:
        cached = state["redis"].get(f"recs:als:{user_id}")
        if cached:
            items = json.loads(cached)[:top_k]
            return {
                "user_id" : user_id,
                "source"  : "als",
                "items"   : items,
                "count"   : len(items)
            }

    # 2. Essayer PyTorch temps réel
    if state["pt_model"] and user_id in state["user_mapping"]:
        user_idx = state["user_mapping"][user_id]
        user_t   = torch.tensor([user_idx] * state["n_items"], dtype=torch.long)
        item_t   = torch.tensor(list(range(state["n_items"])),  dtype=torch.long)

        with torch.no_grad():
            scores = state["pt_model"](user_t, item_t).numpy()

        top_idx   = np.argsort(scores)[::-1][:top_k]
        items     = [
            {"asin": state["item_mapping"][int(i)], "score": round(float(scores[i]), 4)}
            for i in top_idx
        ]
        return {
            "user_id" : user_id,
            "source"  : "pytorch",
            "items"   : items,
            "count"   : len(items)
        }

    # 3. Cold start — top produits populaires
    source = "cold_start"
    if state["redis"]:
        cached = state["redis"].get("recs:cold_start")
        if cached:
            items = json.loads(cached)[:top_k]
            return {
                "user_id" : user_id,
                "source"  : source,
                "items"   : items,
                "count"   : len(items)
            }

    raise HTTPException(status_code=503, detail="Aucune source de recommandation disponible")

@app.get("/popular", response_model=RecommendationResponse)
def get_popular(top_k: int = 10):
    """Top produits populaires (cold start)"""
    if state["redis"]:
        cached = state["redis"].get("recs:cold_start")
        if cached:
            items = json.loads(cached)[:top_k]
            return {
                "user_id" : "anonymous",
                "source"  : "cold_start",
                "items"   : items,
                "count"   : len(items)
            }
    raise HTTPException(status_code=503, detail="Cold start non disponible")

@app.get("/stats")
def get_stats():
    """Statistiques du système"""
    redis_keys = 0
    if state["redis"]:
        try:
            redis_keys = state["redis"].dbsize()
        except Exception:
            pass

    return {
        "n_users"    : state["n_users"],
        "n_items"    : state["n_items"],
        "redis_keys" : redis_keys,
        "models"     : {
            "als"     : "pré-calculé dans Redis",
            "pytorch" : "loaded" if state["pt_model"] else "unavailable"
        }
    }
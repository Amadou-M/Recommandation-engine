"""
Jour 5 — Tests unitaires et d'intégration
"""
import json
import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from src.api.main import app, state
from src.ml.model import EmbeddingModel

client = TestClient(app)

# ── FIXTURES ───────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def setup_state():
    state["n_users"]      = 1000
    state["n_items"]      = 500
    state["user_mapping"] = {f"user_{i}": i for i in range(1000)}
    state["item_mapping"] = {i: f"ASIN_{i:06d}" for i in range(500)}

    model = EmbeddingModel(1000, 500, 64)
    model.eval()
    state["pt_model"] = model

    mock_redis = MagicMock()
    mock_redis.ping.return_value   = True
    mock_redis.get.return_value    = None
    mock_redis.dbsize.return_value = 42
    state["redis"] = mock_redis

    yield

    state["redis"] = None

# ── TESTS MODÈLE ───────────────────────────────────────────────────────
class TestEmbeddingModel:

    def test_model_creation(self):
        model = EmbeddingModel(100, 50, 32)
        assert model is not None

    def test_forward_pass(self):
        model = EmbeddingModel(100, 50, 32)
        model.eval()
        users = torch.tensor([0, 1, 2], dtype=torch.long)
        items = torch.tensor([0, 1, 2], dtype=torch.long)
        with torch.no_grad():
            output = model(users, items)
        assert output.shape == (3,)

    def test_output_range(self):
        model = EmbeddingModel(100, 50, 32)
        model.eval()
        users = torch.tensor([0] * 50, dtype=torch.long)
        items = torch.tensor(list(range(50)), dtype=torch.long)
        with torch.no_grad():
            scores = model(users, items)
        assert scores.shape == (50,)
        assert not torch.isnan(scores).any()

    def test_embedding_dimensions(self):
        model = EmbeddingModel(100, 50, 64)
        assert model.user_embedding.embedding_dim == 64
        assert model.item_embedding.embedding_dim == 64
        assert model.user_embedding.num_embeddings == 100
        assert model.item_embedding.num_embeddings == 50

    def test_model_save_load(self, tmp_path):
        model = EmbeddingModel(100, 50, 32)
        model.eval()  # ← eval() AVANT save
        path  = str(tmp_path / "model.pt")
        torch.save(model.state_dict(), path)
        model2 = EmbeddingModel(100, 50, 32)
        model2.load_state_dict(
            torch.load(path, map_location="cpu", weights_only=True)
        )
        model2.eval()
        users = torch.tensor([0, 1], dtype=torch.long)
        items = torch.tensor([0, 1], dtype=torch.long)
        with torch.no_grad():
            o1 = model(users, items)
            o2 = model2(users, items)
        assert torch.allclose(o1, o2)

# ── TESTS API ──────────────────────────────────────────────────────────
class TestHealthEndpoint:

    def test_health_ok(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "redis"  in data
        assert "models" in data

    def test_health_structure(self):
        response = client.get("/health")
        data = response.json()
        assert "pytorch" in data["models"]
        assert "n_users" in data["models"]
        assert "n_items" in data["models"]

class TestStatsEndpoint:

    def test_stats_ok(self):
        response = client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["n_users"] == 1000
        assert data["n_items"] == 500
        assert "redis_keys" in data
        assert "models"     in data

class TestRecommendationsEndpoint:

    def test_known_user_pytorch(self):
        response = client.get("/recommendations/user_0?top_k=5")
        assert response.status_code == 200
        data = response.json()
        assert data["source"]     == "pytorch"
        assert data["count"]      == 5
        assert len(data["items"]) == 5

    def test_unknown_user_cold_start(self):
        cold = [{"asin": f"ASIN_{i}", "score": float(10 - i)}
                for i in range(10)]
        state["redis"].get.side_effect = lambda k: (
            json.dumps(cold) if k == "recs:cold_start" else None
        )
        response = client.get("/recommendations/unknown_user_xyz?top_k=5")
        assert response.status_code == 200
        data = response.json()
        assert data["source"]     == "cold_start"
        assert len(data["items"]) == 5

    def test_user_from_redis_als(self):
        cached = [{"asin": f"ASIN_{i}", "score": float(5 - i * 0.1)}
                  for i in range(10)]
        state["redis"].get.side_effect = lambda k: (
            json.dumps(cached) if k == "recs:als:user_0" else None
        )
        response = client.get("/recommendations/user_0?top_k=5")
        assert response.status_code == 200
        data = response.json()
        assert data["source"]     == "als"
        assert len(data["items"]) == 5

    def test_top_k_respected(self):
        for k in [5, 10, 20]:
            response = client.get(f"/recommendations/user_1?top_k={k}")
            assert response.status_code == 200
            assert response.json()["count"] == k

    def test_recommendation_structure(self):
        response = client.get("/recommendations/user_2?top_k=3")
        data = response.json()
        assert "user_id" in data
        assert "source"  in data
        assert "items"   in data
        assert "count"   in data
        for item in data["items"]:
            assert "asin"  in item
            assert "score" in item

class TestPopularEndpoint:

    def test_popular_from_redis(self):
        popular = [{"asin": f"ASIN_{i}", "score": float(100 - i)}
                   for i in range(20)]
        state["redis"].get.return_value = json.dumps(popular)
        response = client.get("/popular?top_k=10")
        assert response.status_code == 200
        data = response.json()
        assert data["source"]     == "cold_start"
        assert len(data["items"]) == 10

    def test_popular_no_redis(self):
        state["redis"].get.return_value = None
        response = client.get("/popular?top_k=5")
        assert response.status_code == 503

# ── TESTS DONNÉES ──────────────────────────────────────────────────────
class TestDataFiles:

    def test_mappings_exist(self):
        assert Path("data/delta/mappings/user_mapping.json").exists()
        assert Path("data/delta/mappings/item_mapping.json").exists()

    def test_user_mapping_valid(self):
        with open("data/delta/mappings/user_mapping.json") as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) > 0
        assert isinstance(data[0], str)

    def test_item_mapping_valid(self):
        with open("data/delta/mappings/item_mapping.json") as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) > 0

    def test_model_file_exists(self):
        assert Path("data/delta/models/tf_embeddings/model.pt").exists()

    def test_delta_directories_exist(self):
        for d in ["reviews/beauty", "metadata/beauty", "splits/train",
                  "splits/test", "features/user_profiles",
                  "features/item_profiles"]:
            assert Path(f"data/delta/{d}").exists(), f"Manquant : {d}"
# 🛍️ Moteur de Recommandation E-commerce

> **Projet Big Data & Machine Learning** — Formation Data Engineer  
> **Auteur :** Amadou MAIGA | **Date :** Mars 2026

Système de recommandation de produits Amazon basé sur **ALS (filtrage collaboratif)** et **PyTorch Embeddings (Deep Learning)**, avec pipeline complet de la donnée brute jusqu'au dashboard interactif.

---

## 📊 Résultats obtenus

| Modèle | RMSE | MAE | Type |
|--------|------|-----|------|
| ALS (meilleur) | **2.5457** | — | Filtrage collaboratif |
| **PyTorch Embeddings** | **1.7241** | 1.4485 | Deep Learning ✅ |

> PyTorch est **32% meilleur** qu'ALS sur ce dataset.

**Dataset :** Amazon All_Beauty Reviews 2023
- 701 528 avis · 112 565 produits · 631 986 utilisateurs
- Densité matrice : 0.001% (matrice très creuse)
- Split temporel : 78.8% train / 21.2% test (coupure : 2021-05-02)

---

## 🏗️ Architecture

```
Données Amazon (HuggingFace)
        ↓ 01_load_data.py
Delta Lake (reviews + metadata)
        ↓ 02_build_features.py
Features (mappings, profils, splits 80/20, cold start)
        ↓ 03_als_model.py          ↓ 04_tf_embeddings.py
ALS → recs batch               PyTorch → model.pt
        ↓ 05_export_redis.py
Redis Cache (< 1ms par requête)
        ↓ src/api/main.py
API FastAPI (4 endpoints)
        ↓ src/dashboard/app.py
Dashboard Streamlit (4 pages)
```

---

## 🛠️ Stack technique

| Composant | Technologie | Version |
|-----------|-------------|---------|
| Calcul distribué | Apache Spark | 3.5.1 |
| Stockage données | Delta Lake | 3.1.0 |
| Modèle classique | ALS (PySpark MLlib) | — |
| Modèle Deep Learning | PyTorch | 2.10.0 |
| Suivi expériences | MLflow | ≥ 2.10 |
| Cache | Redis | 7 |
| API | FastAPI + Uvicorn | ≥ 0.110 |
| Dashboard | Streamlit + Plotly | ≥ 1.32 |
| Orchestration | Docker Compose | — |
| Versioning | Git / GitHub | — |

---

## 📁 Structure du projet

```
recommendation-engine/
├── data/
│   ├── raw/                          # Fichiers bruts téléchargés (HuggingFace)
│   └── delta/                        # Données Delta Lake
│       ├── reviews/beauty/           # 701 528 avis
│       ├── metadata/beauty/          # 112 565 produits
│       ├── mappings/
│       │   ├── user_mapping.json     # 631 986 user_id → index
│       │   └── item_mapping.json     # 112 565 asin → index
│       ├── features/
│       │   ├── user_profiles/        # Stats par utilisateur
│       │   └── item_profiles/        # Stats + métadonnées produits
│       ├── splits/
│       │   ├── train/                # 552 944 interactions (78.8%)
│       │   ├── test/                 # 148 584 interactions (21.2%)
│       │   └── all_indexed/          # Dataset complet indexé
│       ├── recommendations/
│       │   ├── als_recs/             # Recs batch ALS (tous les users)
│       │   └── cold_start_popular/   # Top-100 produits populaires
│       └── models/
│           ├── als_best/             # Modèle ALS Spark sauvegardé
│           └── tf_embeddings/
│               └── model.pt          # Modèle PyTorch sauvegardé
│
├── scripts/                          # Pipeline de traitement (à lancer en ordre)
│   ├── 01_load_data.py               # Téléchargement + sauvegarde Delta Lake
│   ├── 02_build_features.py          # Feature engineering + split temporel
│   ├── 03_als_model.py               # Modèle ALS + Grid Search + MLflow
│   ├── 04_tf_embeddings.py           # Modèle PyTorch Embeddings
│   └── 05_export_redis.py            # Export recommandations vers Redis
│
├── src/
│   ├── api/
│   │   └── main.py                   # API FastAPI (4 endpoints)
│   ├── ml/
│   │   └── model.py                  # Classe EmbeddingModel (partagée)
│   └── dashboard/
│       └── app.py                    # Dashboard Streamlit (4 pages)
│
├── tests/
│   └── test_api.py                   # 20 tests pytest (coverage 69%)
│
├── docker-compose.yml                # Redis + API + Dashboard + MLflow
├── Dockerfile.api                    # Image Docker pour l'API
├── Dockerfile.dashboard              # Image Docker pour le dashboard
├── requirements.txt                  # Dépendances Python
└── README.md                         # Ce fichier
```

---

## ⚙️ Prérequis (Windows)

### Java 17 (obligatoire pour Spark)
Télécharger **Eclipse Adoptium JDK 17** : https://adoptium.net

```powershell
# Vérifier l'installation
java -version
# Doit afficher : openjdk version "17.x.x"
```

### Hadoop winutils (obligatoire sur Windows)
Télécharger winutils pour Hadoop 3.x et placer dans `C:\hadoop\bin\`

### Docker Desktop
Télécharger : https://www.docker.com/products/docker-desktop

### Variables d'environnement PowerShell
```powershell
$env:JAVA_HOME   = "C:\Program Files\Eclipse Adoptium\jdk-17.0.18.8-hotspot"
$env:HADOOP_HOME = "C:\hadoop"
$env:PATH        = "$env:JAVA_HOME\bin;$env:HADOOP_HOME\bin;" + $env:PATH
```

---

## 🚀 Installation

```powershell
# 1. Cloner le repository
git clone https://github.com/ton-username/recommendation-engine.git
cd recommendation-engine

# 2. Créer l'environnement virtuel
python -m venv venv
.\venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt
```

---

## ▶️ Lancement du pipeline (ordre obligatoire)

### Étape 1 — Télécharger et stocker les données
```powershell
python scripts/01_load_data.py
# Durée : ~10-15 min (téléchargement HuggingFace)
# Résultat : data/delta/reviews/ et data/delta/metadata/
```

### Étape 2 — Feature engineering
```powershell
python scripts/02_build_features.py
# Durée : ~5-10 min
# Résultat : mappings, profils, splits train/test, cold start
```

### Étape 3A — Modèle ALS
```powershell
python scripts/03_als_model.py
# Durée : ~20-30 min (grid search 4 configs)
# Résultat : modèle ALS + recommandations batch + métriques MLflow
```

### Étape 3B — Modèle PyTorch
```powershell
python scripts/04_tf_embeddings.py
# Durée : ~15-20 min (10 epochs)
# Résultat : data/delta/models/tf_embeddings/model.pt
```

### Étape 4 — Démarrer Redis et exporter le cache
```powershell
# Terminal 1 — Démarrer Redis
docker-compose up -d redis

# Terminal 2 — Exporter les recommandations
python scripts/05_export_redis.py
# Résultat : 631 986 utilisateurs dans Redis (TTL 7 jours)
```

### Étape 5 — Lancer l'API
```powershell
# Dans le venv
python -m uvicorn src.api.main:app --port 8080
# API disponible sur http://localhost:8080
# Documentation Swagger : http://localhost:8080/docs
```

### Étape 6 — Lancer le Dashboard
```powershell
# IMPORTANT : lancer depuis Python GLOBAL (pas le venv)
deactivate
python -m streamlit run src/dashboard/app.py --server.port 8501
# Dashboard disponible sur http://localhost:8501
```

### Tout lancer avec Docker (optionnel)
```powershell
docker-compose up -d
# Lance Redis + API + Dashboard + MLflow en une commande
```

---

## 🌐 URLs

| Service | URL |
|---------|-----|
| API FastAPI | http://localhost:8080 |
| Swagger UI (docs) | http://localhost:8080/docs |
| Dashboard Streamlit | http://localhost:8501 |
| MLflow UI | http://localhost:5000 |

---

## 📡 API Endpoints

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/health` | GET | État du système (Redis, PyTorch, données) |
| `/stats` | GET | Métriques globales (nb users, items, clés Redis) |
| `/recommendations/{user_id}` | GET | Top-N recommandations (cascade ALS→PyTorch→cold start) |
| `/popular` | GET | Top-N produits les plus populaires |

### Exemple de requête
```bash
curl http://localhost:8080/recommendations/AGKHLEW2SOWHNMFQIJGBECAF7INQ?top_k=10
```

### Exemple de réponse
```json
{
  "user_id": "AGKHLEW2SOWHNMFQIJGBECAF7INQ",
  "source": "als",
  "count": 10,
  "items": [
    {"asin": "B001TH2JOW", "score": 4.82},
    {"asin": "B07BHHJ3D3", "score": 4.71}
  ]
}
```

### Stratégie de recommandation en cascade
```
1. Redis (ALS batch)  →  réponse < 1ms    ✅ si user connu et cache chaud
2. PyTorch temps réel →  réponse ~50ms    ✅ si user connu, cache froid
3. Cold Start         →  réponse < 5ms    ✅ si user inconnu (top populaires)
```

---

## 🧪 Tests

```powershell
# Activer le venv et lancer les tests
.\venv\Scripts\activate
python -m pytest tests/test_api.py -v --cov=src --cov-report=term-missing
```

**Résultats :**
```
tests/test_api.py::TestEmbeddingModel::test_model_creation     PASSED
tests/test_api.py::TestEmbeddingModel::test_forward_pass       PASSED
tests/test_api.py::TestEmbeddingModel::test_output_range       PASSED
tests/test_api.py::TestEmbeddingModel::test_embedding_dimensions PASSED
tests/test_api.py::TestEmbeddingModel::test_model_save_load    PASSED
tests/test_api.py::TestHealthEndpoint::test_health_ok          PASSED
tests/test_api.py::TestHealthEndpoint::test_health_structure   PASSED
tests/test_api.py::TestStatsEndpoint::test_stats_ok            PASSED
tests/test_api.py::TestRecommendationsEndpoint::...            PASSED (5)
tests/test_api.py::TestPopularEndpoint::...                    PASSED (2)
tests/test_api.py::TestDataFiles::...                          PASSED (5)

====== 20 passed in Xs ====== Coverage: 69%
```

---

## 🐛 Problèmes connus (Windows)

| Problème | Cause | Solution |
|----------|-------|----------|
| `ChecksumException` sur Delta Lake | Conflit checksums Hadoop Windows | Ajouter `.config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.RawLocalFileSystem")` |
| `checkHadoopHome` au démarrage Spark | HADOOP_HOME non défini | Ajouter `os.environ["HADOOP_HOME"]` avant les imports PySpark |
| Streamlit crash Python 3.14 | Incompatibilité typing | Lancer depuis Python global (`deactivate` d'abord) |
| Port 8080/8501 déjà utilisé | Instance précédente active | Utiliser `--port 8090` et `--server.port 8502` |
| TensorFlow incompatible | Python 3.14 non supporté | Remplacé par PyTorch 2.10.0 (résultat meilleur) |

---

## 📈 Performances

| Métrique | Valeur |
|----------|--------|
| RMSE ALS | 2.5457 |
| RMSE PyTorch | **1.7241** (-32%) |
| MAE PyTorch | 1.4485 |
| Latence Redis (cache chaud) | < 1ms |
| Latence PyTorch (temps réel) | ~50ms |
| Couverture tests | 69% |
| Tests passés | 20/20 ✅ |

---

## 🔀 Branches Git

```
main          ← Production stable (tag v2.0.0)
  └── develop ← Intégration
        └── feature/tests-docker ← Développement actif
```

---

## 📄 Licence

Projet académique — Formation Big Data & Machine Learning — Mars 2026

# Cornac Recommendation System Project

This project implements a recommendation system using the Cornac library for the course project.

## Project Structure
```
.
├── data/
│   ├── raw/              # Raw datasets
│   └── processed/        # Preprocessed data
├── src/
│   ├── models/           # Model implementations
│   ├── utils/            # Utility functions
│   └── config.py         # Configuration file
├── notebooks/            # Jupyter notebooks for exploration
├── experiments/          # Experiment logs
├── results/
│   ├── figures/          # Plots and visualizations
│   └── metrics/          # Evaluation metrics (JSON)
├── tests/                # Unit tests
├── train_baseline.py     # Train baseline models
├── train_model.py        # Train main model
├── evaluate.py           # Evaluate and compare models
└── requirements.txt      # Python dependencies
```

## Installation

### Prerequisites
- Python 3.9 or 3.10
- macOS, Linux, or Windows

### Setup

1. Clone the repository:
```bash
git clone https://github.com/bilalfatian/cornac-recommendation-project.git
cd cornac-recommendation-project
```

2. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate   # On Windows
```

3. Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

### 1. Train Baseline Models
```bash
python train_baseline.py
```

This trains MostPop, UserKNN, and ItemKNN models on MovieLens 100K.

### 2. Train Your Main Model
```bash
python train_model.py
```

(TODO: Update with your specific model details)

### 3. Evaluate and Compare
```bash
python evaluate.py
```

This loads all results and displays a comparison table.

## Dataset

We use the **MovieLens 100K** dataset, which contains:
- 100,000 ratings
- 943 users
- 1,682 movies
- Ratings from 1 to 5

The data is automatically downloaded by Cornac.
## Evaluation Metrics

The models are evaluated using standard top-K ranking metrics:

- **Recall@K**: Proportion of relevant items retrieved among the top-K recommendations.
- **NDCG@K**: Normalized Discounted Cumulative Gain, which accounts for the ranking position of relevant items.
- **Precision@K**: Fraction of relevant items in the top-K recommendations.

Unless stated otherwise, results are reported for **K = 5, 10, and 20**.  
In accordance with the project guidelines, **Recall@10** and **NDCG@10** are used as the primary comparison metrics.

---

## Model Details

### Baseline Models

The following baseline models are implemented for comparison:

1. **MostPop**  
   Recommends items based on global interaction frequency. This non-personalized baseline provides a strong reference point for popularity-driven recommendations.

2. **UserKNN**  
   A user-based collaborative filtering model that recommends items preferred by similar users.

3. **ItemKNN**  
   An item-based collaborative filtering model that recommends items similar to those previously interacted with by the user.

4. **MF**  
   A standard matrix factorization model trained on implicit feedback.

5. **BPR**  
   Bayesian Personalized Ranking, a pairwise ranking-based matrix factorization approach optimized for top-K recommendation.

---

## Proposed Model

### DualHybridVAE

**DualHybridVAE** is a hybrid recommendation model that combines collaborative filtering with semantic item representations derived from a pretrained language model.

The model is based on a variational autoencoder (VAE) operating on user interaction vectors. The decoder is composed of two components:

- A **collaborative decoder**, which learns item representations directly from interaction data.
- A **semantic decoder**, which scores items using fixed text embeddings extracted from movie titles and genres via a pretrained language model.

The final recommendation score is computed as a weighted combination of both decoders, allowing the model to leverage semantic similarity while preserving strong collaborative signals.

---

## Artifacts

Training and evaluation generate several artifacts that are saved to disk for reproducibility and further analysis:

- `artifacts/item_embeddings.npy`  
  Precomputed language model embeddings for each item, extracted from movie titles and genre information.

- `artifacts/item_id_map.json`  
  Mapping between raw item identifiers and internal item indices used by the model.

- `artifacts/DualHybridVAE.pt`  
  Model checkpoint containing the trained parameters of the DualHybridVAE.

- `results/metric



## Authors

Bilal Fatian, Mehdi Ben Barka, Vanessa Meyanga , Adam Chekhab 

## License

This project is for educational purposes as part of a course assignment.

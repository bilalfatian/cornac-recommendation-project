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

- **Recall@K**: Proportion of relevant items retrieved in top-K
- **NDCG@K**: Normalized Discounted Cumulative Gain at K
- **Precision@K**: Precision of top-K recommendations

Default K values: 5, 10, 20

## Model Details

### Baseline Models
1. **MostPop**: Recommends most popular items
2. **UserKNN**: User-based collaborative filtering
3. **ItemKNN**: Item-based collaborative filtering

### Main Model
(TODO: Describe your chosen model - GCN, LightGCN, VAECF+Social, BERT, etc.)

## Results

Results are saved in `results/metrics/` as JSON files.

| Model | Recall@10 | NDCG@10 | Training Time |
|-------|-----------|---------|---------------|
| MostPop | TBD | TBD | TBD |
| UserKNN | TBD | TBD | TBD |
| ItemKNN | TBD | TBD | TBD |
| MainModel | TBD | TBD | TBD |

## Author

Bilal Fatian (the GOAT)

## License

This project is for educational purposes as part of a course assignment.

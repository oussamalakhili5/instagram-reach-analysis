# Instagram Reach Analysis

Portfolio project analyzing the factors associated with Instagram post reach and
presenting the results in an interactive Streamlit dashboard.

## Project overview

The project uses a small Instagram Insights export to:

- clean and profile post-level engagement data;
- explore distributions, correlations, relationships, and text fields;
- segment posts with K-Means clustering;
- predict impressions with a Random Forest regression model;
- expose the analysis through a reusable Streamlit application.

The dataset contains 119 raw posts and 13 columns. After removing 17 duplicate
rows, the analysis uses 102 unique posts. The target is `Impressions`.

## Dashboard

Run the application with:

```bash
python -m streamlit run app/streamlit_app.py
```

The dashboard includes:

| Page | Contents |
| --- | --- |
| Overview | Dynamic KPIs, data audit, and pipeline summary |
| Descriptive Analysis | Data preview, statistics, distributions, correlations, and relationships |
| Post Segmentation | K-Means cluster profiles and interactive visualizations |
| Predictive Analysis | Actual versus predicted impressions |
| Model Performance | R², MAE, RMSE, and model-derived feature importance |
| Insights & Recommendations | Evidence-based interpretation and next steps |
| About | Project and data provenance |

## Methodology

```text
Raw Instagram export
        ↓
Remove missing and duplicate rows
        ↓
Exploratory data analysis
        ↓
Standardize engagement features
        ↓
K-Means clustering (k=3)
        ↓
Random Forest regression
        ↓
Streamlit dashboard
```

### Clustering

K-Means uses standardized `Likes`, `Saves`, `Comments`, `Shares`, `Profile
Visits`, and `Follows`. The three clusters found in the notebook are reproduced
by the application:

| Cluster | Posts | Average impressions |
| ---: | ---: | ---: |
| 0 | 82 | 4,336 |
| 1 | 17 | 9,491 |
| 2 | 3 | 29,003 |

The silhouette score for this fixed clustering is **0.454**. Cluster IDs are
algorithmic labels; the dashboard describes them by relative reach rather than
claiming that a cluster is inherently better.

### Prediction

The Random Forest Regressor uses the same six engagement features to predict
`Impressions`. It uses a fixed 80/20 train/test split (`random_state=42`) and
100 trees.

| Metric | Test-set result |
| --- | ---: |
| R² | 0.847 |
| MAE | 1,448 impressions |
| RMSE | 2,748 impressions |

Feature importance is computed from the fitted model. In the current run, the
ranking is `Follows`, `Likes`, `Profile Visits`, `Saves`, `Comments`, `Shares`.
These are predictive associations, not causal effects.

## Descriptive analysis

The notebook and dashboard cover numeric summaries, distributions, a
correlation heatmap, pairwise relationships with impressions, and WordCloud
visualizations for captions and hashtags. Existing static charts are retained
in [`images/`](images/), while the dashboard provides interactive equivalents
for the numeric analysis.

## Dataset

The source file is [`data/Instagram data.csv`](data/Instagram%20data.csv).
It contains reach sources (`From Home`, `From Hashtags`, `From Explore`, and
`From Other`), engagement (`Likes`, `Saves`, `Comments`, `Shares`), conversion
signals (`Profile Visits`, `Follows`), and text fields (`Caption`, `Hashtags`).

## Installation

```bash
git clone https://github.com/oussamalakhili5/instagram-reach-analysis.git
cd instagram-reach-analysis

python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
python -m streamlit run app/streamlit_app.py
```

## Project structure

```text
instagram-reach-analysis/
├── app/
│   ├── __init__.py
│   └── streamlit_app.py
├── data/
│   └── Instagram data.csv
├── images/
│   └── generated analysis charts
├── notebooks/
│   └── instagram-reach-analysis.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

## Future improvements

- collect a larger and time-aware dataset;
- validate performance with temporal splits;
- tune hyperparameters and compare alternative models;
- add experiment tracking and model monitoring;
- deploy the dashboard with a reproducible CI/CD workflow.

## Author

**Oussama Lakhili**

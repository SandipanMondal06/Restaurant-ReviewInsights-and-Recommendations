
# 🍴 Zomato Review Insights & Recommendations

## 📌 Project Overview
This project analyzes **10,000+ Zomato restaurant reviews** to extract customer sentiment, segment restaurants into business-meaningful clusters, and power a simple **personalized dining recommender**. It helps stakeholders (ops, product, marketing) identify cuisine trends, pricing bands, and service gaps.

---

## 🎯 Objectives
1. **Sentiment Analysis**: Quantify customer satisfaction from review text (VADER).  
2. **Clustering**: Segment 105 restaurants using cuisine mix, cost, and ratings.  
3. **Recommendations**: Simple **rule-based** suggestions that respect user budget and preference for experience.

---

## 🛠️ Tech Stack
- **Language**: Python 3.10+  
- **Core Libraries**: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `nltk`, `textblob`, `spacy`  
- **NLP**: VADER (NLTK), spaCy `en_core_web_sm`  
- **Environment**: Jupyter / Google Colab

---

## 📂 Data
- **Files**  
  - `Restaurant names and Metadata.csv` — Name, Cost, Cuisines, etc.  
  - `Restaurant reviews.csv` — Review text, Rating, Time, Metadata (reviewer stats).  
- **Key engineered fields**  
  - `Cuisines_list`: tokenized cuisines per restaurant  
  - `prev_reviews_count`, `followers_count`: parsed from “Metadata”  
  - `Sentiment`: TextBlob/VADER polarity in [-1, 1]  
  - `Sentiment_Type`: {negative, neutral, positive}  
  - `Cluster`: Agglomerative Clustering/KMeans label

> Note: Replace dataset paths with your actual paths in the notebook.

---

## 📁 Repository Structure
```
.
├── notebooks/
│   └── zomato_analysis.ipynb       # Full EDA, NLP, clustering, recommender
├── data/
│   ├── Restaurant names and Metadata.csv
│   └── Restaurant reviews.csv
├── README.md
└── requirements.txt
```

---

## ⚙️ Setup & Installation

```bash
# 1) Create environment (optional)
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Download spaCy model
python -m spacy download en_core_web_sm

# 4) (If using NLTK VADER)
python -c "import nltk; nltk.download('vader_lexicon')"
```
**requirements.txt**
```
pandas
numpy
scikit-learn
matplotlib
nltk
textblob
spacy
missingno
```

---

## 🔑 End-to-End Pipeline

### 1) Data Cleaning
- Drop near-empty columns (`Collections`) and duplicates.  
- Fix types: `Cost` → int, `Rating` → numeric (coerce non-numbers).  
- Parse reviewer metadata → `prev_reviews_count`, `followers_count`.  
- Normalize cuisines → `Cuisines_list` (lowercased, stripped).

### 2) Sentiment Analysis
- Clean text: remove URLs, digits, and special chars.  
- Compute polarity:
  - **VADER** (primary) for short review sentiments.
  - **TextBlob** (optional) for cross-checking polarity.  
- Map to `Sentiment_Type` (pos/neu/neg).

### 3) Feature Engineering
- **Cuisine one-hot** via `MultiLabelBinarizer`.  
- Merge with **Cost** and **Avg_Ratings** (mean rating per restaurant).  
- Standardize selected features; filter rare cuisines.  
- Optional: cost binning, rating-cost ratio.

### 4) Clustering
- **Modeling**: Try **KMeans** (elbow + silhouette) and **Agglomerative** (Ward).  
- **Selection**: Use silhouette to pick `k`; dendrogram for structure.  
- **Output**: `Cluster` label per restaurant; summarize cluster-wise cuisine mix, median cost, and average ratings.

### 5) Evaluation & Business Insights
- Heatmap: cuisine distribution per cluster.  
- Bar: average **Cost** and **Avg_Ratings** by cluster.  
- Scatter: label top cuisines per cluster.  
- KPI: Ratings/Cost ratio to locate value clusters.

### 6) Simple Rule-Based Recommender
- **Input**: user query e.g., _“best biryani under 700”_.  
- **Logic**:
  1. Parse **budget** (`under 700` → `<= 700`) and **cuisine keywords** (`biryani`).  
  2. **Filter** by cost and cuisine.  
  3. If **choice = "experience"** → sort by `Rating` (and positive `Sentiment`).  
     If **choice = "budget"** → sort by lower `Cost` (tie-break: rating).  
  4. Optional: prioritize helpful clusters (good rating-cost profile).

---

## 🧪 How to Run
1. Open `notebooks/zomato_analysis.ipynb`.  
2. Update dataset paths in the first cell.  
3. Run all cells (EDA → NLP → Clustering → Recommender).  
4. Test queries:
   - `"best biryani under 700", choice="experience"`  
   - `"good chinese under 500", choice="budget"`

---

## 📊 Sample Results (Illustrative)
- **Cluster 2**: Highest **cost & ratings** → likely upscale (Italian/Asian).  
- **Cluster 5**: High ratings, **moderate cost** → strong value (Asian + Continental).  
- **Cluster 1**: Balanced cost, mixed cuisines (desserts/continental/biryani/chinese).  
- **Cluster 4/6**: Lower ratings → improvement areas (service/hygiene/consistency).

> Use cluster summaries to drive **pricing, menu mix, and quality** interventions.

---

## 🧠 Design Choices
- **VADER** for short, informal review text (emojis, intensity).  
- **Agglomerative** to respect hierarchical cuisine similarity, validated with **KMeans**.  
- **Rule-based recommender** for transparency and quick iteration under time constraints.

---

## 📈 Extensions (Future Work)
- **Hybrid Ranking**: Weighted score of sentiment, rating, cluster quality, and cost.  
- **Collaborative Filtering**: If user-item ratings available.  
- **Topic Modeling**: LDA to surface service/food/ambience themes.  
- **A/B Evaluation**: CTR or dwell-time on recommended lists.

---

## ✅ Deliverables
- Notebook with: EDA, Sentiment, Clustering, Recommender.  
- Visuals: missingness matrix, histograms, heatmaps, scatter plots.  
- Utility functions: query parser, budget/cuisine filter, rankers.

---

## 🚀 Quick Snippets

**Cuisine Binarization**
```python
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
X_cuisine = mlb.fit_transform(df['Cuisines_list'])
cuisine_df = pd.DataFrame(X_cuisine, columns=mlb.classes_, index=df.index)
```

**VADER Sentiment**
```python
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
df['sentiment'] = df['Review'].fillna('').apply(lambda t: sia.polarity_scores(t)['compound'])
df['sentiment_type'] = pd.cut(df['sentiment'], bins=[-1, -0.05, 0.05, 1], labels=['negative','neutral','positive'])
```

**Agglomerative Clustering**
```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

X = StandardScaler().fit_transform(feat_df[selected_features])
hc = AgglomerativeClustering(n_clusters=7, linkage='ward')
df['Cluster'] = hc.fit_predict(X)
```

**Rule-Based Recommender**
```python
def recommend(query, choice='experience', top_n=10):
    # parse budget + keywords (simple regex)
    # filter by cost & cuisines
    # rank by rating/sentiment (experience) or cost (budget)
    return ranked_df.head(top_n)[['Name','Cuisines_list','Cost','Rating','Cluster']]
```

---

## 📜 License
This project is released under the **MIT License**. See `LICENSE` for details.

---

## 🙌 Acknowledgements
- Zomato dataset providers and maintainers.  
- Open-source libraries: NLTK, spaCy, scikit-learn, pandas, matplotlib.

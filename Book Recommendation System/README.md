# Folio — Book Recommendation System

A clean, deployable book recommendation web app powered by **Collaborative Filtering** (cosine similarity) and a **Popularity-based** model — trained on the [BookCrossing dataset from Kaggle](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset).

---

## Features

| Feature | Details |
|---|---|
| **Popularity Model** | Top 50 books by weighted average rating (≥300 ratings) |
| **Collaborative Filtering** | Item-item cosine similarity via user-rating pivot table |
| **Autocomplete Search** | Type any book title to get 5 similar recommendations |
| **Polished UI** | Dark editorial design — responsive, animated |
| **REST API** | `/api/recommend` endpoint (JSON) |

---

## Project Structure

```
book-recommender/
├── app.py              # Flask app (routes + API)
├── train_model.py      # One-time ML training script
├── requirements.txt
├── Procfile            # For Render / Railway / Heroku
├── data/               # ← place your CSVs here (not committed)
│   ├── Books.csv
│   ├── Users.csv
│   └── Ratings.csv
├── model/              # ← generated pickle files (commit these)
│   ├── popular_df.pkl
│   ├── pt.pkl
│   ├── books.pkl
│   └── similarity_scores.pkl
└── templates/
    ├── index.html      # Homepage (trending books)
    └── recommend.html  # Search & recommendations page
```

---

## Quick Start (Local)

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/book-recommender.git
cd book-recommender
pip install -r requirements.txt
```

### 2. Get the dataset

Download from Kaggle: https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset

Place the three CSVs into the `data/` folder:
```
data/Books.csv
data/Users.csv
data/Ratings.csv
```

### 3. Train the model

```bash
python train_model.py
```

This creates four `.pkl` files inside `model/`. Training takes ~1–2 minutes.

### 4. Run the app

```bash
python app.py
```

Visit: http://localhost:5000

---

## Deploy to Render (Free)

1. **Push to GitHub** (commit the `model/*.pkl` files — they're small)

   ```bash
   # Update .gitignore to allow pkl files:
   # comment out the "*.pkl" line in .gitignore, then:
   git add .
   git commit -m "Initial commit with trained model"
   git push
   ```

2. **Create a Render Web Service**
   - Go to https://render.com → New → Web Service
   - Connect your GitHub repo
   - Set **Build Command**: `pip install -r requirements.txt`
   - Set **Start Command**: `gunicorn app:app`
   - Choose the free plan → Deploy

3. Done — your app will be live at `https://your-app.onrender.com` 🎉

---

## Deploy to Railway

```bash
npm install -g @railway/cli
railway login
railway init
railway up
```

Railway auto-detects the `Procfile`.

---

## ML Model Details

### Popularity Model
- Merges ratings + books data
- Filters books with **≥ 300 ratings**
- Ranks by average rating → top 50

### Collaborative Filtering
- Keeps users with **≥ 200 book ratings** (active readers)
- Keeps books rated by **≥ 50 users** (well-known books)
- Builds a **User × Book pivot table**
- Computes **cosine similarity** between book vectors
- On query: returns top-5 most similar books

### Why cosine similarity?
It's fast, interpretable, and works excellently on sparse user-rating matrices without needing to train a neural network — making it deployable as a pickle file with no GPU required.

---

## API

### `POST /api/recommend`

**Request:**
```json
{ "book_name": "The Da Vinci Code" }
```

**Response:**
```json
{
  "recommendations": [
    {
      "title": "Angels & Demons",
      "author": "Dan Brown",
      "image": "https://...",
      "score": 0.912
    }
  ]
}
```

---

## App Demo Video

https://github.com/user-attachments/assets/36df07c3-f850-4eac-98e0-790c227bc9c4





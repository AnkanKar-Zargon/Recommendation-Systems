"""
Produces 4 pickle files in ./model/ used by app.py
"""

import os, pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR  = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
os.makedirs(MODEL_DIR, exist_ok=True)


#  1. LOAD

print("Loading CSVs …")

books = pd.read_csv(
    os.path.join(DATA_DIR, "Books.csv"),
    low_memory=False, on_bad_lines="skip",
)
ratings = pd.read_csv(
    os.path.join(DATA_DIR, "Ratings.csv"),
    low_memory=False, on_bad_lines="skip",
)
users = pd.read_csv(
    os.path.join(DATA_DIR, "Users.csv"),
    low_memory=False, on_bad_lines="skip",
)

# Force canonical column names by position (handles any separator/header variant)
ratings.columns = ["User-ID", "ISBN", "Book-Rating"]
books.columns   = [
    "ISBN", "Book-Title", "Book-Author", "Year-Of-Publication",
    "Publisher", "Image-URL-S", "Image-URL-M", "Image-URL-L",
]

print(f"  Books: {len(books):,}  Ratings: {len(ratings):,}  Users: {len(users):,}")

# ── Basic cleaning──
books.dropna(subset=["Book-Title", "Book-Author"], inplace=True)
books.drop_duplicates("ISBN", inplace=True)

ratings["Book-Rating"] = pd.to_numeric(ratings["Book-Rating"], errors="coerce")
ratings.dropna(subset=["Book-Rating"], inplace=True)
ratings["Book-Rating"] = ratings["Book-Rating"].astype(int)


#  2. MERGE ALL THREE TABLES FIRST  (key insight — filter AFTER merging)

print("\nMerging tables …")

# Merge ratings → books → users in one shot so every row has full context
df = (
    ratings
    .merge(books,  on="ISBN",    how="inner")
    .merge(users,  on="User-ID", how="inner")
)
print(f"  Merged dataset: {len(df):,} rows  |  "
      f"{df['ISBN'].nunique():,} unique books  |  "
      f"{df['User-ID'].nunique():,} unique users")


#  3. POPULARITY MODEL  (uses all ratings including implicit 0s for counts)

print("\nBuilding popularity model …")

num_ratings_df = (
    df.groupby("Book-Title")["Book-Rating"]
    .count().reset_index().rename(columns={"Book-Rating": "num_ratings"})
)
avg_ratings_df = (
    df.groupby("Book-Title")["Book-Rating"]
    .mean().reset_index().rename(columns={"Book-Rating": "avg_rating"})
)

# Adaptive threshold: step down until we have ≥ 30 books
for thresh in [500, 250, 100, 50, 20, 10, 5, 1]:
    popular_df = num_ratings_df.merge(avg_ratings_df, on="Book-Title")
    popular_df = popular_df[popular_df["num_ratings"] >= thresh]
    if len(popular_df) >= 30:
        break

popular_df = (
    popular_df
    .sort_values("avg_rating", ascending=False)
    .head(50)
    .merge(books.drop_duplicates("Book-Title"), on="Book-Title")
    [["Book-Title", "Book-Author", "Image-URL-M", "num_ratings", "avg_rating"]]
    .rename(columns={"num_ratings": "Num-Ratings", "avg_rating": "Avg-Rating"})
)
popular_df["Avg-Rating"] = popular_df["Avg-Rating"].round(2)
print(f"  Popular books: {len(popular_df)}  (threshold ≥ {thresh} ratings)")


#  4. COLLABORATIVE FILTERING
#     Filter order matters:
#       a) users with many ratings  (find voracious readers)
#       b) from THAT subset, books rated by many of those users
#     Both filters work on the FULL merged df (not just explicit ratings)

print("\nBuilding collaborative filtering model …")

# ── Step A: keep users who have rated ≥ threshold books 
user_rating_count = df.groupby("User-ID")["Book-Rating"].count()

for u_thresh in [200, 100, 50, 20, 10, 5, 1]:
    active_users = user_rating_count[user_rating_count >= u_thresh].index
    if len(active_users) >= 50:
        break

print(f"  Active users  (≥{u_thresh:>3} ratings): {len(active_users):,}")
df_filtered = df[df["User-ID"].isin(active_users)]

# ── Step B: from those users, keep books with ≥ threshold ratings
book_rating_count = df_filtered.groupby("Book-Title")["Book-Rating"].count()

for b_thresh in [50, 25, 10, 5, 3, 2, 1]:
    popular_books = book_rating_count[book_rating_count >= b_thresh].index
    if len(popular_books) >= 20:
        break

print(f"  Popular books (≥{b_thresh:>2} ratings from active users): {len(popular_books):,}")
df_final = df_filtered[df_filtered["Book-Title"].isin(popular_books)]

print(f"  Final CF dataset: {len(df_final):,} rows  |  "
      f"{df_final['Book-Title'].nunique()} books  |  "
      f"{df_final['User-ID'].nunique()} users")

if df_final.empty:
    raise RuntimeError("CF dataset is empty — check your CSV files.")

# ── Pivot table: books × users
pt = df_final.pivot_table(
    index="Book-Title", columns="User-ID", values="Book-Rating"
)
pt.fillna(0, inplace=True)
print(f"  Pivot table shape: {pt.shape}")

# ── Cosine similarity
similarity_scores = cosine_similarity(pt)


#  5. SAVE

print("\nSaving model artifacts …")
pickle.dump(popular_df,        open(os.path.join(MODEL_DIR, "popular_df.pkl"),        "wb"))
pickle.dump(pt,                open(os.path.join(MODEL_DIR, "pt.pkl"),                "wb"))
pickle.dump(books,             open(os.path.join(MODEL_DIR, "books.pkl"),             "wb"))
pickle.dump(similarity_scores, open(os.path.join(MODEL_DIR, "similarity_scores.pkl"), "wb"))

print(f"""
    Done!
    popular_df       → {len(popular_df)} books
    pt (pivot)       → {pt.shape[0]} books × {pt.shape[1]} users
    similarity_scores → {similarity_scores.shape}
    books            → {len(books):,} entries
""")
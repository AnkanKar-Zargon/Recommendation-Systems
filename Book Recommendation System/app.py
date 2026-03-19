from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

BASE      = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE, "model")

popular_df = pickle.load(open(os.path.join(MODEL_DIR, "popular_df.pkl"), "rb"))
pt         = pickle.load(open(os.path.join(MODEL_DIR, "pt.pkl"),         "rb"))
books      = pickle.load(open(os.path.join(MODEL_DIR, "books.pkl"),      "rb"))
similarity = pickle.load(open(os.path.join(MODEL_DIR, "similarity_scores.pkl"), "rb"))


@app.route("/")
def index():
    return render_template("index.html",
        popular=popular_df.to_dict(orient="records")
    )


@app.route("/recommend")
def recommend_ui():
    return render_template("recommend.html", book_list=list(pt.index))


@app.route("/api/recommend", methods=["POST"])
def recommend_api():
    data      = request.get_json()
    book_name = data.get("book_name", "").strip()

    if book_name not in pt.index:
        return jsonify({"error": "Book not found. Try another title."}), 404

    idx          = np.where(pt.index == book_name)[0][0]
    similar_items = sorted(
        list(enumerate(similarity[idx])), key=lambda x: x[1], reverse=True
    )[1:6]

    results = []
    for i, score in similar_items:
        title = pt.index[i]
        match = books[books["Book-Title"] == title].drop_duplicates("Book-Title")
        if match.empty:
            continue
        row = match.iloc[0]
        results.append({
            "title":  row["Book-Title"],
            "author": row["Book-Author"],
            "year":   str(row.get("Year-Of-Publication", "")),
            "publisher": str(row.get("Publisher", "")),
            "score":  round(float(score), 3),
        })

    return jsonify({"recommendations": results})


@app.route("/api/popular")
def popular_api():
    return jsonify(popular_df.to_dict(orient="records"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
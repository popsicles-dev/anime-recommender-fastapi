from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import sklearn
  # Should match 1.7.0

# Load model artifacts
with open("model/anime_data.pkl", "rb") as f:
    anime_data = pickle.load(f)

with open("model/cosine_sim_matrix.pkl", "rb") as f:
    cosine_sim = pickle.load(f)

with open("model/tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

# Setup FastAPI
app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

# Input model
class AnimeRequest(BaseModel):
    title: str
    top_n: int = 5

@app.post("/predict")
def predict_recommendation(req: AnimeRequest):
    title = req.title.lower()

    if title not in anime_data['name'].str.lower().values:
        raise HTTPException(status_code=404, detail="Anime not found")

    # Find index of the anime
    idx = anime_data[anime_data['name'].str.lower() == title].index[0]
    
    # Get pairwise similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get top N excluding itself
    sim_scores = sim_scores[1:req.top_n+1]
    anime_indices = [i[0] for i in sim_scores]
    
    # Return top similar animes
    results = anime_data.iloc[anime_indices][["name", "genre", "rating"]].to_dict(orient="records")
    return {"recommendations": results}


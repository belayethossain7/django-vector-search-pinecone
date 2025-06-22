from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
from sentence_transformers import SentenceTransformer
import pinecone

pinecone.init(api_key="YOUR_PINECONE_API_KEY", environment="gcp-starter")
index = pinecone.Index("product-index")
model = SentenceTransformer("all-MiniLM-L6-v2")

def search_view(request):
    query = request.GET.get("q", "")
    results = []
    if query:
        vector = model.encode(query).tolist()
        response = index.query(vector=vector, top_k=5, include_metadata=True)
        results = [match['metadata'] for match in response['matches']]
    return render(request, "search.html", {"results": results, "query": query})

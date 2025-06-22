import pinecone
from django.core.management.base import BaseCommand
from datasets import load_dataset
from search_app.models import Product
from sentence_transformers import SentenceTransformer

class Command(BaseCommand):
    help = "Load sample product data from Hugging Face and upload embeddings to Pinecone"

    def handle(self, *args, **kwargs):
        # Load dataset (replace with another if needed)
        dataset = load_dataset("mteb/amazon-en", split="test[:50]")

  # small sample
        model = SentenceTransformer("all-MiniLM-L6-v2")

        pinecone.init(api_key="YOUR_PINECONE_API_KEY", environment="gcp-starter")
        index_name = "product-index"
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(index_name, dimension=384)  # Model has 384 dims

        index = pinecone.Index(index_name)

        for item in dataset:
            name = item['title']
            desc = item['description']

            product = Product.objects.create(name=name, description=desc)
            vector = model.encode(desc).tolist()

            index.upsert([(str(product.id), vector, {
                "name": name,
                "description": desc
            })])

        self.stdout.write(self.style.SUCCESS("Products loaded and embedded successfully."))

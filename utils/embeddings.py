from sentence_transformers import SentenceTransformer
import numpy as np

# Load embedding model only once
model = SentenceTransformer("all-MiniLM-L6-v2")


def create_embeddings(text_chunks):
    """
    Convert list of text chunks into embeddings.
    """

    embeddings = model.encode(
        text_chunks,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return np.array(
        embeddings,
        dtype="float32"
    )


def create_query_embedding(query):
    """
    Convert user query into embedding.
    """

    embedding = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return np.array(
        embedding,
        dtype="float32"
    )
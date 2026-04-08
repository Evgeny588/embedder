from sentence_transformers import SentenceTransformer
from chonkie import SentenceChunker
from dotenv import load_dotenv
import torch
import os

load_dotenv()


model = SentenceTransformer(
    model_name_or_path = os.getenv('CHECKPOINT'),
    cache_folder = 'model_cache/'
)

chunker = SentenceChunker(
    tokenizer = model.tokenizer,
    chunk_size = 450,
    chunk_overlap = 50
)


def embedder(text, model, chunker):
    chunks = chunker.chunk(text)
    chunk_texts = ['passage: ' + c.text for c in chunks]

    with torch.inference_mode():
      embeddings = model.encode(chunk_texts, convert_to_tensor=True)

    mean_emb = torch.mean(embeddings, dim=0)

    max_emb, _ = torch.max(embeddings, dim=0)

    final_vector = (mean_emb + max_emb) / 2
    return final_vector.cpu().tolist()
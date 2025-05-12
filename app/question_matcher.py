from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()

def find_similar(query, corpus):
    query_vec = get_embedding(query)
    scores = [cosine_similarity(query_vec, get_embedding(c))[0][0] for c in corpus]
    return corpus[scores.index(max(scores))]

# Sample use
if __name__ == "__main__":
    corpus = ["What is completing the square?", "How to factor equations?", "Why can't I divide by zero?"]
    print(find_similar("When do I use square completion?", corpus))

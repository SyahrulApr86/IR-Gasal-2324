import numpy as np
import gensim
import lightgbm as lgb
from scipy.spatial.distance import cosine
from gensim.corpora import Dictionary
from gensim.models import LsiModel
from bsbi import BSBIIndex
from compression import VBEPostings
import random

# Define helper functions here
def load_documents(file_path):
    documents = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            doc_id, content = line.strip().split(" ", 1)
            documents[doc_id] = content.split()  # Ubah ini menjadi daftar kata
    return documents


def load_queries(file_path):
    queries = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            q_id, content = line.strip().split(" ", 1)
            queries[q_id] = content.split()
    return queries

def load_relevance_judgments(file_path, queries, documents):
    q_docs_rel = {}  # Grouping by q_id
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            q_id, doc_id, rel = line.strip().split(" ")
            if q_id in queries and doc_id in documents:
                if q_id not in q_docs_rel:
                    q_docs_rel[q_id] = []
                q_docs_rel[q_id].append((doc_id, int(rel)))
    return q_docs_rel


def create_dataset(queries, documents, q_docs_rel):
    NUM_NEGATIVES = 1
    group_qid_count = []
    dataset = []

    for q_id, docs_rels in q_docs_rel.items():
        group_qid_count.append(len(docs_rels) + NUM_NEGATIVES)
        for doc_id, rel in docs_rels:
            dataset.append((queries[q_id], documents[doc_id], rel))
        # Menambahkan negative sample
        negative_sample = random.choice(list(documents.values()))
        dataset.append((queries[q_id], negative_sample, 0))

    return dataset, group_qid_count

def create_lsi_model(documents, num_topics=200):
    dictionary = Dictionary(documents.values())
    bow_corpus = [dictionary.doc2bow(doc) for doc in documents.values()]
    lsi_model = LsiModel(bow_corpus, num_topics=num_topics)
    return lsi_model, dictionary
def vector_representation(text, lsi_model, dictionary):
    bow = dictionary.doc2bow(text)
    lsi_vector = lsi_model[bow]
    return [score for _, score in lsi_vector]

def features(query, doc):
      v_q = vector_representation(query)
      v_d = vector_representation(doc)
      q = set(query)
      d = set(doc)
      cosine_dist = cosine(v_q, v_d)
      jaccard = len(q & d) / len(q | d)
      return v_q + v_d + [jaccard] + [cosine_dist]

def calculate_features(query_vector, doc_vector, query, doc):
    cosine_dist = cosine(query_vector, doc_vector)
    jaccard = len(set(query) & set(doc)) / len(set(query) | set(doc))
    return [cosine_dist, jaccard]

# Fungsi untuk membuat fitur gabungan query dan dokumen
def create_feature_vectors(dataset, lsi_model, dictionary):
    X = []
    Y = []
    for query, doc, rel in dataset:
        query_vector = vector_representation(query, lsi_model, dictionary)
        doc_vector = vector_representation(doc, lsi_model, dictionary)
        features = query_vector + doc_vector + calculate_features(query_vector, doc_vector, query, doc)
        X.append(features)
        Y.append(rel)
    return np.array(X), np.array(Y)


def predict_ranking(query, docs, ranker, lsi_model, dictionary):
    X_unseen = []
    for doc_id, doc in docs:
        query_vector = vector_representation(query.split(), lsi_model, dictionary)
        doc_vector = vector_representation(doc.split(), lsi_model, dictionary)
        features = query_vector + doc_vector + calculate_features(query_vector, doc_vector, query.split(), doc.split())
        X_unseen.append(features)
    X_unseen = np.array(X_unseen)
    scores = ranker.predict(X_unseen)

    return scores

def load_document_content(doc_path):
    with open(doc_path, 'r', encoding='utf-8') as file:
        return file.read().split()

def prepare_docs(SERP):
    docs = []
    for score, doc_path in SERP:
        doc_content = load_document_content(doc_path)
        docs.append((doc_path, ' '.join(doc_content)))  # Simpan path lengkap
    return docs

def rerank_with_letor(ranker, lsi_model, dictionary, SERP, search_query):
    reranked_SERP = []

    for score, doc_path in SERP:
        # Muat konten dokumen
        with open(doc_path, 'r', encoding='utf-8') as file:
            doc_content = file.read().split()

        # Hitung representasi vektor untuk dokumen
        doc_vector = vector_representation(doc_content, lsi_model, dictionary)

        # Hitung fitur gabungan untuk query dan dokumen
        query_vector = vector_representation(search_query.split(), lsi_model, dictionary)
        features = query_vector + doc_vector + calculate_features(query_vector, doc_vector, search_query.split(), doc_content)

        # Prediksi skor dengan LETOR
        letor_score = ranker.predict([features])[0]
        reranked_SERP.append((letor_score, doc_path))  # Simpan skor dan path lengkap

    # Urutkan SERP berdasarkan skor LETOR yang dihasilkan
    reranked_SERP.sort(key=lambda x: x[0], reverse=True)
    return reranked_SERP


def train_letor_model(X, Y, group_qid_count):
    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        boosting_type="gbdt",
        n_estimators=100,
        importance_type="gain",
        metric="ndcg",
        num_leaves=40,
        learning_rate=0.02,
        max_depth=-1
    )
    ranker.fit(X, Y, group=group_qid_count)
    return ranker

def rerank_search_results(search_query, top_k=100):
    # Dapatkan SERP dari bsbi
    BSBI_instance = BSBIIndex(data_dir='collections', postings_encoding=VBEPostings, output_dir='index')
    SERP = BSBI_instance.retrieve_tfidf(search_query, k=top_k)

    # Load dan siapkan data
    documents = load_documents('./qrels-folder/train_docs.txt')
    queries = load_queries('./qrels-folder/train_queries.txt')
    q_docs_rel = load_relevance_judgments('./qrels-folder/train_qrels.txt', queries, documents)
    dataset, group_qid_count = create_dataset(queries, documents, q_docs_rel)

    # Bangun model LSI dan latih LambdaMART
    NUM_LATENT_TOPICS = 200
    lsi_model, dictionary = create_lsi_model(documents, NUM_LATENT_TOPICS)
    X, Y = create_feature_vectors(dataset, lsi_model, dictionary)

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        boosting_type="gbdt",
        n_estimators=100,
        importance_type="gain",
        metric="ndcg",
        num_leaves=40,
        learning_rate=0.02,
        max_depth=-1
    )

    ranker.fit(X, Y, group=group_qid_count)

    # Prediksi dan re-ranking
    docs = prepare_docs(SERP)
    scores = predict_ranking(search_query, docs, ranker, lsi_model, dictionary)

    reranked_SERP = sorted(zip(scores, [doc_path for doc_path, _ in docs]), key=lambda x: x[0],
                           reverse=True)

    # Kembalikan hasil re-ranking
    return reranked_SERP

def train_letor():
    documents = load_documents('./qrels-folder/train_docs.txt')
    queries = load_queries('./qrels-folder/train_queries.txt')
    q_docs_rel = load_relevance_judgments('./qrels-folder/train_qrels.txt', queries, documents)
    dataset, group_qid_count = create_dataset(queries, documents, q_docs_rel)

    # Bangun model LSI dan latih LambdaMART
    NUM_LATENT_TOPICS = 200
    lsi_model, dictionary = create_lsi_model(documents, NUM_LATENT_TOPICS)
    X, Y = create_feature_vectors(dataset, lsi_model, dictionary)

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        boosting_type="gbdt",
        n_estimators=100,
        importance_type="gain",
        metric="ndcg",
        num_leaves=40,
        learning_rate=0.02,
        max_depth=-1
    )

    ranker.fit(X, Y, group=group_qid_count)

    return ranker, lsi_model, dictionary

# Fungsi ini akan dijalankan ketika file diimpor
if __name__ == "__main__":
    search_query = "Terletak sangat dekat dengan khatulistiwa"
    reranked_results = rerank_search_results(search_query, 10)

    print("Query: ", search_query)
    BSBI_instance = BSBIIndex(data_dir='collections', postings_encoding=VBEPostings, output_dir='index')
    SERP = BSBI_instance.retrieve_tfidf(search_query, k=10)

    print("\nSERP:")
    for score, doc_id in SERP:
        print(f"{doc_id}: {score}")

    print("\nReranked SERP:")
    for score, doc_id in reranked_results:
        print(f"{doc_id}: {score}")

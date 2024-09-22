import math
import re
import os
from bsbi import BSBIIndex
from compression import VBEPostings
from tqdm import tqdm

# >>>>> 3 IR metrics: RBP p = 0.8, DCG, dan AP


def rbp(ranking, p=0.8):
    """ menghitung search effectiveness metric score dengan 
        Rank Biased Precision (RBP)

        Parameters
        ----------
        ranking: List[int]
           vektor biner seperti [1, 0, 1, 1, 1, 0]
           gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
           Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                   di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                   di rank-6 tidak relevan

        Returns
        -------
        Float
          score RBP
    """
    score = 0.
    for i in range(1, len(ranking) + 1):
        pos = i - 1
        score += ranking[pos] * (p ** (i - 1))
    return (1 - p) * score


def dcg(ranking):
    """ menghitung search effectiveness metric score dengan 
        Discounted Cumulative Gain

        Parameters
        ----------
        ranking: List[int]
           vektor biner seperti [1, 0, 1, 1, 1, 0]
           gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
           Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                   di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                   di rank-6 tidak relevan

        Returns
        -------
        Float
          score DCG
    """
    score = 0.0
    for i, rel in enumerate(ranking, 1):
        score += (2**rel - 1) / (math.log(i + 1, 2))
    return score


def prec(ranking, k):
    """ menghitung search effectiveness metric score dengan 
        Precision at K

        Parameters
        ----------
        ranking: List[int]
           vektor biner seperti [1, 0, 1, 1, 1, 0]
           gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
           Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                   di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                   di rank-6 tidak relevan

        k: int
          banyak dokumen yang dipertimbangkan atau diperoleh

        Returns
        -------
        Float
          score Prec@K
    """
    top_k = ranking[:k]
    relevant_docs = sum(top_k)
    return relevant_docs / k


def ap(ranking):
    """ menghitung search effectiveness metric score dengan 
        Average Precision

        Parameters
        ----------
        ranking: List[int]
           vektor biner seperti [1, 0, 1, 1, 1, 0]
           gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
           Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                   di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                   di rank-6 tidak relevan

        Returns
        -------
        Float
          score AP
    """
    rel_docs = 0
    cumulative_precision = 0.0
    for k, rel in enumerate(ranking, 1):
        if rel:
            rel_docs += 1
            cumulative_precision += rel_docs / k
    return cumulative_precision / (rel_docs or 1)  # avoid division by zero

# >>>>> memuat qrels


def load_qrels(qrel_file="./qrels-folder/test_qrels.txt"):
    """ 
        memuat query relevance judgment (qrels) 
        dalam format dictionary of dictionary qrels[query id][document id],
        dimana hanya dokumen yang relevan (nilai 1) yang disimpan,
        sementara dokumen yang tidak relevan (nilai 0) tidak perlu disimpan,
        misal {"Q1": {500:1, 502:1}, "Q2": {150:1}}
    """
    with open(qrel_file) as file:
        content = file.readlines()

    qrels_sparse = {}

    for line in content:
        parts = line.strip().split()
        qid = parts[0]
        did = int(parts[1])
        if not (qid in qrels_sparse):
            qrels_sparse[qid] = {}
        if not (did in qrels_sparse[qid]):
            qrels_sparse[qid][did] = 0
        qrels_sparse[qid][did] = 1
    return qrels_sparse

# >>>>> EVALUASI !


def eval_retrieval(qrels, query_file="./qrels-folder/test_queries.txt", k=1000):
    BSBI_instance = BSBIIndex(data_dir='collections',
                              postings_encoding=VBEPostings,
                              output_dir='index')

    # Variasi nilai k1 dan b yang ingin dicoba
    k1_values = [1.2, 1.5, 2.0]
    b_values = [0.5, 0.75, 1.0]

    with open(query_file) as file:
        queries = file.readlines()

    # Evaluasi untuk TF-IDF
    rbp_scores_tfidf = []
    dcg_scores_tfidf = []
    ap_scores_tfidf = []

    for qline in tqdm(queries):
        parts = qline.strip().split()
        qid = parts[0]
        query = " ".join(parts[1:])

        ranking_tfidf = []
        for (score, doc) in BSBI_instance.retrieve_tfidf(query, k=k):
            did = int(os.path.splitext(os.path.basename(doc))[0])
            if (did in qrels[qid]):
                ranking_tfidf.append(1)
            else:
                ranking_tfidf.append(0)

        rbp_scores_tfidf.append(rbp(ranking_tfidf))
        dcg_scores_tfidf.append(dcg(ranking_tfidf))
        ap_scores_tfidf.append(ap(ranking_tfidf))

    print("Hasil evaluasi TF-IDF terhadap 150 queries")
    print("RBP score =", sum(rbp_scores_tfidf) / len(rbp_scores_tfidf))
    print("DCG score =", sum(dcg_scores_tfidf) / len(dcg_scores_tfidf))
    print("AP score  =", sum(ap_scores_tfidf) / len(ap_scores_tfidf))
    print("------------------------------")

    # Evaluasi untuk BM25 dengan variasi k1 dan b
    for k1 in k1_values:
        for b in b_values:
            rbp_scores_bm25 = []
            dcg_scores_bm25 = []
            ap_scores_bm25 = []

            for qline in tqdm(queries):
                parts = qline.strip().split()
                qid = parts[0]
                query = " ".join(parts[1:])

                ranking_bm25 = []
                for (score, doc) in BSBI_instance.retrieve_bm25(query, k1=k1, b=b, k=k):
                    did = int(os.path.splitext(os.path.basename(doc))[0])
                    if (did in qrels[qid]):
                        ranking_bm25.append(1)
                    else:
                        ranking_bm25.append(0)

                rbp_scores_bm25.append(rbp(ranking_bm25))
                dcg_scores_bm25.append(dcg(ranking_bm25))
                ap_scores_bm25.append(ap(ranking_bm25))

            print(f"Hasil evaluasi BM25 dengan k1={k1} dan b={b} terhadap 150 queries")
            print("RBP score =", sum(rbp_scores_bm25) / len(rbp_scores_bm25))
            print("DCG score =", sum(dcg_scores_bm25) / len(dcg_scores_bm25))
            print("AP score  =", sum(ap_scores_bm25) / len(ap_scores_bm25))
            print("------------------------------")


if __name__ == '__main__':
    qrels = load_qrels()

    eval_retrieval(qrels)

from bsbi import BSBIIndex
from compression import VBEPostings

BSBI_instance = BSBIIndex(data_dir='collections',
                          postings_encoding=VBEPostings,
                          output_dir='index')

queries = ["Terletak sangat dekat dengan khatulistiwa", "Jumlah uang terbatas yang telah ditentukan sebelumnya bahwa seseorang harus membayar dari tabungan mereka sendiri"]

for query in queries:
    print(BSBI_instance.retrieve_tfidf(query, k=100))
    print("Query  : ", query)
    print("Results:")
    for (score, doc) in BSBI_instance.retrieve_tfidf(query, k=100):
        print(f"{doc:30} {score:>.3f}")
    print()

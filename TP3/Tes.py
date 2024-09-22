from letor import rerank_search_results

search_query = "Terletak sangat dekat dengan khatulistiwa"
reranked_results = rerank_search_results(search_query)
print(reranked_results)

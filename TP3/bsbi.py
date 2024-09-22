import os
import pickle
import contextlib
import heapq
import math
import re
from collections import defaultdict

import spacy

from index import InvertedIndexReader, InvertedIndexWriter
from util import IdMap, merge_and_sort_posts_and_tfs
from compression import VBEPostings
from tqdm import tqdm

from mpstemmer import MPStemmer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

from operator import itemgetter

nlp = spacy.blank('id')

class BSBIIndex:
    """
    Attributes
    ----------
    term_id_map(IdMap): Untuk mapping terms ke termIDs
    doc_id_map(IdMap): Untuk mapping relative paths dari dokumen (misal,
                    /collection/0/gamma.txt) to docIDs
    data_dir(str): Path ke data
    output_dir(str): Path ke output index files
    postings_encoding: Lihat di compression.py, kandidatnya adalah StandardPostings,
                    VBEPostings, dsb.
    index_name(str): Nama dari file yang berisi inverted index
    """

    def __init__(self, data_dir, output_dir, postings_encoding, index_name="main_index"):
        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.index_name = index_name
        self.postings_encoding = postings_encoding

        # Untuk menyimpan nama-nama file dari semua intermediate inverted index
        self.intermediate_indices = []

    def save(self):
        """Menyimpan doc_id_map and term_id_map ke output directory via pickle"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'wb') as f:
            pickle.dump(self.term_id_map, f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'wb') as f:
            pickle.dump(self.doc_id_map, f)

    def load(self):
        """Memuat doc_id_map and term_id_map dari output directory"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'rb') as f:
            self.term_id_map = pickle.load(f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'rb') as f:
            self.doc_id_map = pickle.load(f)

    def pre_processing_text(self, content):
        """
        Melakukan preprocessing pada text, yakni stemming dan removing stopwords
        """
        # https://github.com/ariaghora/mpstemmer/tree/master/mpstemmer

        stemmer = MPStemmer()
        remover = StopWordRemoverFactory().create_stop_word_remover()
        stop_factory = StopWordRemoverFactory()
        more_stopword = ['dengan', 'ia,' 'bahwa', 'oleh']
        stopword = stop_factory.get_stop_words() + more_stopword

        # Tokenisasi teks
        tokens = re.findall(r'\b\w+\b', content.lower())

        # Proses stemming dan penghapusan stopwords pada setiap token
        preprocessed_tokens = []
        for token in tokens:
            stemmed_token = stemmer.stem(token)
            if stemmed_token not in stopword:
                preprocessed_tokens.append(stemmed_token)

        return ' '.join(preprocessed_tokens)

    def parsing_block(self, block_path):
        """
        Lakukan parsing terhadap text file sehingga menjadi sequence of
        <termID, docID> pairs.

        Gunakan tools available untuk stemming bahasa Indonesia, seperti
        MpStemmer: https://github.com/ariaghora/mpstemmer 
        Jangan gunakan PySastrawi untuk stemming karena kode yang tidak efisien dan lambat.

        JANGAN LUPA BUANG STOPWORDS! Kalian dapat menggunakan PySastrawi 
        untuk menghapus stopword atau menggunakan sumber lain seperti:
        - Satya (https://github.com/datascienceid/stopwords-bahasa-indonesia)
        - Tala (https://github.com/masdevid/ID-Stopwords)

        Untuk "sentence segmentation" dan "tokenization", bisa menggunakan
        regex atau boleh juga menggunakan tools lain yang berbasis machine
        learning.

        Parameters
        ----------
        block_path : str
            Relative Path ke directory yang mengandung text files untuk sebuah block.

            CATAT bahwa satu folder di collection dianggap merepresentasikan satu block.
            Konsep block di soal tugas ini berbeda dengan konsep block yang terkait
            dengan operating systems.

        Returns
        -------
        List[Tuple[Int, Int]]
            Returns all the td_pairs extracted from the block
            Mengembalikan semua pasangan <termID, docID> dari sebuah block (dalam hal
            ini sebuah sub-direktori di dalam folder collection)

        Harus menggunakan self.term_id_map dan self.doc_id_map untuk mendapatkan
        termIDs dan docIDs. Dua variable ini harus 'persist' untuk semua pemanggilan
        parsing_block(...).
        """
        td_pairs = []
        stemmer = MPStemmer()
        stopwords = StopWordRemoverFactory().get_stop_words()

        def tokenize(text):
            return re.findall(r'\b\w+\b', text.lower())

        for doc_name in os.listdir(os.path.join(self.data_dir, block_path)):
            doc_path = os.path.join(self.data_dir, block_path, doc_name)
            with open(doc_path, 'r', encoding='utf-8') as doc_file:
                content = doc_file.read()
                tokens = [stemmer.stem(token) for token in tokenize(content) if token not in stopwords]
                for token in tokens:
                    term_id = self.term_id_map[token]
                    doc_id = self.doc_id_map[doc_path]  # Menggunakan path penuh sebagai unique ID
                    td_pairs.append((term_id, doc_id))

        return td_pairs

    def write_to_index(self, td_pairs, index):
        """
        Melakukan inversion td_pairs (list of <termID, docID> pairs) dan
        menyimpan mereka ke index. Disini diterapkan konsep BSBI dimana 
        hanya di-maintain satu dictionary besar untuk keseluruhan block.
        Namun dalam teknik penyimpanannya digunakan strategi dari SPIMI
        yaitu penggunaan struktur data hashtable (dalam Python bisa
        berupa Dictionary)

        ASUMSI: td_pairs CUKUP di memori

        Di Tugas Pemrograman 1, kita hanya menambahkan term dan
        juga list of sorted Doc IDs. Sekarang di Tugas Pemrograman 2,
        kita juga perlu tambahkan list of TF.

        Parameters
        ----------
        td_pairs: List[Tuple[Int, Int]]
            List of termID-docID pairs
        index: InvertedIndexWriter
            Inverted index pada disk (file) yang terkait dengan suatu "block"
        """
        term_dict = defaultdict(lambda: defaultdict(int))

        for term_id, doc_id in td_pairs:
            term_dict[term_id][doc_id] += 1

        for term_id, doc_freq_dict in term_dict.items():
            doc_ids = sorted(doc_freq_dict.keys())
            tfs = [doc_freq_dict[doc_id] for doc_id in doc_ids]
            index.append(term_id, doc_ids, tfs)

    def merge_index(self, indices, merged_index):
        """
        Lakukan merging ke semua intermediate inverted indices menjadi
        sebuah single index.

        Ini adalah bagian yang melakukan EXTERNAL MERGE SORT

        Gunakan fungsi merge_and_sort_posts_and_tfs(..) di modul util

        Parameters
        ----------
        indices: List[InvertedIndexReader]
            A list of intermediate InvertedIndexReader objects, masing-masing
            merepresentasikan sebuah intermediate inveted index yang iterable
            di sebuah block.

        merged_index: InvertedIndexWriter
            Instance InvertedIndexWriter object yang merupakan hasil merging dari
            semua intermediate InvertedIndexWriter objects.
        """
        # kode berikut mengasumsikan minimal ada 1 term
        merged_iter = heapq.merge(*indices, key=lambda x: x[0])
        curr, postings, tf_list = next(merged_iter)  # first item
        for t, postings_, tf_list_ in merged_iter:  # from the second item
            if t == curr:
                zip_p_tf = merge_and_sort_posts_and_tfs(list(zip(postings, tf_list)),
                                                        list(zip(postings_, tf_list_)))
                postings = [doc_id for (doc_id, _) in zip_p_tf]
                tf_list = [tf for (_, tf) in zip_p_tf]
            else:
                merged_index.append(curr, postings, tf_list)
                curr, postings, tf_list = t, postings_, tf_list_
        merged_index.append(curr, postings, tf_list)

    def retrieve_tfidf(self, query, k=10):
        """
        Melakukan Ranked Retrieval dengan skema TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        w(t, D) = (1 + log tf(t, D))       jika tf(t, D) > 0
                = 0                        jika sebaliknya

        w(t, Q) = IDF = log (N / df(t))

        Score = untuk setiap term di query, akumulasikan w(t, Q) * w(t, D).
                (tidak perlu dinormalisasi dengan panjang dokumen)

        catatan: 
            1. informasi DF(t) ada di dictionary postings_dict pada merged index
            2. informasi TF(t, D) ada di tf_li
            3. informasi N bisa didapat dari doc_length pada merged index, len(doc_length)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.

        """
        # Load the merged index and doc_id_map
        self.load()

        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as index:
            N = len(self.doc_id_map)  # total number of documents

            # Tokenize and preprocess the query
            query_terms = self.pre_processing_text(query).split()

            # Convert query terms to termIDs
            query_term_ids = [self.term_id_map[term] for term in query_terms if term in self.term_id_map]

            # Initialize scores for documents
            doc_scores = defaultdict(float)

            for term_id in query_term_ids:
                if term_id not in index.postings_dict:
                    continue
                start_position, _, len_in_bytes_of_postings, len_in_bytes_of_tf = index.postings_dict[term_id]

                # Seek to the position where tf_list starts
                start_pos_tf_list = start_position + len_in_bytes_of_postings

                # Read and decode the tf_list
                index.index_file.seek(start_pos_tf_list)
                tf_list_encoded = index.index_file.read(len_in_bytes_of_tf)
                tf_list = index.postings_encoding.decode_tf(tf_list_encoded)
                df_t = len(index.postings_dict[term_id])
                w_t_Q = math.log10(N / df_t)

                postings_list, _ = index.get_postings_list(term_id)

                for doc_id, tf in zip(postings_list, tf_list):
                    w_t_D = (1 + math.log10(tf)) if tf > 0 else 0
                    doc_scores[doc_id] += w_t_Q * w_t_D

            # Sort by score and get top-k
            sorted_scores = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:k]

            # Convert doc_ids back to doc names
            results = [(score, self.doc_id_map[doc_id]) for doc_id, score in sorted_scores]

            return results

    def retrieve_bm25(self, query, k=10, k1=1.2, b=0.75):
        """
        Melakukan Ranked Retrieval dengan skema scoring BM25 dan framework TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        """
        self.load()

        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as index:
            N = len(self.doc_id_map)  # total number of documents
            avgdl = sum(index.doc_length.values()) / N

            # Tokenize and preprocess the query
            query_terms = self.pre_processing_text(query).split()

            # Convert query terms to termIDs
            query_term_ids = [self.term_id_map[term] for term in query_terms if term in self.term_id_map]

            # Initialize scores for documents
            doc_scores = defaultdict(float)

            for term_id in query_term_ids:
                if term_id not in index.postings_dict:
                    continue
                start_position, _, len_in_bytes_of_postings, len_in_bytes_of_tf = index.postings_dict[term_id]

                # Seek to the position where tf_list starts
                start_pos_tf_list = start_position + len_in_bytes_of_postings

                # Read and decode the tf_list
                index.index_file.seek(start_pos_tf_list)
                tf_list_encoded = index.index_file.read(len_in_bytes_of_tf)
                tf_list = index.postings_encoding.decode_tf(tf_list_encoded)

                df_t = len(tf_list)
                idf = math.log10((N - df_t + 0.5) / (df_t + 0.5) + 1.0)

                postings_list, _ = index.get_postings_list(term_id)

                for doc_id, tf in zip(postings_list, tf_list):
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 * (1 - b + b * (index.doc_length[doc_id] / avgdl))
                    doc_scores[doc_id] += idf * (numerator / denominator)

            # Sort by score and get top-k
            sorted_scores = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:k]

            # Convert doc_ids back to doc names
            results = [(score, self.doc_id_map[doc_id]) for doc_id, score in sorted_scores]

            return results

    def do_indexing(self):
        """
        Base indexing code
        BAGIAN UTAMA untuk melakukan Indexing dengan skema BSBI (blocked-sort
        based indexing)

        Method ini scan terhadap semua data di collection, memanggil parsing_block
        untuk parsing dokumen dan memanggil write_to_index yang melakukan inversion
        di setiap block dan menyimpannya ke index yang baru.
        """
        # loop untuk setiap sub-directory di dalam folder collection (setiap block)
        for block_dir_relative in tqdm(sorted(next(os.walk(self.data_dir))[1])):
            td_pairs = self.parsing_block(block_dir_relative)
            index_id = 'intermediate_index_'+block_dir_relative
            self.intermediate_indices.append(index_id)
            with InvertedIndexWriter(index_id, self.postings_encoding, directory=self.output_dir) as index:
                self.write_to_index(td_pairs, index)
                td_pairs = None

        self.save()

        with InvertedIndexWriter(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(InvertedIndexReader(index_id, self.postings_encoding, directory=self.output_dir))
                           for index_id in self.intermediate_indices]
                self.merge_index(indices, merged_index)
                # print("Merged index: ", merged_index.postings_dict)


if __name__ == "__main__":

    BSBI_instance = BSBIIndex(data_dir='collections',
                              postings_encoding=VBEPostings,
                              output_dir='index')
    BSBI_instance.do_indexing()  # memulai indexing!

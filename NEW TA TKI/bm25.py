import re
import math
from collections import Counter
from nltk.stem import PorterStemmer
import nltk

# Unduh resource NLTK jika belum ada
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class QueryParsers:
    def __init__(self, query):
        self.query = self._process_query(query)
    
    def _process_query(self, query):
        """Tokenisasi dan stemming query"""
        # Hilangkan karakter non-alphanumeric
        pattern = re.compile(r'\W+')
        cleaned = pattern.sub(' ', query.lower())
        
        # Tokenisasi
        tokens = cleaned.split()
        
        # Stemming
        stemmer = PorterStemmer()
        return [stemmer.stem(token) for token in tokens]

class BuildIndex:
    def __init__(self, documents, k1=1.2, b=0.75):
        """
        Inisialisasi indeks BM25
        
        Args:
            documents: List dokumen/dataset
            k1: Parameter BM25 untuk kontrol term frequency
            b: Parameter BM25 untuk kontrol panjang dokumen
        """
        self.k1 = k1
        self.b = b
        self.documents = documents
        self.N = len(documents)
        self.avgdl = 0
        self.df = {}  # Document frequency
        self.idf = {}  # Inverse document frequency
        self.doc_len = []
        
        # Proses dokumen
        self.processed_docs = []
        self._build_index()
    
    def _build_index(self):
        """Membangun indeks dari kumpulan dokumen"""
        stemmer = PorterStemmer()
        pattern = re.compile(r'\W+')
        total_length = 0
        
        for doc in self.documents:
            # Preprocessing
            cleaned = pattern.sub(' ', doc.lower())
            tokens = cleaned.split()
            stemmed = [stemmer.stem(token) for token in tokens]
            
            self.processed_docs.append(stemmed)
            doc_len = len(stemmed)
            self.doc_len.append(doc_len)
            total_length += doc_len
            
            # Hitung document frequency
            for term in set(stemmed):
                self.df[term] = self.df.get(term, 0) + 1
        
        # Hitung average document length
        self.avgdl = total_length / self.N if self.N > 0 else 0
        
        # Hitung IDF
        for term, freq in self.df.items():
            self.idf[term] = math.log((self.N - freq + 0.5) / (freq + 0.5) + 1)
    
    def _score_document(self, doc_idx, query_terms):
        """Menghitung skor BM25 untuk satu dokumen"""
        score = 0.0
        doc_terms = self.processed_docs[doc_idx]
        doc_len = self.doc_len[doc_idx]
        
        term_freq = Counter(doc_terms)
        
        for term in query_terms:
            if term in term_freq:
                tf = term_freq[term]
                idf = self.idf.get(term, 0)
                
                # Hitung komponen BM25
                numerator = idf * tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                score += numerator / denominator
        
        return score
    
    def ranked_docs(self, query_terms):
        """
        Mendapatkan dokumen terurut berdasarkan skor BM25
        
        Args:
            query_terms: List term query yang sudah diproses
            
        Returns:
            List tuple (doc_index, score) terurut descending
        """
        scores = []
        for i in range(self.N):
            score = self._score_document(i, query_terms)
            scores.append((i, score))
        
        # Urutkan berdasarkan skor tertinggi
        return sorted(scores, key=lambda x: x[1], reverse=True)
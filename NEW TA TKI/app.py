from flask import Flask, render_template, request
import pandas as pd
import os
from collections import namedtuple
from bm25 import BuildIndex, QueryParsers

app = Flask(__name__)

# Konfigurasi
TOP_RESULTS = 200  # Menampilkan 200 berita teratas
RESULTS_PER_PAGE = 20  # Jumlah hasil per halaman
DATASET_PATH = os.path.join(os.path.dirname(__file__), 'Data_latih.csv')

# Field names configuration
FIELD_TITLE = 'judul'
FIELD_ISI = 'narasi'
FIELD_LABEL = 'label'

def load_data():
    try:
        df = pd.read_csv(DATASET_PATH)
        
        # Validasi kolom
        required_columns = [FIELD_TITLE, FIELD_ISI, FIELD_LABEL]
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Dataset harus mengandung kolom: {required_columns}")
            
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return pd.DataFrame(columns=required_columns)  # Return empty DataFrame jika error

# Load data saat aplikasi mulai
df = load_data()

# Prepare data
titles = df[FIELD_TITLE].fillna('').tolist()
docs = df[FIELD_ISI].fillna('').str.lower().tolist()
labels = df[FIELD_LABEL].tolist()

# Initialize BM25
bm25 = BuildIndex(docs)

# Result namedtuple
Result = namedtuple('Result', ['score', 'title', 'content', 'label'])

@app.route('/', methods=['GET', 'POST'])
def index():
    query = request.form.get('query', '') or request.args.get('query', '')
    filter_option = request.args.get('filter', '')
    page = int(request.args.get('page', 1))
    
    results = []
    total_news = len(docs)
    no_results_message = None  # Inisialisasi pesan tidak ada hasil

    if query:
        qlist = QueryParsers(query).query
        ranked = bm25.ranked_docs(qlist)
        
        # Ambil 200 hasil teratas
        for idx, score in ranked[:TOP_RESULTS]:  
            label_val = labels[idx]
            label_str = 'FAKTA' if (label_val == 1 or str(label_val).upper() == 'FAKTA') else 'HOAKS'
            
            # Apply filter
            if (not filter_option or 
                (filter_option == 'hoaks' and label_str == 'HOAKS') or 
                (filter_option == 'fakta' and label_str == 'FAKTA')):
                
                results.append(Result(
                    score=score,
                    title=titles[idx],
                    content=docs[idx],
                    label=label_str
                ))
        
        # Tambahkan logika untuk menangani tidak ada hasil
        if not results:
            no_results_message = "Tidak ada hasil ditemukan untuk pencarian Anda."
    else:
        # Jika tidak ada query, tampilkan semua berita dengan skor 0
        for idx in range(min(TOP_RESULTS, len(docs))):
            label_val = labels[idx]
            label_str = 'FAKTA' if (label_val == 1 or str(label_val).upper() == 'FAKTA') else 'HOAKS'
            
            if (not filter_option or 
                (filter_option == 'hoaks' and label_str == 'HOAKS') or 
                (filter_option == 'fakta' and label_str == 'FAKTA')):
                
                results.append(Result(
                    score=0.0,
                    title=titles[idx],
                    content=docs[idx],
                    label=label_str
                ))
    
    # Pagination logic
    pages = (len(results) + RESULTS_PER_PAGE - 1) // RESULTS_PER_PAGE if results else 1
    start_idx = (page - 1) * RESULTS_PER_PAGE
    end_idx = start_idx + RESULTS_PER_PAGE
    paginated_results = results[start_idx:end_idx]
    
    return render_template('search.html', 
                         results=paginated_results,
                         total_news=total_news,
                         query=query,
                         page=page,
                         pages=pages,
                         total_results=len(results),
                         showing_all=not bool(query),
                         no_results_message=no_results_message)  # Kirim pesan tidak ada hasil

if __name__ == '__main__':
    app.run(debug=True)
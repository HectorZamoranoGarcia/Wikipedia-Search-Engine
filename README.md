# Wikipedia Search Engine 🔍

A high-performance **Information Retrieval System** for Wikipedia articles, featuring binary and semantic search capabilities powered by modern NLP models.

Built with Python, this engine supports full-text indexing, positional queries (phrase search), boolean operators (`NOT`), and semantic similarity search using Sentence-BERT embeddings.

---

## Features

| Feature | Description |
|---|---|
| **Inverted Index** | Efficient term-based inverted index over Wikipedia JSON dumps |
| **Boolean Queries** | Implicit `AND` between terms, explicit `NOT` operator |
| **Phrase Search** | Positional index enables exact phrase matching with `"double quotes"` |
| **Semantic Search** | Dense vector similarity via KDTree with configurable distance threshold |
| **Semantic Reranking** | Re-order binary search results by semantic relevance |
| **Multiple NLP Backends** | Sentence-BERT, BETO (Spanish BERT), or spaCy static vectors |

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  JSON Docs  │────▶│  SAR_Indexer  │────▶│  Binary Index   │
│  (Wikipedia)│     │              │     │  (.bin file)    │
└─────────────┘     └──────┬───────┘     └────────┬────────┘
                           │                      │
                    ┌──────▼───────┐        ┌─────▼────────┐
                    │  Positional  │        │ SAR_Searcher  │
                    │  Index (opt) │        │  (retrieval)  │
                    └──────────────┘        └──────┬───────┘
                                                   │
                    ┌──────────────┐         ┌─────▼───────┐
                    │  KDTree +    │◀───────▶│  Semantic    │
                    │  Embeddings  │         │  Module      │
                    └──────────────┘         └─────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

For semantic search, also install:
```bash
pip install -r requirements-semantic.txt
python -m spacy download es_core_news_lg
```

### 2. Index a Collection

```bash
# Basic index
python SAR_Indexer.py data/sample/ index.bin

# With positional index (enables phrase search)
python SAR_Indexer.py -P data/sample/ index.bin

# With semantic index
python SAR_Indexer.py -P -S data/sample/ index.bin
```

### 3. Search

```bash
# Interactive mode
python SAR_Searcher.py index.bin

# Single query
python SAR_Searcher.py index.bin -Q "inteligencia artificial"

# Phrase search (requires positional index)
python SAR_Searcher.py index.bin -Q '"segunda guerra mundial"'

# Boolean NOT
python SAR_Searcher.py index.bin -Q "isla NOT valencia"

# Semantic search (requires semantic index)
python SAR_Searcher.py index.bin -S 1.5

# Semantic reranking of binary results
python SAR_Searcher.py index.bin -R -Q "machine learning"
```

### 4. Test Against References

```bash
python SAR_Searcher.py index.bin -T tests/100_mixed_positional.ref
```

## Data Format

Each input file is a JSON-lines file where every line is a Wikipedia article:

```json
{
  "url": "https://es.wikipedia.org/wiki/Example",
  "title": "Example Article",
  "summary": "Brief description...",
  "sections": [
    {
      "name": "Section Title",
      "text": "Section content...",
      "subsections": [{"name": "...", "text": "..."}]
    }
  ]
}
```

## Indexing Details

- **Tokenization**: Non-alphanumeric characters removed; case-insensitive
- **Duplicate detection**: Articles with duplicate URLs are skipped
- **Posting list merge**: Uses efficient two-pointer merge algorithms (no set operations)
- **Positional index**: Stores token positions for phrase matching

## Semantic Search Models

| Model | Description | Speed |
|---|---|---|
| `SBERT` | Sentence-BERT (Spanish) — **recommended** | Medium |
| `BetoCLS` | BETO [CLS] token embedding | Slow |
| `Beto` | BETO mean pooling | Slow |
| `Spacy` | spaCy static word vectors (mean) | Fast |

## Project Structure

```
├── SAR_Indexer.py          # CLI indexer entry point
├── SAR_Searcher.py         # CLI search entry point
├── SAR_lib.py              # Core library (indexing + retrieval)
├── SAR_semantics.py        # Semantic embedding models
├── requirements.txt        # Core dependencies
├── requirements-semantic.txt # Semantic search dependencies
├── data/                   # Wikipedia article collections
│   └── sample/
└── tests/                  # Reference query-result pairs
```

## Performance

| Dataset | Articles | Unique Tokens | Index Time | Index Size |
|---|---|---|---|---|
| 100_mixed | 100 | 22,471 | ~0.3s | ~60 KB |
| 1,000_mixed | 969 | 110,309 | ~2.2s | ~600 KB |
| 10,000_mixed | 9,544 | 418,756 | ~25s | ~8 MB |

## License

MIT License — see [LICENSE](LICENSE) for details.

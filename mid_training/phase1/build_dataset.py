"""Build the packed Phase 1 dataset from local books, articles, and EDGAR JSON."""

from __future__ import annotations

from phase1.config import PACKED_DATASET_DIR, TARGET_TOKENS
from phase1.data_articles import load_articles
from phase1.data_books import load_books
from phase1.data_corpus import load_corpus
from phase1.packing import (
    count_tokens,
    load_tokenizer,
    pack_documents,
    print_token_report,
    source_token_count,
)


def build():
    tokenizer = load_tokenizer()
    books = load_books()
    book_texts = [book["text"] for book in books if book["text"]]
    book_tokens = source_token_count(tokenizer, book_texts)
    print(f"\nBook tokens: {book_tokens:,}")

    articles = load_articles()
    article_tokens = source_token_count(tokenizer, articles)
    print(f"Article tokens: {article_tokens:,}")

    remaining = TARGET_TOKENS - book_tokens - article_tokens
    corpus_texts, corpus_tokens = load_corpus(
        remaining,
        lambda text: count_tokens(tokenizer, text),
    )

    documents = book_texts + articles + corpus_texts
    dataset = pack_documents(tokenizer, documents)
    print_token_report(book_tokens, article_tokens, corpus_tokens, len(dataset))
    dataset.save_to_disk(str(PACKED_DATASET_DIR))
    print(f"\nSaved packed dataset to {PACKED_DATASET_DIR}")
    return dataset


if __name__ == "__main__":
    build()

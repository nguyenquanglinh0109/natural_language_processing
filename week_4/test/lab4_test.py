from week_4.src.representation.word_embedder import WordEmbedder

if __name__ == "__main__":
    word_embedder = WordEmbedder("glove-wiki-gigaword-50")
    king_vector = word_embedder.get_vector("king")
    print("King vector:", king_vector)

    king_queen_cosine = word_embedder.get_similarity("king", "queen")
    king_man_cosine = word_embedder.get_similarity("king", "man")

    print("King and Queen cosine similarity:", king_queen_cosine)
    print("King and Man cosine similarity:", king_man_cosine)

    most_similar_computer = word_embedder.get_most_similarity("computer")
    print("Most similar words to 'computer':")
    for word, cosine in most_similar_computer:
        print(f"{word}: {cosine}")

    documents = "The queen rules the country."
    document_vector = word_embedder.embed_document(documents)
    print("Document vector:", document_vector)
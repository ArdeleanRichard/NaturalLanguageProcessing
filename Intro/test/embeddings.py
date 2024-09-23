from gensim.models import KeyedVectors
import numpy as np
from numpy.linalg import norm

file_embeddings = "../../data/embeddings/glove.6B.300d.txt"
glove = KeyedVectors.load_word2vec_format(file_embeddings, no_header=True)
print(glove.vectors.shape)

# Word similarity
# One attribute of word embeddings that makes them useful is the ability to compare them using cosine similarity to
# find how similar they are. KeyedVectors objects provide a method called most_similar() that we can use to find
# the closest words to a particular word of interest. By default, most_similar() returns the 10 most similar words,
# but this can be changed using the topn parameter.

# common noun
print(glove.most_similar("cactus"))
print(glove.most_similar("cake"))

# adjective
print(glove.most_similar("angry"))

# adverb
print(glove.most_similar("quickly"))

# preposition
print(glove.most_similar("between"))

# determiner
print(glove.most_similar("the"))

# Word analogies
# Another characteristic of word embeddings is their ability to solve analogy problems. The same most_similar() method
# can be used for this task, by passing two lists of words: a positive list with the words that should be added and a
# negative list with the words that should be subtracted. Using these arguments, the famous example
#  can be executed as follows:

# king - man + woman
print(glove.most_similar(positive=["king", "woman"], negative=["man"]))

# car - drive + fly
print(glove.most_similar(positive=["car", "fly"], negative=["drive"]))

# Looking under the hood
# Each row corresponds to a 300-dimensional word embedding. These embeddings are not normalized, but normalized
# embeddings can be obtained using the get_normed_vectors() method.
print(glove.vectors.shape)

normed_vectors = glove.get_normed_vectors()
print(normed_vectors.shape)

# The KeyedVectors object has the attributes index_to_key and key_to_index which are a list of words
# and a dictionary of words to indices, respectively.
#glove.index_to_key
#glove.key_to_index





# Word similarity from scratch
# Now we have everything we need to implement a most_similar_words() function that takes a word, the vector matrix,
# the index_to_key list, and the key_to_index dictionary. This function will return the 10 most similar words
# to the provided word, along with their similarity scores.


def most_similar_words(word, vectors, index_to_key, key_to_index, topn=10):
    # retrieve word_id corresponding to given word
    word_id = key_to_index[word]

    # retrieve embedding for given word
    emb = vectors[word_id]

    # calculate similarities to all words in out vocabulary
    similarities = vectors @ emb

    # get word_ids in ascending order with respect to similarity score
    ids_ascending = similarities.argsort()

    # reverse word_ids
    ids_descending = ids_ascending[::-1]

    # get boolean array with element corresponding to word_id set to false
    mask = ids_descending != word_id

    # obtain new array of indices that doesn't contain word_id
    # (otherwise the most similar word to the argument would be the argument itself)
    ids_descending = ids_descending[mask]

    # get topn word_ids
    top_ids = ids_descending[:topn]

    # retrieve topn words with their corresponding similarity score
    top_words = [(index_to_key[i], similarities[i]) for i in top_ids]

    # return results
    return top_words

vectors = glove.get_normed_vectors()
index_to_key = glove.index_to_key
key_to_index = glove.key_to_index
most_similar_words("cactus", vectors, index_to_key, key_to_index)



# Analogies from scratch
# The most_similar_words() function behaves as expected. Now let's implement a function to perform the analogy task.
# We will give it the very creative name analogy. This function will get two lists of words (one for positive words
# and one for negative words), just like the most_similar() method we discussed above.


def analogy(positive, negative, vectors, index_to_key, key_to_index, topn=10):
    # find ids for positive and negative words
    pos_ids = [key_to_index[w] for w in positive]
    neg_ids = [key_to_index[w] for w in negative]
    given_word_ids = pos_ids + neg_ids

    # get embeddings for positive and negative words
    pos_emb = vectors[pos_ids].sum(axis=0)
    neg_emb = vectors[neg_ids].sum(axis=0)

    # get embedding for analogy
    emb = pos_emb - neg_emb

    # normalize embedding
    emb = emb / norm(emb)

    # calculate similarities to all words in out vocabulary
    similarities = vectors @ emb

    # get word_ids in ascending order with respect to similarity score
    ids_ascending = similarities.argsort()

    # reverse word_ids
    ids_descending = ids_ascending[::-1]

    # get boolean array with element corresponding to any of given_word_ids set to false
    given_words_mask = np.isin(ids_descending, given_word_ids, invert=True)

    # obtain new array of indices that doesn't contain any of the given_word_ids
    ids_descending = ids_descending[given_words_mask]

    # get topn word_ids
    top_ids = ids_descending[:topn]

    # retrieve topn words with their corresponding similarity score
    top_words = [(index_to_key[i], similarities[i]) for i in top_ids]

    # return results
    return top_words

positive = ["king", "woman"]
negative = ["man"]
vectors = glove.get_normed_vectors()
index_to_key = glove.index_to_key
key_to_index = glove.key_to_index
analogy(positive, negative, vectors, index_to_key, key_to_index)
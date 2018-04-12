import nltk

from nltk.corpus import movie_reviews
import random

# documents = [(list(movie_reviews.words(fileid)), category)
#              for category in movie_reviews.categories()
#              for fileid in movie_reviews.fileids(category)]
# random.shuffle(documents)
# all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
# word_features = list(all_words)[:2000]
#
#
# def document_features(document):
#     document_words = set(document)
#     features = {}
#     for word in word_features:
#         features['contains({})'.format(word)] = (word in document_words)
#     return features
#
# #documents is list containing tuple
# featuresets = [(document_features(d), c) for (d, c) in documents]
# classifier = nltk.NaiveBayesClassifier.train(featuresets[100:])
#
# print(nltk.classify.accuracy(classifier, featuresets[:100]))
# classifier.show_most_informative_features(5)


from nltk import load_parser
cp = load_parser('grammars/book_grammars/sql0.fcfg')
query = 'What cities are located in China'
tokens = query.split()
for tree in cp.parse(tokens):
    print(tree)
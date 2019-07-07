#CLASSIFYING MOVIE REVIEW: A BINARY CLASSIFICATION EXAMPLE




#STEP 1.1 LOADINF THE IMDB DATASET

'''
The IMDB dataset contains 50,000 highly polarized reviews from
the Internet Movie Database. They're split into 25000 reviews for training
and 25000 for testing.


'''



from keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = 10000)

# the argument num_words = 10000 means you'll only keep the top 10,000 most frequently
# occuring words in the training data.

import numpy as np


def vectorize_sequence(sequences, dimension = 10000):
    results = np.zeros(len(sequences), dimension)
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


x_train = vectorize_sequence(train_data)
x_test = vectorize_sequence(test_data)

print(x_train[0])
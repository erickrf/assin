# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals

"""
Script implementing the overlap baseline for the ASSIN shared task.

It extracts two features from each pair: the amount of words exclusive
to the first sentence and to the second one. Both are fed to an SVM
(for entailment classification) or an SVR (for similarity regression).

It produces an XML file as the output, which can be evaluated with the
`assin-eval.py` script.
"""

import argparse
import numpy as np
from xml.etree import cElementTree as ET
from nltk.tokenize import RegexpTokenizer
from sklearn.linear_model import LinearRegression, LogisticRegression

from commons import read_xml, entailment_to_str


def tokenize_sentence(text):
    '''
    Tokenize the given sentence in Portuguese. The tokenization is done in conformity
    with Universal Treebanks (at least it attempts so).

    :param text: text to be tokenized, as a string
    '''
    tokenizer_regexp = r'''(?ux)
    # the order of the patterns is important!!
    (?:[^\W\d_]\.)+|                  # one letter abbreviations, e.g. E.U.A.
    \d+(?:[.,]\d+)*(?:[.,]\d+)|       # numbers in format 999.999.999,99999
    \w+(?:\.(?!\.|$))?|               # words with numbers (including hours as 12h30),
                                      # followed by a single dot but not at the end of sentence
    \d+(?:[-\\/]\d+)*|                # dates. 12/03/2012 12-03-2012
    \$|                               # currency sign
    -+|                               # any sequence of dashes
    \S                                # any non-space character
    '''
    tokenizer = RegexpTokenizer(tokenizer_regexp)

    return tokenizer.tokenize(text)


def extract_features(pairs):
    '''
    Extract a vector of features from the given pairs and return
    them as a numpy array.
    '''
    features = []
    for pair in pairs:
        t = pair.t.lower()
        tokens1 = tokenize_sentence(t)

        h = pair.h.lower()
        tokens2 = tokenize_sentence(h)

        value1, value2 = words_in_common(tokens1, tokens2)
        features.append((value1, value2))

    return np.array(features)


def words_in_common(sentence1, sentence2):
    '''
    Return the proportion of words in common in a pair.
    Repeated words are ignored.
    '''
    tokenset1 = set(sentence1)
    tokenset2 = set(sentence2)

    num_common_tokens = len(tokenset2.intersection(tokenset1))
    proportion1 = num_common_tokens / len(tokenset1)
    proportion2 = num_common_tokens / len(tokenset2)

    return (proportion1, proportion2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('train', help='XML file with training data')
    parser.add_argument('test', help='XML file with test data')
    parser.add_argument('output', help='Output tagged XML file')
    args = parser.parse_args()

    # extract features and labels
    train_pairs = read_xml(args.train, need_labels=True)
    features = extract_features(train_pairs)
    entailment_target = np.array([pair.entailment for pair in train_pairs])
    similarity_target = np.array([pair.similarity for pair in train_pairs])

    # train models
    classifier = LogisticRegression(class_weight='balanced')
    classifier.fit(features, entailment_target)
    regressor = LinearRegression()
    regressor.fit(features, similarity_target)

    # run models
    test_pairs = read_xml(args.test, need_labels=False)
    features = extract_features(test_pairs)
    predicted_entailment = classifier.predict(features)
    predicted_similarity = regressor.predict(features)

    # write output
    tree = ET.parse(args.test)
    root = tree.getroot()
    for i in range(len(test_pairs)):
        pair = root[i]
        entailment_str = entailment_to_str[predicted_entailment[i]]
        pair.set('entailment', entailment_str)
        pair.set('similarity', str(predicted_similarity[i]))

    tree.write(args.output, 'utf-8')

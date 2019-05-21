#!/usr/bin/env python3

import argparse
import sys
import string

from nltk import FreqDist
from nltk import ChartParser
from nltk import CFG
from nltk import pos_tag
from nltk import ne_chunk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import treebank

def main():
  args = parse_args()
  text = get_text_from_args(args)

  textWithoutPunctuation = remove_punctuation(text)
  words = word_tokenize(textWithoutPunctuation)
  filteredWords = filter_stopwords(words)

  if args.frequency:
    print(get_word_frequencies(filteredWords))
  if args.sentiment:
    print(perform_sentiment_analysis(text))
  if args.syntax:
    print(display_syntax_tree(words))

def parse_args():
  parser = argparse.ArgumentParser(description='Text Analyzer 0.1')

  parser.add_argument('--file', '-f', type=argparse.FileType('r'), help='The file to analyze.')
  parser.add_argument('--sentence', '-s', help='The sentence to analyze.')
  parser.add_argument('--frequency', '-F', action='store_true', default=False, help='Prints the word frequencies of the given input. (stop words are filtered out)')
  parser.add_argument('--sentiment', '-S', action='store_true', default=False, help='Performs a sentiment analysis on the given input and prints the result.')
  parser.add_argument('--syntax', '-SY', action='store_true', default=False, help='Displays the syntax tree of the given input.')

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)

  return parser.parse_args()

def get_text_from_args(args):
  if args.file:
    return str(args.file.read())
  if args.sentence:
    return str(args.sentence)

def remove_punctuation(text):
  return text.translate(str.maketrans('', '', string.punctuation))

def filter_stopwords(words):
  englishStopWords = set(stopwords.words('english'))
  return [word for word in words if word.lower() not in englishStopWords]

def get_word_frequencies(words):
  dist = FreqDist(words)
  return dict((word, freq) for word, freq in dist.items())

def perform_sentiment_analysis(text):
  sid = SentimentIntensityAnalyzer()
  scores = sid.polarity_scores(text)
  strongest = max(scores, key=scores.get)
  sentiment = {'neg': 'Negative', 'neu': 'Neutral', 'pos': 'Positive'}

  return sentiment[strongest]

def display_syntax_tree(words):
  tagged = pos_tag(words)
  tree = ne_chunk(tagged)
  tree.draw()

def query():
  # http://www.ling.helsinki.fi/kit/2009s/clt231/NLTK/book/ch10-AnalyzingTheMeaningOfSentences.html
  return

if __name__ == "__main__":
  main()
import re
from collections import Counter, defaultdict
from nltk.tokenize import word_tokenize
import argparse
import functools
import operator
import sys
import unicodedata


parser = argparse.ArgumentParser()
parser.add_argument("corpus", type=str, help="The name of the corpus to process")
parser.add_argument("-V", "--V", type=int, help="Specify the maximum size the vocabulary can take. Unique tokens that the program sees after this number will be ignored. Default = 10000.", default=10000)

args = parser.parse_args()
basetokens = "abcdefghijklmnopqrstuvwxyz0123456789()[]+-?.,!$%^&"
EndOfWordCharacter = '_' # note this is not the standard underscore character

def build_vocab(corpus: str) -> dict:
    """Step 1. Build vocab from text corpus"""

    # Separate each char in word by space and add mark end of token
    #tokens = [" ".join(word) + " </w>" for word in corpus.split()]
    tokens = [" ".join(word) + EndOfWordCharacter for word in corpus+[basetokens]]

    # Count frequency of tokens in corpus
    vocab = Counter(tokens)  

    return vocab


def get_stats(vocab: dict) -> dict:
    """Step 2. Get counts of pairs of consecutive symbols"""

    pairs = defaultdict(int)
    for word, frequency in vocab.items():
        symbols = word.split()

        # Counting up occurrences of pairs
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += frequency

    return pairs


def merge_vocab(pair: tuple, v_in: dict) -> dict:
    """Step 3. Merge all occurrences of the most frequent pair"""
    
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    
    for word in v_in:
        # replace most frequent pair in all vocabulary
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]

    return v_out

def get_tokens_from_vocab(vocab):
    tokens_frequencies = defaultdict(int)
    vocab_tokenization = {}
    for word, freq in vocab.items():
        word_tokens = word.split()
        for token in word_tokens:
            tokens_frequencies[token] += freq
        vocab_tokenization[''.join(word_tokens)] = word_tokens
    return tokens_frequencies, vocab_tokenization

def flatten(l):
    return functools.reduce(operator.iconcat, l, [])

def bytepairtokenize(s):
  words = [word + EndOfWordCharacter for word in word_tokenize(s)]
  return flatten([m.group() for m in re.finditer(pattern, word) if m.group() != ""] for word in words)

# create vocab

with open(args.corpus+"/corpus", encoding='utf-8') as f:
    text = unicodedata.normalize('NFKD', f.read()).encode('ascii','ignore')
    print (text)
    corpus = word_tokenize(text)

vocab = build_vocab(corpus)  # Step 1

num_merges = args.V - len(basetokens)  # Hyperparameter

for i in range(num_merges):

    pairs = get_stats(vocab)  # Step 2
    
    if not pairs:
        break

    # step 3
    best = max(pairs, key=pairs.get)
    if pairs[best] == 1: # only continue merging if you have seen the sequence at least twice
        sys.stderr.write("Remaining sequence candidates have only been seen once.\n")
        break
    vocab = merge_vocab(best, vocab)
    sys.stderr.write(f"merge {i+1} of {num_merges} done.\n")

toks = [(len(tok), tok) for tok in get_tokens_from_vocab(vocab)[0].keys()]
toks.sort(reverse=True)
toks = [tok for i, tok in toks]


# write vocab

with open(args.corpus+"/vocab", "w", encoding='utf-8') as vocabfile:   
  print ("\n".join(toks), file=vocabfile)

pattern = "|".join(toks) + "|\ |\n"
pattern = re.sub("\?", "\\\?", pattern)
pattern = re.sub("\.", "\\\.", pattern)
pattern = re.sub("\[", "\\\[", pattern)
pattern = re.sub("\]", "\\\]", pattern)
pattern = re.sub("\+", "\\\+", pattern)
pattern = re.sub("\*", "\\\*", pattern)
pattern = re.sub("\(", "\\\(", pattern)
pattern = re.sub("\)", "\\\)", pattern)
pattern = re.sub("\_", "\\\_", pattern)
pattern += "|_"


for line in open(args.corpus+"/corpus").readlines():
    print (" ".join(bytepairtokenize(line)))

import re
from collections import Counter, defaultdict
from nltk.tokenize import word_tokenize
import functools
import operator
import sys
from unidecode import unidecode
import pickle

basetokens = "abcdefghijklmnopqrstuvwxyz0123456789()[]+-?.,!$%^&_"
EndOfWordCharacter = '_' # note this is not the standard underscore character
EndOfWordCharacter = '_' 

def build_vocab(corpus: str) -> dict:
    """Step 1. Build vocab from text corpus"""

    # Separate each char in word by space and add mark end of token
    #tokens = [" ".join(word) + " </w>" for word in corpus.split()]
    tokens = [EndOfWordCharacter+" ".join(word) + EndOfWordCharacter for word in corpus+[basetokens]]

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

def tokenizeword(word, pattern, vocabdict):
    if word in vocabdict:
        return vocabdict[word]
    else:
        word = EndOfWordCharacter + word + EndOfWordCharacter
        return " ".join([m.group() for m in re.finditer(pattern, word) if m.group() != ""])

def bytepairtokenize(s, pattern, vocablist):
    
    words = [tokenizeword(word, pattern, vocablist) for word in word_tokenize(s)]
    return words
  #words = [EndOfWordCharacter + word + EndOfWordCharacter for word in word_tokenize(s)]
  #return flatten([m.group() for m in re.finditer(pattern, word) if m.group() != ""] for word in words)

def learn (corpus, V):
  # create vocab

  with open(corpus+"/corpus", encoding='utf-8') as f:
      #text = unicodedata.normalize('NFKD', f.read()).encode('ascii','ignore')
      text = unidecode(f.read()).lower()
      toks = word_tokenize(text)

  vocab = build_vocab(toks)  # Step 1

  num_merges = V - len(basetokens)  # Hyperparameter

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
      if i % 100 == 0:
          sys.stderr.write(f"merge {i+1} of up to {num_merges} done.\n")

  toks = [(len(tok), tok) for tok in get_tokens_from_vocab(vocab)[0].keys()]
  toks.sort(reverse=True)
  toks = [tok for i, tok in toks]


  # write vocab

  with open(corpus+"/vocab", "w", encoding='ascii') as vocabfile:   
    print ("\n".join(toks), file=vocabfile)

  vocabdict = {}
  for word in vocab:
      vocabdict["".join(word.split())[1:-1]] = word
  with open(corpus+"/vocabpickle", "wb") as vocabpicklefile:   
      pickle.dump(vocabdict, vocabpicklefile)

  

def tokenize(corpus):

    # get vocabdict from vocabpickle file

    with open(corpus+"/vocabpickle", "rb") as vocabpicklefile:   
        vocabdict = pickle.load(vocabpicklefile)

    # get vocab from vocab file

    toks = []
    for line in open(corpus+"/vocab", "r", encoding='ascii'):   
        toks.append(line.strip())

    # create a regular expression with the toks

    pattern = "|".join(re.escape(tok) for tok in toks) + "|\ |\n"

    # should compile pattern

    # go through corpus a line at a time tokenizing


    print (f"tokenizing {corpus}/corpus => {corpus}/corpus.tok")
    with open(corpus+"/corpus.tok", "w") as corpusfile:
        for line in open(corpus+"/corpus"):
            line = unidecode(line).lower()
            print (" ".join(bytepairtokenize(line, pattern, vocabdict)), file=corpusfile)

def bpstr(l):
    result = ""
    for i in range(len(l)):
        tok = l[i]
        if tok[-1] == "_":
            if i == len(l)-1:
                tok = tok[0:-1]
            else:
                tok = tok[0:-1] + " "
        if tok[0] == "_":
            tok = tok[1:]
        result += tok
    return result


#print (bpstr(["_april_", "_is_", "_beg", "in", "ing_", "_today_"]))
#s = ['_april_', '_is_', '_the_', '_sea', 's', 'on', 'al_', '_equivalent_', '_of_', '_october_', '_in_', '_the_', '_year_', '_,_', '_and_', '_in_', '_leap_', '_years_', '_,_', '_april_', '_fin', 'ishes_', '_on_', '_the_', '_same_', '_day_', '_of_']
#print (bpstr(s))

from numpy import exp, array, zeros, ones, dot, set_printoptions, log10, argmax, clip, sqrt
from collections import Counter
from scipy.sparse import load_npz, lil_matrix, csr_matrix, csc_matrix
import shelve
from numpy.random import choice
set_printoptions(precision=2, edgeitems = 10, linewidth=200)
import warnings
from bytepair import bpstr
warnings.filterwarnings("ignore", message="Changing the sparsity structure of a csr_matrix is expensive.")

def sigmoid(v):
    try:
        return 1./(1.+exp(-v))
    except:
        print(v)

def softmax(v):
    if type(v) is lil_matrix:
        v = v.toarray()
    v = v-v.mean() # to avoid overflow errors
    es = exp(v)
    output = es/(es.sum()+0.00000001)
    return output

def sparsemax(v):
    if type(v) is lil_matrix:
        v = v.toarray()
    z = v.copy()
    z.sort()
    K = z.shape[1]
    z = z[0,::-1]
    k = 1
    total = z[0, 0]
    try:
        while k < z.shape[1] and 1 + k * z[0, k] > total:
            total += z[0, k]
            k += 1
    except Exception as e:
        print(z.shape, total.shape)
        print (e)
        exit()
    tau = (total-1)/k
    p = v-tau
    p[p<0] = 0.0
    return p

class SP():

    CowanBufferLength = 4
    MillerBufferLength = 7
    HoneyHassonBufferLength = 20
    KintschBufferLength = 150
    epsilon = 0.00000000001
    lidstone = 0.000001

    def __init__(self, corpus, constraints="X BX AX XB XA ABX XBA CX MX HX KX CI MI HI KI"):
        self.corpus = corpus
        self.constraints = set(constraints.split())  # which parameters to change when learning
        self.lam = 0.0001
        self.vocab = [w.strip() for w in open(corpus+"/vocab", encoding='utf-8').readlines()]
        self.V = len(self.vocab)
        self.I = dict([(w.strip(), i) for i, w in enumerate(self.vocab)])
        self.bigramvocab = [w.strip() for w in open(self.corpus+"/bigramvocab", encoding='utf-8').readlines()]
        self.bigramV = len(self.bigramvocab)
        self.bigramI = dict([(w.strip(), i) for i, w in enumerate(self.bigramvocab)])
        self.CountsLoaded = False
        self.initParams()
        self.loadParams()

        # init net input variables

        self.netBX = None
        self.netAX = None
        self.netXA = None
        self.netXB = None
        self.netABX = None
        self.netXBA = None
        self.netX = None
        self.netCX = None
        self.netMX = None
        self.netHX = None
        self.netKX = None
        self.netCI = None
        self.netMI = None
        self.netHI = None
        self.netKI = None

    def loadCounts(self):
        self.BX = load_npz(self.corpus+f"/BX.npz") * 1000000000
        self.X = self.BX.sum(axis=0)
        self.AX = load_npz(self.corpus+f"/AX.npz") * 1000000000
        self.ABX = load_npz(self.corpus+f"/ABX.npz") * 1000000000
        self.XBA = load_npz(self.corpus+f"/XBA.npz") * 1000000000
        self.CX = load_npz(self.corpus+f"/CX.npz") * 1000000000
        self.MX = load_npz(self.corpus+f"/MX.npz") * 1000000000
        self.HX = load_npz(self.corpus+f"/HX.npz") * 1000000000
        self.KX = load_npz(self.corpus+f"/KX.npz") * 1000000000

        self.BX.data = log10(self.BX.data+1.0)
        self.X = log10(self.X+1.0)
        self.AX.data = log10(self.AX.data+1.0)
        self.ABX.data = log10(self.ABX.data+1.0)
        self.XBA.data = log10(self.XBA.data+1.0)
        self.CX.data = log10(self.CX.data+1.0)
        self.MX.data = log10(self.MX.data+1.0)
        self.HX.data = log10(self.HX.data+1.0)
        self.KX.data = log10(self.KX.data+1.0)

        #self.BX = log10(self.BX.todense()+SP.lidstone)
        #self.X = log10(self.X+SP.lidstone)
        #self.AX = log10(self.AX.todense()+SP.lidstone)
        #self.ABX = log10(self.ABX.todense()+SP.lidstone)
        #self.XBA = log10(self.XBA.todense()+SP.lidstone)
        #self.CX = log10(self.CX.todense()+SP.lidstone)
        #self.MX = log10(self.MX.todense()+SP.lidstone)
        #self.HX = log10(self.HX.todense()+SP.lidstone)
        #self.KX = log10(self.KX.todense()+SP.lidstone)
        self.CountsLoaded = True

    def initParams(self):
        self.Gax = 1.0
        self.Gbx = 1.0
        self.Gxa = 1.0
        self.Gxb = 1.0
        self.Gci = -1.0 # Cowan scale inhibition
        self.Gmi = -1.0 # Miller scale inhibition
        self.Ghi = -1.0 # Honey Hasson scale inhibition
        self.Gki = -1.0 # Kintsch scale inhibition
        self.Gx = -1.0
        self.Gabx = 1.0
        self.Gxba = 1.0
        #self.Gcx = ones((self.V, 1)) / self.CowanBufferLength
        #self.Gmx = ones((self.V, 1)) / self.MillerBufferLength
        #self.Ghx = ones((self.V, 1)) / self.HoneyHassonBufferLength
        #self.Gkx = ones((self.V, 1)) / self.KintschBufferLength
        self.Gcx = 1.0
        self.Gmx = 1.0
        self.Ghx = 1.0
        self.Gkx = 1.0

    def strVec(self, v, NumToReport=14, NoZeros = False):
  
      if type(v) is lil_matrix or type(v) is csr_matrix or type(v) is csc_matrix:
          v = v.toarray()
      v = array(v)
      v2 = v.flatten()
      c = Counter(dict(enumerate(v2[0:len(self.vocab)])))
      res = ""
      for i,n in c.most_common(int(NumToReport/2)):
          if n != 0 or not NoZeros:
              res += "{0} {1:1.3f} ".format(self.vocab[i], n)
      res += " ... "
      for i,n in c.most_common()[:-int(NumToReport/2)-1:-1][::-1]:
          if n != 0 or not NoZeros:
              res += "{0} {1:1.3f} ".format(self.vocab[i], n)
      return res

    def saveParams(self):

        params = shelve.open(self.corpus+"/params")
        params["Gbx"] = self.Gbx
        params["Gax"] = self.Gax
        params["Gxb"] = self.Gxb
        params["Gxa"] = self.Gxa
        params["Gabx"] = self.Gabx
        params["Gxba"] = self.Gxba
        params["Gcx"] = self.Gcx
        params["Gmx"] = self.Gmx
        params["Ghx"] = self.Ghx
        params["Gkx"] = self.Gkx
        params["Gci"] = self.Gci
        params["Gmi"] = self.Gmi
        params["Ghi"] = self.Ghi
        params["Gki"] = self.Gki
        params["Gx"] = self.Gx
        params.close()

    def loadParams(self):

        params = shelve.open(self.corpus+"/params")
        if "Gax" in params.keys():
            self.Gax = params["Gax"]
        if "Gbx" in params.keys():
            self.Gbx = params["Gbx"]
        if "Gxa" in params.keys():
            self.Gxa = params["Gxa"]
        if "Gxb" in params.keys():
            self.Gxb = params["Gxb"]
        if "Gx" in params.keys():
            self.Gx = params["Gx"]
        if "Gabx" in params.keys():
            self.Gabx = params["Gabx"]
        if "Gxba" in params.keys():
            self.Gxba = params["Gxba"]
        if "Gcx" in params.keys():
            self.Gcx = params["Gcx"]
        if "Gmx" in params.keys():
            self.Gmx = params["Gmx"]
        if "Gci" in params.keys():
            self.Gci = params["Gci"]
        if "Gmi" in params.keys():
            self.Gmi = params["Gmi"]
        if "Ghi" in params.keys():
            self.Ghi = params["Ghi"]
        if "Gki" in params.keys():
            self.Gki = params["Gki"]
        if "Ghx" in params.keys():
            self.Ghx = params["Ghx"]
        if "Gkx" in params.keys():
            self.Gkx = params["Gkx"]
        params.close()
        if "X" not in self.constraints: self.Gx = 0.0
        if "BX" not in self.constraints: self.Gbx = 0.0
        if "AX" not in self.constraints: self.Gax = 0.0
        if "ABX" not in self.constraints: self.Gabx = 0.0
        if "XB" not in self.constraints: self.Gxb = 0.0
        if "XA" not in self.constraints: self.Gxa = 0.0
        if "XBA" not in self.constraints: self.Gxba = 0.0
        if "CX" not in self.constraints: self.Gcx = 0.0
        if "MX" not in self.constraints: self.Gmx = 0.0
        if "HX" not in self.constraints: self.Ghx = 0.0
        if "KX" not in self.constraints: self.Gkx = 0.0
        if "CI" not in self.constraints: self.Gci = 0.0
        if "MI" not in self.constraints: self.Gmi = 0.0
        if "HI" not in self.constraints: self.Ghi = 0.0
        if "KI" not in self.constraints: self.Gki = 0.0

    def strParams(self, NumToReport=15):
        result =  f"Gabx = {self.Gabx:1.3f} Gxba = {self.Gxba:1.3f} Gax = {self.Gax:1.3f} Gbx = {self.Gbx:1.3f} Gxa = {self.Gxa:1.3f} Gxb = {self.Gxb:1.3f} Gx = {self.Gx:1.3f}\n"
        result += f"Gci = {self.Gci:1.3f} Gmi = {self.Gmi:1.3f} Ghi = {self.Ghi:1.3f} Gki = {self.Gki:1.3f}\n"
        result +=  f"Gcx = {self.Gcx:1.3f} Gmx = {self.Gmx:1.3f} Ghx = {self.Ghx:1.3f} Gkx = {self.Gkx:1.3f}\n"
        return result

    def strNets(self, cowan, miller, honeyhasson, kintsch):
        t = zeros((1,self.V)) # temporary storage of attention weights
        result = "Weighted Nets:\n"
        result += f"all: {self.strVec(self.net)}\n\n"
        result += f"X({self.Gx:1.3f}): {self.strVec(self.Gx * self.netX, NoZeros = True)}\n"
        result += f"BX({self.Gbx:1.3f}): {self.strVec(self.Gbx * self.netBX)}\n"
        if self.netXB is not None: 
            result += f"XB({self.Gxb:1.3f}): {self.strVec(self.Gxb * self.netXB)}\n"
        result += f"AX({self.Gax:1.3f}): {self.strVec(self.Gax * self.netAX)}\n"
        if self.netXA is not None: 
            result += f"XA({self.Gxa:1.3f}): {self.strVec(self.Gxa * self.netXA)}\n"
        if self.netABX is not None: 
            result += f"ABX({self.Gabx:1.3f}): {self.strVec(self.Gabx * self.netABX)}\n"
        if self.netXBA is not None: 
            result += f"XBA({self.Gxba:1.3f}): {self.strVec(self.Gxba * self.netXBA)}\n"
        result += "\n"
        result += f"CX({self.Gcx:1.3f}): {self.strVec(self.Gcx * self.netCX)}\n"
        result += f"MX({self.Gmx:1.3f}): {self.strVec(self.Gmx * self.netMX)}\n"
        result += f"HX({self.Ghx:1.3f}): {self.strVec(self.Ghx * self.netHX)}\n"
        result += f"KX({self.Gkx:1.3f}): {self.strVec(self.Gkx * self.netKX)}\n"

        result += "\n"
        result += f"CI({self.Gci:1.3f}): {self.strVec(self.Gci * self.netCI, NoZeros = True)}\n"
        result += f"MI({self.Gmi:1.3f}): {self.strVec(self.Gmi * self.netMI, NoZeros = True)}\n"
        result += f"HI({self.Ghi:1.3f}): {self.strVec(self.Ghi * self.netHI, NoZeros = True)}\n"
        result += f"KI({self.Gki:1.3f}): {self.strVec(self.Gki * self.netKI, NoZeros = True)}\n"

        # attention weights
        result += "\n"
        result += "Attention Weights:\n"

        if len(cowan) > 0:
            attention = (sparsemax(-self.X[:, cowan]))
            t[:,:] = 0.0
            t[:, cowan] = attention
            result += "Cowan: " + self.strVec(t, NoZeros = True) + "\n"
        if len(miller) > 0:
            attention = (sparsemax(-self.X[:, miller]))
            t[:,:] = 0.0
            t[:, miller] = attention
            result += "Miller: " + self.strVec(t, NoZeros = True) + "\n"
        if len(honeyhasson) > 0:
            attention = (sparsemax(-self.X[:, honeyhasson]))
            t[:,:] = 0.0
            t[:, honeyhasson] = attention
            result += "HoneyHasson: " + self.strVec(t, NoZeros = True) + "\n"
        if len(kintsch) > 0:
            attention = (sparsemax(-self.X[:, kintsch]))
            t[:,:] = 0.0
            t[:, kintsch] = attention
            result += "Kintsch: " + self.strVec(t, NoZeros = True) + "\n"
        return result

    def select(self, words, vec):
        if type(vec) == csr_matrix:
            vec = vec.todense()
        wordvec = zeros((1,self.V))
        wordvec[0, words] = vec[0,words]
        return wordvec

    def trace(self, words, cowan, miller, honeyhasson, kintsch):
        result = "Weighted Nets:\n"
        result += f"all: {self.strVec(self.select(words, self.net), NoZeros=True)}\n\n"
        result += f"X({self.Gx:1.3f}): {self.strVec(self.select(words, self.Gx * self.netX), NoZeros = True)}\n"
        result += f"BX({self.Gbx:1.3f}): {self.strVec(self.select(words, self.Gbx * self.netBX), NoZeros=True)}\n"
        if self.netXB is not None: 
            result += f"XB({self.Gxb:1.3f}): {self.strVec(self.select(words, self.Gxb * self.netXB), NoZeros=True)}\n"
        result += f"AX({self.Gax:1.3f}): {self.strVec(self.select(words, self.Gax * self.netAX), NoZeros=True)}\n"
        if self.netXA is not None: 
            result += f"XA({self.Gxa:1.3f}): {self.strVec(self.select(words, self.Gxa * self.netXA), NoZeros=True)}\n"
        if self.netABX is not None: 
            result += f"ABX({self.Gabx:1.3f}): {self.strVec(self.select(words, self.Gabx * self.netABX), NoZeros=True)}\n"
        if self.netXBA is not None: 
            result += f"XBA({self.Gxba:1.3f}): {sel.strVec(self.select(words, self.Gxba * self.netXBA), NoZeros=True)}\n"
        result += "\n"
        result += f"CX({self.Gcx:1.3f}): {self.strVec(self.select(words, self.Gcx * self.netCX), NoZeros=True)}\n"
        result += f"MX({self.Gmx:1.3f}): {self.strVec(self.select(words, self.Gmx * self.netMX), NoZeros=True)}\n"
        result += f"HX({self.Ghx:1.3f}): {self.strVec(self.select(words, self.Ghx * self.netHX), NoZeros=True)}\n"
        result += f"KX({self.Gkx:1.3f}): {self.strVec(self.select(words, self.Gkx * self.netKX), NoZeros=True)}\n"

        result += "\n"
        result += f"CI({self.Gci:1.3f}): {self.strVec(self.select(words, self.Gci * self.netCI), NoZeros = True)}\n"
        result += f"MI({self.Gmi:1.3f}): {self.strVec(self.select(words, self.Gmi * self.netMI), NoZeros = True)}\n"
        result += f"HI({self.Ghi:1.3f}): {self.strVec(self.select(words, self.Ghi * self.netHI), NoZeros = True)}\n"
        result += f"KI({self.Gki:1.3f}): {self.strVec(self.select(words, self.Gki * self.netKI), NoZeros = True)}\n"

        # attention weights

        t = zeros((1,self.V)) # temporary storage of attention weights
        result += "\n"
        result += "Attention Weights:\n"

        if len(cowan) > 0:
            attention = (sparsemax(-self.X[:, cowan]))
            t[:,:] = 0.0
            t[:, cowan] = attention
            result += "Cowan: " + self.strVec(t, NoZeros = True) + "\n"
        if len(miller) > 0:
            attention = (sparsemax(-self.X[:, miller]))
            t[:,:] = 0.0
            t[:, miller] = attention
            result += "Miller: " + self.strVec(t, NoZeros = True) + "\n"
        if len(honeyhasson) > 0:
            attention = (sparsemax(-self.X[:, honeyhasson]))
            t[:,:] = 0.0
            t[:, honeyhasson] = attention
            result += "HoneyHasson: " + self.strVec(t, NoZeros = True) + "\n"
        if len(kintsch) > 0:
            attention = (sparsemax(-self.X[:, kintsch]))
            t[:,:] = 0.0
            t[:, kintsch] = attention
            result += "Kintsch: " + self.strVec(t, NoZeros = True) + "\n"
        return result

    def prob(self, a, b, bafter, aafter, cowan, miller, honeyhasson, kintsch):

        # lazy load counts and do log transform
        if not self.CountsLoaded:
            self.loadCounts()

        abkey = self.vocab[a]+"_"+self.vocab[b]
        if abkey in self.bigramI:
            ab = self.bigramI[abkey]
        else:
            ab = -1
        if bafter != -1 and aafter != -1:
            bakey = self.vocab[bafter]+"_"+self.vocab[aafter]
            if bakey in self.bigramI:
                baafter = self.bigramI[bakey]
            else:
                baafter = -1
        else:
            baafter = -1
    
        self.netAX = self.AX[a,:]
        self.netBX = self.BX[b,:]
        if aafter != -1:
            self.netXA = self.AX.T[aafter,:]
        if bafter != -1:
            self.netXB = self.BX.T[bafter,:]
        if ab != -1:
            self.netABX = self.ABX[ab,:]
        if baafter != -1:
            self.netXBA = self.XBA[baafter,:]
        self.netX = self.X

        if len(cowan) > 0:
            attention = (sparsemax(-self.X[:, cowan]))
            self.netCX = attention * self.CX[cowan,:]
        else:
            self.netCX = zeros((1,self.V))
        if len(miller) > 0:
            attention = (sparsemax(-self.X[:, miller]))
            self.netMX = attention * self.MX[miller,:]
        else:
            self.netMX = zeros((1,self.V))
        if len(honeyhasson) > 0:
            attention = (sparsemax(-self.X[:, honeyhasson]))
            self.netHX = attention * self.HX[honeyhasson,:]
        else:
            self.netHX = zeros((1,self.V))
        if len(kintsch) > 0:
            attention = (sparsemax(-self.X[:, kintsch]))
            self.netKX = attention * self.KX[kintsch,:]
        else:
            self.netKX = zeros((1,self.V))
        #self.netMX = self.MX[miller,:]
        #self.netHX = self.HX[honeyhasson,:]
        #self.netKX = self.KX[kintsch,:]

        self.netCI = csr_matrix((1, self.V))
        self.netMI = csr_matrix((1, self.V))
        self.netHI = csr_matrix((1, self.V))
        self.netKI = csr_matrix((1, self.V))
        self.netCI[0, cowan] = 1.
        self.netMI[0, miller] = 1.
        self.netHI[0, honeyhasson] = 1.
        self.netKI[0, kintsch] = 1.

        self.net = self.Gax * self.netAX 
        self.net += self.Gbx * self.netBX
        self.net += self.Gx * self.netX
        self.net += self.Gcx * self.netCX
        self.net += self.Gci * self.netCI
        self.net += self.Gmx * self.netMX
        self.net += self.Gmi * self.netMI
        self.net += self.Ghx * self.netHX
        self.net += self.Ghi * self.netHI
        #if len(honeyhasson) > 0:
        #    v = zeros((1, self.V))
        #    attention = (sparsemax(-self.X[:, honeyhasson]))
        #    for j in range(len(honeyhasson)):
        #        v[0, honeyhasson[j]] += attention[0, j]
        #    print (self.strVec(v))
        self.net += self.Gkx * self.netKX
        #if len(kintsch) > 0:
        #    v = zeros((1, self.V))
        #    attention = (sparsemax(-self.X[:, kintsch]))
        #    for j in range(len(kintsch)):
        #        v[0, kintsch[j]] += attention[0, j]
        #    print (self.strVec(v))
        self.net += self.Gki * self.netKI
        if ab != -1:
            self.net += self.Gabx * self.netABX
        if aafter != -1:
            self.net += self.Gxa * self.netXA
        if bafter != -1:
            self.net += self.Gxb * self.netXB
        if baafter != -1:
            self.net +=  self.Gxba * self.netXBA

        s = sparsemax(self.net)
        return s

    def samplerGibbs(self, prefix, BufferLength = 8, Threshold = 6, stopSampler = 1000):
        buffer = ones(BufferLength, int) * -1
        bufferprobs = zeros(BufferLength)
        for i in range(len(prefix)):
            buffer[i] = self.I[prefix[i]]
        counts = Counter()
        iteration = 0
        while len(counts) == 0 or counts.most_common(1)[0][1] < Threshold and iteration < stopSampler:
            for j in range(len(prefix), len(buffer)):
                cowan = buffer[max(0, j - SP.CowanBufferLength):j]
                miller = buffer[max(0, j - SP.MillerBufferLength):j]
                honeyhasson = buffer[max(0, j - SP.HoneyHassonBufferLength):j]
                kintsch = buffer[max(0, j - SP.KintschBufferLength):j]
                if j == len(buffer)-2:
                    ps = array(self.prob(buffer[j-2], buffer[j-1], buffer[j+1], -1, cowan, miller, honeyhasson, kintsch))
                elif j == len(buffer)-1:
                    ps = array(self.prob(buffer[j-2], buffer[j-1], -1, -1, cowan, miller, honeyhasson, kintsch))
                else:
                    ps = array(self.prob(buffer[j-2], buffer[j-1], buffer[j+1], buffer[j+2], cowan, miller, honeyhasson, kintsch))
                c = choice(range(self.V), 1, p=ps.ravel())[0]
                buffer[j] = c
                bufferprobs[j] = ps[0, c]
            key = " ".join(self.vocab[w] for w in buffer)
            print (" ".join(bpstr([self.vocab[buffer[k]]]) if k < len(prefix) else f"{bpstr([self.vocab[buffer[k]]])} ({bufferprobs[k]:1.2f})" for k in range(len(buffer))))
            counts[key] += 1
            iteration += 1
        if iteration >= stopSampler:
            return None
        else:
            return counts.most_common(1)[0][0]

    def sampler(self, prefix, BufferLength = 8):
        buffer = ones(BufferLength, int) * -1
        bufferprobs = zeros(BufferLength)
        for i in range(len(prefix)):
            buffer[i] = self.I[prefix[i]]
        for j in range(len(prefix), len(buffer)):
            cowan = buffer[max(0, j - SP.CowanBufferLength):j]
            miller = buffer[max(0, j - SP.MillerBufferLength):j]
            honeyhasson = buffer[max(0, j - SP.HoneyHassonBufferLength):j]
            kintsch = buffer[max(0, j - SP.KintschBufferLength):j]
            if j == len(buffer)-2:
                ps = array(self.prob(buffer[j-2], buffer[j-1], buffer[j+1], -1, cowan, miller, honeyhasson, kintsch))
            elif j == len(buffer)-1:
                ps = array(self.prob(buffer[j-2], buffer[j-1], -1, -1, cowan, miller, honeyhasson, kintsch))
            else:
                ps = array(self.prob(buffer[j-2], buffer[j-1], buffer[j+1], buffer[j+2], cowan, miller, honeyhasson, kintsch))
            c = argmax(ps.ravel())
            buffer[j] = c
            #bufferprobs[j] = ps[0, c]
            #key = " ".join(self.vocab[w] for w in buffer)
            #print (" ".join(bpstr([self.vocab[buffer[k]]]) if k < len(prefix) else f"{bpstr([self.vocab[buffer[k]]])} ({bufferprobs[k]:1.2f})" for k in range(len(buffer))))
        return buffer



    def learnOnePass (self, changeweights = True, verbose=False, regularise=False):
        se = 0.0
        sa = 0.0
        count = 0
        correct = 0
        for i in range(len(self.words)-4):
            cowan = self.words[max(0, i - SP.CowanBufferLength):i]
            miller = self.words[max(0, i - SP.MillerBufferLength):i]
            honeyhasson = self.words[max(0, i - SP.HoneyHassonBufferLength):i]
            kintsch = self.words[max(0, i - SP.KintschBufferLength):i]
            output = self.prob(self.words[i], self.words[i+1], self.words[i+3], self.words[i+4], cowan, miller, honeyhasson, kintsch)
            c = self.words[i+2]
          
            delta = -output 
            delta [0,c] += 1
            se += dot(delta, delta.T)[0,0]
            sa += dot(output, output.T)[0,0]
            if argmax(output) == c:
                correct += 1
            count += 1
            if verbose:
                print (f"{dot(delta, delta.T)[0,0]:1.3f} {self.vocab[self.words[i+2]]}: {self.strVec(output, NoZeros=True)}")

            if changeweights:
                delta = csr_matrix(delta)
                if "BX" in self.constraints:
                    self.Gbx += self.lam * dot(delta, self.netBX.T)[0,0]
                    if regularise: self.Gbx -= self.lam * self.Gbx
                if "AX" in self.constraints:
                    self.Gax += self.lam * dot(delta, self.netAX.T)[0,0]
                    if regularise: self.Gax -= self.lam * self.Gax
                if "XA" in self.constraints:
                    self.Gxa += self.lam * dot(delta, self.netXA.T)[0,0]
                    if regularise: self.Gxa -= self.lam * self.Gxa
                if "XB" in self.constraints:
                    self.Gxb += self.lam * dot(delta, self.netXB.T)[0,0]
                    if regularise: self.Gxb -= self.lam * self.Gxb
                if self.netABX is not None:
                    if "ABX" in self.constraints:
                        self.Gabx += self.lam * dot(delta, self.netABX.T)[0,0]   
                        if regularise: self.Gabx -= self.lam * self.Gabx
                if self.netXBA is not None:
                    if "XBA" in self.constraints:
                        self.Gxba += self.lam * dot(delta, self.netXBA.T)[0,0]
                        if regularise: self.Gxba -= self.lam * self.Gxba
                if len(cowan) > 0:
                    if "CX" in self.constraints:
                        self.Gcx += self.lam * dot(self.netCX, delta.T.todense())[0,0]
                        if regularise: self.Gcx -= self.lam * self.Gcx
                    if "CI" in self.constraints:
                        self.Gci += self.lam * dot(self.netCI, delta.T)[0,0]
                        if regularise: self.Gci -= self.lam * self.Gci
                if len(miller) > 0:
                    if "MX" in self.constraints:
                        self.Gmx += self.lam * dot(self.netMX, delta.T.todense())[0,0]
                        if regularise: self.Gmx -= self.lam * self.Gmx
                    if "MI" in self.constraints:
                        self.Gmi += self.lam * dot(self.netMI, delta.T)[0,0]
                        if regularise: self.Gmi -= self.lam * self.Gmi
                if len(honeyhasson) > 0:
                    if "HX" in self.constraints:
                        self.Ghx += self.lam * dot(self.netHX, delta.T.todense())[0,0]
                        if regularise: self.Gkx -= self.lam * self.Gkx
                    if "HI" in self.constraints:
                        self.Ghi += self.lam * dot(self.netHI, delta.T)[0,0]
                        if regularise: self.Ghi -= self.lam * self.Ghi
                if len(kintsch) > 0:
                    if "KX" in self.constraints:
                        self.Gkx += self.lam * dot(self.netKX, delta.T.todense())[0,0]
                        if regularise: self.Gkx -= self.lam * self.Gkx
                    if "KI" in self.constraints:
                        self.Gki += self.lam * dot(self.netKI, delta.T)[0,0]
                        if regularise: self.Gki -= self.lam * self.Gki

                if "X" in self.constraints:
                    self.Gx += self.lam * (delta * self.netX.T)[0,0]
                    if regularise: self.Gx -= self.lam * self.Gxa

        return sqrt(se/count), sqrt(sa/count), correct/count
    

    def learn(self, NumberOfIterations, CorpusSize):

        self.words = []
        with open(self.corpus+"/corpus.tok", encoding='ascii') as f:
            for line in f:
                self.words += [self.I[w] for w in line.lower().split()]
                if len(self.words) >= CorpusSize:
                    break
        self.words = self.words[0:CorpusSize]

        for iteration in range(NumberOfIterations):
            rmse, rmsa, correct = self.learnOnePass()
            print (f"{iteration+1}/{NumberOfIterations}: rmse: {rmse:1.4f} rmsa: {rmsa:1.4f} correct: {correct:1.2f} {self.strParams()}", flush=True)
            self.saveParams()

    def test(self, size=None, verbose=False):

        with open(self.corpus+"/corpus.tok", encoding='ascii') as f:
            self.words = [self.I[w] for w in f.read().lower().split()]

        if size:
            self.words = self.words[0:size]
        rmse, rmsa, correct = self.learnOnePass(changeweights = False, verbose=verbose)   
        print (self.strParams())
        print (f"rmse: {rmse:1.4f} rmsa:{rmsa:1.4f} correct:{correct:1.2f}")

if __name__ == "__main__":
    s = SP("small")
    a = s.I["australia"]
    b = s.I["is"]
    aafter = -1
    bafter = -1
    cowan = [s.I[w] for w in "australia is".split()]
    miller = cowan
    honeyhasson = cowan
    kintsch = cowan
    ps = s.prob(a, b, bafter, aafter, cowan, miller, honeyhasson, kintsch)
    print (s.strVec(ps))

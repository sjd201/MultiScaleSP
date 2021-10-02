from numpy import exp, array, zeros, ones, dot, set_printoptions, log10, argmax, clip, sqrt
from collections import Counter
from scipy.sparse import load_npz, lil_matrix, csr_matrix
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
        self.BX = load_npz(self.corpus+f"/BX.npz") * 10000000
        self.X = self.BX.sum(axis=0)
        self.AX = load_npz(self.corpus+f"/AX.npz") * 10000000
        self.ABX = load_npz(self.corpus+f"/ABX.npz") * 10000000
        self.XBA = load_npz(self.corpus+f"/XBA.npz") * 10000000
        self.CX = load_npz(self.corpus+f"/CX.npz") * 10000000
        self.MX = load_npz(self.corpus+f"/MX.npz") * 10000000
        self.HX = load_npz(self.corpus+f"/HX.npz") * 10000000
        self.KX = load_npz(self.corpus+f"/KX.npz") * 10000000

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
        self.Gcx = ones((self.V, 1)) / self.CowanBufferLength
        self.Gmx = ones((self.V, 1)) / self.MillerBufferLength
        self.Ghx = ones((self.V, 1)) / self.HoneyHassonBufferLength
        self.Gkx = ones((self.V, 1)) / self.KintschBufferLength

    def strVec(self, v, NumToReport=14):
  
      if type(v) is lil_matrix or type(v) is csr_matrix:
          v = v.toarray()
      v = array(v)
      v2 = v.flatten()
      c = Counter(dict(enumerate(v2[0:len(self.vocab)])))
      res = ""
      for i,n in c.most_common(int(NumToReport/2)):
          if n != 0.0:
              res += "{0} {1:1.3f} ".format(self.vocab[i], n)
      res += " ... "
      for i,n in c.most_common()[:-int(NumToReport/2)-1:-1][::-1]:
          if n != 0.0:
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

    def strParams(self, NumToReport=15):
        result =  f"Gabx = {self.Gabx:1.3f} Gxba = {self.Gxba:1.3f} Gax = {self.Gax:1.3f} Gbx = {self.Gbx:1.3f} Gxa = {self.Gxa:1.3f} Gxb = {self.Gxb:1.3f} Gx = {self.Gx:1.3f}\n"
        result += f"Gci = {self.Gci:1.3f} Gmi = {self.Gmi:1.3f} Ghi = {self.Ghi:1.3f} Gki = {self.Gki:1.3f}\n"
        result +=  f"Gcx = {self.strVec(self.Gcx, NumToReport)}\n"
        result +=  f"Gmx = {self.strVec(self.Gmx, NumToReport)}\n"
        result +=  f"Ghx = {self.strVec(self.Ghx, NumToReport)}\n"
        result +=  f"Gkx = {self.strVec(self.Gkx, NumToReport)}\n"
        return result

    def strNets(self, cowan, miller, honeyhasson, kintsch):
        result = "Weighted Nets:\n"
        result += f"all: {self.strVec(self.net)}\n"
        #result += f"AX: {self.strVec(self.netAX)}\n"
        result += f"AX: {self.strVec(self.Gax * self.netAX)}\n"
        #result += f"BX: {self.strVec(self.netBX)}\n"
        result += f"BX: {self.strVec(self.Gbx * self.netBX)}\n"
        #result += f"ABX: {self.strVec(self.netABX)}\n"
        if self.netABX is not None: 
            result += f"ABX: {self.strVec(self.Gabx * self.netABX)}\n"
        #for i, w in enumerate(cowan):
        #    #result += f"CX({self.vocab[w]}): {self.strVec(self.netCX[i])}\n"
        #    result += f"weighted CX({self.vocab[w]}): {self.strVec(self.Gcx[w, 0] * self.netCX[i, :])}\n"
        #for i, w in enumerate(miller):
        #    #result += f"MX({self.vocab[w]}): {self.strVec(self.netMX[i])}\n"
        #    result += f"weighted MX({self.vocab[w]}): {self.strVec(self.Gmx[w, 0] * self.netMX[i, :])}\n"
        #for i, w in enumerate(kintsch):
        #    #result += f"KX({self.vocab[w]}): {self.strVec(self.netKX[i])}\n"
        #    result += f"weighted KX({self.vocab[w]}): {self.strVec(self.Gkx[w, 0] * self.netKX[i, :])}\n"
        result += f"CX: {self.strVec(self.Gcx[cowan, 0] * self.netCX)}\n"
        result += f"MX: {self.strVec(self.Gmx[miller, 0] * self.netMX)}\n"
        result += f"HX: {self.strVec(self.Ghx[honeyhasson, 0] * self.netHX)}\n"
        result += f"KX: {self.strVec(self.Gkx[kintsch, 0] * self.netKX)}\n"

        result += f"CI: {self.strVec(self.Gci * self.netCI)}\n"
        result += f"MI: {self.strVec(self.Gmi * self.netMI)}\n"
        result += f"HI: {self.strVec(self.Ghi * self.netHI)}\n"
        result += f"KI: {self.strVec(self.Gki * self.netKI)}\n"
        if self.netXBA is not None: 
        #    result += f"XBA: {self.strVec(self.netXBA)}\n"
            result += f"XBA: {self.strVec(self.Gxba * self.netXBA)}\n"
        #result += f"X: {self.strVec(self.netX)}\n"
        result += f"X: {self.strVec(self.Gx * self.netX)}\n"
        if self.netXA is not None: 
        #    result += f"XA: {self.strVec(self.netXA)}\n"
            result += f"XA: {self.strVec(self.Gxa * self.netXA)}\n"
        if self.netXB is not None: 
        #    result += f"XB: {self.strVec(self.netXB)}\n"
            result += f"XB: {self.strVec(self.Gxb * self.netXB)}\n"
        # show weights
        result += "\nWeights\n"
        cowanweights = " ".join(f"{self.vocab[i]}: {self.Gcx[i, 0]:1.3f}" for i in cowan)
        result += f"Cowan: {cowanweights}\n"
        millerweights = " ".join(f"{self.vocab[i]}: {self.Gmx[i, 0]:1.3f}" for i in miller)
        result += f"Miller: {millerweights}\n"
        honeyhassonweights = " ".join(f"{self.vocab[i]}: {self.Ghx[i, 0]:1.3f}" for i in honeyhasson)
        result += f"HoneyHasson: {honeyhassonweights}\n"
        kintschweights = " ".join(f"{self.vocab[i]}: {self.Gkx[i, 0]:1.3f}" for i in kintsch)
        result += f"Kintsch: {kintschweights}\n"
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

        self.netCX = self.CX[cowan,:]
        self.netMX = self.MX[miller,:]
        self.netHX = self.HX[honeyhasson,:]
        self.netKX = self.KX[kintsch,:]

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
        #self.net += self.Gcx[cowan, 0] * self.netCX
        self.net += self.Gci * self.netCI
        #self.net += self.Gmx[miller, 0] * self.netMX
        self.net += self.Gmi * self.netMI
        #self.net += self.Ghx[honeyhasson, 0] * self.netHX
        self.net += self.Ghi * self.netHI
        if len(honeyhasson) > 0:
            v = zeros((1, self.V))
            attention = (sparsemax(-self.X[:, honeyhasson]))
            for j in range(len(honeyhasson)):
                v[0, honeyhasson[j]] += attention[0, j]
            print (self.strVec(v))
        #self.net += self.Gkx[kintsch, 0] * self.netKX
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

    def samplerGibbs(self, prefix, BufferLength = 8, Threshold = 6):
        buffer = ones(BufferLength, int) * -1
        bufferprobs = zeros(BufferLength)
        for i in range(len(prefix)):
            buffer[i] = self.I[prefix[i]]
        counts = Counter()
        while len(counts) == 0 or counts.most_common(1)[0][1] < Threshold:
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
            print (key)
            print (" ".join(bpstr([self.vocab[buffer[k]]]) if k < len(prefix) else f"{bpstr([self.vocab[buffer[k]]])} ({bufferprobs[k]:1.2f})" for k in range(len(buffer))))
            counts[key] += 1
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



    def learnOnePass (self, changeweights = True, verbose=False):
        se = 0.0
        sa = 0.0
        count = 0
        for i in range(len(self.words)-4):
            cowan = self.words[max(0, i - SP.CowanBufferLength):i]
            miller = self.words[max(0, i - SP.MillerBufferLength):i]
            honeyhasson = self.words[max(0, i - SP.HoneyHassonBufferLength):i]
            kintsch = self.words[max(0, i - SP.KintschBufferLength):i]
            output = self.prob(self.words[i], self.words[i+1], self.words[i+3], self.words[i+4], cowan, miller, honeyhasson, kintsch)
            if verbose:
                print (self.vocab[self.words[i+2]], ": ", self.strVec(output))
            c = self.words[i+2]
          
            delta = -output 
            delta [0,c] += 1
            se += dot(delta, delta.T)[0,0]
            sa += dot(output, output.T)[0,0]
            count += 1

            if changeweights:
                delta = csr_matrix(delta)
                if "BX" in self.constraints:
                    self.Gbx += self.lam * dot(delta, self.netBX.T)[0,0]
                if "AX" in self.constraints:
                    self.Gax += self.lam * dot(delta, self.netAX.T)[0,0]
                if "XA" in self.constraints:
                    self.Gxa += self.lam * dot(delta, self.netXA.T)[0,0]
                if "XB" in self.constraints:
                    self.Gxb += self.lam * dot(delta, self.netXB.T)[0,0]
                if self.netABX is not None:
                    if "ABX" in self.constraints:
                        self.Gabx += self.lam * dot(delta, self.netABX.T)[0,0]   
                if self.netXBA is not None:
                    if "XBA" in self.constraints:
                        self.Gxba += self.lam * dot(delta, self.netXBA.T)[0,0]
                if len(cowan) > 0:
                    if "CX" in self.constraints:
                        self.Gcx[cowan] += self.lam * dot(self.netCX, delta.T)
                        #for i in cowan:
                        #    if self.Gcx[i] < 0.0:
                        #        self.Gcx[i] = 0.0
                    if "CI" in self.constraints:
                        self.Gci += self.lam * dot(self.netCI, delta.T)[0,0]
                        #if self.Gci > 0.0:
                        #    self.Gci = 0.0
                if len(miller) > 0:
                    if "MX" in self.constraints:
                        self.Gmx[miller] += self.lam * dot(self.netMX, delta.T)
                        #for i in miller:
                        #    if self.Gmx[i] < 0.0:
                        #        self.Gmx[i] = 0.0
                    if "MI" in self.constraints:
                        self.Gmi += self.lam * dot(self.netMI, delta.T)[0,0]
                        #if self.Gmi > 0.0:
                        #    self.Gmi = 0.0
                if len(honeyhasson) > 0:
                    if "HX" in self.constraints:
                        self.Ghx[honeyhasson] += self.lam * dot(self.netHX, delta.T)
                        #for i in honeyhasson:
                        #    if self.Ghx[i] < 0.0:
                        #        self.Ghx[i] = 0.0
                    if "HI" in self.constraints:
                        self.Ghi += self.lam * dot(self.netHI, delta.T)[0,0]
                        #if self.Ghi > 0.0:
                        #    self.Ghi = 0.0
                if len(kintsch) > 0:
                    if "KX" in self.constraints:
                        self.Gkx[kintsch] += self.lam * dot(self.netKX, delta.T)
                        #for i in kintsch:
                        #    if self.Gkx[i] < 0.0:
                        #        self.Gkx[i] = 0.0
                    if "KI" in self.constraints:
                        self.Gki += self.lam * dot(self.netKI, delta.T)[0,0]
                        #if self.Gki > 0.0:
                        #    self.Gki = 0.0
                #self.netX = csr_matrix(self.netX)

                if "X" in self.constraints:
                    self.Gx += self.lam * (delta * self.netX.T)[0,0]
        return sqrt(se/count), sqrt(sa/count)
    

    def learn(self, NumberOfIterations, CorpusSize):

        self.words = []
        with open(self.corpus+"/corpus.tok", encoding='ascii') as f:
            for line in f:
                self.words += [self.I[w] for w in line.lower().split()]
                if len(self.words) >= CorpusSize:
                    break
        self.words = self.words[0:CorpusSize]

        for iteration in range(NumberOfIterations):
            rmse, rmsa = self.learnOnePass()
            print (f"{iteration+1}/{NumberOfIterations}: rmse: {rmse:1.4f} rmsa: {rmsa:1.4f} {self.strParams()}")
            self.saveParams()

    def test(self, size=None, verbose=False):

        with open(self.corpus+"/corpus.tok", encoding='ascii') as f:
            self.words = [self.I[w] for w in f.read().lower().split()]

        if size:
            self.words = self.words[0:size]
        rmse, rmsa = self.learnOnePass(changeweights = False, verbose=verbose)   
        print (self.strParams())
        print (f"rmse: {rmse:1.4f} rmsa:{rmsa:1.4f}")

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

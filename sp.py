from numpy import exp, array, zeros, log, ones, dot
from collections import Counter
from scipy.sparse import load_npz, lil_matrix
import shelve
from numpy.random import choice

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

class SP():

    CowanBufferLength = 4
    MillerBufferLength = 7
    HoneyHassonBufferLength = 20
    KintschBufferLength = 150
    epsilon = 0.00000001

    def __init__(self, corpus):
        self.corpus = corpus
        self.lam = 0.0001
        self.vocab = [w.strip() for w in open(corpus+"/vocab", encoding='utf-8').readlines()]
        self.V = len(self.vocab)
        self.I = dict([(w.strip(), i) for i, w in enumerate(self.vocab)])
        with open(corpus+"/corpus", encoding='utf-8') as f:
            self.words = [self.I[w] for w in f.read().lower().split()]

        self.bigramvocab = [w.strip() for w in open(self.corpus+"/bigramvocab", encoding='utf-8').readlines()]
        self.bigramV = len(self.bigramvocab)
        self.bigramI = dict([(w.strip(), i) for i, w in enumerate(self.bigramvocab)])
  
        self.BC = load_npz(self.corpus+f"/BC.npz")
        self.C = self.BC.sum(axis=0)
        self.AC = load_npz(self.corpus+f"/AC.npz")
        self.ABC = load_npz(self.corpus+f"/ABC.npz")
        self.CBA = load_npz(self.corpus+f"/CBA.npz")
        self.CX = load_npz(self.corpus+f"/CX.npz")
        self.MX = load_npz(self.corpus+f"/MX.npz")
        self.HX = load_npz(self.corpus+f"/HX.npz")
        self.KX = load_npz(self.corpus+f"/KX.npz")

        self.BC = log(self.BC.todense()+SP.epsilon)
        self.C = log(self.C+SP.epsilon)
        self.AC = log(self.AC.todense()+SP.epsilon)
        self.ABC = log(self.ABC.todense()+SP.epsilon)
        self.CBA = log(self.CBA.todense()+SP.epsilon)
        self.CX = log(self.CX.todense()+SP.epsilon)
        self.MX = log(self.MX.todense()+SP.epsilon)
        self.HX = log(self.HX.todense()+SP.epsilon)
        self.KX = log(self.KX.todense()+SP.epsilon)

        self.initParams()
        self.loadParams()

        # init net input variables

        self.netAC = None
        self.netCX = None
        self.netMX = None
        self.netHX = None
        self.netKX = None
        self.netCI = None
        self.netMI = None
        self.netHI = None
        self.netKI = None
        self.netBC = None
        self.netCA = None
        self.netCB = None
        self.netABC = None
        self.netCBA = None
        self.netC = None

    def initParams(self):
        self.Gac = 0.0
        self.Gbc = 0.0
        self.Gca = 0.0
        self.Gcb = 0.0
        self.Gci = 0.0 # Miller scale inhibition
        self.Gmi = 0.0 # Miller scale inhibition
        self.Ghi = 0.0 # Miller scale inhibition
        self.Gki = 0.0 # Miller scale inhibition
        self.Gc = 0.0
        self.Gabc = 0.0
        self.Gcba = 0.0
        self.Gcx = zeros((self.V, 1))
        self.Gmx = zeros((self.V, 1))
        self.Ghx = zeros((self.V, 1))
        self.Gkx = zeros((self.V, 1))
        self.Gi = 0. # self inhibition - could be at multiple scales also
                     # also might want to remove diagonals in count matrices

    def strVec(self, v, NumToReport=14):
  
      if type(v) is lil_matrix:
          v = v.toarray()
      v = array(v)
      v2 = v.flatten()
      c = Counter(dict(enumerate(v2[0:len(self.vocab)])))
      res = ""
      for i,n in c.most_common(int(NumToReport/2)):
            res += "{0} {1:1.3f} ".format(self.vocab[i], n)
      res += " ... "
      for i,n in c.most_common()[:-int(NumToReport/2)-1:-1][::-1]:
            res += "{0} {1:1.3f} ".format(self.vocab[i], n)
      return res

    def saveParams(self):

        params = shelve.open(self.corpus+"/params")
        params["Gbc"] = self.Gbc
        params["Gac"] = self.Gac
        params["Gcb"] = self.Gcb
        params["Gca"] = self.Gca
        params["Gabc"] = self.Gabc
        params["Gcba"] = self.Gcba
        params["Gcx"] = self.Gcx
        params["Gmx"] = self.Gmx
        params["Gci"] = self.Gci
        params["Gmi"] = self.Gmi
        params["Ghi"] = self.Ghi
        params["Gki"] = self.Gki
        params["Ghx"] = self.Ghx
        params["Gkx"] = self.Gkx
        params["Gc"] = self.Gc
        params.close()

    def loadParams(self):

        params = shelve.open(self.corpus+"/params")
        if "Gac" in params.keys():
            self.Gac = params["Gac"]
        if "Gbc" in params.keys():
            self.Gbc = params["Gbc"]
        if "Gca" in params.keys():
            self.Gca = params["Gca"]
        if "Gcb" in params.keys():
            self.Gcb = params["Gcb"]
        if "Gc" in params.keys():
            self.Gc = params["Gc"]
        if "Gabc" in params.keys():
            self.Gabc = params["Gabc"]
        if "Gcba" in params.keys():
            self.Gcba = params["Gcba"]
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
        result =  f"Gabc = {self.Gabc:1.3f} Gcba = {self.Gcba:1.3f} Gac = {self.Gac:1.3f} Gbc = {self.Gbc:1.3f} Gca = {self.Gca:1.3f} Gcb = {self.Gcb:1.3f} Gc = {self.Gc:1.3f}\n"
        result += f"Gci = {self.Gci:1.3f} Gmi = {self.Gmi:1.3f} Ghi = {self.Ghi:1.3f} Gki = {self.Gki:1.3f}\n"
        result +=  f"Gcx = {self.strVec(self.Gcx, NumToReport)}\n"
        result +=  f"Gmx = {self.strVec(self.Gmx, NumToReport)}\n"
        result +=  f"Ghx = {self.strVec(self.Ghx, NumToReport)}\n"
        result +=  f"Gkx = {self.strVec(self.Gkx, NumToReport)}\n"
        return result

    def strNets(self, cowan, miller, honeyhasson, kintsch):
        result = "Weighted Nets:\n"
        #result += f"AC: {self.strVec(self.netAC)}\n"
        result += f"AC: {self.strVec(self.Gac * self.netAC)}\n"
        #result += f"BC: {self.strVec(self.netBC)}\n"
        result += f"BC: {self.strVec(self.Gbc * self.netBC)}\n"
        #result += f"ABC: {self.strVec(self.netABC)}\n"
        if self.netABC is not None: 
            result += f"ABC: {self.strVec(self.Gabc * self.netABC)}\n"
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
        if self.netCBA is not None: 
        #    result += f"CBA: {self.strVec(self.netCBA)}\n"
            result += f"CBA: {self.strVec(self.Gcba * self.netCBA)}\n"
        #result += f"C: {self.strVec(self.netC)}\n"
        result += f"C: {self.strVec(self.Gc * self.netC)}\n"
        if self.netCA is not None: 
        #    result += f"CA: {self.strVec(self.netCA)}\n"
            result += f"CA: {self.strVec(self.Gca * self.netCA)}\n"
        if self.netCB is not None: 
        #    result += f"CB: {self.strVec(self.netCB)}\n"
            result += f"CB: {self.strVec(self.Gcb * self.netCB)}\n"
        return result

    def prob(self, a, b, bafter, aafter, cowan, miller, honeyhasson, kintsch):
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
    
        self.netAC = self.AC[a,:]
        self.netCX = self.CX[cowan,:]
        self.netMX = self.MX[miller,:]
        self.netCI = zeros((1, self.V))
        self.netMI = zeros((1, self.V))
        self.netHI = zeros((1, self.V))
        self.netKI = zeros((1, self.V))
        self.netCI[0, cowan] = 1.
        self.netMI[0, miller] = 1.
        self.netHI[0, honeyhasson] = 1.
        self.netKI[0, kintsch] = 1.
        self.netHX = self.HX[honeyhasson,:]
        self.netKX = self.KX[kintsch,:]
        self.netBC = self.BC[b,:]
        if aafter != -1:
            self.netCA = self.AC.T[aafter,:]
        if bafter != -1:
            self.netCB = self.BC.T[bafter,:]
        if ab != -1:
            self.netABC = self.ABC[ab,:]
        if baafter != -1:
            self.netCBA = self.CBA[baafter,:]
        self.netC = self.C
        net = self.Gac * self.netAC 
        net += self.Gbc * self.netBC
        net += self.Gc * self.netC
        net += self.Gcx[cowan, 0] * self.netCX
        net += self.Gci * self.netCI
        net += self.Gmx[miller, 0] * self.netMX
        net += self.Gmi * self.netMI
        net += self.Ghx[honeyhasson, 0] * self.netHX
        net += self.Ghi * self.netHI
        net += self.Gkx[kintsch, 0] * self.netKX
        net += self.Gki * self.netKI
        if ab != -1:
            net += self.Gabc * self.netABC
        if aafter != -1:
            net += self.Gca * self.netCA
        if bafter != -1:
            net += self.Gcb * self.netCB
        if baafter != -1:
            net +=  self.Gcba * self.netCBA

        return softmax(net)

    def sampler(self, prefix, BufferLength = 8, Threshold = 6):
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
                #print (self.vocab[c], ": ", self.strVec(ps))
                buffer[j] = c
                bufferprobs[j] = ps[0, c]
            key = " ".join(self.vocab[w] for w in buffer)
            print (" ".join(self.vocab[buffer[k]] if k < len(prefix) else f"{self.vocab[buffer[k]]} ({bufferprobs[k]:1.2f})" for k in range(len(buffer))))
            counts[key] += 1
        return counts.most_common(1)[0][0]


    def learnOnePass (self, changeweights = True, verbose=False):
        loglik = 0.0
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
            loglik += log(output[0,c])
            count += 1
          
            if changeweights:
                delta = -output 
                delta [0,c] += 1

                self.Gac += self.lam * dot(delta, self.netAC.T)[0,0]
                self.Gbc += self.lam * dot(delta, self.netBC.T)[0,0]
                self.Gca += self.lam * dot(delta, self.netCA.T)[0,0]
                self.Gcb += self.lam * dot(delta, self.netCB.T)[0,0]
                self.Gabc += self.lam * dot(delta, self.netABC.T)[0,0]   
                self.Gcba += self.lam * dot(delta, self.netCBA.T)[0,0]
                if len(cowan) > 0:
                    self.Gcx[cowan] += self.lam * dot(self.netCX, delta.T)
                    self.Gci += self.lam * dot(self.netCI, delta.T)[0,0]
                if len(miller) > 0:
                    self.Gmx[miller] += self.lam * dot(self.netMX, delta.T)
                    self.Gmi += self.lam * dot(self.netMI, delta.T)[0,0]
                if len(honeyhasson) > 0:
                    self.Ghx[honeyhasson] += self.lam * dot(self.netHX, delta.T)
                    self.Ghi += self.lam * dot(self.netHI, delta.T)[0,0]
                if len(kintsch) > 0:
                    self.Gkx[kintsch] += self.lam * dot(self.netKX, delta.T)
                    self.Gki += self.lam * dot(self.netKI, delta.T)[0,0]
                self.Gc += self.lam * dot(delta, self.netC.T)[0,0]
        return loglik/count
    

    def learn(self, NumberOfIterations):

        for iteration in range(NumberOfIterations):
            meannetlik = self.learnOnePass()
            print (f"{iteration+1}/{NumberOfIterations}: perplexity: {2**(-meannetlik):1.4f} max: {self.V} {self.strParams()}")
            self.saveParams()

    def test(self, verbose = False):
      meannetlik = self.learnOnePass(changeweights = False, verbose=verbose)   
      print (self.strParams())
      print (f"perplexity: {2**(-meannetlik):1.4f} max: {self.V}")

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

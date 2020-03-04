
# Sıfırdan Yapay Sinir Ağı Yazma

import numpy as np

def sigmoid(x):
   return 1 / (1 + np.exp(-x))

n = 0.5                       # n:öğrenme oranı
i = np.array([[0.05,0.10]], dtype=np.float64)  
w1 = np.array([[0.15,0.25],[0.20,0.30]], dtype=np.float64)
b1 = 0.35

h = (np.dot(i,w1)+ b1)        # h1:0.3775 , h2:0.3925 , h:gizli katman çıktısı
h1sigmoid = sigmoid(h)
print("sigmoid sonucu [h1,h2] : {} ".format(h1sigmoid))

w2 = np.array([[0.40,0.50], [0.45,0.55]], dtype=np.float64)
b2 = 0.60
neto1 = np.array([[0.59326999, 0.59688438]], dtype = np.float64)

o = (np.dot(neto1,w2)+ b2)    # o1:1.10590597 , o2:1.2249214
o1sigmoid = sigmoid(o)
print("sigmoid sonucu [o1,o2] : {} ".format(o1sigmoid))

# Hata değeri hesaplama

hedef = np.array([[0.01, 0.99]], dtype = np.float64)
gercek = np.array([[0.75136507, 0.77292847]], dtype = np.float64)

cıktı = (gercek - hedef) 
toplamhata = 1/2 * np.sum(cıktı**2)
print("toplamhata : {} ".format([toplamhata]))

# Geriye yayılım algoritması

degisim = np.dot(h1sigmoid.T, -(hedef-gercek) * o1sigmoid* (1-o1sigmoid)) 
print("degisim : {} ".format(degisim))

günceldegerler = w2 - n*degisim
print("günceldegerler : {} ".format(günceldegerler))


import numpy as np
import pandas as pd

#mush=df
#features = kolom_data_X
#classes = klasifikasi
#probs = prob_semua_kolom_semua_atribut_terhadap_semua_y
#probcl = prob_data_y
#mushcl=df_x_terhadap_y
#target = 'income'
#col = nama_kolom
#colp = prob_semua_atribut
#clsp = prob_semua_kolom_terhadap_semua_atribut
#val = nama_atribut
#cnt = jumlah_nama_atribut
#tot = panjang_data_per_Y
#pr = probabilitas_antar_atribut
#cl = isi_dari_class


mush = pd.read_csv("mushrooms.csv")
mush.replace('?',np.nan,inplace=True)
print(len(mush.columns),"columns, after dropping NA,",len(mush.dropna(axis=1).columns))
mush.dropna(axis=1,inplace=True)

#pengecekan kolom mana yang dipake buat X
target = 'class'
features = mush.columns[mush.columns != target]

#untuk masukin Y nya
classes = mush[target].unique()

#nge drop data make frac , dipisagin antara data train sama data test
test = mush.sample(frac=0.3)
mush = mush.drop(test.index)

#membuat object buat nampung
probs = {}
probcl = {}

#ini buat looping berdasarkan Ynya
for x in classes:
#    ini buat ngambil data X yang cocok sama Ynya aja 
    mushcl = mush[mush[target]==x][features]
    clsp = {}
#    ini buat ngitung total panjang data X terhadap data Y
    tot = len(mushcl)
#    ini looping buat ngitung jumlah kolom data X nya terhadap data Y
    for col in mushcl.columns:
         colp = {}
#         ini looping untuk menghitung jumlah tiap valuenya yang ada di dalam data X
         for val,cnt in mushcl[col].value_counts().iteritems():
#             ini untuk menghitung probabilitas dari suatu X terhadap data Y
             pr = cnt/tot
#             memasukan hari perhitungan di atas ke dalam objek colp
             colp[val] = pr
#         memasukan semua perhitungan probabilistic dari colp ke dalam objek clsp
         clsp[col] = colp
#    ini total perhitungan semua probabilitas dari semua value data X terhadap data Y
    probs[x] = clsp
#    probabilitas dari data Y
    probcl[x] = len(mushcl)/len(mush)


def probabs(x):
    #X - pandas Series with index as feature
    if not isinstance(x,pd.Series):
        raise IOError("Arg must of type Series")
    probab = {}
#    ini buat looping terhadap data Ynya
    for cl in classes:
#        ini perhitungan probabilistic dari data Y nya
        pr = probcl[cl]
        for col,val in x.iteritems():
            try:
                pr *= probs[cl][col][val]
            except KeyError:
                pr = 0
        probab[cl] = pr
    return probab



def classify(x):
    probab = probabs(x)
    mx = 0
    mxcl = ''
    for cl,pr in probab.items():
        if pr > mx:
            mx = pr
            mxcl = cl
    return mxcl


#Train data
b = []
#print(classify(mush.loc[0,features]),mush.loc[0,target])
for i in mush.index:
#    print(classify(mush.loc[i,features]),mush.loc[i,target])
    b.append(classify(mush.loc[i,features]) == mush.loc[i,target])
print(sum(b),"correct of",len(mush))
print("Accuracy:", sum(b)/len(mush))

#Test data
b = []
for i in test.index:
    #print(classify(mush.loc[i,features]),mush.loc[i,target])
    b.append(classify(test.loc[i,features]) == test.loc[i,target])
print(sum(b),"correct of",len(test))
print("Accuracy:",sum(b)/len(test))
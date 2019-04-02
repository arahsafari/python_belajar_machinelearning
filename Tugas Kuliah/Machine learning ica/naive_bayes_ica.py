import numpy as np
import pandas as pd

df = pd.read_csv("datatrain.csv")
datatest = pd.read_csv("datatest.csv")

#pengecekan kolom mana yang dipake buat X
kolom_data_X = df.iloc[:,1:8].columns
#kolom_data_X = df.columns[df.columns != 'income']

#untuk memisahkan klasifikasinya / data Y nya
klasifikasi = df['income'].unique()

#nge drop data make frac , dipisagin antara data train sama data test
#test = df.sample(frac=0.3,random_state=0)
#Y_test = test.iloc[:,8:9]
#df = df.drop(test.index)

#membuat object buat nampung
prob_semua_kolom_semua_atribut_terhadap_semua_y = {}
prob_data_y = {}

#ini buat looping berdasarkan Ynya
for x in klasifikasi:
#    ini buat ngambil data X yang cocok sama Ynya aja 
    df_x_terhadap_y = df[df['income']==x][kolom_data_X]
    prob_semua_kolom_terhadap_semua_atribut = {}
#    ini buat ngitung total panjang data X terhadap data Y
    panjang_data_per_Y = len(df_x_terhadap_y)
    #    ini looping buat ngitung jumlah kolom data X nya terhadap data Y
    for nama_kolom in df_x_terhadap_y.columns:
        prob_semua_atribut = {}
        for nama_atribut,jumlah_nama_atribut in df_x_terhadap_y[nama_kolom].value_counts().iteritems():
            #             ini untuk menghitung probabilitas dari suatu X terhadap data Y
             probabilitas_antar_atribut = jumlah_nama_atribut/panjang_data_per_Y
             #             memasukan semua perhitungan di atas ke dalam objek prob_semua_atribut
             prob_semua_atribut[nama_atribut] = probabilitas_antar_atribut
        #memasukan semua perhitungan probabilistic dari colp ke dalam objek clsp
        prob_semua_kolom_terhadap_semua_atribut[nama_kolom] = prob_semua_atribut
    #    ini total perhitungan semua probabilitas dari semua value data X terhadap data Y
    prob_semua_kolom_semua_atribut_terhadap_semua_y[x] = prob_semua_kolom_terhadap_semua_atribut
    #    probabilitas dari data Y
    prob_data_y[x] = len(df_x_terhadap_y)/len(df)
    
def probabs(x):
    #pengecekan tipe data x
    if not isinstance(x,pd.Series):
        raise IOError("tipe yang dimasukin bukan make pandas")
    probab = {}
#    ini buat looping terhadap data Ynya
    for isi_dari_class in klasifikasi:
#        ini perhitungan probabilistic dari data Y nya
        probabilitas_antar_class = prob_data_y[isi_dari_class]
        for nama_kolom,nama_atribut in x.iteritems():
            try:
                probabilitas_antar_class *= prob_semua_kolom_semua_atribut_terhadap_semua_y[isi_dari_class][nama_kolom][nama_atribut]
            except KeyError:
                probabilitas_antar_class = 0
        probab[isi_dari_class] = probabilitas_antar_class
    return probab

def classify(x):
    probab = probabs(x)
#    print(probab.items())
    if probab['>50K'] < probab['<=50K']:
        hasil = "<=50K"
#        hasil = 0
    elif probab['>50K'] > probab['<=50K']:
        hasil = ">50K"
        #        hasil = 1
    return hasil

##Train data
#b = []
#for i in df.index:
#    b.append(classify(df.loc[i,kolom_data_X]))

##Test data
#jawab_data_test = []
#for i in test.index:
#    jawab_data_test.append(classify(test.loc[i,kolom_data_X]))    
    
#JAWABAN
jawaban = []
for i in datatest.index:
    jawaban.append(classify(datatest.loc[i,kolom_data_X]))
    
pd.DataFrame(jawaban).to_csv("tebakanfile.csv",header=None, index=None)


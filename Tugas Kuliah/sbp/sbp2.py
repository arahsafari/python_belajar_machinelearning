import numpy as np
import pandas as pd
import math

df = pd.read_csv("datasetsbp.csv",header=None)
datatest = pd.read_csv("datatestsbp.csv",header=None)


#pengecekan kolom mana yang dipake buat X
kolom_data_X = df.iloc[:,:-1].columns
#kolom_data_X = df.columns[df.columns != 'income']

#untuk memisahkan klasifikasinya / data Y nya
klasifikasi = df[5].unique()

#nge drop data make frac , dipisagin antara data train sama data test
# test = df.sample(frac=0.2,random_state=0)
Y_test = datatest.iloc[:,5:6]
# df = df.drop(test.index)

# kalkulasi buat atribut kontinu
def calculate_prob(mean, std, x):
    exp = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(std, 2))))
    return (1 / (math.sqrt(2 * math.pi) * std)) * exp

#membuat object buat nampung
prob_semua_kolom_semua_atribut_terhadap_semua_y = {}
prob_data_y = {}

#ini buat looping berdasarkan Ynya
for x in klasifikasi:
#    ini buat ngambil data X yang cocok sama Ynya aja 
    df_x_terhadap_y = df[df[5]==x][kolom_data_X]
    prob_semua_kolom_terhadap_semua_atribut = {}
#    ini buat ngitung total panjang data setiap Ynya
    panjang_data_per_Y = len(df_x_terhadap_y)
    #    ini looping buat ngitung jumlah kolom data X nya terhadap data Y
    continuous_attributes = list(df_x_terhadap_y.dtypes[df_x_terhadap_y.dtypes != 'object'].index)

#    ini looping sebanyak adanya kolom Xnya
    for nama_kolom in df_x_terhadap_y.columns:
            # ini perhitungan jika atributnya adalah diskret bukan kontinu
            prob_semua_atribut = {}
    #        ini looping sebanyak atribut yang ada di dalem tiap kolom Xnya dan MENGHITUNG tiap atributnya ada berapa
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
#        variable buat pengecekan jika atributnya bukan data diskret
        continuous_attributes = list(df_x_terhadap_y.dtypes[df_x_terhadap_y.dtypes != 'object'].index)
#        ini perhitungan probabilistic dari data Y nya
        probabilitas_antar_class = prob_data_y[isi_dari_class]
        for nama_kolom,nama_atribut in x.iteritems():
#            jika datanya adalah kontinu masuk kondisi ini
            if(nama_kolom in continuous_attributes):
                np_arr = np.array(df_x_terhadap_y[nama_kolom])
                rata_rata = np.mean(np_arr)
                std= np.std(np_arr)
                for nama_atribut, jumlah_nama_atribut in df_x_terhadap_y[nama_kolom].value_counts().iteritems():
#                    kalkulasi probabilitas buat bilangan kontinu
                    probabilitas_antar_class *= calculate_prob(rata_rata, std, nama_atribut)
#            jika datanya diskret masuk kondisi ini
            else:
#                kalkulasi probabilitas untuk data diskret
                probabilitas_antar_class *= prob_semua_kolom_semua_atribut_terhadap_semua_y[isi_dari_class][nama_kolom][nama_atribut]
#            memasukan hasil dari kalkulasi probabilitas tiap classnya
        probab[isi_dari_class] = probabilitas_antar_class
    return probab

def classify(x):
    probab = probabs(x)
    print(probab)
    if probab['+'] < probab['-']:
        hasil = "-"
#        hasil = 0
        return hasil

    elif probab['+'] > probab['-']:
        hasil = "+"
        #        hasil = 1
        return hasil

        
#clasifier naive bayes
#JAWABAN
jawaban = []
for i in datatest.index:
   jawaban.append(classify(datatest.loc[i,kolom_data_X]))



from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(Y_test, jawaban))
print(metrics.classification_report(Y_test, jawaban))
print(metrics.confusion_matrix(Y_test, jawaban))


print(jawaban)
pd.DataFrame(jawaban).to_csv("testsbp.csv",header=None, index=None)


import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Membaca data training dari file Excel
data_train = pd.read_excel("F:\\TUGAS KULIAH\\semester 5\\project\code\\project\\fixed_project\\data_train (4).xlsx")

# Lakukan preprocessing data yang diperlukan (misalnya: encoding label)
en = LabelEncoder()
data_train['kelas'] = en.fit_transform(data_train['kelas'])
dataset_train = data_train.drop(["file"], axis=1)

# Pisahkan fitur dan label
x_train = dataset_train.iloc[:, :-1].values
y_train = dataset_train.iloc[:, -1].values

sc = StandardScaler()
x_train = sc.fit_transform(x_train)

# Latih model Naive Bayes
your_trained_model = GaussianNB()
your_trained_model.fit(x_train, y_train)

# Simpan model yang sudah dilatih menggunakan joblib
joblib.dump(your_trained_model, 'model.joblib')

# Simpan juga StandardScaler
joblib.dump(sc, 'standard_scaler.joblib')

# Laporan Proyek Machine Learning - Muh. Wira

## Domain Proyek

### Background

Biaya medis merupakan aspek penting dalam sistem perawatan kesehatan. Di era modern ini, biaya perawatan kesehatan terus meningkat, membuat perencanaan keuangan menjadi tantangan bagi individu dan penyedia layanan kesehatan. Masalah yang ingin diatasi dalam proyek ini adalah ketidakmampuan untuk secara akurat memprediksi biaya medis individu berdasarkan faktor demografis dan kesehatan. Ketidakakuratan dalam prediksi biaya medis dapat menyebabkan perencanaan anggaran yang tidak efisien, peningkatan biaya tidak terduga bagi pasien, dan pengelolaan sumber daya yang kurang optimal di rumah sakit dan klinik.

### Urgency of the Problem

Urgensi dari masalah ini terletak pada dampak finansial yang signifikan bagi individu dan sistem kesehatan. Tanpa prediksi yang akurat, individu mungkin tidak siap secara finansial untuk menghadapi biaya medis yang tinggi, sementara penyedia layanan kesehatan mungkin menghadapi kesulitan dalam mengalokasikan sumber daya dengan tepat. Dengan menggunakan model machine learning untuk memprediksi biaya medis, kita dapat membantu mengurangi ketidakpastian ini dan memberikan alat yang berguna untuk perencanaan keuangan dan pengelolaan anggaran yang lebih baik.

### Dataset Relevance and Results

Dataset yang digunakan dalam proyek ini mencakup berbagai faktor demografis dan kesehatan, seperti usia, BMI, jumlah anak, status merokok, jenis kelamin, dan wilayah geografis. Dengan menganalisis dan memodelkan data ini, kita dapat mengidentifikasi pola dan hubungan antara faktor-faktor tersebut dan biaya medis.

### Importance of the Problem

Masalah ini perlu diatasi untuk membantu penyedia layanan kesehatan dalam perencanaan anggaran dan sumber daya yang lebih baik. Dengan prediksi biaya medis yang lebih akurat, rumah sakit dan klinik dapat mengalokasikan sumber daya mereka secara lebih efisien dan efektif, memastikan bahwa dana dan tenaga medis tersedia sesuai kebutuhan. Selain itu, hasil dari model prediksi ini dapat memberikan wawasan yang berharga bagi perusahaan asuransi dalam menentukan premi yang lebih adil dan berbasis risiko, sehingga membantu menghindari overcharging atau undercharging pelanggan. Terakhir, individu dapat menggunakan informasi ini untuk merencanakan keuangan mereka dengan lebih baik, mempersiapkan diri untuk biaya medis yang mungkin timbul di masa depan dan mengurangi ketidakpastian finansial terkait kesehatan mereka.

## Business Understanding

### Problem Statements

- Bagaimana memprediksi biaya medis seseorang berdasarkan faktor demografi dan kesehatan?

### Goals

- Mengembangkan model prediksi yang mampu memprediksi biaya medis dengan akurasi tinggi, diukur dengan _Mean Squared Error_ (MSE) yang rendah. Model yang baik diharapkan memiliki MSE di bawah ambang batas tertentu yang ditentukan berdasarkan analisis data awal.

### Solution Statement

#### Algorithms yang digunakan:

1. **_K-Nearest Neighbors_ (KNN)**:
   - **Alasan Pemilihan**: KNN dipilih karena kesederhanaannya dan kemampuannya dalam menangkap hubungan non-linear antara fitur-fitur dan target. KNN bekerja dengan mencari sejumlah tetangga terdekat dari data yang ingin diprediksi dan menghitung rata-rata dari nilai target tetangga tersebut.
2. **_Random Forest_ (RF)**:
   - **Alasan Pemilihan**: Random Forest dipilih karena keandalannya dalam menangani data dengan banyak fitur dan menangkap hubungan non-linear. Algoritma ini membangun banyak pohon keputusan selama pelatihan dan menggabungkan hasilnya untuk meningkatkan akurasi prediksi dan mengontrol overfitting.
3. **_Boosting_**:
   - **Alasan Pemilihan**: Boosting dipilih karena kemampuannya untuk meningkatkan kinerja model prediksi dengan menggabungkan kekuatan beberapa model sederhana (weak learners) menjadi satu model yang kuat (strong learner). Algoritma ini secara iteratif menambahkan model-model sederhana untuk mengurangi kesalahan yang ada.

#### Improvements

- Melakukan hyperparameter tuning untuk meningkatkan kinerja model. Hyperparameter tuning adalah proses pencarian kombinasi parameter terbaik untuk setiap algoritma, yang dapat membantu meningkatkan akurasi prediksi model dan mengurangi overfitting.

#### Evaluation Metrics

- **Mean Squared Error (MSE)**: Metrik ini dipilih karena mengukur rata-rata kuadrat dari kesalahan prediksi. MSE memberikan gambaran seberapa jauh nilai prediksi model dari nilai aktual, dengan penalti lebih besar untuk kesalahan yang lebih besar. MSE yang lebih rendah menunjukkan kinerja model yang lebih baik.

## Data Understanding

### Dataset Information

- Jumlah data: 1338
- Kondisi data: tidak ada missing values
- Dataset source: [Medical Cost Personal Dataset](https://www.kaggle.com/mirichoi0218/insurance)

Dataset ini berisi informasi tentang biaya medis individu yang ditanggung oleh asuransi kesehatan, bersama dengan beberapa atribut demografis dan kesehatan yang terkait dengan setiap individu. Data ini digunakan untuk membangun model prediksi biaya medis berdasarkan faktor-faktor tersebut.

### Descriptive and Statistical Analysis

Untuk memberikan pemahaman yang lebih mendalam tentang dataset, analisis deskriptif dan statistik dilakukan pada setiap variabel yang ada.

#### Descriptive statistics

Berikut adalah beberapa statistik deskriptif dasar untuk variabel numerik dalam dataset:

```python
import pandas as pd

# Membaca dataset
url = 'https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv'
df = pd.read_csv(url)

# Menampilkan statistik deskriptif
df.describe()
```

![image](https://github.com/muhwira38/sub/assets/127601088/d19173e1-2051-4f73-9401-618d5f13bd0d)

Output dari `data.describe()` memberikan informasi seperti mean, standard deviation, min, dan max untuk setiap variabel numerik. Ini membantu memahami distribusi data dan rentang nilai untuk setiap fitur.

### Types and Data Types

Berikut adalah jenis dan tipe data untuk setiap variabel dalam dataset:

- **Usia (age)**: Numerik (integer)
- **Jenis kelamin (sex)**: Kategorikal (string)
- **BMI (bmi)**: Numerik (float)
- **Jumlah anak (children)**: Numerik (integer)
- **Status merokok (smoker)**: Kategorikal (string)
- **Wilayah (region)**: Kategorikal (string)
- **Biaya medis (charges)**: Numerik (float)

### Variables in the Dataset

1. **Usia (age)**:

   - **Deskripsi**: Usia penerima manfaat utama.
   - **Analisis**: Usia dalam dataset ini berkisar antara 18 hingga 64 tahun. Usia rata-rata adalah sekitar 39 tahun. Tidak ada nilai yang hilang.

2. **Jenis kelamin (sex)**:

   - **Deskripsi**: Jenis kelamin penerima manfaat (laki-laki atau perempuan).
   - **Analisis**: Variabel ini merupakan data kategorikal dengan dua kategori: 'male' dan 'female'. Distribusi jenis kelamin dalam dataset ini relatif seimbang.

3. **BMI (bmi)**:

   - **Deskripsi**: Indeks Massa Tubuh, sebuah ukuran yang dihitung dari berat dan tinggi badan.
   - **Analisis**: Nilai BMI dalam dataset ini berkisar antara 15.96 hingga 53.13. Rata-rata BMI adalah sekitar 30. Ini menunjukkan bahwa banyak individu dalam dataset ini berada dalam kisaran overweight atau obesitas.

4. **Jumlah anak (children)**:

   - **Deskripsi**: Jumlah anak yang tercakup dalam asuransi kesehatan.
   - **Analisis**: Variabel ini merupakan data numerik dengan nilai antara 0 hingga 5. Rata-rata jumlah anak adalah sekitar 1.

5. **Status merokok (smoker)**:

   - **Deskripsi**: Apakah penerima manfaat adalah perokok atau tidak.
   - **Analisis**: Variabel ini merupakan data kategorikal dengan dua kategori: 'yes' dan 'no'. Hanya sebagian kecil individu dalam dataset ini yang merupakan perokok, yang dapat memiliki pengaruh signifikan terhadap biaya medis.

6. **Wilayah (region)**:

   - **Deskripsi**: Wilayah tempat tinggal penerima manfaat.
   - **Analisis**: Variabel ini merupakan data kategorikal dengan empat kategori: 'northeast', 'northwest', 'southeast', dan 'southwest'. Distribusi wilayah cukup merata.

7. **Biaya medis (charges)**:
   - **Deskripsi**: Biaya medis yang ditanggung oleh asuransi.
   - **Analisis**: Nilai biaya medis dalam dataset ini berkisar antara 1121.87 hingga 63770.43. Rata-rata biaya medis adalah sekitar 13270.42. Distribusi biaya medis menunjukkan skewness positif, dengan beberapa nilai outlier yang sangat tinggi.

### Visualisasi dan Insight

1. **Boxplot untuk Kolom 'charges'**
   ![image](https://github.com/muhwira38/sub/assets/127601088/11fcceb8-c8ca-4b25-abee-d00dd44dc550)

   Terlihat ada beberapa outlier di bagian atas distribusi. Hal ini menunjukkan bahwa ada beberapa individu dengan biaya medis yang sangat tinggi dibandingkan yang lain. Mayoritas data berkumpul di bawah sekitar 20.000, dengan beberapa nilai yang jauh lebih tinggi.

2. **Boxplot untuk Kolom 'bmi'**

   ![image](https://github.com/muhwira38/sub/assets/127601088/97369057-3ed0-48c5-8028-b24ce130e762)

   Ada beberapa outlier di bagian atas distribusi BMI. Mayoritas data berada di kisaran 25 hingga 35, yang menunjukkan bahwa banyak individu berada dalam kategori overweight atau obesitas. Outlier ini mungkin mempengaruhi analisis dan prediksi biaya medis.

3. **Boxplot untuk Kolom 'age'**

   ![image](https://github.com/muhwira38/sub/assets/127601088/77912e98-fd13-4ea4-8aee-f41e3047211d)

   Tidak ada outlier yang signifikan dalam distribusi usia. Usia tersebar merata dari 18 hingga 64 tahun, dengan median di sekitar 39 tahun.

4. **Boxplot untuk Kolom 'children'**

   ![image](https://github.com/muhwira38/sub/assets/127601088/6358d79f-c841-413a-8f44-53040e66c6e8)

   Tidak ada outlier dalam distribusi jumlah anak. Mayoritas individu memiliki 0 hingga 2 anak, dengan beberapa individu memiliki hingga 5 anak.

5. **Barplot untuk Fitur 'sex'**

   ![image](https://github.com/muhwira38/sub/assets/127601088/f37590a6-e77a-4d14-87bf-fdd382055749)

   Distribusi jenis kelamin cukup seimbang antara laki-laki dan perempuan, dengan sedikit lebih banyak perempuan dalam dataset ini. Ini menunjukkan bahwa jenis kelamin tidak terlalu miring dan dapat digunakan secara adil dalam model prediksi.

6. **Barplot untuk Fitur 'smoker'**

   ![image](https://github.com/muhwira38/sub/assets/127601088/1888e3f7-5444-455c-b02f-fb45129ff9c4)

   Mayoritas individu dalam dataset ini adalah non-perokok. Hanya sebagian kecil yang merupakan perokok, yang dapat memiliki dampak signifikan terhadap biaya medis mereka.

7. **Barplot untuk Fitur 'region'**

   ![image](https://github.com/muhwira38/sub/assets/127601088/c201c3cb-54f0-4543-a0c1-3c6af0042d76)

   Distribusi wilayah tempat tinggal cukup merata di antara empat kategori ('northeast', 'northwest', 'southeast', 'southwest'). Ini menunjukkan bahwa data geografis cukup seimbang dan dapat digunakan dalam analisis tanpa bias wilayah yang signifikan.

8. **Histogram untuk Distribusi Variabel Numerik**

   ![image](https://github.com/muhwira38/sub/assets/127601088/8c0a9663-f7e7-41aa-9bd6-68c2ffefccc0)

   - **Usia**: Distribusi usia menunjukkan puncak pada usia sekitar 20-an dan kemudian menurun seiring bertambahnya usia.
   - **BMI**: Distribusi BMI berbentuk lonceng, dengan puncak di sekitar 30, menunjukkan banyak individu berada dalam kategori overweight atau obesitas.
   - **Jumlah Anak**: Distribusi jumlah anak menunjukkan mayoritas individu memiliki 0 hingga 2 anak.
   - **Biaya Medis**: Distribusi biaya medis menunjukkan skewness positif, dengan sebagian besar biaya berada di bawah 20.000 dan beberapa biaya yang sangat tinggi.

9. **Average 'charges' Relative to - sex**

   ![image](https://github.com/muhwira38/sub/assets/127601088/a6316b8b-ef19-40d4-bedf-e37bf8d327db)

   Tidak ada perbedaan signifikan dalam biaya medis rata-rata antara laki-laki dan perempuan. Ini menunjukkan bahwa jenis kelamin mungkin bukan faktor penentu utama dalam variasi biaya medis.

10. **Average 'charges' Relative to - smoker**

    ![image](https://github.com/muhwira38/sub/assets/127601088/7d987ff5-e845-4982-a852-d5e29f31f2cb)

    Perokok memiliki biaya medis rata-rata yang jauh lebih tinggi dibandingkan non-perokok. Ini menunjukkan bahwa merokok adalah faktor risiko yang signifikan yang meningkatkan biaya medis.

11. **Average 'charges' Relative to - region**

    ![image](https://github.com/muhwira38/sub/assets/127601088/bfd6761b-2a87-4087-865b-469eda2d11a3)

    Tidak ada perbedaan signifikan dalam biaya medis rata-rata di antara wilayah yang berbeda. Ini menunjukkan bahwa wilayah tempat tinggal mungkin tidak memiliki pengaruh besar terhadap biaya medis dalam dataset ini.

12. **Pairplot untuk Hubungan Antar Variabel Numerik**

    ![image](https://github.com/muhwira38/sub/assets/127601088/41ffb21b-97bf-41cf-a321-3a09a08ab142)

    Ada korelasi positif yang jelas antara usia dan biaya medis, serta antara BMI dan biaya medis. Ini menunjukkan bahwa individu yang lebih tua dan memiliki BMI lebih tinggi cenderung memiliki biaya medis yang lebih tinggi.

13. **Correlation Matrix for Numerical Features**

    ![image](https://github.com/muhwira38/sub/assets/127601088/51aa7f68-2f34-449f-b3f9-36b5b702c23d)

    Usia menunjukkan korelasi positif yang cukup kuat dengan biaya medis (0.44), menunjukkan bahwa usia lebih tua cenderung berhubungan dengan biaya medis yang lebih tinggi. BMI dan jumlah anak menunjukkan korelasi yang lebih lemah dengan biaya medis.

### Analysis and Interpretation

- **Outlier**: Identifikasi outlier dalam 'charges' dan 'bmi' penting karena dapat mempengaruhi hasil model prediksi. Outlier ini mungkin perlu ditangani secara khusus, misalnya dengan transformasi data atau teknik robust regression.
- **Distribusi Variabel**: Pemahaman tentang distribusi variabel seperti usia, BMI, dan jumlah anak membantu dalam mengidentifikasi karakteristik umum dari data. Misalnya, banyaknya individu dengan BMI tinggi dapat menunjukkan risiko kesehatan yang lebih besar.
- **Fitur Kategorikal**: Distribusi fitur kategorikal seperti jenis kelamin, status merokok, dan wilayah menunjukkan keseimbangan data dan memberikan wawasan tentang kelompok mana yang mungkin memiliki pengaruh lebih besar terhadap biaya medis.
- **Hubungan Antara Variabel**: Visualisasi ini juga membantu dalam memahami hubungan potensial antara variabel demografis dan kesehatan dengan biaya medis, yang akan menjadi dasar dalam pengembangan model prediksi yang akurat.

## Data Preparation

### Data Preparation Techniques

#### 1. **_One-Hot Encoding_**

One-Hot Encoding digunakan untuk mengubah fitur kategorikal menjadi format numerik yang dapat diproses oleh algoritma machine learning. Dalam dataset ini, fitur 'sex', 'smoker', dan 'region' adalah fitur kategorikal yang perlu di-encode.

**Penerapan**:

```python
df = pd.concat([df, pd.get_dummies(df['sex'], prefix='sex', dtype='int')],axis=1)
df = pd.concat([df, pd.get_dummies(df['smoker'], prefix='smoker', dtype='int')],axis=1)
df = pd.concat([df, pd.get_dummies(df['region'], prefix='region', dtype='int')],axis=1)
df.drop(['sex','smoker','region'], axis=1, inplace=True)
df.head()
```

**Alasan**:
Algoritma machine learning tidak dapat memproses data kategorikal secara langsung. One-Hot Encoding mengubah kategori menjadi kolom biner yang terpisah, memungkinkan algoritma untuk memproses informasi ini secara efektif.

#### 2. Standardization

Standardization dilakukan untuk memastikan fitur numerik berada pada skala yang sama. Ini penting untuk model yang sensitif terhadap skala fitur seperti K-Nearest Neighbors (KNN) dan algoritma berbasis gradient descent.

**Penerapan**:

```python
from sklearn.preprocessing import StandardScaler

numerical_features = ['age', 'bmi']
scaler = StandardScaler()
scaler.fit(X_train[numerical_features])
X_train[numerical_features] = scaler.transform(X_train.loc[:, numerical_features])
```

**Alasan**:
Standardization membantu model dalam konvergensi lebih cepat dan performa yang lebih baik, terutama pada model yang sensitif terhadap skala data. Ini juga membantu dalam menghindari dominasi fitur dengan skala yang lebih besar.

#### 3. Train-Test Split

Train-Test Split digunakan untuk membagi data ke dalam training dan test set dengan rasio 80:20. Hal ini penting untuk mengevaluasi kinerja model pada data yang tidak terlihat selama pelatihan.

**Penerapan**:

```python
from sklearn.model_selection import train_test_split

# Split dataset with ration 80:20
X = df.drop(["charges"],axis =1)
y = df["charges"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)
```

**Alasan**:
Train-Test Split memastikan bahwa model dievaluasi pada data yang belum pernah dilihat selama pelatihan, memberikan gambaran yang lebih akurat tentang kinerja model pada data dunia nyata.

#### 4. Outlier Handling

Outlier dapat mempengaruhi performa model. Outlier pada fitur 'charges' dan 'bmi' telah diidentifikasi melalui boxplot. Teknik penanganan outlier seperti penggunaan Interquartile Range (IQR) dapat diterapkan.

**Penerapan**:

```python
# Calculate Q1 (first quartile) and Q3 (third quartile) for each numeric column
numeric_cols = ['bmi', 'charges']
Q1 = df[numeric_cols].quantile(0.25)
Q3 = df[numeric_cols].quantile(0.75)

# Calculate IQR (Interquartile Range)
IQR = Q3 - Q1

# Determine the lower and upper limits for detecting outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Delete rows that contain values ​​outside the lower limit and upper limit for a numeric column
df = df[~((df[numeric_cols] < lower_bound) | (df[numeric_cols] > upper_bound)).any(axis=1)]
```

**Alasan**:
Outlier dapat mengganggu pelatihan model dan menghasilkan prediksi yang tidak akurat. Dengan menangani outlier, kita dapat memastikan bahwa model dilatih pada data yang representatif.

### Conclusion

Teknik-teknik persiapan data ini diterapkan untuk memastikan bahwa data dalam kondisi optimal untuk pemodelan machine learning. One-Hot Encoding memungkinkan pemrosesan fitur kategorikal, Standardization memastikan skala yang seragam untuk fitur numerik, Train-Test Split memberikan evaluasi yang adil dari model, dan penanganan outlier menghindari gangguan pada pelatihan model. Semua langkah ini bertujuan untuk meningkatkan kinerja dan akurasi model prediksi biaya medis.

## Modeling

### Modeling Stages

#### 1. **_K-Nearest Neighbors_** (KNN)

1. **Grid Parameter**:

   - **Parameter yang digunakan**:
     - `n_neighbors`: [5, 10, 15, 20]
     - `weights`: ['uniform', 'distance']
     - `algorithm`: ['auto', 'ball_tree', 'kd_tree', 'brute']
   - **Alasan Pemilihan**: Parameter ini dipilih untuk mengeksplorasi berbagai kombinasi yang mungkin mempengaruhi kinerja model KNN. Parameter `n_neighbors` menentukan jumlah tetangga terdekat yang akan dipertimbangkan, sementara `weights` mengatur bobot jarak tetangga, dan `algorithm` menentukan algoritma yang digunakan untuk menghitung tetangga terdekat.

2. **_GridSearchCV_**:

   - **Penerapan**: Objek `GridSearchCV` digunakan untuk melakukan hyperparameter tuning dengan cross-validation untuk menemukan kombinasi hyperparameter terbaik.
   - **Alasan Pemilihan**: `GridSearchCV` membantu dalam menemukan parameter terbaik dengan mencoba setiap kombinasi yang ditentukan dan memilih yang menghasilkan kinerja terbaik berdasarkan cross-validated score.

3. **Estimator Terbaik dan Perhitungan MSE**:
   - **Model terbaik dipilih berdasarkan kinerja cross-validated**: Model yang memberikan MSE terendah pada data validasi dipilih sebagai model terbaik.
   - **_Mean Squared Error_ (MSE) dihitung untuk dataset latih dan uji**: MSE digunakan sebagai metrik evaluasi utama untuk menilai kinerja model.

#### 2. **_Random Forest_** (RF)

1. **Grid Parameter**:

   - **Parameter yang digunakan**:
     - `n_estimators`: [50, 100, 200]
     - `max_depth`: [10, 16, 20, None]
     - `min_samples_split`: [2, 5, 10]
     - `min_samples_leaf`: [1, 2, 4]
   - **Alasan Pemilihan**: Parameter ini dipilih untuk mengeksplorasi berbagai pengaturan pada Random Forest. `n_estimators` menentukan jumlah pohon dalam hutan, `max_depth` membatasi kedalaman pohon, `min_samples_split` menentukan jumlah minimum sampel yang diperlukan untuk membagi node, dan `min_samples_leaf` menetapkan jumlah minimum sampel yang harus ada di node daun.

2. **GridSearchCV**:

   - **Penerapan**: Objek `GridSearchCV` digunakan untuk melakukan hyperparameter tuning dengan cross-validation untuk menemukan kombinasi hyperparameter terbaik.
   - **Alasan Pemilihan**: `GridSearchCV` membantu dalam menemukan kombinasi parameter yang optimal untuk meningkatkan kinerja model Random Forest.

3. **Estimator Terbaik dan Perhitungan MSE**:
   - **Model terbaik dipilih berdasarkan kinerja cross-validated**: Model dengan kombinasi parameter terbaik yang menghasilkan MSE terendah pada data validasi dipilih sebagai model terbaik.
   - **_Mean Squared Error_ (MSE) dihitung untuk dataset latih dan uji**: MSE digunakan untuk menilai kinerja model pada data latih dan uji.

#### 3. **_AdaBoost_**

1. **Grid Parameter**:

   - **Parameter yang digunakan**:
     - `n_estimators`: [50, 100, 200]
     - `learning_rate`: [0.01, 0.05, 0.1, 0.5, 1]
   - **Alasan Pemilihan**: Parameter ini dipilih untuk mengeksplorasi pengaturan yang berbeda pada _AdaBoost_. `n_estimators` menentukan jumlah estimator, dan `learning_rate` mengontrol kontribusi masing-masing estimator.

2. **_GridSearchCV_**:

   - **Penerapan**: Objek `GridSearchCV` digunakan untuk melakukan hyperparameter tuning dengan cross-validation untuk menemukan kombinasi hyperparameter terbaik.
   - **Alasan Pemilihan**: `GridSearchCV` memungkinkan pencarian parameter yang optimal untuk meningkatkan kinerja model AdaBoost.

3. **Estimator Terbaik dan Perhitungan MSE**:
   - **Model terbaik dipilih berdasarkan kinerja cross-validated**: Model dengan kombinasi parameter terbaik yang memberikan MSE terendah pada data validasi dipilih sebagai model terbaik.
   - **_Mean Squared Error_ (MSE) dihitung untuk dataset latih dan uji**: MSE digunakan sebagai metrik untuk mengevaluasi kinerja model.

### Reasons for Model Selection

Model-model yang digunakan, yaitu _K-Nearest Neighbors_ (KNN), _Random Forest_ (RF), dan _AdaBoost_, dipilih karena memiliki keunggulan masing-masing dalam menangani masalah prediksi biaya medis berdasarkan faktor demografis dan kesehatan:

1. **_K-Nearest Neighbors_ (KNN)**:

   - **Kelebihan**: Kesederhanaan dan kemampuan menangkap hubungan non-linear antara fitur-fitur dan target.
   - **Alasan Pemilihan**: KNN dipilih untuk mengeksplorasi hubungan lokal dalam data, yang dapat berguna dalam memprediksi biaya medis berdasarkan tetangga terdekat dengan karakteristik serupa.

2. **_Random Forest_ (RF)**:

   - **Kelebihan**: Keandalannya dalam menangani data dengan banyak fitur dan kemampuannya menangkap hubungan non-linear.
   - **Alasan Pemilihan**: Random Forest dipilih karena kemampuannya dalam mengurangi overfitting melalui penggabungan hasil dari banyak pohon keputusan, memberikan prediksi yang lebih stabil dan akurat.

3. **_AdaBoost_**:
   - **Kelebihan**: Kemampuannya untuk meningkatkan kinerja model dengan menggabungkan kekuatan beberapa model sederhana menjadi satu model yang kuat.
   - **Alasan Pemilihan**: AdaBoost dipilih untuk mengeksplorasi pendekatan ensemble yang dapat memperbaiki kesalahan model-model sederhana, memberikan prediksi yang lebih baik melalui iterasi dan penyesuaian.

### Modeling Process

1. **Pemilihan Algoritma**: Memilih KNN, _Random Forest_, dan _AdaBoost_ berdasarkan karakteristik data dan kebutuhan prediksi.
2. **Hyperparameter Tuning**: Menggunakan _GridSearchCV_ untuk menemukan kombinasi parameter terbaik yang meningkatkan kinerja model.
3. **Evaluasi Model**: Menggunakan _Mean Squared Error_ (MSE) sebagai metrik utama untuk mengevaluasi kinerja model pada data latih dan uji.
4. **Pemilihan Model Terbaik**: Memilih model dengan MSE terendah sebagai model terbaik untuk prediksi biaya medis.

### Conclusion

Pemilihan dan penerapan model ini didasarkan pada kemampuan masing-masing model dalam menangani data dengan fitur demografis dan kesehatan yang kompleks. Dengan melakukan hyperparameter tuning dan evaluasi yang tepat, model yang dihasilkan diharapkan dapat memberikan prediksi biaya medis yang akurat dan berguna untuk perencanaan keuangan dan pengelolaan anggaran dalam sistem perawatan kesehatan.

## Evaluation

### Evaluation Metrics Used

- **_Mean Squared Error_ (MSE)**: Mengukur perbedaan kuadrat rata-rata antara nilai aktual dan prediksi. MSE adalah metrik evaluasi yang penting karena memberikan gambaran tentang seberapa besar kesalahan prediksi model. Semakin rendah nilai MSE, semakin baik kinerja model.

### Project Results Based on Evaluation Metrics

Berikut adalah hasil MSE untuk masing-masing model:

- **_K-Nearest Neighbors_ (KNN)**:

  - **Train MSE**: 19,999.06
  - **Test MSE**: 82,381.79
  - **Interpretasi**: Nilai MSE yang sangat tinggi pada data latih dan uji menunjukkan bahwa model KNN tidak mampu menangkap pola dalam data dengan baik, mengindikasikan underfitting. Model ini tidak sesuai untuk masalah prediksi biaya medis dalam konteks data ini.

- **_Random Forest_ (RF)**:

  - **Train MSE**: 13,254.32
  - **Test MSE**: 66,183.99
  - **Interpretasi**: Meskipun MSE pada data latih relatif rendah, MSE pada data uji masih cukup tinggi, menunjukkan adanya overfitting. Model Random Forest mampu menangkap pola dalam data latih dengan baik tetapi gagal menggeneralisasi dengan baik pada data uji.

- **_AdaBoost_**:
  - **Train MSE**: 19,282.28
  - **Test MSE**: 44,678.88
  - **Interpretasi**: Meskipun performa AdaBoost lebih baik daripada KNN, model ini masih menunjukkan MSE yang tinggi pada data latih dan uji, menunjukkan adanya potensi overfitting.

### Comparison and Interpretation

Berikut adalah perbandingan MSE dari ketiga model:

| Model         | Train MSE | Test MSE  |
| ------------- | --------- | --------- |
| KNN           | 19,999.06 | 82,381.79 |
| Random Forest | 13,254.32 | 66,183.99 |
| AdaBoost      | 19,282.28 | 44,678.88 |

- **KNN**: Model ini menunjukkan kinerja yang sangat buruk dengan MSE yang sangat tinggi baik pada data latih maupun data uji. Hal ini menunjukkan bahwa KNN tidak cocok untuk masalah prediksi biaya medis dalam dataset ini karena tidak mampu menangkap pola yang relevan dalam data.

- **_Random Forest_**: Model ini menunjukkan MSE yang lebih rendah pada data latih tetapi masih cukup tinggi pada data uji. Ini menunjukkan bahwa Random Forest mampu menangkap pola pada data latih dengan baik tetapi mengalami overfitting, sehingga tidak mampu memberikan prediksi yang baik pada data baru.

- **_AdaBoost_**: Model ini menunjukkan MSE yang lebih rendah dibandingkan dengan KNN tetapi masih cukup tinggi. _AdaBoost_ memiliki performa yang lebih baik dalam menangkap pola pada data latih dibandingkan KNN tetapi juga mengalami overfitting.

### Practical Impact and Conclusion

- **KNN**: Dengan MSE yang sangat tinggi, KNN tidak berhasil dalam menyelesaikan masalah prediksi biaya medis. Model ini tidak mampu mencapai tujuan proyek untuk memberikan prediksi yang akurat.

- **_Random Forest_**: Meskipun _Random Forest_ memiliki MSE yang lebih rendah pada data latih, MSE yang tinggi pada data uji menunjukkan bahwa model ini mengalami overfitting. Model ini tidak mampu memberikan prediksi yang dapat diandalkan pada data baru, sehingga belum berhasil mencapai tujuan proyek.

- **_AdaBoost_**: _AdaBoost_ menunjukkan performa yang lebih baik dibandingkan KNN tetapi masih belum memadai. MSE yang tinggi menunjukkan bahwa model ini juga belum mampu memberikan prediksi yang akurat dan andal.

## Conclusion

Proyek ini bertujuan untuk mengembangkan model prediksi biaya medis yang akurat menggunakan data demografi dan kesehatan. Tiga algoritma utama yang digunakan adalah _K-Nearest Neighbors_ (KNN), _Random Forest_ (RF), dan _Boosting_. Hasil evaluasi menunjukkan bahwa semua model mengalami kesulitan dalam memberikan prediksi yang akurat, dengan MSE yang tinggi pada data uji.

### Key Conclusions:

- Model KNN tidak cocok untuk masalah ini karena menghasilkan MSE yang sangat tinggi.
- Model _Random Forest_ menunjukkan adanya overfitting, dengan MSE yang lebih rendah pada data latih tetapi masih tinggi pada data uji.
- Model _AdaBoost_ memiliki performa yang lebih baik tetapi masih belum memadai untuk memberikan prediksi yang andal.

### The next step:

- Melakukan data augmentation dan feature engineering.
- Melanjutkan hyperparameter tuning dan mengeksplorasi metode ensemble lainnya.
- Menggunakan lebih banyak data untuk meningkatkan kemampuan generalisasi model.

Dengan langkah-langkah ini, diharapkan dapat memperbaiki kinerja model dan mencapai tujuan proyek untuk memberikan prediksi biaya medis yang akurat dan andal.

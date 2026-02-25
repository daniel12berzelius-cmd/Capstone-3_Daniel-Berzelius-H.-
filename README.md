# Capstone-3_Daniel-Berzelius-H.-
Model Machine Learning untuk menganalisis data historis reservasi dan memprediksi probabilitas pembatalan hotel.

Model dijalankan melalui Jupyter Notebook

Project Overview:
Hotel adalah salah satu jenis akomodasi komersial yang menyediakan layanan penginapan bagi masyarakat umum, baik untuk tujuan perjalanan wisata, bisnis, maupun kebutuhan lainnya. Selain menyediakan kamar untuk bermalam, hotel umumnya juga menawarkan berbagai fasilitas pendukung seperti restoran, ruang pertemuan, layanan kebersihan, serta fasilitas rekreasi guna memberikan kenyamanan dan kemudahan bagi para tamu. Seiring perkembangan industri pariwisata dan mobilitas masyarakat, hotel tidak hanya berfungsi sebagai tempat menginap, tetapi juga menjadi bagian penting dalam mendukung aktivitas ekonomi, sosial, dan budaya di berbagai daerah.

Namun, seiring meningkatnya kebutuhan dan ekspektasi tamu, layanan hotel kini tidak hanya berhenti pada penyediaan kamar dan fasilitas, tetapi juga pada fleksibilitas layanan pemesanan. Sistem pemesanan (booking) hotel modern memungkinkan tamu untuk melakukan reservasi jauh hari, mengubah jadwal menginap, mengajukan permintaan khusus sesuai kebutuhan, hingga membatalkan pemesanan apabila terjadi perubahan rencana. Fleksibilitas ini memberikan kenyamanan bagi tamu, namun di sisi lain juga menimbulkan tantangan bagi pihak hotel dalam mengelola ketersediaan kamar, permintaan khusus, serta risiko pembatalan yang dapat memengaruhi operasional dan pendapatan hotel. Oleh karena itu, pengelolaan sistem pemesanan menjadi isu penting yang perlu dikaji lebih lanjut dalam analisis ini.

**Problem Statement**

Hotel Lonestar Oriental adalah hotel terkemuka di Portugal. Sebagai salah satu hotel yang menjadi ikon di Portugal, Hotel Lonestar Oriental selalu berusaha untuk memberikan pelayanan terbaik bagi para pengunjung hotel yang beragam dari seluruh dunia. Untuk meningkatkan kepercayaan pengunjung, Hotel Lonestar Oriental memberikan fleksibilitas para pengunjung untuk menyesuaikan kebutuhan ruang mereka sesuai dengan keinginan pengunjung. Hotel Lonestar Oriental memberikan fleksibilitas kepada pengunjung dalam bentuk:
- kebebasan dalam pemesanan
- ukuran kelompok pengunjung yang menginap
- 10 tipe kamar yang bebas dipilih pengunjung
- lapangan parkir kendaraan yang luas
- fleksibilitas dalam waktu tunggu penyewaan ruang

Dengan fleksibilitas yang ditawarkan oleh Hotel Lonestar Oriental, persentase calon pengunjung yang membatalkan pemesanan ruang di hotel masih di angka yang tinggi. 36.9% dari calon pengunjung Hotel Lonestar Oriental berpotensi untuk membatalkan pesanan ruang mereka.

Tim marketing Hotel Lonestar Oriental melihat masalah ini dan ingin bertindak cepat untuk menekan angka pembatalan yang cukup tinggi. Angka ini bukan hanya berpengaruh terhadap kapital yang menjaga hotel tetap berjalan, namun juga memiliki dampak signifikan terhadap kepercayaan wisatawan; baik lokal maupun mancanegara terhadap Hotel Lonestar Oriental. Tim marketing ingin mencari faktor-faktor penting apa yang mempengaruhi pembatalan, serta faktor-faktor apa yang mempengaruhi tidak terjadinya pembatalan oleh calon pengunjung.

**Tujuan**
1. Tim Marketing akan bekerja sama dengan Tim Data untuk menggali, menganalisis, dan memberi gambaran mengenai layanan yang diberikan terhadap calon pengunjung berdasarkan pola-pola aktivitas pengunjung.
2. Menggali dan memberikan alasan untuk faktor-faktor yang berdampak terhadap suksesnya penyewaan kamar serta faktor-faktor yang berdampak terhadap pembatalan pemesanan ruangan.
3. Tujuan akhir dari analisis ini adalah memberikan gambaran terhadap faktor-faktor penting yang dapat ditingkatkan untuk menekan angka-angka pembatalan ruang yang cukup tinggi; yang berpotensi untuk menaikkan keuntungan  terhadap Hotel Lonestar Oriental.


#Daftar Library yang digunakan dalam model ini:
# 1. Data Manipulation, Visualization & Utilities
pandas
matplotlib
seaborn
missingno
country_converter
plotly.express
sys

# 2. Preprocessing & Pipeline
import category_encoders as ce
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils import resample

# 3. Model Selection, Validation & Evaluation
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score, cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, make_scorer, precision_recall_curve

# 4. Machine Learning Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import lightgbm as lgb

# 5. Imbalanced Data Handling (Imblearn)
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, RandomOverSampler, SMOTENC
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours, NearMiss, OneSidedSelection
from imblearn.combine import SMOTEENN

# 6. Statistical Tests & Analysis
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from scipy.stats import uniform, randint, shapiro, kstest, anderson, chi2_contingency, normaltest


Bagian A: Business Understanding
Mendalami mengenai bisnis, menemukan masalah dan target yang menjadi pondasi dalam proyek ini

Bagian B: Analytical Approach
Setelah melakukan business understanding, mencari strategi untuk menerapkan target yang akan dieksekusi

Bagian C: Data Understanding
Mengetahui fitur-fitur yang ada di dataframe

Bagian D: Data Cleaning
Melakukan cleaning singkat mengenai data-data yang akan digunakan

Bagian E: Exploratory Data Analysis
Melakukan analisis data untuk menemukan gambaran terhadap dataframe

Bagian F: Machine Learning
Melakukan persiapan data dan optimisasi data; serta melanjutkannya ke proses Machine Learning

Bagian G: Feature Importance
mencari tahu fitur yang berpengaruh besar terhadap hasil machine learning

Bagian H: Data Limitaion
menjelaskan batasan dataframe yang terjadi dalam proses machine learning ini

Bagian I: Save Model
Menyimpan model machine learning lewat pickle, dengan output format .pkl

Bagian J: Business Impact
Kesimpulan akhir, memberikan saran-saran yang disintesis setelah melakukan studi mengenai prediksi pembatalah kamar hotel. 
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from scipy.stats import uniform, randint, shapiro, kstest, anderson, chi2_contingency, normaltest

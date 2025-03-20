import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from PIL import Image
import seaborn as sns
import io

st.markdown(
    """
    <style>
        .stApp {
            background-color: white !important;
        }
        body, .stMarkdown, .stTextInput, .stButton, .stSelectbox, .stSlider, .stDataFrame, .stTable {
            color: black !important;
        }
        h1, h2, h3, h4, h5, h6, .stHeader, .stSubheader {
            color: black !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Monte Carlo Simulation, Markov Chain, and Hidden Markov Model")
st.markdown("""
- Nama : Naufal Fadhlullah
- NIM : 20234920001""")

st.header("flowchart")


st.header("Data : dataset yang digunakan merupakan data saham selama rentang 2015-2016")
image = r"D:\Matana\Semester 4\komputasi statistik\UTS\dataa.png"
st.image(image, caption = "Sumber Data: https://www.kaggle.com/code/bturan19/financial-time-series-monte-carlo-simulation-arma/input")
st.markdown(""" 
data ini merupakan data saham AAPL dari rentang tahun 2015 hingga 2016. data asli sebenarnya merupakan data stocks 5 tahun dari berbagai perusahaan saham,
namun karena terkendala ukuran yang besar, jadi kami memutuskan untuk hanya mengambil saham AAPL selama rentang tersebut. berikut variabel dalam datasetnya.
1. date -> tanggal transaksi saham
2. open -> harga pembukaan saham di hari tersebut
3. high -> harga tertinggi di hari tsb
4. low -> harga terendah di hari tsb
5. close -> harga penutupan saham di hari tsb
6. volume -> jumlah saham yang diperdagangkan di hari tsb
7. name -> nama saham.""")
st.markdown(""" untuk nama saham yang digunakan saat ini adalah saham AAPL untuk Apple Inc.""")

st.subheader("Data Ekplorasi")
image = r"D:\Matana\Semester 4\komputasi statistik\UTS\sum_strdata.png"
st.image(image)
st.markdown("""berikut merupakan summary dari data yang ada (max, min, 1st Quartile, median, mean, 3rd quartile), dan juga berisi struktur dari data yang ada.""")

st.subheader("missing value")
kode_r ="""
```r
missing_values <- colSums(is.na(data))
print(missing_values)"""

st.markdown(kode_r)
image = r"D:\Matana\Semester 4\komputasi statistik\UTS\MV.png"
st.image(image)
st.markdown("""tidak terdapat missing value pada data yang ada""")

st.subheader("outlier")
kode_r ="""
```r
ggplot(data, aes(y = close)) + 
  geom_boxplot() + 
  labs(title = "Boxplot of Closing Prices") + 
  theme_minimal()"""

st.markdown(kode_r)
image = r"D:\Matana\Semester 4\komputasi statistik\UTS\outlier.png"
st.image(image)
st.markdown("""terlihat pada boxplot tidak terdapat titik yang berada di luar garis vertikal atas maupun bawah box, biasanya outlier terdeteksi di tempat yang disebutkan. dengan begini, data yang digunakan tidak terdapat outlier.""")

st.subheader("visualisasi")
kode_r ="""
```r
ggplot(data, aes(x = date, y = close)) + 
  geom_line(color = 'blue') + 
  labs(title = "AAPL Closing Prices", x = "Date", y = "Close Price") + 
  theme_minimal()"""

st.markdown(kode_r)
image = r"D:\Matana\Semester 4\komputasi statistik\UTS\visualisasi.png"
st.image(image)


st.subheader("Feature Engineering")
kode_r ="""
```r
data <- data %>% 
  arrange(date) %>% 
  mutate(
    Return = (close - lag(close)) / lag(close),
    LogReturn = log(close / lag(close)),
    Volatility = rollapply(LogReturn, width = 10, FUN = sd, fill = NA, align = "right")
  )
head(data)"""

st.markdown(kode_r)
image = r"D:\Matana\Semester 4\komputasi statistik\UTS\FE.png"
st.image(image)
st.markdown("""
1. pada return dan LogReturn, terdapat nilai NA dikarenakan data pertama tidak memiliki historis nilai sebelumnya untuk dihitung return di baris pertama.
2. sedangkan untuk volality karena penggunaan window (width) 10, maka 9 hari pertama data belum cukup untuk dihitung standar deviasinya.""")

st.subheader("Monte Carlo Simulation")
kode_r ="""
```r
set.seed(123)
n_simulations <- 1000
n_days <- 30  # Simulate for the next 30 days
mu <- mean(data$LogReturn, na.rm = TRUE)
sigma <- sd(data$LogReturn, na.rm = TRUE)

simulated_prices <- matrix(NA, nrow = n_simulations, ncol = n_days)
starting_price <- tail(data$close, 1)

for (i in 1:n_simulations) {
  shocks <- rnorm(n_days, mean = mu, sd = sigma)
  simulated_prices[i, ] <- starting_price * exp(cumsum(shocks))
}"""
st.markdown(kode_r)

kode_r ="""
```r
time_horizon <- seq(1, n_days)
data_sim <- data.frame(time_horizon, t(simulated_prices))

ggplot(data_sim, aes(x = time_horizon)) +
  geom_line(aes(y = X1), alpha = 0.2) +
  labs(title = "Monte Carlo Simulation for AAPL", x = "Days", y = "Simulated Price") +
  theme_minimal()"""
st.markdown(kode_r)
image = r"D:\Matana\Semester 4\komputasi statistik\UTS\visualisasi_dist.png"
st.image(image)
st.markdown("""1. pada bagian pertama, grafik menunjukkan simulasi monte carlo mengenai prediksi harga saham beberapa hari ke depan yang tiap garisnya melambangkan skenario berbeda yang hasilnya didapat dari distribusi log return pada data."
yang diasumsikan return harian saham mengikuti distribusi normal rata-rata dan standar deviasi."
selanjutnya, bentuk distribusi dari histogram menunjukkan kemungkinan harga saham kedepannya."
2. asumsi dalam monte carlo yang diterapkan adalah dimana return harian berdistribusi normal, perubahan saham yang terjadi tanpa ada faktor eksternal lain mengikuti adanya proses stokastik,"
dan tidak terdapat perubahan struktural pada pola harga saham selama periode simulasi yang dilakukan.""")

st.subheader("Markov Chain")
kode_r ="""
```r
data$State <- ifelse(data$LogReturn > 0.005, "Up", ifelse(data$LogReturn < -0.005, "Down", "Stable"))

markov_model <- markovchainFit(data$State)$estimate
print(markov_model)"""
st.markdown(kode_r)
image = r"D:\Matana\Semester 4\komputasi statistik\UTS\markov.png"
st.image(image)
st.markdown("""Dengan pendekatan LogReturn, perhitungan nilai didasarkan pada harga sebelumnya. Jika nilainya mencapai 0.005, maka itu adalah “UP”, jika itu adalah -0.005, maka itu adalah “DOWN”, dan jika nilainya pada keduanya, itu adalah “STABLE”. Berdasarkan apa di atas, “nilai terdeteksi dalam “DOWN “, 35.58% mungkin “DOWN”, 26.99% mungkin “ STABLE”, dan 37.42% mungkin “UP”. “untuk nilai terdeteksi “STABLE”, 33.54% mungkin menjadi “DOWN”, 31.05% mungkin 30.40% mungkin “UP”. Dan akhirnya, “jika mungkin” UP”. Sebanyak 28.65% mungkin “DOWN”, 37.64% mungkin “STABLE”, dan 33.70% mungkin. Dapat dilihat bahwa model ini berguna untuk mengetahui pola pergerakan harga saham menggunakan analisis probabilitas transisi."""
)

st.subheader("visualisasi")
kode_r ="""
```r
transition_matrix <- as.matrix(markov_model@transitionMatrix)
colnames(transition_matrix) <- rownames(transition_matrix) <- c("Down", "Stable", "Up")

library(reshape2)
library(ggplot2)
transition_df <- melt(transition_matrix)
colnames(transition_df) <- c("From", "To", "Probability")

ggplot(transition_df, aes(x = From, y = To, fill = Probability)) +
  geom_tile() +
  geom_text(aes(label = round(Probability, 2)), color = "white") +
  scale_fill_gradient(low = "blue", high = "red") +
  labs(title = "Markov Chain Transition Matrix", x = "From State", y = "To State") +
  theme_minimal()"""
st.markdown(kode_r)
image = r"D:\Matana\Semester 4\komputasi statistik\UTS\visual_MC.png"
st.image(image)

st.markdown("""
berdasarkan visualisasi, didapatkan bahwa probabilitas nilai yang down berpotensi tetap down sebesar 36%, down menjadi stable sebesar 34%, down menjadi up sebesar 29%.
selanjutnya, stable menjadi down sebesar 27%, stable tetap stable 31%, dan stable menjadi up sebesar 38%.
dan berikutnya, up menjadi down sebesar 37%, up menjadi stable 35%, dan up tetap akan up sebesar 34%.
terdapat perbedaan pembulatan dari hasil sebelumnya.""")

st.subheader("Hidden Markov Model")
kode_r ="""
```r
hmm_model <- depmix(response = LogReturn ~ 1, data = data, nstates = 2, family = gaussian())
hmm_fit <- fit(hmm_model)
summary(hmm_fit)"""
st.markdown(kode_r)
image = r"D:\Matana\Semester 4\komputasi statistik\UTS\HMM.png"
st.image(image)
st.markdown("""Dari hasil yang dihasilkan oleh model HMM mengungkapkan bahwa terdapat dua keadaan tersembunyi yang ada di balik data saham. Keadaan pertama state1 adalah Bearish dengan return rata-rata -0.001 dan volalitas 0.022 yang berarti bahwa pasar dalam keadaan tidak stabil dan downtrend. Sedangkan keadaan kedua adalah Bullish dengan return rata-rata 0.001 dan volalitas 0.009 yang dikatakan bahwa pasar dalam keadaan stabil dan naik. Adapun model ini menunjukkan bahwa pola transisi dua state ini dapat terlihat bahwa pada keadaan bearish, terdapat probabilitas 0.839 dan 83.9% bahwa pasar tetap pada keadaan sama pada periode selanjutnya dan 16.1% pasaran beralih ke bullish. Tetapi apabila keadaan pasar tidak ada di state2, yaitu kondisi bearish, probabilitas 0.883 atau 88.3% untuk bertahan dan probabilitas 11.7% probabilitas digunakan apabila beralih ke state1 yaitu bullish. Dari probabilitas yang dihasilkan, adalah kecenderungan pasar untuk konsisten dalam satu posisi dalam periode yang sama, pada probabilitas yang lebih kecil terjadi transisi antarkeadaan. 
""")

kode_r ="""
```r
data$HMM_State <- posterior(hmm_fit)$state

ggplot(data, aes(x = date, y = close, color = factor(HMM_State))) +
  geom_line() +
  labs(title = "Hidden Markov Model States", x = "Date", y = "Close Price", color = "State") +
  theme_minimal()"""
st.markdown(kode_r)
image = r"D:\Matana\Semester 4\komputasi statistik\UTS\HMM_states.png"
st.image(image)

st.subheader("Evaluation and Discussion")
st.markdown("""Adapun terdapat tiga metode untuk memahami pergerakan harga saham tersebut yaitu Monte Carlo Simulation, Markov Chain, dan Hidden Markov Model. 
            Monte Carlo Simulation adalah metode yang digunakan untuk memberikan proyeksi harga saham dengan melihat distribusi return historisnya. 
            Hanya saja, metode ini tidak dapat mempertimbangkan pola atau tren high-low price tertentu. 
            Oleh karena itu, metode tersebut hanya dapat berfungsi sebagai prediksi scaling harga yang memungkinkan terjadi di masa depan tanpa memahami alur prediksi yang lebih dahulu. 
            Lebih lanjut, metode tersebut melahirkan markov Chain untuk menangkap transisi pola antar state harga saham, seperti bahwa harga akan cenderung ingin naik atau turun atau stabil dalam kurun waktu dekat.
             Namun, metode ini juga gagal menangkap berbagai ukuran setiap state sehingga juga tidak bisa dilibatkan dalam mengerti faktor yang mendalam dan dipertimbangkan faktor harga. 
            Sementara itu, HMM menawarkan pendekatan yang lebih canggih karena dapat menunjukkan state tersembunyi yang tidak bisa diamati dan memberikan probabilitas transisi antar state melalui probabilitas emisi dan trasnisi meskipun dilakukan tidak langsung. 
            Dengan demikian, investor akan lebih memahami tentang pemahaman dinamika pasar yang sebenarnya.
             Dengan mempertimbangkan kegunaan ketiga jenis ini, monte carlo sangat membantu, tetapi tidak bisa memprediksi tren atau momentum.
             Markov akan sangat membantu untuk pola transisi jangka pendek tetapi tidak lebih dari detik. 
            Sementara HMM benar-benar membantu dalam memahami kondisi pasar seperti yang bisa dilihat.
             Sebagai hasil akhir, ini semua adalah kombinasi dari beberapa komponen datas yang memberikan insight yang lebih baik kepada investor.
             Dari hasil gabungan metode tersebut, investor akan lebih memahami trend investasinya.""")
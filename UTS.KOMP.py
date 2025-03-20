import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from PIL import Image
import seaborn as sns
import io


st.title("Monte Carlo Simulation, Markov Chain, and Hidden Markov Model")
st.markdown("""
- Nama : Naufal Fadhlullah
- NIM : 20234920001""")

##Flowchart

st.header("Data : dataset yang digunakan merupakan data saham selama rentang 2015-2016")
image = r"D:\Matana\Semester 4\komputasi statistik\UTS\data.png"
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

st.subheader("Data Exploration")
file_path = r"D:/Matana/Semester 4/komputasi statistik/UTS/stocks_AAPL.csv"  # Pastikan path ini benar
df = pd.read_csv(file_path)
st.write("Data Saham AAPL")
st.dataframe(df)

st.write("Distribusi data, missing values, dan outliers")
st.write("### Tampilkan & Analisis Data")
st.write(df.head())
st.write("**Statistik Deskriptif:**")
st.write(df.describe())
st.write("**Missing Values:**")
st.write(df.isnull().sum())

st.subheader("Visualisasi")
kode_r = """
```r
volume_plot <- ggplot(data_smc, aes(x = date, y = volume)) +
  geom_line(color = "red", size = 1) +
  labs(title = "Volume Perdagangan Saham AAPL", x = "Tanggal", y = "Volume") +
  theme_minimal()
volume_plot"""
st.markdown(kode_r)
st.image(r"D:/Matana/Semester 4/komputasi statistik/UTS/perdagangan.png")

kode_r = """
```r
scatter_plot <- ggplot(data_smc, aes(x = open, y = close)) +
  geom_point(aes(color = volume), alpha = 0.6, size = 2) +
  labs(title = "Scatter Plot: Open vs Close Harga Saham AAPL",
       x = "Open Price",
       y = "Close Price") +
  theme_minimal() +
  scale_color_gradient(low = "khaki", high = "darkgreen")
scatter_plot
"""
st.markdown(kode_r)
st.image(r"D:/Matana/Semester 4/komputasi statistik/UTS/openVsclose.png")
st.markdown("Scatter Plot Open Vs Close Harga Saham AAPL")

st.subheader("Feature Engineering")
st.markdown("data return")
kode_r = """
```r
data_smc$return <- c(NA, diff(data_smc$close)) / dplyr::lag(data_smc$close)"""
st.markdown(kode_r)
st.image(r"D:/Matana/Semester 4/komputasi statistik/UTS/return.png")

st.markdown("moving average")
kode_r = """
```r
data_smc <- data_smc %>%
  mutate(
    MA_5 = zoo::rollmean(close, k = 5, fill = NA, align = "right"),
    MA_20 = zoo::rollmean(close, k = 20, fill = NA)
  )
data_smc"""
st.markdown(kode_r)
st.image(r"D:/Matana/Semester 4/komputasi statistik/UTS/movAvg.png")

st.markdown("volatilitas")
kode_r = """
```r
data_smc <- data_smc %>%
  mutate(volatility = zoo::rollapply(return, width = 20, FUN = sd, fill = NA, align = "right"))
data_smc"""
st.markdown(kode_r)
st.image(r"D:/Matana/Semester 4/komputasi statistik/UTS/volatilitas.png")

st.markdown("volume perdagangan")
kode_r = """
```r 
data_smc <- data_smc %>%
  mutate(volume_change = (volume - lag(volume)) / lag(volume))
data_smc"""
st.markdown(kode_r)
st.image(r"D:/Matana/Semester 4/komputasi statistik/UTS/vol.per.png")

st.markdown("menghapus NA dan buat csv")
kode_r = """
```r
data_smc <- na.omit(data_smc)
write.csv(data_smc, "D:/Matana/Semester 4/komputasi statistik/UTS/stocks_FE.csv", row.names = FALSE)
data_smc"""
st.markdown(kode_r)

file_path = r"D:/Matana/Semester 4/komputasi statistik/UTS/stocks_FE.csv"  # Pastikan path ini benar
df = pd.read_csv(file_path)
st.write("Data setelah Feature Engineering")
st.dataframe(df)


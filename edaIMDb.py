                                                ### Film Analizi (Movie Analysis) ###


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Veri Setini Yükleme

df = pd.read_csv("C:\\Users\\murat\\.vscode\\Projects\\Film Analizi\\imdb-top-1000-movies\\imdb_clean.csv")


# İlk Bakış
print("Şekil(satır, sütun):", df.shape)
print("\nSütunlar:", list(df.columns))
print("\nİlk 5 Satır:")
print(df.head())


# Veri Tipleri ve Eksik Değerler

print("\nDtype Bilgisi:")
print(df.info())
print("\nEksik Değer Sayıları:")
print(df.isnull().sum())

# Sütun isimlerini normalize etme:

df.columns = (
    df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^0-9a-z]+", "_", regex=True)
)
print("Normalize Edilmiş Sütunlar:", list(df.columns))


# Temel tip dönüşümleri:

# Yıl

for ycol in ["released_year", "year"]:
    if ycol in df.columns:
        df[ycol] = pd.to_numeric(df[ycol], errors="coerce").astype("Int64")

#Puanlar

for rcol in ["imdb_rating", "rating", "metascore", "votes", "gross", "runtime", "runtime_mins", "duration_mins"]:
    if rcol in df.columns:
        #Para/virgül gibi karakterleri temizlemek için(varsa)
        df[rcol] = (
            df[rcol]
            .astype(str)
            .str.replace(r"[^\d.]", "", regex=True)
        )
        df[rcol] = pd.to_numeric(df[rcol], errors="coerce")

print(df.dtypes)

df.to_csv("imdb_work.csv", index=False)
print("Kaydedildi: iimdb_work.csv")


#Tür sütununu inceleme

for col in df.columns:
    if "genre" in col.lower():
        print("Tür sütunu bulundu", col)
        print(df[col].head())

#Bulduğumuz sütun adını aşağıdaki yere yazma

genre_col = "genre" #Bulduğumuz ad ile değiştirme

#1) Virgülle ayrılmış türleri listeye çevirme

df[genre_col] = df[genre_col].astype(str) #String yap
df["genre_list"] = df[genre_col].str.split(",")

#2) explode ile her tür ayrı satır olacak şekilde açma

df_exploded = df.explode("genre_list")

#3) Boşlukları temizleme

df_exploded["genre_list"] = df_exploded["genre_list"].str.strip()

#İlk 10 satırı kontrol etme
print(df_exploded[["title", "genre_list"]].head(10))


# Görselleştirmeler için varsayılan stil ayarları

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = [12, 7]
plt.rcParams["font.size"] = 10


            ###########################################################################
                            #--- 1) Türlerin Popülerliği ---#

print("\n--- En Yaygın Film Türleri Analizi ---")

# Her türün kaç filmden geçtiğini sayma
genre_counts = df_exploded["genre_list"].value_counts()

print("En yaygın 15 film türü:")
print(genre_counts.head(15))

# Sonuçları görselleştirme

plt.figure(figsize=(15, 8))
sns.barplot(x=genre_counts.head(15).index, y=genre_counts.head(15).values, palette="viridis")
plt.title("Top 15 En Yaygon Film Türü", fontsize=16)
plt.xlabel("Türler",fontsize=12)
plt.ylabel("Film Sayısı", fontsize=12)
plt.xticks(rotation=45)
plt.show()


            ###########################################################################
                            #--- 2) Türlere Göre Ortalama Puanlar ---#


print("\n--- Türlere Göre Ortalama IMDb Puanları ---")

# Türlere göre ortalama IDMb puanını hesaplama ve sıralama

genre_ratings = df_exploded.groupby("genre_list")["rating"].mean().sort_values(ascending=False)

print("Türlere göre ortalama IMDb puanları (En yüksekten düşüğe)")
print(genre_ratings.head(15))

# Sonuçları görselleştirme

plt.figure(figsize=(15, 8))
sns.barplot(x = genre_ratings.head(15).index, y= genre_ratings.head(15).values, palette="mako")
plt.title("Top 15 Türün Ortalama IMDb Puanları", fontsize= 16)
plt.xlabel("Türler", fontsize = 12)
plt.ylabel("Ortalama IMDb Puanı", fontsize=12)
plt.xticks(rotation=45)
plt.ylim(genre_ratings.min() - 0.1, genre_ratings.max() + 0.1) # Y ekseni limitlerini ayarlama
plt.show()


            ###########################################################################
                            #--- 3) Yıllara Göre Film Üretim Trendleri ---#

print("\n--- Yıllara Göre Film Üretimi ---")

# Yıllara göre film sayısı hesaplama
movies_per_year = df["release_year"].value_counts().sort_index()

# Sonuçları görselleştirme (Çizgi Grafiği)

plt.figure(figsize=(15, 8))
movies_per_year.plot(kind="line", marker="o", linestyle="-")
plt.title("Yıllara Göre Çekilen Film Sayısı ( Top 1000 Listesi)", fontsize=16)
plt.xlabel("Yıl", fontsize=12)
plt.ylabel("Film Sayısı", fontsize=12)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()


# En popüler 5 türün yıllara göre trendini inceleme

print("\n--- Popüler Türlerin Yıllara Göre Trendi ---")
top_5_genres = genre_counts.head(5).index

# Bu 5 türü içeren filmleri filtreleme

df_top_genres = df_exploded[df_exploded["genre_list"].isin(top_5_genres)]

# Pivot table oluşturarak her yıl her türden kaç film çekildiğini bulma
genre_trends = df_top_genres.groupby(["release_year", "genre_list"]).size().unstack(fill_value=0)

# Sonuçları görselleştirme

genre_trends.plot(kind="line", figsize=(15, 8))
plt.title("En Popüler 5 Türün Yıllara Göre Trendi", fontsize=16)
plt.xlabel("Yıl", fontsize=12)
plt.ylabel("Film Sayısı", fontsize=12)
plt.legend(title="Tür")
plt.show()


            ###########################################################################
                            #--- 4) Sayısal Değişkenler Arasındaki Korelasyon Analizi ---#

print("\n--- Sayısal Değişkenler Arası Korelasyon ---")

# Korelasyon için gerekli sütunları seçme

correlation_df = df[["rating", "metascore", "gross_m_", "runtime"]].dropna()

correlation_matrix = correlation_df.corr()

print("Korelasyon Matrisi:")
print(correlation_matrix)

# Korelasyon matrisini ısı haritası ile görselleştirme

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
plt.title("Sayısal Değişkenler Arası Korelasyon Isı Haritası", fontsize=16)
plt.show()

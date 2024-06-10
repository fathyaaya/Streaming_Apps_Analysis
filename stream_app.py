
import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import time

df_d = pd.read_csv('disney_testing.csv')
df_d_l = pd.read_csv('disney_testing_label.csv')

x_d = df_d['stem_review']
y_d = df_d_l['polarity']

model_d = pickle.load(open('sentiment_disney.pkl', 'rb'))
vectorizer_d = pickle.load(open('vectorizer_disney.pkl', 'rb'))

model_n = pickle.load(open('sentiment_netflix.pkl', 'rb'))
vectorizer_n = pickle.load(open('vectorizer_netflix.pkl', 'rb'))

x_pred_d = vectorizer_d.transform(x_d.values.astype('U')).toarray()
y_pred_d = model_d.predict(x_pred_d)
acc_d = accuracy_score(y_pred_d,y_d)
acc_d = round((acc_d*100),2)

df_n = pd.read_csv('netflix_testing.csv')
df_n_l = pd.read_csv('netflix_testing_label.csv')

x_n = df_n['stem_review']
y_n = df_n_l['polarity']

x_pred_n = vectorizer_n.transform(x_n.values.astype('U')).toarray()
y_pred_n = model_n.predict(x_pred_n)
acc_n = accuracy_score(y_pred_n,y_n)
acc_n = round((acc_n*100),2)

st.set_page_config(
    page_title = "Streaming Apps Review Sentiment Analysis", 
    page_icon = "📺"
)

#st.title('Astro Apps Review Sentiment Analysis')
selected = option_menu(
        menu_title = None,
        options = ['Informasi Aplikasi Netflix & Disney Hotstar','Perbandingan Performa Model','Model Analisa Sentimen Netflix','Model Analisa Sentimen Disney'],
        icons = ['tv','plus-slash-minus','camera-reels','cast'],
        default_index = 0,
        orientation = 'horizontal',
        styles={
            "nav-link-selected":{"background-color":"#E50914"},
        },
    )

if selected == 'Informasi Aplikasi Netflix & Disney Hotstar':
    st.title('Informasi Aplikasi Netflix & Disney Hotstar')
    st.markdown("""---""")
    st.subheader('Netflix', divider='rainbow')
    st.image('netflix.png')
    st.write('')
    st.write('Netflix adalah layanan aliran video sesuai permintaan berbasis langganan yang berasal dari Amerika Serikat. Layanan ini menawarkan beragam film dan acara televisi, termasuk produksi orisinal dan yang diperoleh dari pihak lain, yang mencakup berbagai genre dan tersedia dalam banyak bahasa secara internasional.\n\n Diluncurkan pada 16 Januari 2007, hampir sepuluh tahun setelah Netflix, Inc. memulai bisnis penyewaan film melalui DVD, Netflix telah berkembang menjadi layanan aliran video atas permintaan dengan jumlah pelanggan terbesar. Per 2022, layanan ini memiliki 238,39 juta keanggotaan berbayar di lebih dari 190 negara.')
    st.write('')
    st.markdown("""---""")
    st.subheader('Disney Hotstar', divider='rainbow')
    st.image('disney.jpeg')
    st.write('')
    st.write('Disney+ Hotstar (dikenal sebagai Hotstar di Singapura, Kanada dan Britania Raya) adalah layanan streaming video over-the-top asal India yang dimiliki oleh Disney Star, sebuah anak perusahaan dari The Walt Disney Company.[1] Sebelumnya layanan ini diluncurkan sebagai Hotstar, sebelum diakusisi oleh layanan Disney+ pada April 2020.[2] Pada Februari 2020, setelah pembelian perusahaan induk Star India, 21st Century Fox, oleh Disney pada 2019, Disney mengumumkan rencana integrasi layanan video sesuai permintaan Disney+ dengan Hotstar pada April 2020 untuk memanfaatkan infrastruktur dan pengguna Hotstar. Pada 3 April 2020, platform Hotstar resmi digabungkan dengan Disney+.')

if selected == 'Perbandingan Performa Model':
    st.title('Perbandingan Model Sentiment Anaylsis')
    st.subheader('', divider='rainbow')
    st.write('Pada penelitian ini penulis membuat model analisis sentimen yang dilatih dengan 2 dataset berdeda. Model ini dirancang untuk menganalisis sentimen dari ulasan pengguna aplikasi streaming Netflix dan Disney Hotstar. Dengan menggunakan model ini, pengguna dapat memahami apakah ulasan yang diberikan oleh pengguna lain bersifat positif atau negatif. Model analisis sentimen ini dibuat menggunakan algoritma Multinomial Naive Bayes, yang merupakan salah satu metode yang efektif dalam pengolahan teks dan analisis sentimen.')
    st.write('')
    st.subheader('**Perbedaan DataSet:**')
    st.write('Dataset yang digunakan untuk melatih model ini adalah data ulasan aplikasi dari kedua aplikasi yang diambil langsung dari Google Playstore. Data ini mencakup berbagai ulasan dari pengguna, yang kemudian diproses dan dianalisis untuk membangun model yang akurat dan andal.')
    col1, col2 = st.columns(2)
    col1.image('netflix_distri.png',caption="Distribusi dataset Netflix", use_column_width=True)
    col2.image('disney_distri.png',caption="Distribusi dataset Disney Hotstar", use_column_width=True)
    st.write('Pada Gambar di atas memperlihatkan dataset yang berisi ulasan aplikasi Astro dari user sejumlah 733 ulasan terbagi menjadi 81 ulasan negatif dan 652 ulasan positif. Data ini nantinya akan dipisah menjadi data latih dan data uji. Pada penelitian ini dataset akan dipecah dalam bentuk 90% data latih dan 10% data uji.')
    st.write('')
    st.subheader('**Hasil Uji:**')
    col1, col2 = st.columns(2)
    col1.image('pie_netflix.png',caption="Hasil Pengujian Model Ulasan Netflix", use_column_width=True)
    col2.image('pie_disney.png',caption="Hasil Pengujian Model Ulasan Disney Hotstar",use_column_width=True)
    st.write('')
    st.write('Dari 74 ulasan yang termasuk dalam data uji, model ini memprediksi bahwa 2 ulasan memiliki sentimen negatif dan 72 ulasan memiliki sentimen positif. Deskripsi ini menggambarkan kemampuan model dalam mengkategorikan sentimen ulasan pengguna dengan cukup baik.')
    st.subheader('**Confusion Matrix:**')
    col1, col2 = st.columns(2)
    col1.image('cm_netflix.png',caption="Confusion Matrix Model Ulasan Netflix", use_column_width=True)
    col2.image('cm_disney.png',caption="Confusion Matrix Model Ulasan Disney Hotstar", use_column_width=True)
    st.write('')
    st.write('bla bla bla bla bla')

if selected == 'Model Analisa Sentimen Netflix':
    st.title('Netflix Apps Review Sentiment Analysis')
    st.write(f"**_Accuracy Model Analisa Sentimen Ulasan Aplikasi Netflix_** :  :green[**{acc_n}**]%")
    st.subheader('', divider='rainbow')
    st.title('Single-Predict Model Demo')
    coms = st.text_input('Masukan ulasan anda terhadap aplikasi Netflix')

    submit = st.button('Predict')

    if submit:
        start = time.time()
        # Transform the input text using the loaded TF-IDF vectorizer
        transformed_text = vectorizer_n.transform([coms]).toarray()
        #st.write('Transformed text shape:', transformed_text.shape)  # Debugging statement
        # Reshape the transformed text to 2D array
        transformed_text = transformed_text.reshape(1, -1)
        #st.write('Reshaped text shape:', transformed_text.shape)  # Debugging statement
        # Make prediction
        prediction = model_n.predict(transformed_text)
        end = time.time()
        st.write('Prediction time taken: ', round(end-start, 2), 'seconds')

        print(prediction[0])
        if prediction[0] == 1:
            st.success("👍 Sentimen review anda positif")
        else:
            st.error("👎 Sentimen review anda negatif")
    
    st.markdown("""---""")
    st.title('Multi-Predict Model Demo')
    sample_csv = df_n.iloc[:5].to_csv(index=False).encode('utf-8')

    st.write("")
    st.download_button("Download CSV Example", data=sample_csv, file_name='sample_review.csv', mime='text/csv')

    st.write("")
    st.write("")
    file_uploaded = st.file_uploader("Upload a CSV file", type='csv')

    if file_uploaded:
        uploaded_df = pd.read_csv(file_uploaded)
        conv_df = vectorizer_n.transform(uploaded_df['stem_review'].values.astype('U')).toarray()
        prediction_arr = model_n.predict(conv_df)

        bar = st.progress(0)
        status_text = st.empty()

        for i in range(1, 70):
            status_text.text(f"{i}% complete")
            bar.progress(i)
            time.sleep(0.01)

        result_arr = []

        for prediction in prediction_arr:
            if prediction == 1:
                result = "Sentimen positif"
            else:
                result = "Sentimen Negatif"
            result_arr.append(result)

        uploaded_result = pd.DataFrame({'Prediction Result': result_arr})

        for i in range(70, 101):
            status_text.text(f"{i}% complete")
            bar.progress(i)
            time.sleep(0.01)
            if i == 100:
                time.sleep(1)
                status_text.empty()
                bar.empty()

        col1, col2 = st.columns([1, 2])

        with col1:
            st.dataframe(uploaded_result)
        with col2:
            st.dataframe(uploaded_df)
if selected == 'Model Analisa Sentimen Disney':
    st.title('Disney Hotstar Apps Review Sentiment Analysis')
    st.write(f"**_Accuracy Model Analisa Sentimen Ulasan Aplikasi Disney Hotstar_** :  :green[**{acc_d}**]%")
    st.subheader('', divider='rainbow')
    st.title('Single-Predict Model Demo')
    coms = st.text_input('Masukan ulasan anda terhadap aplikasi Disney')

    submit = st.button('Predict')

    if submit:
        start = time.time()
        # Transform the input text using the loaded TF-IDF vectorizer
        transformed_text = vectorizer_d.transform([coms]).toarray()
        #st.write('Transformed text shape:', transformed_text.shape)  # Debugging statement
        # Reshape the transformed text to 2D array
        transformed_text = transformed_text.reshape(1, -1)
        #st.write('Reshaped text shape:', transformed_text.shape)  # Debugging statement
        # Make prediction
        prediction = model_d.predict(transformed_text)
        end = time.time()
        st.write('Prediction time taken: ', round(end-start, 2), 'seconds')

        print(prediction[0])
        if prediction[0] == 1:
            st.success("👍 Sentimen review anda positif")
        else:
            st.error("👎 Sentimen review anda negatif")
    
    st.markdown("""---""")
    st.title('Multi-Predict Model Demo')
    sample_csv = df_d.iloc[:5].to_csv(index=False).encode('utf-8')

    st.write("")
    st.download_button("Download CSV Example", data=sample_csv, file_name='sample_review.csv', mime='text/csv')

    st.write("")
    st.write("")
    file_uploaded = st.file_uploader("Upload a CSV file", type='csv')

    if file_uploaded:
        uploaded_df = pd.read_csv(file_uploaded)
        conv_df = vectorizer_d.transform(uploaded_df['stem_review'].values.astype('U')).toarray()
        prediction_arr = model_d.predict(conv_df)

        bar = st.progress(0)
        status_text = st.empty()

        for i in range(1, 70):
            status_text.text(f"{i}% complete")
            bar.progress(i)
            time.sleep(0.01)

        result_arr = []

        for prediction in prediction_arr:
            if prediction == 1:
                result = "Sentimen positif"
            else:
                result = "Sentimen Negatif"
            result_arr.append(result)

        uploaded_result = pd.DataFrame({'Prediction Result': result_arr})

        for i in range(70, 101):
            status_text.text(f"{i}% complete")
            bar.progress(i)
            time.sleep(0.01)
            if i == 100:
                time.sleep(1)
                status_text.empty()
                bar.empty()

        col1, col2 = st.columns([1, 2])

        with col1:
            st.dataframe(uploaded_result)
        with col2:
            st.dataframe(uploaded_df)

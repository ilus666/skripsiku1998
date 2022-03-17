import streamlit as st
import joblib

st.title('Prediksi Sentiment')
st.subheader('Implementasi Sentiment Analysis Berdasarkan Tweets Masyarakat Terhadap Kinerja Presiden dalam Aspek Penanganan Covid-19')
st.text('Algoritma SVM OneVSRest')

#input
my_form = st.form(key="form1")
name = my_form.text_input(label = "Masukkan teks berbahasa indonesia:")
submit = my_form.form_submit_button(label = 'submit')
teks = name.title()

#sistem
model = joblib.load(open('model_B.pkl', 'rb'))
tfidf = joblib.load(open('tf_idf_B.pkl', 'rb'))
data = tfidf.transform([teks])
hasil = model.predict(data)
hasil1 = " ".join(hasil)

#output
if hasil ==  'positif':
    st.write('positif')
elif hasil1 == 'netral':
    st.write('netral')
else:
    st.write('negatif')


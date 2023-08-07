import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
# port_stem = PorterStemmer()
# vectorization = TfidfVectorizer()

load_model = tf.keras.models.load_model('RNN_02.h5')


def fake_news_det(news):
    input_data = [news]
    vectorized_input_data = TfidfVectorizer.transform(input_data)
    prediction = load_model.predict(vectorized_input_data)
    print(prediction)


if __name__ == '__main__':
    st.title('Fake News Classification app ')
    st.subheader("Input the News content below")
    sentence = st.text_area("Enter your news content here", "",height=200)
    predict_btt = st.button("predict")
    if predict_btt:
        prediction_class=fake_news_det(sentence)
        print(prediction_class)
        if prediction_class == [0]:
            st.success('Reliable')
        if prediction_class == [1]:
            st.warning('Unreliable')
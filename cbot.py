# import nltk
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity      
import pandas as pd
import streamlit as st
import warnings
warnings.filterwarnings('ignore')
# import spacy
lemmatizer = nltk.stem.WordNetLemmatizer()
# Download required NLTK data
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')

vb = pd.read_csv('Samsung Dialog.txt', sep = ':', header = None)
vb.head(10)

cust = vb.loc[vb[0] == 'Customer']
sales = vb.loc[vb[0] == 'Sales Agent']

cust.rename(columns = {1: 'Customer'}, inplace = True)
cust = cust[['Customer']].reset_index(drop = True)

sales.rename(columns = {1: 'Sales Agent'}, inplace = True)
sales = sales[['Sales Agent']].reset_index(drop = True)

datax = pd.concat([cust, sales], axis = 1)

# Define a function for text preprocessing (including lemmatization)
def preprocess_text(text):
    global tokens
    # Identifies all sentences in the datax
    sentences = nltk.sent_tokenize(text)
    
    # Tokenize and lemmatize each word in each sentence
    preprocessed_sentences = []
    for sentence in sentences:
        tokens = [lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(sentence) if word.isalnum()]
        # Turns to basic root - each word in the tokenized word found in the tokenized sentence - if they are all alphanumeric 
        # The code above does the following:
        # Identifies every word in the sentence 
        # Turns it to a lower case 
        # Lemmatizes it if the word is alphanumeric

        preprocessed_sentence = ' '.join(tokens)
        preprocessed_sentences.append(preprocessed_sentence)
    
    return ' '.join(preprocessed_sentences)


datax['tokenized Customer'] = datax['Customer'].apply(preprocess_text)
# datax.head()

corpus = datax['tokenized Customer'].to_list()
# corpus

Tfidf_Vectorizer = TfidfVectorizer()
X = Tfidf_Vectorizer.fit_transform(corpus)
# print(X)

# ---------------------------STREAMLIT DESIGN--------------------------
st.markdown("<h1 style = 'color: #EE9322; text-align: center; font-family: helvetica;'>Samsung Devices Chatbot</h1>", unsafe_allow_html = True)

st.markdown("<h4 style = 'margin: -27px; color: #B0A695; text-align: center; font-family: helvetica;'>Created by Ayodeji</h4>", unsafe_allow_html = True)

st.markdown("<br> <br>", unsafe_allow_html= True)
col1, col2 = st.columns(2)
col1.image('cbo.png', caption = 'Samsung Chabot')


def response(user_input):
    user_input_processed = preprocess_text(user_input)
    v_input = Tfidf_Vectorizer.transform([user_input_processed])
    most_similar = cosine_similarity(v_input, X)
    most_similar_index = most_similar.argmax()
    
    return datax['Sales Agent'].iloc[most_similar_index]

chatbot_greeting = [
    "Hello there, welcome to Ayodeji Bot. Pls enjoy your usage",
    "Hi user, this bot is created by Ayodeji, enjoy your usage",
    "Hi HI, How you dey my nigga",
    "Alaye mi, Abeg enjoy your usage",
    "Hey Hey, Pls enjoy your usage"
]

user_greeting = ["hi", "hello there", "hey", "hi there"]
exit_word = ['bye', 'thanks bye', 'exit', 'goodbye']


# print(f'\t\t\t\t\tWelcome To Ayodeji ChatBot\n\n')
# while True:
#     user_q = input('Pls ask your mental illness related question: ')
#     if user_q in user_greeting:
#         print(random.choice(chatbot_greeting))
#     elif user_q in exit_word:
#         print('Thank you for your usage. Bye')
#         break
#     else:
#         responses = response(user_q)
#         print(f'ChatBot:  {responses}') 

# st.write(f'\t\t\t\t\tWelcome To Ayodeji ChatBot\n\n')
# while True:
user_q = col2.text_input('Pls ask questions about our Samsung devices: ')
if user_q in user_greeting:
    col2.write(random.choice(chatbot_greeting))
elif user_q in exit_word:
    col2.write('Thank you for your usage. Bye')
elif user_q == '':
    st.write('')
else:
    responses = response(user_q) 
    col2.write(f'ChatBot:  {responses}') 

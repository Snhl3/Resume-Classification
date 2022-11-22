import pandas as pd
import numpy as np
import streamlit as st
import docx2txt,textract
import pdfplumber
import re, os, docx, spacy
import nltk
from nltk.probability import FreqDist
import plotly.express as px
from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot  as plt
stop=set(stopwords.words('english'))
from pickle import load
import aspose.words as aw
import pickle
vectors = pickle.load(open('vector.pkl','rb'))
model = pickle.load(open('RandomForestClassifier.pkl','rb'))

nltk.download('wordnet')
nltk.download('stopwords')

def add_bg_image():
    st.markdown(
          f"""
          <style>
         .stApp {{
             background-image: url("https://img.freepik.com/free-vector/abstract-background-with-squares_23-2148995948.jpg?w=996&t=st=1663219978~exp=1663220578~hmac=aee3da925492e169a7f9fb7d1aa1577c58a7db3849d8be3f448114080d42a7a7");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True)
     
add_bg_image()

def display(doc_file):

    
    resume = []
    if doc_file.endswith('.docx'):
        resume.append(docx2txt.process(doc_file))

    elif doc_file.endswith('.pdf'):
        with pdfplumber.open(doc_file) as pdf:
            pages=pdf.pages[0]
            resume.append(pages.extract_text())

    elif doc_file.endswith('.doc'):
        fullText = ''
        output = aw.Document()
        output.remove_all_children()
        inputs = aw.Document(doc_file)
        output.append_document(inputs, aw.ImportFormatMode.KEEP_SOURCE_FORMATTING)
        output.save("Output.docx");
        doc = docx.Document('Output.docx')
        
        for para in doc.paragraphs:
            fullText = fullText + para.text
        resume.append(fullText[79:])
            
    return resume

def preprocess(sentence):
    sentence=str(sentence)
    sentence = sentence.lower()
    sentence=sentence.replace('{html}',"") 
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url=re.sub(r'http\S+', '',cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)  
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    lemma_words=[lemmatizer.lemmatize(w) for w in filtered_words]
    return " ".join(lemma_words)

def mostcommon_words(cleaned,i):
    tokenizer = RegexpTokenizer(r'\w+')
    words=tokenizer.tokenize(cleaned)
    mostcommon=FreqDist(cleaned.split()).most_common(i)
    return mostcommon

def display_wordcloud(mostcommon):
    wordcloud=WordCloud(width=1000, height=600, background_color='black').generate(str(mostcommon))
    plt.figure(figsize=(15,15))
    a=px.imshow(wordcloud)
    st.plotly_chart(a)
    plt.axis('off')

def display_words(mostcommon_small):
    x,y=zip(*mostcommon_small)
    chart=pd.DataFrame({'keys': x,'values': y})
    fig=px.bar(chart,x=chart['keys'],y=chart['values'],height=700,width=700)
    st.plotly_chart(fig)

target = {0:'Peoplesoft',1:'SQL Developer',2:'React JS Developer',3:'Workday'}

def main():
    html_temp = f"""
          <style>
         .stApp {{
             background-image: url("https://media.istockphoto.com/vectors/graphic-of-dots-connected-by-lines-in-blue-and-white-tones-vector-id455445559?k=20&m=455445559&s=170667a&w=0&h=YL8P0GXOz5bAAd7RkZ17z25i1KFX4InFQJAiE-fH3mg=");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """
      
     
    st.markdown(html_temp, unsafe_allow_html = True)

    file_type=pd.DataFrame([], columns=['Uploaded File', 'Predicted Profile'])
    filename = []
    predicted = []
    experience = []
    skills = []

    
    
        
    with st.form(key="form1"):
            st.warning(body="Supported file Formats: '.docx' or '.doc' or '.pdf'")
            folder_path =st.text_input(label= "Enter the absolute folder path below in the box")
            st.markdown("Eg.  C:\\Users\\Admin\\Documents\\foldername consisting resumes")
            submit = st.form_submit_button(label="Click Me!")                           

   
            if submit:

                 for files in os.listdir(folder_path):
                    if files.endswith('.docx') or files.endswith('.doc') or files.endswith('.pdf'):
                        file_path  = os.path.join(folder_path, files)
                        filename.append(files)
                        cleaned=preprocess(display(file_path))
                        prediction = model.predict(vectors.transform([cleaned]))[0]
                        predicted.append(target.get(prediction))
                        st.subheader('WORDCLOUD for '+ files)
                        display_wordcloud(mostcommon_words(cleaned,60))

                        st.header('Frequency of 10 Most Common Words for '+ files)
                        display_words(mostcommon_words(cleaned,10))
    if len(predicted) > 0:
        file_type['Uploaded File'] = filename
        file_type['Predicted Profile'] = predicted
        file_type


    

if  __name__ == '__main__':
     main()

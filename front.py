import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import PyPDF2

from plotly import graph_objs as go
from sklearn.linear_model import LinearRegression
from streamlit_card import card

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation 

st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Drive-EZ")
st.sidebar.title("PARAMETERS")
select = st.sidebar.selectbox("Type of File :",["CSV","TXT","PNG","JPEG","PDF","MISC"])
files_list=["Front_End.csv","Data_Science.csv","Back_End.csv","Position_Salaries.csv","Social_Network_Ads.csv"]
if select == "CSV" :
    file_name=st.sidebar.selectbox(
        "Select the file u want to read:",
        files_list
    )
    df1 = pd.read_csv(file_name)
    list1 = list(df1.columns.values)
    
    view_query = st.sidebar.radio("Query",["YES","NO"])
    
    if view_query == "YES" :
        col_name=st.selectbox(
            "Select the column you want to query from:",
            list1
        )
        
        column_query=st.multiselect(
            "Select the : "+col_name,
            options=df1[col_name].unique(),
            default=df1[col_name].unique()
        )
        str = col_name+"==@column_query"
        df_selection=df1.query(str)

        st.dataframe(df_selection)
        
    graph_select=st.sidebar.selectbox(
            "Select between graphs and predictions :",
            ["Graphs","Predictions"]
        )
        
    if graph_select == "Predictions" :
    
        model_list = ["Linear Regression","Polynomial Regression","K Nearest Neighbors","Decision Tree","Logistic Regression","Random Forest","Naive Bayes"]
        
        Linear = ["Data_Science.csv","Front_End.csv","Back_End.csv"]
        Polynomial = ["Position_Salaries.csv"]
        All = ["Social_Network_Ads.csv"]
        
        if file_name in Linear :
            model_list = ["Linear Regression"]
        elif file_name in Polynomial :
            model_list = ["Polynomial Regression"]
        elif file_name in All :
            model_list = ["K Nearest Neighbors","Decision Tree","Logistic Regression","Random Forest","Naive Bayes"]
        
        model_name = st.sidebar.selectbox(
            "Select the Model : ",
            model_list
        )
        
        if model_name == "Linear Regression" :
            x = np.array(df1['experience']).reshape(-1,1)
            fr = LinearRegression()
            fr.fit(x,np.array(df1['salary']))
            
            val = st.number_input("Enter your value : ",0.00,20.00,step = 0.25)
            val = np.array(val).reshape(1,-1)
            pred =fr.predict(val)[0]

            if st.button("Predict"):
                st.success(f"Your predicted salary is {round(pred)}")
        elif model_name == "Polynomial Regression" :
            from polynomialreg import *
            val1 = st.number_input("Enter your value : ",1.00,10.00,step = 0.25,key = "first")
            if st.button("Predict") :
                st.success(f"Your predicted salary is : {round(polreg(file_name,val1))}")
        elif model_name == "K Nearest Neighbors" :
            from knn import *
            val1 = st.number_input("Enter your value : ",key = "first")
            val2 = st.number_input("Enter your value : ",key = "sec")
            if st.button("Predict") :
                st.info(knn(file_name,val1,val2))
                if knn(file_name,val1,val2)[0] == 1 :
                    st.success(f"You can purchase this house !!!")
                else :
                    st.error(f"You can't purchase this house !!!")
        elif model_name == "Decision Tree" :
            from decision_tree import *
            val1 = st.number_input("Enter your value : ",key = "first")
            val2 = st.number_input("Enter your value : ",key = "sec")
            if st.button("Predict") :
                st.info(dec_tree(file_name,val1,val2))
                if list(dec_tree(file_name,val1,val2))[1] == 1 :
                    st.success(f"You can purchase this house !!!")
                else :
                    st.error(f"You can't purchase this house !!!")
        elif model_name == "Logistic Regression" :
            from logistic_reg import *
            val1 = st.number_input("Enter your value : ",key = "first")
            val2 = st.number_input("Enter your value : ",key = "sec")
            if st.button("Predict") :
                st.info(log_reg(file_name,val1,val2))
                if list(log_reg(file_name,val1,val2))[0] == 1 :
                    st.success(f"You can purchase this house !!!")
                else :
                    st.error(f"You can't purchase this house !!!")
        elif model_name == "Random Forest" :
            from random_forest import *
            val1 = st.number_input("Enter your value : ",key = "first")
            val2 = st.number_input("Enter your value : ",key = "sec")
            if st.button("Predict") :    
                st.info(ranfor(file_name,val1,val2))
                if list(ranfor(file_name,val1,val2))[0] == 1 :
                    st.success(f"You can purchase this house !!!")
                else :
                    st.error(f"You can't purchase this house !!!")
            #ranfor(file_name)
        elif model_name == "Naive Bayes" :
            from naive_bayes import *
            val1 = st.number_input("Enter your value : ",key = "first")
            val2 = st.number_input("Enter your value : ",key = "sec")
            if st.button("Predict") :
                st.info(naive_bayes(file_name,val1,val2))
                if list(naive_bayes(file_name,val1,val2))[0] == 1 :
                    st.success(f"You can purchase this house !!!")
                else :
                    st.error(f"You can't purchase this house !!!")
            #naive_bayes(file_name)
    else :
        graph_type = ["Scatter plot","Line plot","Bar plot"]
        
        graph_name = st.sidebar.selectbox(
            "Select the graph type :",
            graph_type
        )
        
        result = df1.select_dtypes(include='number')
        list1 = list(result.columns.values)
        col_name_x = st.selectbox(
                "Select the column you want at x axis :",
                list1
            )
        
        col_name_y = st.selectbox(
                "Select the column you want at y axis :",
                list1
            )
            
        if graph_name == "Scatter plot" :
            plt.scatter(df1[col_name_x],df1[col_name_y])
        if graph_name == "Line plot" :
            plt.plot(df1[col_name_x],df1[col_name_y])
        if graph_name == "Bar plot" :
            plt.bar(df1[col_name_x],df1[col_name_y])
        #plt.ylim(0)
        plt.xlabel(col_name_x)
        plt.ylabel(col_name_y)
        #plt.tight_layout()
        st.pyplot()

elif select =="TXT":
    
    file_txt = ["chess.txt","computer.txt","volcano.txt","word_edit.txt"]
    txt_file=st.sidebar.selectbox(
            "Select the txt file :",
            file_txt
        )

    text = open(txt_file,"r")
    text = text.read()
    stopwords = list(STOP_WORDS)
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    tokens = [token.text for token in doc]
    punctuation = punctuation + '\n'
    word_frequencies = {}
    for word in doc:
        if word.text.lower() not in stopwords:
            if word.text.lower() not in punctuation:
                if word.text not in word_frequencies.keys():
                    word_frequencies[word.text] = 1
                else:
                    word_frequencies[word.text] += 1

    max_frequency = max(word_frequencies.values())

    for word in word_frequencies.keys():
        word_frequencies[word] = word_frequencies[word]/max_frequency

    sentence_tokens = [sent for sent in doc.sents]

    sentence_scores = {}
    for sent in sentence_tokens:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent] += word_frequencies[word.text.lower()]

    from heapq import nlargest
    select_length = int(len(sentence_tokens)*0.3)
    summary = nlargest(select_length, sentence_scores, key = sentence_scores.get)
    final_summary = [word.text for word in summary]
    summary = ' '.join(final_summary)
    display=st.selectbox(
            "Select between summarised and full text:",
            ["full","summary"]
        )
    
    #txt=st.text_area("Your Text HERE")

    if display=="full":
        txt=st.text_area("",text,height = 400)
        #txt.write(text)
    else:
        txt=st.text_area("",summary,height = 400)
        #txt.write(summary)
        
elif select == "PNG":
    arr = ['hal.png']
    for i in arr:
        image=Image.open(i)
        st.image(image, caption=i) 
elif select=='JPEG':

    arr=["riv.jpeg","him.jpeg"]
    for i in arr:
        image=Image.open(i)
        st.image(image, caption=i)

elif select=='PDF':
    
    file_pdf = ["pdfsample.pdf","samplepdf.pdf"]
    pdf_file = st.sidebar.selectbox(
            "Select the pdf file :",
            file_pdf
        )
    
    def pdf_to_txt(pdf):
        pdffileobj=open(pdf,'rb')
        pdfreader=PyPDF2.PdfFileReader(pdffileobj)
        x=pdfreader.numPages
        pageobj=pdfreader.getPage(x-1)
        text=pageobj.extractText()
        file1=open(r"1.txt","w")
        file1.writelines(text)
        #file_txt.append(file1)
        file1.close()
    
    pdf_to_txt(pdf_file)
    
    text = open(r'1.txt',"r")
    text = text.read()
    stopwords = list(STOP_WORDS)
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    tokens = [token.text for token in doc]
    punctuation = punctuation + '\n'
    word_frequencies = {}
    for word in doc:
        if word.text.lower() not in stopwords:
            if word.text.lower() not in punctuation:
                if word.text not in word_frequencies.keys():
                    word_frequencies[word.text] = 1
                else:
                    word_frequencies[word.text] += 1

    max_frequency = max(word_frequencies.values())

    for word in word_frequencies.keys():
        word_frequencies[word] = word_frequencies[word]/max_frequency

    sentence_tokens = [sent for sent in doc.sents]

    sentence_scores = {}
    for sent in sentence_tokens:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent] += word_frequencies[word.text.lower()]

    from heapq import nlargest
    select_length = int(len(sentence_tokens)*0.3)
    summary = nlargest(select_length, sentence_scores, key = sentence_scores.get)
    final_summary = [word.text for word in summary]
    summary = ' '.join(final_summary)
    display=st.selectbox(
            "Select between summarised and full text:",
            ["full","summary"]
        )
    
    #txt=st.text_area("Your Text HERE")

    if display=="full":
        txt=st.text_area("",text,height = 400)
        #txt.write(text)
    else:
        txt=st.text_area("",summary,height = 400)
        #txt.write(summary)    
    
elif select=='MISC':
    arr=['logging.h','mutex.h']
    for i in arr:
        image=Image.open('place.jpg')
        st.image(image, caption=i)

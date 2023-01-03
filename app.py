

import time
import streamlit as st
from time import sleep
from stqdm import stqdm # for getting animation after submit event
import pandas as pd
from transformers import pipeline
import json
import spacy
import spacy_streamlit
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import plotly.express as px
import altair as alt
import seaborn as sns
import datetime
from datetime import date
import spacy_streamlit
import asent
import streamlit.components.v1 as components
import base64

from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image

import requests

from skimage.transform import resize
import numpy as np
import time






file_ = open("images/dwf-robo.gif", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()


text, logo = st.columns(2)
with logo:
    st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="Glossa" width="180" height="250">',
        unsafe_allow_html=True
        )

with text:
    st.markdown("<H1 style='text-align: left; color:grey;'>Glossa</p>", unsafe_allow_html=True)



#
# from PIL import Image
# image = Image.open('/Users/jekedum/Documents/apps/env/ibm-watson-logo.gif')
#
# # st.sidebar.image(image, use_column_width=True)
# st.image(image, caption ='This Natural Language Processing Based Web App can do anything u can imagine with Text 😱', use_column_width=True )


st.markdown("<h4 style='text-align: center; color: grey;'>A general-purpose application that can perform a number of natural language tasks such as questioning and answering, language translation, generative text and more </h4>", unsafe_allow_html=True)




def main():
    menu = ["--Select--","Summarizer","Named Entity Recognition","Sentiment Analysis","Question Answering","Computer Vision"]
    choice = st.sidebar.selectbox("Choose your ML/NLP task!", menu)

    if choice=="--Select--":

        st.markdown("<H1 style='text-align: left; color:grey;'>Glossa Use Cases </p>", unsafe_allow_html=True)

        col1, col2 = st.columns([4,4])
        col3,col4  = st.columns([4,4])


        with col1:
            st.markdown("<H4 style='text-align: left; color:grey;'>Text summarization</H4>", unsafe_allow_html=True)
            st.write("Create shorter text without removing the semantic structure of text.")



        with col2:
            st.markdown("<H4 style='text-align: left; color:grey;'>Named entity Recognition</H4>", unsafe_allow_html=True)
            st.write("Identify and categorizing key information (entities) in text")



        with col3:
            st.markdown("<H4 style='text-align: left; color:grey;'>Question Answering</H4>", unsafe_allow_html=True)
            st.write("Provide answers to questions in a simple to consume form.")



        with col4:
            st.markdown("<H4 style='text-align: left; color:grey;'>Text Completion</H4>", unsafe_allow_html=True)
            st.write("automated sentence completion")

    # select choice 2
    elif choice=="Summarizer":
        st.markdown("<H4 style='text-align: center; color:orange;'>Text Summarization</H4>", unsafe_allow_html=True)
        raw_text = st.text_area("","Enter the text you want to summarize !", height = 200)
        num_words = st.number_input("Enter number of words in summarized output (max words required is 50) ")
        btnResult = st.button('Analyze Text')


        if raw_text!="" and num_words is not None and btnResult:
            num_words = int(num_words)
            with st.spinner("Wait..Analyzing Sentiment.."):
                time.sleep(5)
                summarizer = pipeline('summarization')
                summary = summarizer(raw_text, min_length=num_words,max_length=50)
                result_summary = summary[0]['summary_text'].strip().capitalize()
                st.write(f"Here's your text summary: {result_summary}")





    elif choice=="Named Entity Recognition":
        # !python -m spacy download en
        nlp = spacy.load("en_core_web_sm")
        st.subheader("Text Based Named Entity Recognition")
        raw_text = st.text_area("Your Text","Enter the Text below To extract Named Entities !",height = 200)
        btnResult = st.button('Analyze Text')
        if raw_text !="Enter Text Here" and btnResult:

            doc = nlp(raw_text)
            with st.spinner("Wait..generating entities.."):
                time.sleep(5)
            # for _ in stqdm(range(50), desc="Please wait a bit. The model is fetching the results !!"):
            #     sleep(0.1)
                spacy_streamlit.visualize_ner(doc, labels=nlp.get_pipe("ner").labels, title= "List of Entities")


    elif choice=="Sentiment Analysis":
            st.subheader("Sentiment Analysis")
            raw_text = st.text_area("","Enter text to find out it's sentiments!", height = 200)
            click = st.button('Analyze Text')
            if raw_text !="Enter Text Here" and click:
                nlp = spacy.blank('en')
                nlp.add_pipe('sentencizer')
                # add the rule-based sentiment model
                nlp.add_pipe('asent_en_v1')
                doc = nlp(raw_text)

                with st.spinner("Wait..Analyzing Sentiment.."):
                    time.sleep(5)
                #df = pd.DataFrame({'Negative Sentiment Score': [doc._.polarity.negative],'Positve Sentiment Score':[doc._.polarity.positive], 'Compound Score':[doc._.polarity.compound]})
                # st.dataframe(df)
                col1,col2,col3 = st.columns(3)
                with col1:
                    st.write(f" Positive Sentiment: { round(doc._.polarity.positive,3)}  🤗")
                with col2:
                    st.write(f" Negative Sentiment: { round(doc._.polarity.negative,2) } 😤")

                with col3:
                    st.write(f" Compund: { round(doc._.polarity.neutral,2) }  😐")


                st.subheader("Visualizer")


                components.html(asent.visualize(doc, style='prediction'), height = 300)

    elif choice=="Question Answering":
            st.subheader("Question Answering")
            context = st.text_area("Context","Enter the Context Here",height = 200)
            question = st.text_area("Your Question","Enter your Question Here")
            click = st.button('Analyze Text')

            if context !="Enter Text Here" and question!="Enter your Question Here" and click:


                question_answering = pipeline(model="deepset/roberta-base-squad2")

                with st.spinner("Wait..Lets generate answers to your question.."):
                    time.sleep(10)


                result = question_answering(question=question, context=context)



                st.write(result['answer'].strip().capitalize())

                st.markdown("<H4 style='text-align: center; color:grey;'> Probability associated to the answer </h4>", unsafe_allow_html=True)

                st.write(round(result['score'],2))


    elif choice=="Computer Vision":
            st.write('\n')
            st.subheader("Image Classification")
            # For newline
            st.write('\n')
            # image = Image.open('/Users/jekedum/Downloads/black-5KDJBNZ.jpg')
            # show = st.image(image, use_column_width=True)
            # st.sidebar.title("Upload Image")
            #Disabling warning
            st.set_option('deprecation.showfileUploaderEncoding', False)
            # image = Image.open('/Users/jekedum/Downloads/black-5KDJBNZ.jpg')
            # show = st.image(image,use_column_width=True)


            #Choose your own image
            uploaded_file = st.file_uploader("Choose an image",type=['png', 'jpg', 'jpeg'] )


            if uploaded_file is not None:
                u_img = Image.open(uploaded_file)
                st.image(u_img, 'Uploaded Image', use_column_width=True)
                # We preprocess the image to fit in algorithm.
                image = np.asarray(u_img)/255
                st.markdown("""
                <style>.stSpinner > div > div {
                    border-top-color: #0f0;
                }</style>
                """, unsafe_allow_html=True)

                if st.button("Classify image"):
                    with st.spinner('Wait for it...'):
                        feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
                        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

                        inputs = feature_extractor(images=image, return_tensors="pt")
                        outputs = model(**inputs)
                        logits = outputs.logits
                        # model predicts one of the 1000 ImageNet classes
                        predicted_class_idx = logits.argmax(-1).item()
                        st.success(model.config.id2label[predicted_class_idx])



if __name__ == '__main__':
    main()













# # dashboard analysis
#
# import pandas as import pd
# import numpy as np
#
#
#
# df_survey = pd.read_excel('/Users/jekedum/Documents/survey_data/ux survey june2022.xlsx')
#
#
#
#
#
# # number of people who had surveyed administered.
#
# df_survey.Visitorid.nunique()
#
#
#
# # progress made
#
# def progress_encode(col):
#     if col == '[1,2,3]':
#         return '3'
#
#     elif col == '[1,2]':
#         return '2'
#
#     else:
#         return '1'
#
# df_survey['Progress_made'] = df_survey.Progress.apply(progress_encode)
#
#
#
# # nps bucket
# def nps_bucket(x):
#     if x > 8:
#         bucket = 'promoter'
#     elif x > 6:
#         bucket = 'passive'
#     elif x>= 0:
#         bucket = 'detractor'
#     else:
#         bucket = 'no score'
#     return bucket
#
# df_survey['NPS Bucket'] = df_survey['How likely are you to recommend Adobe Marketo Engage to a colleague or friend? '].apply(nps_bucket)
#
# df_survey['NPS Bucket'].value_counts()
#
#
# # nps score
#
#
# # nps score
# def nps_score(col):
#     return (col.str.contains('promoter').sum() - col.str.contains('detractor').sum()) / (col.str.contains('promoter').sum() + col.str.contains('passive').sum() + col.str.contains('detractor').sum())
#
# nps_score(df_survey['NPS Bucket'])
#
#
#
# # satisfaction score
#
#
# def csat(col, data = df_survey):
#     return df_survey[col].gt(4).sum() / df_survey[col].count()
#
# csat('How satisfied are you with the capabilities of Adobe Marketo Engage? ')
#
#
#
#
#
#
# #import necessary libraries
#
# #for importing data and wrangling
# import pandas as pd
# import numpy as np
#
# #for plotting images & adjusting colors
# import matplotlib.pyplot as plt
# import matplotlib as mpl
#
# from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
#
# from PIL import Image
#
#
#
#
#
#
#
#
#
#
# open_responses  =  df_survey['What are some additional improvements we could make that would increase your satisfaction?'].dropna()
# text = open_responses.tolist()
# # join the list and lowercase all the words
# text = ' '.join(text).lower()
#
# #create the wordcloud object
# wordcloud = WordCloud(stopwords = STOPWORDS, collocations=True).generate(text)
# #plot the wordcloud object
# plt.imshow(wordcloud, interpolation='bilInear')
# plt.axis('off')
# plt.show()
#
#
#
#
#
#
#
#
#
#
#
# import spacy
# import asent
#
#
#
# df_survey.head()
#
#
#
#
#
# def sentiment_analysis():
#     nlp = spacy.blank('en')
#     nlp.add_pipe('sentencizer')
#     nlp.add_pipe('asent_en_v1')
#     for i in
#
#
#
# # load spacy pipeline
# nlp = spacy.blank('en')
# nlp.add_pipe('sentencizer')
#
# # add the rule-based sentiment model
# nlp.add_pipe('asent_en_v1')
#
# # try an example
# text = 'I am not very happy, but I am also not especially sad'
# doc = nlp(text)
#
# doc._.polarity
#
#
# # print polarity of document, scaled to be between -1, and 1
# print(doc._.polarity)
# # neg=0.0 neu=0.631 pos=0.369 compound=0.7526
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# wc = WordCloud(background_color="peachpuff",
#             max_words=110,
#                width=500,
#                height=500,
#                random_state=143,
#                stopwords=STOPWORDS,
#                color_func=lambda *args, **kwargs: (255,255,255)).generate(text)
# plt.figure(figsize=(30,20))
# plt.axis("off")
# plt.imshow(wc)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# # sentiment analysis
#
# import spacy
# import asent
#
# # load spacy pipeline
# nlp = spacy.blank('en')
# nlp.add_pipe('sentencizer')
#
# # add the rule-based sentiment model
# nlp.add_pipe('asent_en_v1')
#
# # try an example
# text = 'I am not very happy, but I am also not especially sad'
# doc = nlp(text)
#
# # print polarity of document, scaled to be between -1, and 1
# print(doc._.polarity)
# # neg=0.0 neu=0.631 pos=0.369 compound=0.7526
#
# # Naturally, a simple score can be quite unsatisfying, thus Asent implements a series of visualizer to interpret the results:
# asent.visualize(doc, style='prediction')
#  # or
# asent.visualize(doc[:5], style='analysis')
#
#
#
#
#
#
#
#
#
#
#
#
#
# import spacy
# from spacytextblob.spacytextblob import SpacyTextBlob
#
# nlp = spacy.load('en_core_web_sm')
#
# nlp.add_pipe('spacytextblob')
#
# text = 'I had a really horrible day. It was the worst day ever! But every now and then I have a really good day that makes me happy.'
# doc = nlp(text)
# doc._.blob.polarity                            # Polarity: -0.125
# doc._.blob.subjectivity                        # Subjectivity: 0.9
# doc._.blob.sentiment_assessments.assessments   # Assessments: [(['really', 'horrible'], -1.0, 1.0, None), (['worst', '!'], -1.0, 1.0, None), (['really', 'good'], 0.7, 0.6000000000000001, None), (['happy'], 0.8, 1.0, None)]
# doc._.blob.ngrams()
#
#



# Delighted will take all responses that are “agree” (5, 6, or 7) and divide that by the total number of responses.


# Calculating CSAT
# To calculate a CSAT score from your survey data, you’ll use the responses of 4 (satisfied) and 5 (very satisfied). It has been shown that using the two highest values on feedback surveys is the most accurate predictor of customer retention.
#
# To do this calculation you’ll also need to know the total number of responses you’ve received. This number is easy to locate if you’re running your customer feedback program through a centralized platform.
#
# Use this formula to arrive at a percentage score:
#
# (Number of satisfied customers (4 and 5) / Number of survey responses) x 100 = % of satisfied customers



# df_survey['NPS Bucket'].apply(nps_score)
#
#
# df_survey['nps']
















# !pip install scrubadub_spacy
# import scrubadub, scrubadub_spacy
#
# scrubber = scrubadub.Scrubber()
#
# scrubber.add_detector(scrubadub_spacy.detectors.SpacyEntityDetector)
# scrubber.clean("My name is Alex, I work at LifeGuard in London, and my eMail is alex@lifeguard.com btw. my super secret twitter login is username: alex_2000 password: g-dragon180888")
#
#
#





# My name is {{NAME}}, I work at {{ORGANIZATION}} in {{LOCATION}}, and my eMail is {{EMAIL}} btw. my super secret twitter login is username: {{USERNAME}} password: {{PASSWORD}}







# @st.cache
# def convert_df(df):
#     # IMPORTANT: Cache the conversion to prevent computation on every rerun
#     return df.to_csv().encode('utf-8')
#
# csv = convert_df(my_large_df)
#
# st.download_button(
#     label="Download data as CSV",
#     data=csv,
#     file_name='large_df.csv',
#     mime='text/csv',
# )
#
#

#
#
#
#
# pd.DataFrame
#












#
# Probability associated to the answer
#


#
# if sentiment =="POSITIVE":
#                 st.write(f"# This text has a Positive Sentiment.{doc._.polarity.positive}🤗")
#             elif sentiment =="NEGATIVE":
#                 st.write("""# This text has a Negative Sentiment. 😤""")
#             elif sentiment =="NEUTRAL":
#                 st.write("""# This text seems Neutral ... 😐""")
#
#
#








#
#
# def draw_all(key, plot=False):
#     st.write(
#         """
#         # PCA NLP Web App
#
        # This Natural Language Processing Based Web App can do anything u can imagine with Text. 😱
        #
        # This App is built using pretrained transformers which are capable of doing wonders with the Textual data.
        #
        # ```python
        # # Key Features of this App.
        # 1. Advanced Text Summarizer
        # 2. Named Entity Recognition
        # 3. Sentiment Analysis
        # 4. Question Answering
        # 5. Text Completion
#
#         ```
#         """
#     )



#
# with st.sidebar:
#     draw_all("sidebar")




#
# def main():
#     st.title("NLP Web App")
#     menu = ["--Select--","Summarizer","Named Entity Recognition","Sentiment Analysis","Question Answering","Text Completion","File Upload"]
#     choice = st.sidebar.selectbox("Choose What u wanna do !!", menu)
#
#     if choice=="--Select--":
#
#         st.write("""
#
#                  This is a Natural Language Processing Based Web App that can do anything u can imagine with the Text.
#         """)
#
#         st.write("""
#
#                  Natural Language Processing (NLP) is a computational technique to understand the human language in the way they spoke and write.
#         """)
#
#         st.write("""
#
#                  NLP is a sub field of Artificial Intelligence (AI) to understand the context of text just like humans.
#         """)


        # st.image('banner_image.jpg')


#
#     elif choice=="Summarizer":
#         st.subheader("Text Summarization")
#         st.write(" Enter the Text you want to summarize !")
#         raw_text = st.text_area("Your Text","Enter Your Text Here")
#         num_words = st.number_input("Enter Number of Words in Summary")
#
#         if raw_text!="" and num_words is not None:
#             num_words = int(num_words)
#             summarizer = pipeline('summarization')
#             summary = summarizer(raw_text, min_length=num_words,max_length=50)
#             s1 = json.dumps(summary[0])
#             d2 = json.loads(s1)
#             result_summary = d2['summary_text']
#             result_summary = '. '.join(list(map(lambda x: x.strip().capitalize(), result_summary.split('.'))))
#             st.write(f"Here's your Summary : {result_summary}")
#
#
#     elif choice=="Named Entity Recognition":
#         nlp = spacy.load("en_core_web_sm")
#         st.subheader("Text Based Named Entity Recognition")
#         st.write(" Enter the Text below To extract Named Entities !")
#
#         raw_text = st.text_area("Your Text","Enter Text Here")
#         if raw_text !="Enter Text Here":
#             doc = nlp(raw_text)
#             for _ in stqdm(range(50), desc="Please wait a bit. The model is fetching the results !!"):
#                 sleep(0.1)
#             spacy_streamlit.visualize_ner(doc, labels=nlp.get_pipe("ner").labels, title= "List of Entities")
#
#     elif choice=="Sentiment Analysis":
#         st.subheader("Sentiment Analysis")
#         sentiment_analysis = pipeline("sentiment-analysis")
#         st.write(" Enter the Text below To find out its Sentiment !")
#
#         raw_text = st.text_area("Your Text","Enter Text Here")
#         click = st.button('Run')
#         if raw_text !="Enter Text Here" and click:
#             result = sentiment_analysis(raw_text)[0]
#             sentiment = result['label']
#             for _ in stqdm(range(50), desc="Please wait a bit. The model is fetching the results !!"):
#                 sleep(0.1)
#             if sentiment =="POSITIVE":
#                 st.write("""# This text has a Positive Sentiment.  🤗""")
#             elif sentiment =="NEGATIVE":
#                 st.write("""# This text has a Negative Sentiment. 😤""")
#             elif sentiment =="NEUTRAL":
#                 st.write("""# This text seems Neutral ... 😐""")
#
#     elif choice=="Question Answering":
#         st.subheader("Question Answering")
#         st.write(" Enter the Context and ask the Question to find out the Answer !")
#         question_answering = pipeline("question-answering")
#
#
#         context = st.text_area("Context","Enter the Context Here")
#
#         question = st.text_area("Your Question","Enter your Question Here")
#
#         if context !="Enter Text Here" and question!="Enter your Question Here":
#             result = question_answering(question=question, context=context)
#             s1 = json.dumps(result)
#             d2 = json.loads(s1)
#             generated_text = d2['answer']
#             generated_text = '. '.join(list(map(lambda x: x.strip().capitalize(), generated_text.split('.'))))
#             st.write(f" Here's your Answer :\n {generated_text}")
#
#     elif choice=="Text Completion":
#         st.subheader("Text Completion")
#         st.write(" Enter the uncomplete Text to complete it automatically using AI !")
#         text_generation = pipeline("text-generation")
#         message = st.text_area("Your Text","Enter the Text to complete")
#
#
#         if message !="Enter the Text to complete":
#             generator = text_generation(message)
#             s1 = json.dumps(generator[0])
#             d2 = json.loads(s1)
#             generated_text = d2['generated_text']
#             generated_text = '. '.join(list(map(lambda x: x.strip().capitalize(), generated_text.split('.'))))
#             st.write(f" Here's your Generate Text :\n   {generated_text}")
#
#
#
# if __name__ == '__main__':
# 	main()


#
#
# nlp = spacy.load("en_core_web_sm")
# raw_text = "Our current PCA clusters only account for core AA usage (125K users), not CJA usage (1K users and growing). If our cluster definitions are not updated to account for CJA, our clusters will be missing out on critical usage data from some of our key customers".
# doc = nlp(raw_text)
#
# spacy_streamlit.visualize_ner(doc, labels=nlp.get_pipe("ner").labels, title= "List of Entities")
#
#
#
#
#
#         raw_text = st.text_area("Your Text","Enter Text Here")
#         if raw_text !="Enter Text Here":
#             doc = nlp(raw_text)
#             for _ in stqdm(range(50), desc="Please wait a bit. The model is fetching the results !!"):
#                 sleep(0.1)
#             spacy_streamlit.visualize_ner(doc, labels=nlp.get_pipe("ner").labels, title= "List of Entities")




#
#
# spacy_streamlit.visualize_ner(doc, labels=nlp.get_pipe("ner").labels, title= "List of Entities")
#
#
#
#
# doc = nlp(text)
#
#
#
#
#
#
# text ="""
# Shawn Corey Carter (born December 4, 1969), known professionally as Jay-Z,[a] is an American rapper, record producer, entrepreneur, and founder of Manhattan-based conglomerate talent and entertainment agency Roc Nation. He is regarded as one of the most influential hip-hop artists.[8] He was the CEO of Def Jam Recordings and he has been central to the creative and commercial success of artists including Kanye West, Rihanna, and J. Cole.[9][10]
#
# Born and raised in New York City, Jay-Z first began his musical career in the late 1980s; he co-founded the record label Roc-A-Fella Records in 1995 and released his debut studio album Reasonable Doubt in 1996. The album was released to widespread critical success, and solidified his standing in the music industry. He went on to release twelve additional albums, including the acclaimed albums The Blueprint (2001), The Black Album (2003), American Gangster (2007), and 4:44 (2017).[11][12] He also released the full-length collaborative albums Watch the Throne (2011) with Kanye West and Everything Is Love (2018) with his wife Beyoncé, respectively.[13]
#
# Through his business ventures Jay-Z has become the first hip-hop billionaire.[14][15] In 1999, he founded the clothing retailer Rocawear,[16] and in 2003, he founded the luxury sports bar chain 40/40 Club. Both businesses have grown to become multi-million-dollar corporations, and allowed him to start up Roc Nation in 2008. In 2015, he acquired the tech company Aspiro and took charge of their media streaming service Tidal.[17][18] In 2020, he launched "Monogram", a line of cannabis products.[19]
#
# One of the world's best-selling music artists, with over 140 million records sold, Jay-Z has won 24 Grammy Awards, the joint-most Grammy awards of any rapper along with Kanye West. Jay-Z also holds the record for the most number-one albums by a solo artist on the Billboard 200 (14).[20][21] Additionally, he is a recipient of the NAACP's President's Award, a Primetime Emmy Award, and a Sports Emmy Award; and has also received a nomination for a Tony Award. Ranked by Billboard and Rolling Stone"""















#
#
#
# # load spacy pipeline
# nlp = spacy.blank('en')
# nlp.add_pipe('sentencizer')
# # add the rule-based sentiment model
# nlp.add_pipe('asent_en_v1')
#
# # try an example
# text = "I am not very happy, but I am also not especially sad. I'm grateful and blessed but my grandma died yesterday"
#
#
# doc = nlp(text)
#
# # print polarity of document, scaled to be between -1, and 1
# doc._.polarity.negative
#
#
#
#
# # neg=0.0 neu=0.631 pos=0.369 compound=0.7526
#
# # Naturally, a simple score can be quite unsatisfying, thus Asent implements a series of visualizer to interpret the results:
#
#
#
#
#
# spacy_streamlit
# #
#  # or
# asent.visualize(doc[:5], style='analysis')
#
#
#
#
# sentiment_analysis = pipeline("sentiment-analysis")
#
#
#
# elif choice=="Sentiment Analysis":
#         st.subheader("Sentiment Analysis")
#         st.write(" Enter the Text below To find out its Sentiment !")
#         raw_text = st.text_area("Your Text","Enter Text Here")
#         click = st.button('Run')
#         if raw_text !="Enter Text Here" and click:
#             nlp = spacy.blank('en')
#             nlp.add_pipe('sentencizer')
#             # add the rule-based sentiment model
#             nlp.add_pipe('asent_en_v1')
#             doc = nlp(raw_text)
#             st.write(f"Here's your sentiment analysis: {doc._.polarity}")
#





            # print polarity of document, scaled to be between -1, and 1









#
#
#
#
#
#
# import eng_spacysentiment
# nlp = eng_spacysentiment.load()
# text = "Welcome to Arsenals official YouTube channel Watch as we take you closer and show you the personality of the club"
# doc = nlp(text)
# print(doc.cats)
# # {'positive': 0.29878824949264526, 'negative': 0.7012117505073547}
#
#
#
#
#
#
# # pip install PyTextRank
# import spacy
# import pytextrank
#
# # example text
# text = """Compatibility of systems of linear constraints over the set of natural numbers.
# Criteria of compatibility of a system of linear Diophantine equations, strict inequations,
# and nonstrict inequations are considered. Upper bounds for components of a minimal set of
# solutions and algorithms of construction of minimal generating sets of solutions for all types
# of systems are given. These criteria and the corresponding algorithms for constructing a minimal
# supporting set of solutions can be used in solving all the considered types systems and systems of mixed types."""
#
# # load a spaCy model, depending on language, scale, etc.
# nlp = spacy.load("en_core_web_sm")
# # add PyTextRank to the spaCy pipeline
# nlp.add_pipe("textrank")
#
# doc = nlp(text)
#
#
# # examine the top-ranked phrases in the document
# for phrase in doc._.phrases:
#     print(phrase.text)
#     print(phrase.rank, phrase.count)
#     print(phrase.chunks)
#
#
#
#
# import scrubadub, scrubadub_spacy
#
# scrubber = scrubadub.Scrubber()
# scrubber.add_detector(scrubadub_spacy.detectors.SpacyEntityDetector)
# print(scrubber.clean("My name is Alex, I work at LifeGuard in London, and my eMail is alex@lifeguard.com btw. my super secret twitter login is username: alex_2000 password: g-dragon180888"))
#
#
#
# # My name is {{NAME}}, I work at {{ORGANIZATION}} in {{LOCATION}}, and my eMail is {{EMAIL}} btw. my super secret twitter login is username: {{USERNAME}} password: {{PASSWORD}}\\\\\\









#The probability associated to the answer

#
#
# from transformers import pipeline
#
# oracle = pipeline(model="deepset/roberta-base-squad2")
#
# result = oracle(question="What is my name?", context="My name is Wolfgang and I live in Berlin")
#
# summary[0]['summary_text'].strip().capitalize()
#
# result
#
# {'score': 0.9191, 'start': 34, 'end': 40, 'answer': 'Berlin'}
#

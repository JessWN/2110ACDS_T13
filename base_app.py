"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os			#Loading the model & accessing OS File System
from PIL import Image		#Importing logo Image

import matplotlib

from wordcloud import WordCloud


#import webpage image
img = Image.open('resources\imgs\performing-twitter-sentiment-analysis1.png') 
#Set the Page Title
st.set_page_config(page_title= 'JHUST Inc.: Climate Change Sentiment Classification', page_icon= img )


# Data dependencies
import pandas as pd

# Tokenizing the train dataset
from sklearn.feature_extraction.text import TfidfVectorizer


# Prediction Model
news_vectorizer = open("resources/BNBModel.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your predictive model from the pkl file

# Load your train_df data
train_df = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	# options = ["Prediction", "Information"]
	# selection = st.sidebar.selectbox("Choose Option", options)

	# options_2 = ["PredicT", "Information"]
	# selection_2 = st.sidebar.selectbox("Choose Options", options_2)





	# if st.checkbox('Show train_df data'): # data is hidden if box is unchecked
	# 	st.write(train_df[['sentiment', 'message']]) # will write the df to the page

	# st.select_slider('Display Values', ['this', 'these', 'that'])

	# Creates a main title and subheader on your page -
	# these are static across all pages

	# Introductory Page, describing the solution
	def introduction_page():

		st.title("Climate Change Sentiment Classification")
		st.markdown('---')
		st.subheader("Developed by JHUST Inc. -Team 13")
		st.markdown('### Development Team')
		st.markdown(' - Jessica Njuguna \n - Hunadi Mawela \n - Uchenna Unigwe \n - Stanley Agbo \n - Teddy Waweru \n')
		st.markdown('## Introduction')
		st.markdown('---')

		#Climate Change Narrative
		st.markdown(
			"""
			##### Climate Change
			Species on earth are dependent on optimum conditions for their survival, & human civilization is not exempted. Having
			the right _temperature_, _sufficient water_, _clean air_, and _enough food_ are the **basic needs** to ensure for survival.
			These needs are dependent on the **earth's climatic stability**.

			**Climate change** refers to shifts in the _earth's weather patterns_. Both **human activity** and **natural occurrences** contribute to climate change,
			but the former has been the main driver since _industrialization in the 19th Century_. ie. With increased deforestation, combustion of fossil
			fuels and livestock farming, the concentration of **greenhouse gases** has increased. These gases cause a greenhouse effect by trapping the sun's heat and escalating **global warming**.

			**Warmer temperatures** disrupt the _natural climate cycles_ of the earth. These disruptions degrade the quality of the Earth's air and water, and is bound to become exacerbated if the
			human race does not reverse its course.

			##### Strategic Action

			In a bid to curb the rate of climate change, global organizations are invested into building their brands with emphasis on ensuring that their operations,
			products & services are environmentally friendly and sustainable.
			
			To support market research on climate change, **JHUST Inc.** developed a **Machine Learning Classification model**, whose purpose is to determine people's **perception** of climate change,
			and whether or not they believe it is a real threat. 
			
			Providing a robust ML solution will enable our clients to access to a broad base of **consumer sentiment**, spanning multiple demographic and geographic categories,
			thus increasing their insights and informing **future marketing strategies**.

			"""

		)

		#Project Links
		st.markdown(
			"""
			#### Project Links
			[Github Repository][1] |
			[Google Slides Presentation][2]

			[1]: https://github.com/JessWN/2110ACDS_T13
			[2]: https://docs.google.com/presentation/d/1-AIbZcDdUDmvVoIB4WoJcIZslbbdb6S9bujMNEgpuHw/edit?usp=sharing
			[3]: 
			[4]: 

			"""
		)

		st.text('')
		st.text('')


		if st.checkbox('Load Exploratory Data Analysis'):
			BROWSE_PAGES['Exploratory Data Analysis']()
			# pass

			# pass

	# Insights from EDA of the training dataset
	def exploratory_data_analysis():
		st.title('Exploratory Data Analysis')

		st.image('resources/imgs/word_cloud.png'

		)

		if st.checkbox('Load Modelling'):
			# BROWSE_PAGES['Exploratory Data Analysis']()
			BROWSE_PAGES['Modelling']()
			# pass


	def modelling():
		st.title("Tweet Classification Models")

		pass

	def prediction():
		pass


	#Dictionary of radio buttons & functions that are loaded depending on page selected
	BROWSE_PAGES = {
		'Introduction': introduction_page,
		'Exploratory Data Analysis': exploratory_data_analysis,
		'Modelling': modelling,
		'Prediction': prediction

	}

	#Page Navigation Title & Radio BUttons
	st.sidebar.title('Navigation')
	page = st.sidebar.radio('Go to:',list(BROWSE_PAGES.keys()))

	#Load function depending on radio selected above.
	BROWSE_PAGES[page]()

	# # Building out the "Information" page
	# if selection == "Information":
	# 	st.info("General Information")
	# 	# You can read a markdown file from supporting resources folder
	# 	st.markdown("Some information here")

	# 	st.subheader("train_df Twitter data and label")

	# 	if st.checkbox('Show train_df data'): # data is hidden if box is unchecked
	# 		st.write(train_df[['sentiment', 'message']]) # will write the df to the page

	# # Building out the predication page
	# if selection == "Prediction":
	# 	st.info("Prediction with ML Models")
	# 	# Creating a text box for user input
	# 	tweet_text = st.text_area("Enter Text","Type Here")

	# 	if st.button("Classify"):
	# 		# Transforming user input with vectorizer
	# 		vect_text = tweet_cv.transform([tweet_text]).toarray()
	# 		# Load your .pkl file with the model of your choice + make predictions
	# 		# Try loading in multiple models to give the user a choice
	# 		predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
	# 		prediction = predictor.predict(vect_text)

	# 		# When model has successfully run, will print prediction
	# 		# You can use a dictionary or similar structure to make this output
	# 		# more human interpretable.
	# 		st.success("Text Categorized as: {}".format(prediction))

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()




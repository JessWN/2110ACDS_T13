

import streamlit as st
import time

# Streamlit dependencies
import joblib,os			#Loading the model & accessing OS File System
from PIL import Image		#Importing logo Image
from io import BytesIO		#Buffering Images

import numpy as np


import matplotlib.pyplot as plt

from wordcloud import WordCloud


# Data dependencies
import pandas as pd


# Tokenizing the train dataset
from sklearn.feature_extraction.text import TfidfVectorizer

# Load Website's photo clip art
img = Image.open('resources\imgs\performing-twitter-sentiment-analysis1.png') 
#Set the Page Title
st.set_page_config(page_title= 'JHUST Inc.: Climate Change Sentiment Classification',
					page_icon= img,
					layout="wide",
					menu_items = {
							'Report a Bug': 'https://www.google.com'

					})

page_bg_img = '''
<style>
body {
background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)


# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	SENTIMENT_DICT = {
		0: 'Anti-Climate Change', 1: 'Neutral',
		2: 'Pro-Climate Change', 3: 'Climate Change News Related'
	}



	@st.experimental_singleton
	def load_vectorizer():
		with open('resources/models/tfidfvect.pkl') as vectorizer:
			vect = joblib.load(vectorizer)
		return vect

	@st.experimental_singleton
	def load_models():
		# Prediction Model
		with open("resources/models/BNBModel.pkl","rb") as model:
			bnb_model = joblib.load(model) # loading your predictive model from the pkl file

		with open("resources/models/220405_MNB.pkl","rb") as model:
			mnb_model = joblib.load(model) # loading your predictive model from the pkl file

		return bnb_model, mnb_model

	@st.experimental_singleton
	def load_mask():

		# mask
		mask = np.array(Image.open('resources/imgs/callout_1.png'))
		def transform_format(val):
			if val == 0:
				return 255
			else:
				return val


		trans_mask = np.ndarray((mask.shape[0],mask.shape[1]), np.int32)
		for k in range(len(mask)):
			# print(mask[k])
			trans_mask[k] = list(map(transform_format,mask[k]))

		return trans_mask


	@st.experimental_singleton
	def load_datasets():
		# global train_df, grouped_sent_df
		# Load the train_df dataset
		train_df = pd.read_csv("resources/train.csv")

		# Load the grouped sentiment data
		grouped_sent_df = pd.read_csv("resources/grouped_sentiment.csv")

		return train_df, grouped_sent_df
		

	@st.experimental_singleton
	def prepare_word_cloud_data(sentiment) -> float:


		_, df = load_datasets()
		print('{} to load msk'.format(time.time()))
		mask = load_mask()
		print('{} to finish load msk'.format(time.time()))

		vect = TfidfVectorizer(stop_words='english',token_pattern = '[a-z]+\w*')
		vecs = vect.fit_transform([df.loc[[sentiment],'clean_msg'].values[0]])
		print()

		feature_names = vect.get_feature_names_out()

		dense = vecs.todense()
		lst = dense.tolist()

		df = pd.DataFrame(lst, columns=feature_names)

		df = df.T.sum(axis=1)

		cloud = WordCloud(background_color= 'white',
							max_words=100,
							scale = 2,
							width= 600,
							height= 300,
							mask = mask).generate_from_frequencies(df)

		return cloud


	@st.experimental_singleton
	def prep_pred_text(text):
		vectorizer = load_vectorizer()
		df = pd.DataFrame(
			{
				'message': text,
				'tweet_len':len(text)

			}
		) 

	# if st.checkbox('Show train_df data'): # data is hidden if box is unchecked
	# 	st.write(train_df[['sentiment', 'message']]) # will write the df to the page

	# st.select_slider('Display Values', ['this', 'these', 'that'])

	# Creates a main title and subheader on your page -
	# these are static across all pages

	# Introductory Page, describing the solution
	# @st.experimental_memo
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
		st.markdown(
			"""
			EDA involves gathering insights on the state of our training data.
			This would then create the basis of our approach in creating an effective
			machine learning model.
			"""
		)

		st.markdown(
			"""
			##### Data Distribution
			"""
		)

		col1, col2 = st.columns([3,6])

		with col1:
			st.markdown('#### Word Distribution in Corpus')
		with col2:
			st.write('Common Words in **{}** sentiments'.format(SENTIMENT_DICT[0]))
			fig = plt.figure(figsize=(10,8))


			plt.imshow(prepare_word_cloud_data(1), interpolation='bilinear')
			# plt.title('Common Words in {} sentiments'.format(SENTIMENT_DICT[0]))
			plt.axis('off')
			plt.tight_layout()

			buf = BytesIO()
			fig.savefig(buf, format='png')
			st.image(buf)
		# st.pyplot(fig = fig)




		if st.checkbox('Load Modelling'):
			# BROWSE_PAGES['Exploratory Data Analysis']()
			BROWSE_PAGES['Modelling']()
			# pass


	def modelling():
		st.title("Tweet Classification Models")

		model_options = ['Model 1:', 'Model 2:']

		model_selection = st.selectbox("Choose Option", model_options)

		if model_selection == 'Model 1:':
			st.markdown(
				"""
				The model works as such with the folowing parameters. Weutlized the following paramters to achieve an accuracy score of X.
				"""

			)

		elif model_selection == 'Model 2:':
			st.markdown(
				"""
				The model works as such with the folowing parameters. Weutlized the following paramters to achieve an accuracy score of X.
				"""

			)


	def prediction():
		col1, col2 = st.columns([3,8])

		with col1:
			st.markdown('#### Model Selection')
			# model_options = ['Model 1:', 'Model 2:']
			model_options = [str(i) for i in load_models()]
			model_selection = st.selectbox('Select Model to Test:', model_options)
			st.markdown('---')
			if model_selection == 'BernoulliNB()':
				model,_ = load_models()
				st.markdown('{}'.format('this model'))
			elif model_selection == 'MultinomialNB()':
				_, model = load_models()
				st.markdown('{}'.format('this other model'))

		with col2:
			tweet_text_area = st.empty()
			tweet_text = tweet_text_area.text_area("Enter Tweet",placeholder="Type here.", key = 'tweet_text_area')
			st.markdown('Else, select Random sample from test data.')

			if st.button('Select Random'):
				text = tweet_text_area.text_area('Random Tweet', value = 'Random Text from Training data')

			st.markdown('---')

			if st.button('Predict Text'):
				pred = model.predict(tweet_text)


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
	print('----Loading Website-----')

	print(time.time())

	main()

	print(time.time())




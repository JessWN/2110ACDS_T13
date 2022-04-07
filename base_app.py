"""
--------------------------------------------------------
The following application was developed by Team 13: 2110ACDS_T13
For the Advanced Classification Sprint at Explore Data Science Academy.

The application is intended as a text sentiment predictr fr tweet messages.

Authors: Teddy Waweru, Jessica Njuguna, Hunadi Mawela, Uchenna Unigwe, Stanley Agbo 

Github Link: https://github.com/JessWN/2110ACDS_T13
Official Presentation Link:	https://docs.google.com/presentation/d/1-AIbZcDdUDmvVoIB4WoJcIZslbbdb6S9bujMNEgpuHw/edit?usp=sharing

The content is under the GNU icense & is free-to-use.

"""

import time
import streamlit as st

# Streamlit dependencies
import joblib,os			#Loading the model & accessing OS File System
from PIL import Image		#Importing logo Image
from io import BytesIO		#Buffering Images

# Mathematic Computation
import numpy as np

# Plotting of graphs
import matplotlib.pyplot as plt
from wordcloud import WordCloud


# Data dependencies
import re
import string
import pandas as pd
import emoji
import contractions
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer


# Tokenizing the train dataset
from sklearn.feature_extraction.text import TfidfVectorizer


#-------------------------------------------------------------------
#START
#-------------------------------------------------------------------


# Load Website's photo clip art
clip_art = Image.open('resources/imgs/performing-twitter-sentiment-analysis1.png') 

#Set the Pages Initial Configuration Settings
st.set_page_config(page_title= 'JHUST Inc.: Climate Change Sentiment Classification',
					page_icon= clip_art,
					layout="wide",
					menu_items = {
							'Report a Bug': 'https://www.google.com'
					})

#Style the pagge Background
page_bg_img = '''
<style>
body {
background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
background-size: cover;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)


# Main function of the Web application
def main():
	"""Tweet Classifier App with Streamlit """

	# Declare constant variable dict for sentiment values
	SENTIMENT_DICT = {
		0: 'Anti-Climate Change', 1: 'Neutral',
		2: 'Pro-Climate Change', 3: 'Climate Change News Related'
	}

	# Function to load the text Vectorizers
	@st.experimental_singleton				#Enable function caching
	def load_vectorizer():
		# with open('resources/models/tfidf_vectorizer.pkl', encoding="utf8") as vectorizer:
		# 	vect = joblib.load(vectorizer)
		vectorizer = joblib.load('resources/models/tfidf_vect.pkl')


		return vectorizer

	# Function to load ML models
	@st.experimental_singleton
	def load_models():
		with open("resources/models/bnb_model.pkl","rb") as model:
			bnb_model = joblib.load(model) # loading your predictive model from the pkl file

		with open("resources/models/mnb_model.pkl","rb") as model:
			mnb_model = joblib.load(model) # loading your predictive model from the pkl file

		with open("resources/models/lr_model.pkl","rb") as model:
			lr_model = joblib.load(model) # loading your predictive model from the pkl file

		return lr_model, bnb_model, mnb_model

	# Function to lad word cloud mas image
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

	# Function to load datasets
	@st.experimental_singleton
	def load_datasets():
		# global train_df, grouped_sent_df
		# Load the train_df dataset
		train_df = pd.read_csv("resources/data/train.csv")

		# Load the train_df dataset
		test_df = pd.read_csv("resources/data/test.csv")

		# Load the grouped sentiment data
		grouped_sent_df = pd.read_csv("resources/grouped_sentiment.csv")

		return train_df, grouped_sent_df, test_df
		

	# Function to generate word cloud
	@st.experimental_singleton
	def prepare_word_cloud_data(sentiment) -> float:

		_, df,_ = load_datasets()
		print('{} to load mask'.format(time.time()))
		mask = load_mask()
		print('{} to finish load mask'.format(time.time()))

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

	# Function to prepare text to be predicted & carry out prediction
	#Input: tweet text
	#Output: predicted sentiment
	# @st.experimental_singleton
	def prep_pred_text(tweet,model):
		vectorizer = load_vectorizer()
		        #remove urls
		tweet = re.sub(
                r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+',
                'url-web', tweet)
        #remove digits and words with digits
		tweet = re.sub('\w*\d\w*','', tweet)
		# print(str(model))

        #make text lowercase
		tweet = tweet.lower() # lower case

        #expand contractions
		tweet = contractions.fix(tweet)

        #remove punctuation
		tweet = re.sub('[%s]' % re.escape(string.punctuation), '', tweet) # strip punctuation

        #remove emojis
		tweet = emoji.replace_emoji(tweet, replace = '')

		#lemmatize the text
		token_words = TreebankWordTokenizer().tokenize(tweet)
		lemmatizer = WordNetLemmatizer()
		lem_sentence=[]
		for word in token_words:
			#just lemmmatize
			lem_word = lemmatizer.lemmatize(word)
			lem_sentence.append(lem_word)
			lem_sentence.append(" ")
		prep_tweet = "".join(lem_sentence)

		prep_tweet_series = pd.Series(prep_tweet)

		prep_tweet = vectorizer.transform(prep_tweet_series)
		print(prep_tweet.shape)

		pred = model.predict(prep_tweet)
		


		return pred, prep_tweet

		# vectorizer = load_vectorizer()
		# df = pd.DataFrame(
		# 	{
		# 		'message': text,
		# 		'tweet_len':len(text)

		# 	}
		# ) 

	# if st.checkbox('Show train_df data'): # data is hidden if box is unchecked
	# 	st.write(train_df[['sentiment', 'message']]) # will write the df to the page

	# st.select_slider('Display Values', ['this', 'these', 'that'])

	# Creates a main title and subheader on your page -
	# these are static across all pages


	# Introductory Page
	# @st.experimental_memo
	def introduction_page():


		st.title("Climate Change Sentiment Classification")
		st.markdown('---')
		st.subheader("Developed by JHUST Inc. -Team 13")


		col1, col2, col3 = st.columns([1,8,1])
		# st.markdown(' - Jessica Njuguna \n - Hunadi Mawela \n - Uchenna Unigwe \n - Stanley Agbo \n - Teddy Waweru \n')
		with col1:
			pass
		with col2:
			# st.markdown('---')
			st.markdown('### Development Team')
			team_members = Image.open('resources/imgs/team_members.png')
			st.image(team_members)

		with col3:
			pass
			
		st.markdown('## Introduction')
		st.markdown('---')


		col1, col2 = st.columns([10,6])

		with col1:
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
				"""
			)

		with col2:
			st.text('')
			st.text('')
			st.text('')
			st.text('')
			climate_change = Image.open('resources/imgs/climate_change.jpg')

			st.image(climate_change)

		col1, col2 = st.columns([6,10])

		with col1:
			st.text('')
			st.text('')
			st.text('')
			twitter_logo = Image.open('resources/imgs/healthy_world.jpg')

			st.image(twitter_logo)
		with col2:
			# Strategic Action Narrative
			st.markdown(
				"""
				##### Strategic Action: Sentiment Analysis

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


			# Plot WordCloud
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

	# Model Description Page
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

	# Prediction Page
	def prediction():
		global model
		lr_model, bnb_model, mnb_model = load_models()

		st.title('Prediction')
		st.markdown("""
			In this section, users are able to generate a sentiment prediction based on the models that are included in this web application.

			You may provide a text series for prediction, or select a randomly selected one from the test dataset that has been included.
		""")
		col1, col2 = st.columns([3,8])

		with col1:
			st.markdown('#### Model Selection')
			# model_options = ['Model 1:', 'Model 2:']
			model_options = ['Model: \t{}'.format(str(j)) for i,j in enumerate(load_models())]
			model_selection = st.selectbox('Select Model to Test:', model_options)

			if 'Logistic' in model_selection:
				st.text('Multinomial Naive Bayes Model')
				model = lr_model

			elif 'Bernoulli' in model_selection:
				st.text('Bernoulli Naive Bayes Model')
				model = bnb_model

			elif 'Multinomial' in model_selection:
				st.text('Multinomial Naive Bayes Model')
				model = mnb_model

			st.markdown('---')

		with col2:
			tweet_text_area = st.empty()
			tweet_text = tweet_text_area.text_area("Enter Tweet",placeholder="Type here.", key = 'tweet_text_area')
			st.markdown('Else, select Random sample from test data.')

			if st.button('Select Random'):
				train_df, _, test_df = load_datasets()
				idx = int(np.random.randint(0,len(train_df),size=1))
				text = tweet_text_area.text_area('Random Text from Training data', value = train_df.loc[idx, 'message'])

			st.markdown('---')

			if st.button('Predict Text'):
				pred, prep_tweet = prep_pred_text(tweet_text,model)

				st.markdown('{} and \n {}'.format(pred,prep_tweet))





	#Dictionary of radio buttons & functions that are loaded depending on page selected
	BROWSE_PAGES = {
		'Home Page': introduction_page,
		'Data Insights': exploratory_data_analysis,
		'Models Description': modelling,
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




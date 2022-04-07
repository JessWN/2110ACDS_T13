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
from pathlib import Path
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
import plotly.express as px


# Data dependencies
import re
import string
import pandas as pd
import emoji
import contractions
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


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
		2: 'Pro-Climate Change', 3: 'Factual Pro-Climate Change'
	}

	SENTIMENT_DICT_ ={
		-1: 'Anti-Climate Change', 0: 'Neutral',
		1: 'Pro-Climate Change', 2: 'Factual Pro-Climate Change'
	}

	SENTIMENT_DICT_SHORT ={
		-1: 'Anti', 0: 'Neutral',
		1: 'Pro', 2: 'Factual'
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
                '', tweet)
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
			if word in stopwords.words('english') or not word.isascii() or  word == 'rt':
				continue
			#just lemmmatize
			lem_word = lemmatizer.lemmatize(word)
			lem_sentence.append(lem_word)
			lem_sentence.append(" ")
		prep_tweet = "".join(lem_sentence)

		prep_tweet_series = pd.Series(prep_tweet)

		prep_tweet_trans = vectorizer.transform(prep_tweet_series)

		pred = model.predict(prep_tweet_trans)
		


		return int(pred), prep_tweet


	# Read Markdown files
	def read_markdown(file):
		return Path(file).read_text()


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
			st.markdown(read_markdown('resources/markdowns/climate_change.md'))


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
			st.markdown(read_markdown('resources/markdowns/strategic_action.md'))
		

		#Project Links
		st.markdown(read_markdown('resources/markdowns/project_links.md'))

		st.text('')
		st.text('')


		if st.checkbox('Load Data Insights'):
			BROWSE_PAGES['Data Insights']()


	# Insights from EDA of the training dataset
	# @st.experimental_singleton(suppress_st_warning=True)
	def data_insights():

		train_df,_,_ = load_datasets()
		global sentiment_select

		st.title('Exploratory Data Analysis')
		st.markdown(
			"""
			EDA involves gathering insights on the state of our training data.
			This would then create the basis of our approach in creating an effective
			machine learning model.
			"""
		)
		st.text('')

		#Plot Pie Chart for Sentiment Distribution using matplotlib
		# fig = plt.figure(figsize=(8,6))
		# mycolors = ["navy", "cornflowerblue", "blue", "aqua"]
		# train_df['sentiment'].value_counts(ascending = True).plot(kind = 'pie', 
        #                                                   title = 'Sentiment Distribution', 
        #                                                   xlabel = 'Sentiments',
        #                                                  colors = mycolors
        #                                                   )
		# #setting the label names
		# plt.legend(['Anti', 'Nuetral', 'Factual', 'Pro'], 
		# 		loc ="lower right", bbox_to_anchor =(1.5, 0.15))
		#show the plot
		# st.pyplot(fig)

		col1, col2 = st.columns([3,6])

		with col1:
			#Sentiment Distribution Narrative
			st.markdown(read_markdown('resources/markdowns/sentiment_distribution.md'))


		with col2:
			#Plot Pie Chart for Sentiment Distribution
			fig = px.pie(train_df['sentiment'],
						values=train_df['sentiment'].value_counts().values,
						# names=train_df['sentiment'].value_counts().index)
						names= [SENTIMENT_DICT_[i] for i in train_df['sentiment'].value_counts().index])
			fig.update_traces(hoverinfo='label+percent', textinfo='value+percent')

			fig.update_layout(legend = dict(
				yanchor= 'top', y = 1,
				xanchor = 'left', x = -0.1,
				bgcolor = 'rgba(0,0,0,0)'
			))

			st.plotly_chart(fig)

		col1, col2 = st.columns([3,6])

		with col1:
			duplicated_sentiments = train_df[train_df.duplicated(['message'])]

			fig = px.bar(duplicated_sentiments,
						y = duplicated_sentiments['sentiment'].value_counts().values,
						x= [SENTIMENT_DICT_SHORT[i] for i in duplicated_sentiments['sentiment'].value_counts().index],
						width=300, height=300)
			fig.update_layout(xaxis_title = 'Sentiments',
								yaxis_title= 'Count of Duplicates',
								bargap=0.3,
								margin = dict(l=20, r=20,t=0,b=20)
								)

			st.plotly_chart(fig)
		with col2:
			st.markdown(
				"""
				##### Observation Duplicates
				It was noted that the dataset held some duplicate values, especially favoring the 
				Pro Climate Change Category.
				It was necessary to deal with duplicates, to ensure our models receive unbiased data.
				
				"""
			)






		col1, col2 = st.columns([3,6])

		with col1:
			st.markdown("""
			#### Word Distribution in Corpus
			It is anticipated that different sentiments would have different keywords in the messages.
			In this case, we generated wordclouds that highlight the most common words depending on sentiment.
			Select  a sentiment below to generate the WordCloud for that sentiment
			> It is anticipated that the major words would be similar, ie. 'global, climate'
			""")

			sentiment_options = [SENTIMENT_DICT[i] for i in [0,1,2,3]]
			sentiment_select = st.selectbox('Select Sentiment', sentiment_options)

		with col2:
			val = list(SENTIMENT_DICT.keys())[list(SENTIMENT_DICT.values()).index(sentiment_select)]

			st.text('')
			st.text('')
			st.markdown('Common Words in **{}** sentiments'.format(SENTIMENT_DICT[val]))

			# Plot WordCloud
			fig = plt.figure(figsize=(10,5))
			plt.imshow(prepare_word_cloud_data(val), interpolation='bilinear')
			# plt.title('Common Words in {} sentiments'.format(SENTIMENT_DICT[0]))
			plt.axis('off')
			plt.tight_layout()

			buf = BytesIO()
			fig.savefig(buf, format='png')
			st.image(buf)
			# st.pyplot(fig = fig)

		col1, col2 = st.columns([6,3])

		with col1:
			train_df['tweet_len'] = train_df['message'].astype(str).apply(len)

			fig = px.histogram(train_df, train_df['tweet_len'],
								nbins = 20,
								# title = 'Distribution of Tweet Lengths',
								width=600,height=300)
			fig.update_layout(xaxis_title = 'Tweet Lengths',
								margin = dict(l=20, r=20,t=10,b=20)
								)

			st.plotly_chart(fig)


			#Plot Word Count Bar Graph
			# train_df['word_count'] = train_df['message'].apply(lambda x : len(re.findall(r'/w+',x)))
			# fig.update_layout(xaxis_title = 'Tweet Lengths',
			# 					margin = dict(l=20, r=20,t=0,b=20)
			# 					)
			# fig = px.bar(
			# 	train_df,
			# 	x = [SENTIMENT_DICT_SHORT[i] for i in train_df['sentiment'].value_counts().index],
			# 	y = train_df.groupby('sentiment')['word_count'].mean()
			# )
			# fig.update_layout(
			# 	xaxis_title = 'Sentiment', yaxis_title = ''
			# )

			# st.plotly_chart(fig)


		with col2:
			# st.markdown(
			# 	"""
			# 	##### Additional Insights
			# 	The following are basic insights to get from the data as well, & would show some
			# 	significant differences that we exploited during model building.
			# 	> Include charts for: Tweet lengths, stop words

			# 	"""
			# )	
			st.markdown(
				"""
				##### Tweet Lengths
				We provided a histogram of the length of tweets in the dataframe, which 
				would come in handy once we carry out feature engineering during analysis of each sentiment independently.

				> The feature engineering section is covered in the accompanying noteboo, accessible in the [Github repo][1]

				[1]: https://github.com/JessWN/2110ACDS_T13
				"""
			)	







		if st.checkbox('Load Modelling'):
			# BROWSE_PAGES['Exploratory Data Analysis']()
			BROWSE_PAGES['Models Performance']()
			# pass


	# Prediction Page
	def models_performance():
		global model, model_title, text
		lr_model, bnb_model, mnb_model = load_models()

		st.title('Prediction')
		st.markdown("""
			After training Machine Learning algorithms based on the available dataset, the resultant models were included
			here for application.

			In this section:
			- The model types are outlined below, as well as relevant performance metrics
			- Users are able to generate a sentiment prediction based on the availed models.

			> You can provide a text series for prediction, or select a randomly selected one from the dataset that has been included.
		""")
		col1, col2 = st.columns([6,8])

		with col1:
			st.markdown('#### Model Selection')
			#Model selections from list of loaded models
			model_options = ['Model: \t{}'.format(str(j)) for i,j in enumerate(load_models())]
			model_selection = st.selectbox('Select Model:', model_options)

			#Logistic Regression
			if 'Logistic' in model_selection:
				model = lr_model
				model_title = 'Logistic Regression Model'

				st.markdown(
					"""
					##### Logistic Regression Model
					The model utilizes a logistic functionality to compare the probability of an even occurrence.

					For multivariable predictions, the model applies either a one-vs-many or one-vs-other workflow.
					The model's basis is on making a naive assumption that the features provided are independent.

					"""
				)

			#Bernoulli Naives Bayes
			elif 'Bernoulli' in model_selection:
				model = bnb_model
				model_title = 'Bernoulli Naive Bayes Model'
				st.markdown(
					"""
					##### Binomial Naive Bayes Model
					The model's basis is on making a naive assumption that the features provided are independent.

					It also assumes that the features are drawn from a simple binomial distribution.
					It is highly applicable to text data
					"""
				)

			#Multinomial Naive Bayes
			elif 'Multinomial' in model_selection:
				model = mnb_model
				model_title = 'Multinomial Naive Bayes Model'

				st.markdown(
					"""
					##### Multinomial Naive Bayes Model
					The model's basis is on making a naive assumption that the features provided are independent.

					It also assumes that the features are drawn from a simple multinomial distribution ie, multiple binomial distributions
					It is highly applicable to text data
					"""
				)

			st.markdown('---')

			people_sent = Image.open('resources/imgs/people_sentiment.jpeg')

			st.image(people_sent)

			st.markdown('{}🌍'.format(read_markdown('resources/markdowns/conclusion.md')))


		#Prediction Column
		with col2:
			st.markdown('---')

			st.markdown('#### Sentiment Prediction using {}'.format(model_title))

			#Text Area for the tweet to be predicted
			tweet_text_area = st.empty()
			tweet_text = tweet_text_area.text_area("Enter Tweet",placeholder="Type here.", key = 'tweet_text_area')
			text = tweet_text
			st.markdown('Else, select Random sample from test data.')

			train_df, _, test_df = load_datasets()
			if st.checkbox('Select Random'):
				idx = int(np.random.randint(0,len(train_df),size=1))
				tweet_text = tweet_text_area.text_area('Random Text from Training data', value = train_df.loc[idx, 'message'])
				text = tweet_text


			if st.button('Predict Text'):

				pred, prep_tweet = prep_pred_text(text,model)

				st.markdown("""Predicted Sentiment: {}
				
				 {}""".format(pred,SENTIMENT_DICT_[pred]))
				st.text_area('Stripped Text','{}'.format(prep_tweet))





	#Dictionary of radio buttons & functions that are loaded depending on page selected
	BROWSE_PAGES = {
		'Home Page': introduction_page,
		'Data Insights': data_insights,
		'Models Performance': models_performance,
	}

	#Page Navigation Title & Radio BUttons
	st.sidebar.title('Navigation')
	page = st.sidebar.radio('Go to:',list(BROWSE_PAGES.keys()))

	#Load function depending on radio selected above.
	#Used to navigate through pages
	BROWSE_PAGES[page]()




# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	print('----Loading Website-----')

	print(time.time())

	main()

	print(time.time())




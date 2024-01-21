import pandas as pd
import spacy
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from textblob import TextBlob

class TextAnalyzer:
	def __init__(self, data_file):
		self.data = pd.read_csv(data_file)
		self.nlp = spacy.load("en_core_web_sm")
		self.labels = ['neutral', 'surprise', 'sadness', 'joy', 'fear', 'disgust', 'anger']
		self.categories = {
			'all': self.labels,
			'neutral': ['neutral'],
			'negative': ['sadness', 'fear', 'disgust', 'anger'],
			'positive': ['joy', 'surprise']
		}
		self.dict = {}

	def get_word_frequencies(self, text_data):
		vectorizer = CountVectorizer(stop_words='english')
		word_count = vectorizer.fit_transform(text_data)
		sum_words = word_count.sum(axis=0)
		word_freq = [(word, int(sum_words[0, idx])) for word, idx in vectorizer.vocabulary_.items()]
		return sorted(word_freq, key=lambda x: x[1], reverse=True)
	
	def get_sentiment(self, text_data):
		sentiments = [TextBlob(text).sentiment.polarity for text in text_data]
		return sum(sentiments) / len(sentiments) if sentiments else 0

	def get_named_entities(self, text_data):
		entities = []
		for doc in self.nlp.pipe(text_data, disable=["tagger", "parser"]):
			entities.extend([(ent.text, ent.label_) for ent in doc.ents])
		return entities

	def get_tfidf_word_frequencies(self, text_data):
		vectorizer = TfidfVectorizer(stop_words='english')
		tfidf_matrix = vectorizer.fit_transform(text_data)
		feature_names = vectorizer.get_feature_names_out()
		dense = tfidf_matrix.todense()
		denselist = dense.tolist()
		df = pd.DataFrame(denselist, columns=feature_names)
		data = df.mean(axis=0).sort_values(ascending=False).reset_index().rename(columns={0: 'score'})
		data = data.to_records(index=False)
		data = list(data)
		return data

	def analyze(self):
		for key in tqdm(self.categories.keys()):
			filtered_data = self.data[self.data['label'].isin(self.categories[key])]['text']
			self.dict[f'{key}_word_freq'] = self.get_word_frequencies(filtered_data)
			self.dict[f'{key}_word_tfidf'] = self.get_tfidf_word_frequencies(filtered_data)
			self.dict[f'{key}_sentiment'] = self.get_sentiment(filtered_data)
			self.dict[f'{key}_named_entities'] = self.get_named_entities(filtered_data)
			
		print("Done loading data")
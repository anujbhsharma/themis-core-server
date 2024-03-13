from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import requests
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import ssl
tfidf_vectorizer = pickle.load(open('/Users/anujsharma/Desktop/themisCore/themis-core-server/tfidf_vectorizer.pkl', 'rb'))
tfidf_matrix = pickle.load(open('/Users/anujsharma/Desktop/themisCore/themis-core-server/tfidf_matrix.pkl','rb'))
law_list = pickle.load(open('/Users/anujsharma/Desktop/themisCore/themis-core-server/law_dict.pkl','rb'))
df = pd.DataFrame(law_list)

# Initialize NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Initialize Porter Stemmer
porter_stemmer = PorterStemmer()
app = Flask(__name__)
CORS(app)

@app.route('/api/get_data', methods=['POST'])
def get_data():
	# try:
		data = request.json
		data = request.get_json()
		keywords = data.get('keywords')
		year = (int)(data.get('year'))
		dataset_name = data.get('court')
#   keywords = data.get('keywords')
#         year = data.get('year')
#         court = data.get('court')
		if dataset_name == "ALL" or dataset_name == "":
			print("All")
			dataset_name = None
			print(dataset_name)
			print(year)
		
		if year is not None and dataset_name is not None:
			filtered_df = df[(df['year'] == year) & (df['dataset'] == dataset_name)]
		elif year is not None:
			filtered_df = df[df['year'] == year]
		elif dataset_name is not None:
			filtered_df = df[df['dataset'] == dataset_name]
		else:
			filtered_df = df.copy()  # No filtering if year and dataset_name are None

		if filtered_df.empty:
			return [], []  # Return empty lists if filtered dataframe is empty

	# Split query into individual keywords
		keywords = keywords.split(',')
	# Preprocess and tokenize each keyword
		preprocessed_keywords = [preprocess_text(keyword) for keyword in keywords]
	# Vectorize query
		query_vector = tfidf_vectorizer.transform(preprocessed_keywords)

	# Get indices of filtered documents in tfidf_matrix
		filtered_indices = filtered_df.index
	# Calculate cosine similarity between query vector and filtered document vectors
		cosine_similarities = cosine_similarity(query_vector, tfidf_matrix[filtered_indices])
		# Sum similarity scores across all keywords for each document
		combined_scores = cosine_similarities.sum(axis=0)
		# Get indices of top n similar documents
		top_indices = combined_scores.argsort()[-5:][::-1]

		response = []
		for index in top_indices:
			case_name = filtered_df.loc[filtered_indices[index], "name"]
			case_year = int(filtered_df.loc[filtered_indices[index], "year"])
			case_url = filtered_df.loc[filtered_indices[index], "source_url"]
			case_text = filtered_df.loc[filtered_indices[index], "unofficial_text"]
			response.append({"name": case_name, "year": case_year, "url": case_url, "text":case_text})
		return response

	# except Exception as e:
		
	#  	return jsonify({'error': str(e)}), 500

def preprocess_text(text):
	text = text.lower()
	# Remove non-alphanumeric characters
	text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
	# Tokenize text
	tokens = word_tokenize(text)
	# Remove stopwords
	stop_words = set(stopwords.words('english'))
	tokens = [word for word in tokens if word not in stop_words]
	# Apply stemming
	tokens = [porter_stemmer.stem(word) for word in tokens]
	# Join tokens back into a string
	preprocessed_text = ' '.join(tokens)
	return preprocessed_text

if __name__ == '__main__':
	app.run(debug=True, port=8080)
	
	

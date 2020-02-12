import spacy
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
import re
from bs4 import BeautifulSoup
from contractions import CONTRACTION_MAP
import unicodedata


nlp = spacy.load('en_core_web_sm', parse=True, tag=True, entity=True)
#nlp_vec = spacy.load('en_vecs', parse = True, tag=True, #entity=True)
tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')


##	Removes HTML tags
def strip_html_tags(text):
	soup = BeautifulSoup(text, "html.parser")
	stripped_text = soup.get_text()
	# file_text+=stripped_text
	# print(file_text)
	return stripped_text

# strip_html_tags('<html><h2>Some important text</h2></html>')



##	remove accented texts
def remove_accented_chars(text):
	text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
	return text

# remove_accented_chars('Sómě Áccěntěd těxt')



##	expanding contracted words
def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
	
	contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
									  flags=re.IGNORECASE|re.DOTALL)
	def expand_match(contraction):
		match = contraction.group(0)
		first_char = match[0]
		expanded_contraction = contraction_mapping.get(match)\
								if contraction_mapping.get(match)\
								else contraction_mapping.get(match.lower())                       
		expanded_contraction = first_char+expanded_contraction[1:]
		return expanded_contraction
		
	expanded_text = contractions_pattern.sub(expand_match, text)
	expanded_text = re.sub("'", "", expanded_text)
	return expanded_text

# expand_contractions("Y'all can't expand contractions I'd think")



##	removing special characters (also digits)
def remove_special_characters(text, remove_digits=False):
	pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
	text = re.sub(pattern, '', text)
	return text

# remove_special_characters("Well this was fun! What do you think? 123#@!",remove_digits=True)



##	stemming (reducinng words to their base form, which might not exist in dictionary)

def simple_stemmer(text):
	ps = nltk.porter.PorterStemmer()
	text = ' '.join([ps.stem(word) for word in text.split()])
	return text

# simple_stemmer("My system keeps crashing his crashed yesterday, ours crashes daily")



##	Lemmatization (reducinng words to their root form)

def lemmatize_text(text):
	text = nlp(text)
	text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
	return text

# lemmatize_text("My system keeps crashing! his crashed yesterday, ours crashes daily")



##	Removing Stopwords (The, if, are)

def remove_stopwords(text, is_lower_case=False):
	tokens = tokenizer.tokenize(text)
	tokens = [token.strip() for token in tokens]
	if is_lower_case:
		filtered_tokens = [token for token in tokens if token not in stopword_list]
	else:
		filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
	filtered_text = ' '.join(filtered_tokens)    
	return filtered_text

# remove_stopwords("The, and, if are stopwords, computer is not")



##	Text Normalization (All the above functions combined)
def normalize_corpus(corpus, html_stripping=True, contraction_expansion=True,
					 accented_char_removal=True, text_lower_case=True, 
					 text_lemmatization=True, special_char_removal=True, 
					 stopword_removal=True, remove_digits=True, named_ent_recog=True):
	
	normalized_corpus = ""
	file_text = ""
	clean_text = ""
	# normalize each document in the corpus
	for doc in corpus:
		# strip HTML
		if html_stripping:
			doc = strip_html_tags(doc)
			file_text+=doc
		# remove accented characters
		if accented_char_removal:
			doc = remove_accented_chars(doc)
		# expand contractions    
		if contraction_expansion:
			doc = expand_contractions(doc)
		# lowercase the text    
		if text_lower_case:
			doc = doc.lower()
		# remove extra newlines
		doc = re.sub(r'[\r|\n|\r\n]+', ' ',doc)

		# lemmatize text
		if text_lemmatization:
			clean_text += doc
			doc = lemmatize_text(doc)
		# remove special characters and\or digits    
		if special_char_removal:
			# insert spaces between special characters to isolate them    
			special_char_pattern = re.compile(r'([{.(-)!}])')
			doc = special_char_pattern.sub(" \\1 ", doc)
			doc = remove_special_characters(doc, remove_digits=remove_digits)  
		# remove extra whitespace
		doc = re.sub(' +', ' ', doc)
		# remove stopwords
		if stopword_removal:
			doc = remove_stopwords(doc, is_lower_case=text_lower_case)
		
		# named entity recognition
		if named_ent_recog:
			sentence_nlp = nlp(doc)
			# print named entities in article
			# normalized_corpus.append([(word, word.ent_type_) for word in sentence_nlp if word.ent_type_])
			# normalized_corpus.append([word for word in sentence_nlp if word.ent_type_])
			normalized_corpus+=	str([word for word in sentence_nlp if word.ent_type_])

	return normalized_corpus, file_text, clean_text




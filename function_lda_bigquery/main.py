import spacy
import re
import json
#data cleaning libraries
import langdetect
from textblob import TextBlob

#language modelling libraries
import gensim
from gensim.utils import simple_preprocess
import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords')
from nltk.corpus import stopwords
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from gensim.parsing.preprocessing import preprocess_string, strip_punctuation, strip_numeric

#google cloud libraries
from google.cloud.storage import Client

def _insert_into_bigquery(bucket_name, file_name):
    blob = Client().get_bucket(bucket_name).blob(file_name)
    row = json.loads(blob.download_as_string())
    return row
    # table = BQ.dataset(BQ_DATASET).table(BQ_TABLE)
    # errors = BQ.insert_rows_json(table,
    #                              json_rows=[row],
    #                              row_ids=[file_name],
    #                              retry=retry.Retry(deadline=30))
    # if errors != []:
    #     raise BigQueryError(errors)

class topic_modelling():

    def __init__(self, train_dataframe):
        self.data = train_dataframe.text.values.tolist()
        self.data_words = list(self.sent_to_words(self.data))
        self.bigram = gensim.models.Phrases(self.data_words, min_count=5, threshold=100)
        self.bigram_mod = gensim.models.phrases.Phraser(self.bigram)
        self.stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
        self.stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
    def sent_to_words(self, sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
    
    def remove_stopwords(self, texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in self.stop_words] for doc in texts]
    def make_bigrams(self, texts):
        return [self.bigram_mod[doc] for doc in texts]
    
    def lemmatization(self, nlp, texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent)) 
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out

    def train(self):
        data_words_nostops = self.remove_stopwords(self.data_words)
        # Form Bigrams
        data_words_bigrams = self.make_bigrams(data_words_nostops)
        # Do lemmatization keeping only noun, adj, vb, adv
        nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
        data_lemmatized = self.lemmatization(nlp, data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
        # print(data_lemmatized[:1])
        # Create Dictionary
        id2word = corpora.Dictionary(data_lemmatized)
        # Create Corpus
        texts = data_lemmatized
        # Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in texts]
        num_tops = 10
        lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=num_tops, 
                                                random_state=100,
                                                chunksize=100,
                                                passes=10,
                                                per_word_topics=True)



        # Print the Keyword in the 10 topics
        lda_topics = lda_model.show_topics(num_words=5)

        topics = []
        filters = [lambda x: x.lower(), strip_punctuation, strip_numeric]

        for topic in lda_topics:
            topics.append(preprocess_string(topic[1], filters))

        print(topics)
    
class clean_data:

    def __init__(self, dataframe):
        self.df = dataframe

    def language(self, text):
        try:
            a = langdetect.detect(text)
            return a
        except:
            return 'na'
    def lower(self, text):
        textwords = text.split()
        resultwords  = [word.lower() for word in textwords if word.lower() != 'rt']
        result = ' '.join(resultwords)
        return result

    def remove_URL(self, text):
        url = re.compile(r'https?://\S+|www\.\S+')
        return url.sub(r'', text)

    def remove_html(self, text):
        html = re.compile(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
        return re.sub(html, '', text)

    def remove_emoji(self, text):
        emoji_pattern = re.compile(
        '['
        u'\U0001F600-\U0001F64F'  # emoticons
        u'\U0001F300-\U0001F5FF'  # symbols & pictographs
        u'\U0001F680-\U0001F6FF'  # transport & map symbols
        u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
        u'\U00002702-\U000027B0'
        u'\U000024C2-\U0001F251'
        ']+',
        flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)
    
    def process(self):
        self.df['text'] = self.df['text'].apply(lambda x: self.remove_URL(str(x)))
        self.df['text'] = self.df['text'].apply(lambda x: self.remove_emoji(str(x)))
        self.df['text'] = self.df['text'].apply(lambda x: self.remove_html(str(x)))
        self.df['text'] = self.df.text.str.replace('#','')  #remove hashtags
        self.df['text'] = self.df['text'].apply(lambda x : re.sub("@[A-Za-z0-9]+","",x))
        self.df['text'] = self.df['text'].apply(lambda x : re.sub(r'\\ud...', '', x))
        self.df['text'] = self.df['text'].apply(lambda x : re.sub('[^A-Za-z0-9{" "}@{#}]+', '', x))
        self.df['text'] = self.df['text'].apply(lambda x : re.sub(r'img', '', x))
        self.df['text'] = self.df.text.str.replace(r'\bimg$', '', regex=True).str.strip()
        self.df['text'] = self.df['text'].apply(lambda x : re.sub(r'src"', '', x))
        self.df['text'] = self.df['text'].apply(lambda x: self.lower(str(x)))
        self.df.dropna()       
        self.df['language'] = self.df['text'].apply(lambda x: self.language(x))
        self.df = self.df.loc[self.df['language'] == 'en']
        return self.df

def twitter_lda(event, context):
    """Triggered by a change to a Cloud Storage bucket.
    Args:
         event (dict): Event payload.
         context (google.cloud.functions.Context): Metadata for the event.
    """
    file = event
    print("Processing file:", file['name'])

    bucket_name = file['bucket']
    file_name = file['name']
    raw_dataframe = _insert_into_bigquery(bucket_name, file_name)
    clean = clean_data(raw_dataframe)
    clean_dataframe = clean.process()
    print(clean_dataframe.iloc[0])

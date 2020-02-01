#Importing modules
import pandas as pd
import os
import re
import numpy
from wordcloud import WordCloud

# Load the library with the CountVectorizer method
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
#%matplotlib inline

import warnings
warnings.simplefilter("ignore", DeprecationWarning)

# Load the LDA model from sk-learn
from sklearn.decomposition import LatentDirichletAllocation as LDA

from pyLDAvis import sklearn as sklearn_lda
import pickle 
import pyLDAvis

os.chdir(r'C:\OSU 2019-2021\Semester - 2\Company Project')
os.getcwd()

# Read data into papers
textfile = pd.read_csv('project_data.csv')

# Print head
textfile.head()
textfile.columns

# Remove the columns
comment = pd.DataFrame(textfile['Comments'])

# Print out the first rows of papers
comment.head()

type(comment['Comments'])

comment.to_csv('comments.csv')

textdata = pd.read_csv('comments.csv')
textdata = pd.DataFrame(text['Comments'])
type(textdata['Comments'])

# Remove punctuation and spaces in the beginning and end
textdata["Comments_pr"] = textdata['Comments'].str.replace('[^\w\s]','')
textdata["Comments_ps"] = textdata["Comments_pr"].str.strip()

# Convert the titles to lowercase
textdata["Comments_lower"] = textdata["Comments_ps"].str.lower()

# Print out the first rows of papers
textdata.head()

#Convert comment column into string
long_string = ','.join(list(textdata["Comments_lower"].values))

# Create a WordCloud object
wordcloud = WordCloud(background_color="white", max_words=1000, contour_width=3, contour_color='steelblue')
wordcloud.generate(long_string)

# Visualize the word cloud
#wordcloud.to_image() <- not working alternate method used

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

def plot_10_most_common_words(count_data, count_vectorizer):
    import matplotlib.pyplot as plt
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts+=t.toarray()[0]
    
    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words)) 
    
    plt.figure(2, figsize=(15, 15/1.6180))
    plt.subplot(title='10 most common words')
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=90) 
    plt.xlabel('words')
    plt.ylabel('counts')
    plt.show()

# Initialise the count vectorizer with the English stop words
count_vectorizer = CountVectorizer(stop_words='english')

# Fit and transform the processed titles
count_data = count_vectorizer.fit_transform(textdata["Comments_lower"])

# Visualise the 10 most common words
plot_10_most_common_words(count_data, count_vectorizer)

# Helper function
def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
        
# Tweak the two parameters below (use int values below 15)
number_topics = 10
number_words = 20

# Create and fit the LDA model
lda = LDA(n_components=number_topics)
lda.fit(count_data)

# Print the topics found by the LDA model
print("Topics found via LDA:")
print_topics(lda, count_vectorizer, number_words)

# Visualize the topics
#pyLDAvis.enable_notebook() Only for IPython or Spyder or Jupiter Notebook

LDAvis_data_filepath = os.path.join('./ldavis_prepared_'+str(number_topics))

if 1 == 1:
    LDAvis_prepared = sklearn_lda.prepare(lda, count_data, count_vectorizer)
    with open(LDAvis_data_filepath, 'w') as f:
        pickle.dump(LDAvis_prepared, f)
        
# load the pre-prepared pyLDAvis data from disk
with open(LDAvis_data_filepath) as f:
    LDAvis_prepared = pickle.load(f)

pyLDAvis.save_html(LDAvis_prepared, './ldavis_prepared_'+ str(number_topics) +'.html')

LDAvis_prepared
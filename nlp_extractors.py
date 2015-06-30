import random
import cPickle as pickle
from collections import defaultdict
from pprint import pprint
from nltk.tag import pos_tag
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from nltk.corpus import movie_reviews
from sklearn.decomposition import NMF
from nltk.tokenize import word_tokenize
from textblob.classifiers import NaiveBayesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

def create_sentiment():
    """
        Train sentiment model and save.

        Input type: None 
        Output: Model as pickle 
    """

    random.seed(1)

    test = [
        ("The dude presenting Unravel seems like one of the most genuine game developers Ive ever seen I really hope this game works out for him",'pos'),
        ("His hands are shaking Dude looks so stoked and scared at the same time",'pos'),
        ("Right I just felt like I was watching his dream come true It was nice The game looks very well done as well Good for him",'pos'),
        ("Seriously Unravel looks really good actually and honestly seeing him so happy about what hes made is contagious I want to see more of Unravel ",'pos'),
        ("He was so nervous shaking all over his voice quivering",'neg'),
        ("The game looked nice too very cute art style ",'pos'),
        ("You could tell he genuinely wanted to be there it looked like he was even shaking from the excitement  I hope it works out for them aswell",'pos'),
        ("However following that up with the weird PvZ thing was odd To say the least",'neg'),
        ("Haha The game did look nice though Im definitely going to keep an eye on it I enjoy supporting such hopeful developers",'pos'),
        ("Very personable This looks like a buy for me As a dev in a other sector I appreciate this passion",'pos'),
        ("I want to give him a cookie",'pos'),
        ("Im getting a copy Im gonna support my indie devs",'pos'),
        ("The twitch leak was accurate It was like a play by play you start speaking French then switch to English",'neg'),
        ("yep exactly what i was thinking lol its important to note that the twitch leak never had them saying it was Dishonored 2 but that they were honored to be here very different",'neg'),
        ("Honored  Im 100 sure that was intentional",'neg'),
        ("oh yea for sure but wasnt solid enough evidence imo to be like dishonored 2 confirmed just based off that",'neg'),
        ("The confirmation was who was talking not what they were talking about ",'neg'),
        ("How awkward is it for a pop singer to perform at a video game conference",'neg'),
        ("Oh god did they warn him that he will get zero reaction",'neg'),
        ("I really hope so",'pos'),
        ("Almost as bad as Aisha fucking up her dialogue constantly Shes doing alright though E3 is really becoming a mainstream media event Hollywood has nothing like this ComicCon is the only comparison and they dont dazzle it up like E3",'neg')
        ]


    # Grab review data
    reviews = [
        (list(movie_reviews.words(fileid)), category)
        for category in movie_reviews.categories()
        for fileid in movie_reviews.fileids(category)
        ]
    random.shuffle(reviews)

    # Divide into 10% train/test splits
    new_train, new_test = reviews[:1900], reviews[1900:]

    # Train the NB classifier on the train split
    cl = NaiveBayesClassifier(new_train)

    # Compute accuracy
    accuracy = cl.accuracy(test + new_test)
    print("Accuracy: {0}".format(accuracy))

    # Show 5 most informative features
    cl.show_informative_features(5)

    # Save model for use in creating social model sentiment
    with open('sentiment_clf_full.pkl', 'wb') as pk:
        pickle.dump(cl, pk)
    print 'done saving model'


def noun_tokenizer(text, format='string'):
    """
        Tokenize words as nouns from input text string
        Input type: str 
        Output: str or list 
    """

    lower_cased_lst = []

    #Split text into list of words with word tokenizer
    tokens = word_tokenize(text)

    #Tag all words in tokenized list
    tagged = pos_tag(tokens)
    nouns = [word for word,pos in tagged\
            if (pos == 'NN' or\     #Noun tag
                pos == 'NNP' or\    #Proper Noun tag
                pos == 'NNS' or\    #Plural Noun tag
                pos == 'NNPS' or\   #Plural Proper Noun tag
                pos == 'CD' or\     #Numbers tag
                pos == 'FW'         #Foreign Word tag
            )]       

    #Remove metatag for subreddit
    if len(nouns) >= 1 and nouns[0] == 'rGames':
        nouns = nouns[1:]
    
    #Lowercase strings of list    
    lower_cased_lst.extend([x.lower() for x in nouns])

    #Join strings into topic string
    joined = " ".join(lower_cased_lst).encode('utf-8')
    if format=='list':
        return lower_cased_lst
    else:
        return joined


def verb_tokenizer(text, format='string'):
    """
    Tokenize words as verbs from string
    Input type: str 
    Output: str or list 

    Tag list of verbs:
        VB :   Verb, base form
        VBD:   Verb, past tense
        VBG:   Verb, gerund or present participle
        VBN:   Verb, past participle
        VBP:   Verb, non-3rd person singular present
        VBZ:   Verb, 3rd person singular present
        RB :   Adverb
        RBR:   Adverb, comparative
        RBS:   Adverb, superlative
        RP :   Particle
    """

    lower_cased_lst = []

    #Split text into list of words with word tokenizer    
    tokens = nltk.word_tokenize(text)

    #Tag all words in tokenized list
    tagged = nltk.pos_tag(tokens)
    
    #Filter out nouns
    not_nouns = [word for word,pos in tagged\
            if (pos != 'NN'\
                or pos != 'NNP'\
                or pos != 'NNS'\
                or pos != 'NNPS')]

    #Lowercase strings of list  
    lower_cased_lst.extend([x.lower() for x in not_nouns])

    #Join strings into topic string
    joined = " ".join(lower_cased_lst).encode('utf-8')
    if format=='list':
        return lower_cased_lst
    else:
        return joined


def tfidf_vectorizer(codex,\
                     max_df=1,\
                     min_df=0,\
                     stop_words='english',\
                     train_split=False
                     ):
    """
        Calculate term frequency for words in all comments 

        Input:  text string (nouns only from noun_tokenizer)
        Output: transformed input, term list from tfidf, model
    """

    #Select english stopwords
    cachedStopWords = set(stopwords.words("english"))

    #Add words to stopwords list
    cachedStopWords.update(('and','I','A','And','So','arnt','This','When','It',\
                            'many','Many','so','cant','Yes','yes','No','no',\
                            'These','these','',' ','ok','na', 'edit','idk',\
                            'gon','wasnt','yt','sure','watch','whats','youre',\
                            'theyll','anyone'
                            ))
    if train_split:
        #Initialize model
        vectorizer = TfidfVectorizer(max_df=max_df,\
                                     min_df=min_df,\
                                     stop_words=cachedStopWords\
                                     )
        x_train, x_test = train_test_split(codex)

        #Transform codex to vectors and calculate TFIDFs
        X = vectorizer.fit_transform(x_train)

        #Get all word tokens
        terms = vectorizer.get_feature_names()
        return X, terms, vectorizer
    else:
        #Initialize model
        vectorizer = TfidfVectorizer(max_df=max_df,\
                                     min_df=min_df,\
                                     stop_words=cachedStopWords
                                     )
        
        #Transform codex to vectors and calculate TFIDFs
        X = vectorizer.fit_transform(codex)

        #Get all word tokens
        terms = vectorizer.get_feature_names()
        return X, terms, vectorizer


def do_nltk_lemmatizer(text):
    """
        NLTK package lemmatizer to get word roots 

        Input:  text string
        Output: list of strings
    """
    #Lowercase all text
    text = text.lower()

    #Initialize model
    stemmer = WordNetLemmatizer()

    #Strip out all punctuation and non-words
    text = text.encode('utf-8').encode('ascii', 'ignore')
    text = text.translate(string.maketrans("",""), string.punctuation)

    #Strip out stopwords, numbers and run lemmatizer model
    stop = stopwords.words('english')
    stop.extend(map(str,[0,1,2,3,4,5,6,7,8,9]))
    words = [stemmer.lemmatize(word) for word in text.split() if word not in stop]

    return map(str, words)


def Kmeans_tkn_words(X,\
                     max_iter=100,\
                     init='k-means++',\
                     n_init=1,\
                     verbose=True
                     ):
    """
        K-means to group tokenized words into topic clusters.

        Input: word list
        Output: topic centroids
    """

    #Set k to return many small clusters
    k=len(X)/3

    #Initialize model
    km = KMeans(n_clusters=k, init=init, max_iter=max_iter, n_init=n_init,\
                verbose=verbose)

    #Select english stopwords
    cachedStopWords = set(stopwords.words("english"))
    #Add custom words
    cachedStopWords.update(('and','I','A','And','So','arnt','This','When','It',\
                            'many','Many','so','cant','Yes','yes','No','no',\
                            'These','these','',' ','ok','na', 'edit','idk',\
                            'gon','wasnt','yt','sure','watch','whats','youre',\
                            'theyll','anyone'
                            ))
    vectorizer = TfidfVectorizer(stop_words=cachedStopWords)
    
    # Tranfrom into TFIDF
    X = vectorizer.fit_transform(X)

    # Fit TFIDF vectors to model
    km.fit(X)

    print("Top terms per cluster:")
    ordered_centroids = km.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    for i in range(k):
        pprint("Cluster %d:" % i)
        for ind in ordered_centroids[i]:
            print ' %s' % terms[ind]
        print

    return ordered_centroids, km


def NMF_tkn_words(X,\
                  n_features=10000,\
                  n_topics=5,\
                  n_top_words=5,\
                  random_state=1
                  ):
    """
        Non-negative matrix factorization to group tokenized words into
        topic clusters.

        Input: word list
        Output: feature list, topic-strings list
    """

    #Set return size to be shape of input
    n_samples = X.shape[0]

    #Select english stopwords
    cachedStopWords = set(stopwords.words("english"))
    
    #Update stopwords
    cachedStopWords.update(('and','I','A','And','So','arnt','This','When','It',\
                            'many','Many','so','cant','Yes','yes','No','no',\
                            'These','these','',' ','ok','na', 'edit','idk',\
                            'gon','wasnt','yt','sure','watch','whats','youre',\
                            'theyll','anyone'))

    #Initialize model
    vectorizer = TfidfVectorizer(min_df=0.02,stop_words=cachedStopWords)

    #Run fit-tranform to TFIDF vectors
    X = vectorizer.fit_transform(X).toarray()

    #Fit the NMF model
    print("Fitting the NMF model with n_samples=%d and n_features=%d..."
          % (n_samples, n_features))

    #Initialize NMF model and fit 
    nmf = NMF(n_components=n_topics, random_state=random_state).fit(X)

    #Get all tokenized words
    feature_names = vectorizer.get_feature_names()

    for topic_idx, topic in enumerate(nmf.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]\
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))

    return feature_names, nmf
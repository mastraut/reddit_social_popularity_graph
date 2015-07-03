import operator
import gensim
from gensim import corpora, models, similarities
from gensim.models.doc2vec import LabeledSentence

def label_lines(series):
    """
       Takes a DataFrame series and converts it to a list of gensim LabeledSentences 
        Input: Series
        Output: List(LabeledSentence)
    """
    
    labeled_list = []
    for uid, val in enumerate(series):
        L = LabeledSentence(words=str(val).split(), labels=['TOPIC_%s' % uid])
        labeled_list.append(L)
    return labeled_list


def train_d2v(retrain=True, user_input_topic='e3', topics=None):
    """
        Sorted rank of cluster topics based on
        cosine similarity with user_topic.
        
        Input:
            retrain: True to rebuild model.  False to use previously built model.
            user_input_topic: str or List(str)
            topics: Complete list of topics in codex
        Output: Tagged list of similar topics
    """


    if topics is not None:
        labeled_list = label_lines(topics)
        if retrain:
            print 'loading model'
            model = models.Doc2Vec(labeled_list, size=len(labeled_list), window=2, min_count=1, workers=4)
            print 'model loaded'
            # model.save('d2v_model')
            # print 'model saved'
        else:
            model = models.Doc2Vec.load('d2v_model')
            print "model loaded"
        matches = model.most_similar(user_input_topic)
        tagged = list(filter(lambda x: 'TOPIC_' in x[0], matches))
        print matches
        print
        print tagged
        return tagged
    else:
        print 'no data found...'
        pass

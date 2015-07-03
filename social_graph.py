import operator
import numpy as np
import pandas as pd
from itertools import izip, combinations
from collections import defaultdict, Counter
import dill
from pprint import pprint
from igraph import *
from doc2vec_topic_sorting import label_lines, train_d2v
from NLP_extractors import Kmeans_tkn_words,NMF_tkn_word
from NLP_extractors import noun_tokenizer, verb_tokenizer, tfidf_vectorizer


class VerySocial():

    def __init__(self, graph_obj=None, topics=None):
        """
            Reddit to social graph class using iGraph!
        """
        self.graph_obj = graph_obj
        self.topics = topics

    def test_example(self):
        """
            Toy example for testing graph initialization

            Input: None
            Output: DataFrame
        """
        df = pd.DataFrame([[1,'bob','mike','starfox',\
                            'e3 event today','biggest event of the year'],\
                           [2,'mike','andrew','LOL','Japanese video games',\
                            'what do i know about Japanese video games'],\
                           [2,'mike','andrew','COD','best game shooter ever',\
                            'COD shooter. I want.']],\
                           index = [1,2,3],\
                           columns=['score','sub_author','author','topics','title','text']
                         )
        return df


    def df_to_edge_metadata(self, df, test=False):
        """
            Morph edge_data into format for iGraph edges...
            Edge list is { (user_pair): metadata{(topic, score)}}

            Input: Dataframe
            Output: Edge list 
        """
        
        with open('sentiment_clf_mini.pkl', 'rb') as pk:
            clf = dill.load(pk)
        print 'Sentiment model loaded'

        #Take first 3000 datapoints for quick testing
        if test:
            df = df.loc[:3000,:]

        #Run noun nlp_extractors.noun_tokenizer on titles    
        df['topics'] = df.title.apply(lambda x: noun_tokenizer(x));
        print 'Title column tokenized....'

        #Calculate sentiment on all comment text strings
        df['sentiments'] = df.text.apply(lambda x: clf.classify(x));
        df.sentiments = df.sentiments.apply(lambda x: 1 if x=='pos' else -1)
        print 'Comment text column sentiment extracted'

        #Aggregate reddit karma for all edge pairs
        edges = df.groupby(['sub_author','author']).agg({'score': 'sum'})
        print 'Popularity score for comments aggregated'

        #Readjust scores to remove popularity of negative sentiment comments
        edges['topics'] = df.groupby(['sub_author','author'])['topics'].unique()
        edges.topics = edges.topics.apply(lambda x: str(x[0]))
        edges['sentiments'] = df.groupby(['sub_author','author'])['sentiments'].sum()
        edges['score'][edges.sentiments < 0] = - edges['score']

        print 'Finished edges prep....'
        self.topics = edges.topics.values
        return edges.index.values, edges.topics.values, edges.score.values


    def nmf_topic_extractor(self, df):
        """
            Non-negative matrix factorization for
            topic extraction from post title

            Input: Dataframe
            Output: topic list of lists
        """

        x = df.topics.values
        
        #NMF topic extractor
        print 'starting nmf....'
        feature_names, nmf = NMF_tkn_words(x)
        print
        print '10 first feature names....'
        pprint(map(str,feature_names)[:10])
        
        topics = []
        for topic_idx, topic in enumerate(nmf.components_):
            topics.append([feature_names[i]\
                            for i in topic.argsort()[:-n_top_words - 1:-1]])
        return topics


    def df_to_vertices(self, df):
        """
            Morph node_data into format for iGraph vertices...

            Input: DataFrame
            Output: Array
        """

        vertices = np.concatenate((df.sub_author.unique(),\
                                   df.author.unique()),\
                                   axis=1
                                   )

        verts = np.unique(vertices)
        print
        print '%d vertices created' %(len(verts))
        return verts


    def create_graph(self, vertices, edges):
        """
            Take vertices and edges and create iGraph graph.
            Edge list is { (user_pair): metadata{(topic, score)}}.
            Vertices list is unique user list.

            Input: Vertices list, Edges list
            Output: None
        """

        print edges
        print
        edge_list = []
        def map_int(vertices, node):
            for i,val in enumerate(vertices):
                if val==node:
                    return i
                else:
                    pass

        #Map edges into integer mapping of vertices
        for edge in edges[0]:
            edge_list.append(( map_int(vertices,edge[0]), map_int(vertices,edge[1])))

        #Create iGraph graph object of size # vertices
        g = Graph(len(vertices))

        #Label vertices with user ids    
        g.vs['user'] = list(vertices)

        #Add edges to graph
        g.add_edges(edge_list)

        #Add metadata to iGraph object
        g.es['topics'] = list(edges[1])
        g.es['weight'] = list(edges[2])

        print
        print '%d edges added to graph.' %(len(edge_list))
        print
        self.graph_obj = g

    def degree_centrality(self):
        """
            Calculate degree centrality of graph.
            Degree centrality, which is defined as the number of links
                incident upon a node (i.e., the number of ties that a node has).
                The degree can be interpreted in terms of the immediate risk
                of a node for catching whatever is flowing through the network
                (topic info).

            Input: None
            Output: None
        """

        # simplify
        sg=self.graph_obj.simplify()

        # find largest degree value
        degMax=max(sg.degree())

        # get list of all degrees
        allDegs=sg.degree()

        # create list to store results
        calcList=[]

        # loop for subtracting degree from
        # maximum degree for all nodes
        for x in allDegs:
            calc=degMax-x
            calcList.append(calc)

        # sum all results for enumerator
        degr=sum(calcList)

        # calculate denominator
        deno=(sg.vcount()-1.0)*(sg.vcount()-2.0)

        # divide them
        cDeg=degr/deno

        # format results
        gCentr="Graph degree centrality: "+str("%.3f" % cDeg)

        # print results
        print gCentr
        print


    def degree_cluster(self):
        """
            Degree of vertices in cluster.  
            Could be able to use for edges too.

            Input: None
            Output: None

        """

        vertice_degree = []
        for i in range(self.graph_obj.vcount()):
            vertice_degree.append((self.graph_obj.vs[i]['user'],\
                                   self.graph_obj.degree(i)
                                   ))
        print 'Graph vertice degree...first 5'
        print vertice_degree[:5]
        print


    def community_infomap(self, trials=3):
        """
            Finds the community structure of the network according to the
            Infomap method of Martin Rosvall and Carl T.

            Input: # times to check community structure
            Output: subgraph objects
        """
        
        print 'Starting communities infomap algorithm...'
        communes = self.graph_obj.community_infomap(trials=trials)
        
        #Print out density of first 5 communities for reference
        count = 0
        for i in communes:
            if count < 6:
                print 'Community size: ', len(i)
                sub = self.graph_obj.subgraph(i)
                print 'Community density: ', sub.density()
                print
            count += 1
        print 'stopped at 5....'
        print

        return communes


    def grab_communities_topics(self):
        """
            Return topics for all communities for KNN_communities_by_topic

            Input: None
            Output: list of lists
        """

        #Get all communities
        communes = self.community_infomap()
        subs = []
        for i in communes:
            subs.append(self.graph_obj.subgraph(i))

        #Aggregate topic lists per community
        tpl_subgraphs = []
        commune = 0
        for sub_graph in subs:
            edge_list = sub_graph.get_edgelist()
            topic_counter = Counter()
            if len(edge_list) > 1:
                for edg in edge_list[:-1]:
                    tpcs = sub_graph.es[sub_graph.get_eid(edg[0],\
                                                          edg[1])]['topics']
                    topic_counter[tpcs] += 1
                tpl_subgraphs.append((commune,topic_counter))
            else:
                pass
            commune +=1
            
        print 'communities... ', len(tpl_subgraphs)
        return tpl_subgraphs


    def KNN_communities_by_topic(self,\
                                 sgraphs,\
                                 topic='e3',\
                                 k=10
                                 ):
        """
            K-Nearest-Neighbors based clustering algorithm based on weighted 
            graph metadata (topics, adjusted weights).

            Input: Graph obj, topic to cluster on, k=# of clusters
            Output: K clusters 
        """

        #Create LabeledSentence obj list for each topic
        labeled_list = label_lines(self.topics)

        #Train doc2vec model and run Naive Bayes 
        tagged_list = train_d2v(retrain=True, user_input_topic=topic,\
                                topics=self.topics)

        #Rank tagged list returned by Naive Bayes
        ranked_tagged_list = sorted(tagged_list,\
                                key=operator.itemgetter(1),reverse=True)[:k]

        #Extract tag from each topic
        indices = []
        for i in range(len(ranked_tagged_list)):
            indices.append(ranked_tagged_list[i][0][6:])

        #Extract topic from LabeledSentences based on topic tag    
        matched_topics = []
        for ix in map(int,indices):
            matched_topics.append(' '.join(labeled_list[ix].words))
        matched_topics = np.unique(matched_topics)
        print matched_topics

        print 'Starting match search....'
        tpl_matched = []
        for topic_str in matched_topics:
            for sub in sgraphs:
                for t,cnt in sub[1].iteritems():
                    print t
                    if topic_str == t:
                        print 'topic_str == t: ',\
                               topic_str, '==',\
                               t
                        tpl_matched.append( (sub[0], cnt, topic_str) )

        #Sort results by number of comments about topic
        srt_topic_counts = sorted(tpl_matched,key=operator.itemgetter(1),reverse=True)
        
        #Recursive if no best community found
        if len(srt_topic_counts) == 0:
            return self.KNN_communities_by_topic(sgraphs,topic,k)
        print
        print srt_topic_counts
        return srt_topic_counts


    def laplacian_centrality(self, vs=None):
        """
            Calculated laplacian centrality reduction effect on graph.

            Input: Vertice count
            Output: None
        """

        if vs is None:
            vs = xrange(self.graph_obj.vcount())
        degrees = self.graph_obj.degree(mode="all")
        result = []
        for v in vs:
            neis = self.graph_obj.neighbors(v, mode="all")
            result.append(degrees[v]**2 + degrees[v]\
                        + 2 * sum(degrees[i] for i in neis))
        print 'Laplacian centrality...'
        pprint(result[:2])
        print


    def rich_club(self,\
                  fraction=0.1,\
                  highest=True,\
                  scores=None,\
                  indices_only=False
                  ):
        """
            Extracts the "rich club" of the given graph, 
            i.e. the subgraph spanned
            between vertices having the top X% of some score.
            Scores are given by the vertex degrees by default.

            Input:
                Graph:    The graph to work on
                Fraction: The fraction of vertices to extract; 
                            must be between 0 and 1.
                Highest:  Whether to extract the subgraph spanned by the 
                            highest or lowest scores.
                Scores:   The scores themselves.  Uses the vertex degrees.
                Indices_only: Whether to return the vertex indices only 
                                (and not the subgraph)
            Output: None
        """

        if scores is None:
            scores = self.graph_obj.degree()

        indices = range(self.graph_obj.vcount())
        indices.sort(key=scores.__getitem__)

        n = int(round(self.graph_obj.vcount() * fraction))
        if highest:
            indices = indices[-n:]
        else:
            indices = indices[:n]

        if indices_only:
            return indices
        print 'Most influenctial community....'

        self.cluster_social_summary(self.graph_obj.subgraph(indices))


    def betweenness_centralization(self):
        """
            Calculated betweenness centralization for graph.

            Input: None
            Output: None
        """

        #Get total vertice count
        vnum = self.graph_obj.vcount()

        if vnum < 3:
            raise ValueError("graph must have at least three vertices")
        
        denom = (vnum-1)*(vnum-2)
        temparr = [2*i/denom for i in self.graph_obj.betweenness()]
        max_temparr = max(temparr)

        print 'Graph betweenness_centralization...'
        print sum(max_temparr-i for i in temparr)/(vnum-1)
        print


    def cluster_social_summary(self, g):
        """
            Summary information for communities.

            Input: iGraph object
            Output: None
        """

        #Graph summary
        print g.summary()

        print
        print 'Verticies...'
        print
        #check labels
        count = 0
        for i in g.vs.indices:
            if count < 10:
                print 'user: ', ''.join(map(str,g.vs[i]['user']))
                print
            count +=1
        print


    def big_man_on_campus(self):
        """
            Find most infuencial user in graph.

            Input: None
            Output: None
        """

        d = dict()

        #Check all edges
        for e in range(self.graph_obj.ecount()):
            (i,j) = self.graph_obj.es[e].tuple

            #Return all neighbors to both vertices of edge
            nei_i = self.graph_obj.neighbors(i, mode='all')
            nei_j = self.graph_obj.neighbors(j, mode='all')

            #Iterate through all possible edges in neighborhoods,
            #   saving weight of edges to dictionary
            for n in range(len(nei_i)):
                edg = self.graph_obj.get_eid(i,nei_i[n])
                if i in d.keys():
                    d[i] += self.graph_obj.es[edg]['weight']
                else:
                    d[i] = self.graph_obj.es[edg]['weight']

                if n in d.keys():
                    d[n] += self.graph_obj.es[edg]['weight']
                else:
                    d[n] = self.graph_obj.es[edg]['weight']

            for n in range(len(nei_j)):
                edg = self.graph_obj.get_eid(j,nei_j[n])
                if j in d.keys():
                    d[j] += self.graph_obj.es[edg]['weight']
                else:
                    d[j] = self.graph_obj.es[edg]['weight']

                if n in d.keys():
                    d[n] += self.graph_obj.es[edg]['weight']
                else:
                    d[n] = self.graph_obj.es[edg]['weight']

        print 'Most influenctial user....', d
        m = self.graph_obj.vs[max(d.iteritems(), key=operator.itemgetter(1))[0]]
        print 'User: ', m['user']
        print 'User neighborhood size: ', self.graph_obj.neighborhood_size(vertices=m.index, order=1, mode=ALL)
        print


    def social_summary(self):
        """
            Generate summary statistics for graph and communities.

            Input: None
            Output: None
        """
        #Igraph summary function
        print self.graph_obj.summary()

        print
        print 'Verticies...'
        
        #View labels
        count = 0
        for i in self.graph_obj.vs.indices:
            if count < 10:
                print 'user: ', ''.join(map(str,self.graph_obj.vs[i]['user']))
                print
            count += 1
        print

        print 'Edge parameters...'
        #View labels
        count = 0
        for i in self.graph_obj.es.indices:
            if count < 10:
                print '{',\
                       self.graph_obj.es[i].tuple,\
                       ': {topics: ', self.graph_obj.es[i]['topics'],\
                       ';weight: ', self.graph_obj.es[i]['weight'],\
                       '})'
                print
                count += 1
        print

        #Show degree centrality calculation for graph
        self.degree_centrality()

        #Show degree of vertices in graph
        self.degree_cluster()

        #Show modularity maximization algorithm results
        self.community_infomap()

        #Show laplacian centrality reduction algorithm results
        self.laplacian_centrality()

        #Show most infuencial community in graph
        self.rich_club() #Warning: graph returned without edge labels

        #Show betweenness centralization algorithm results
        self.betweenness_centralization()


if __name__ == "__main__":

    #Initialize class obj
    S = VerySocial()
    
    #Read in data
    # df = S.test_example()
    df = pd.read_csv('full_games.csv')

    #Created edges list w/ or w/o limiting to test set
    edges = S.df_to_edges(df, test=False)

    #Create vertices list
    vertices = S.df_to_vertices(df)

    #Initialize graph
    S.create_graph(vertices, edges)

    #Return topic of interest list for all communities
    com_topics = S.grab_communities_topics()

    #Calculate most interesting community by topic arg
    ranked_communities = S.KNN_communities_by_topic(sgraphs=com_topics,\
                                                    topic='e3',\
                                                    k=2
                                                    )

    #Overview graph summary
    S.social_summary()

    #Most infuencial user in graph algorithm
    S.big_man_on_campus()

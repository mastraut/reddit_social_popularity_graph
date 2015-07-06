# Reddit Social Popularity Graph
<i> - Reddit social mapping for community detection of marketable customer segments - </i>

Reddit Social Popularity Graph (RSPG) is a Python module for customer segmentation and popularity analysis of sub-reddit communities.

The project was started in June 2015 by Matt Strautmann as the capstone project for the Galvanize Data Science Immersive Program (galvanize.com). 
See the SlideDeck.pdf file for a presentation of key points and results of the project.

##Project Overview
        The goal of the project was to model the social relationships of online communities to 
        detect communities in these networks. These communities would from frameworks to extract 
        the topics of interest of each of the members, their relative popularity in the community, 
        and their sentiment toward the the communities common topics of interest.

        To accomplish this goal, I needed a community on reddit with a big population that frequently 
        commented and participated in multiple the conversations.  This was determined by trying to 
        extract signal from common topics from subreddits on the front page of Reddit, the so-called 
        "default" subreddits.  I believe this is because the default subreddits are seen by everyone who
        comes to the website not just parties interested in a specific topic. To overcome this problem, 
        I set my criteria to be a subreddit that was in the top 50 on Reddit with a niche topic. For 
        this project, that was the subreddit "Games."  The code works on any social media source with 
        posts and comment trees.

        With my social relationship graph and the strong opinions of Reddit users!, I was able to see
        the customer segmentation of reddit users' opinions of games, releases/announcements, and game
        consoles. This segmentation gave me a community-level view of the entire subreddit user-base 
        with a rich meta-data about the popularity and sentiments of each community and the members of
        the communities.

##In-depth Process

####Scraping Reddit 
        A relational database must come before any graph building. Using a Python wrapper for the 
        Reddit API, I wrote a scraper class I called "Rabid Reddit" to grab the submission posts 
        then flatten the comment tree and scrape the metadata and text from each comment. The 
        scraped posts and comments were stored in a PostgreSQL database using the Psycopg2
        module after data cleaning to regularize the data formats and remove deleted comments.
        
        Now that I had popularity scores for each comment and the comment text bodies, I adjusted 
        the popularity scores by calculating the sentiment of each comment using term frequency
        vectorizing (TF-IDF) and polarizing them using a Naive Bayes model.  This pos/neg 
        polarization from the Naive Bayes model was then applied to the popularity score to 
        adjust for the edge case where the sentiment of the comment was opposite to the 
        sentiment of the original post i.e. a negative comment about a positive sentiment posting
        would have a negative popularity for the comment.

####Building a Social Graph
        A graph is the framework to store the information about the Reddit users: their actions
        and relationships. I used the iGraph module for this project. The nodes (or vertices) of 
        my graph were the unique users scraped from r/Games. The edges represent the relationships
        between users. The graph is formed by adding edges between users who talked about the same
        topics. In my project they were games. Now that I have the framework of who talked to who,
        I wanted to be able to detect popular members of communities and see what topics interested
        them.  I wrote, as part of my graph master class, an add edges function that adds metadata
        to each edge.  I added the popularity of the comments in common to the two users(calculated
        from the Reddit karma score: the net upvote/downvote score of the comment by other users) as 
        well as the all the topics the two users talked about. This allowed me to query the graph
        at a community level and extract the topics talked about in the community.

####Community Detection
        A graph is not very useful without communities. Using the maximum modularity algorithm, I 
        divided the graph into communities.  I used the betweenness centrality theorem of maximum
        modularity as my measurement.  This separates communities by maximizing the number of edges 
        within each community and minimizing edges connecting communities together.
        
        I then wrote a custom distance scoring function for the K-Nearest Neighbors function to 
        allow for finding similar communities based on Doc2Vec similarity of topics discussed in
        the communities.
        
####Customer Segmentation
        The complete graph now shows me which games and topics are of interested to each community.
        It can also tell the dynamics of each group such as most influential member.  And which 
        users had positive or negative perspectives about the topics of interest to the community.

##Important links:

    PRAW documentation: https://praw.readthedocs.org/en/v3.0.0/
    Gensim documentation: http://gensim.readthedocs.org/en/latest/

##Dependencies

RSPG is tested to work under Python 2.7.

The following are required dependencies with the version tested on for the project:

    gensim (0.11.1.post1)
    gnumpy (0.2)
    graphviz (0.4.4)
    pygraphviz (1.2)
    matplotlib (1.4.3)
    nltk (3.0.3)
    numpy (1.9.2)
    pandas (0.16.1)
    praw (3.0.0)
    python-igraph (0.7.1.post6)
    scikit-learn (0.16.1)
    scipy (0.15.1)
    psycopg2 (2.6)

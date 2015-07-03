import string
import psycopg2

def parse_submission_data(submission):
    """
        Parses Reddit API wrapper, PRAW, to extract post submission data.
        Returns a list of relevant submission data for export to database.
        
        Input: PRAW submission generator-object
        Output: List
    """
    
    subreddit_name = str(submission.subreddit)
    subreddit_id = submission.subreddit_id
    subscriber_count = submission.subreddit.subscribers # number of redditers subscribed to subreddit
    sub_id = submission.id
    title = submission.title.encode('ascii','ignore')
    title = str(title).translate(string.maketrans("",""), string.punctuation)
    author = str(submission.author) # account name of poster, null if promotional link
    creation_date = submission.created_utc # date utc
    comment_count = submission.num_comments
    score = submission.score # The net-score of the link.
    nsfw = submission.over_18 # Subreddit marked NSFW

    return [str(subreddit_id),
            subreddit_name,
            subscriber_count,
            str(sub_id),
            title,
            author,
            creation_date,
            comment_count,
            score,
            nsfw]


def to_postgres_submissions(sub_test):
    """
        Psycopg2 dump scraped list to PostgreSQL Database table "submissions"
        
        Input: List
        Output: Psycopg2 commit to "Submission" table
    """
    
    pconn = psycopg2.connect(dbname=[Insert DB name], user=[Insert local user], password='', host='localhost')
    pcur = pconn.cursor()

    sql_stmt ="""INSERT INTO\
              submissions (subreddit_id,
                           subreddit_name,
                           subscriber_count,
                           sub_id,
                           title,
                           author,
                           creation_date,
                           comment_count,
                           score,
                           nsfw) VALUES\
                           ('{subreddit_id}',
                            '{subreddit_name}',
                             {subscriber_count},
                            '{sub_id}',
                            '{title}',
                            '{author}',
                             {creation_date},
                             {comment_count},
                             {score},
                            '{nsfw}')""".format(subreddit_id =sub_test[0],
                                                subreddit_name=sub_test[1],
                                                subscriber_count=sub_test[2],
                                                sub_id=sub_test[3],
                                                title=sub_test[4],
                                                author=sub_test[5],
                                                creation_date=sub_test[6],
                                                comment_count=sub_test[7],
                                                score=sub_test[8],
                                                nsfw=sub_test[9])
    print
    print sql_stmt
    print
    pcur.execute(sql_stmt)
    pconn.commit()


def parse_comment_data(comment):
    """
        Parses Reddit API wrapper, PRAW, to extract post comment data.
        Returns a list of relevant submission data for export to database.
        
        Input: PRAW comment generator-object
        Output: List
    """
    
    comment_id = comment.id
    author = comment.author
    parent_id = comment.parent_id
    submission_id = comment.submission.id
    subreddit_id = str(comment.subreddit)
    creation_date = comment.created_utc
    banned_by = comment.banned_by
    if banned_by is None:
        banned_by = 'NA'
    score = comment.score
    if score is None:
        score = 0
    gilded = comment.gilded # number of times comment received reddit gold
    if gilded is None:
        gilded = 0
    likes = comment.likes
    if likes is None:
        likes = 0
    controversiality = comment.controversiality
    if controversiality is None:
        controversiality = 'NA'
    text = comment.body.encode('ascii','ignore')
    text = str(text).translate(string.maketrans("",""), string.punctuation)
    if text is None:
        text = 'NA'
    return [str(comment_id),
            str(author),
            str(parent_id),
            str(submission_id),
            subreddit_id,
            creation_date,
            str(banned_by),
            score,
            str(gilded),
            likes,
            controversiality,
            text]


def to_postgres_comments(com_test):
        """
        Psycopg2 dump scraped list to PostgreSQL Database table "comments"
        
        Input: List
        Output: Psycopg2 commit to "Submission" table
    """
    
    pconn = psycopg2.connect(dbname=[Insert DB name], user=[Insert local user], password='', host='localhost')
    pcur = pconn.cursor()

    sql_stmt ="""INSERT INTO\
              comments (comment_id,
                       author,
                       parent_id,
                       submission_id,
                       subreddit_id,
                       creation_date,
                       banned_by,
                       score,
                       guilded,
                       likes,
                       controversiality,
                       text) VALUES\
                        ('{comment_id}',
                         '{author}',
                         '{parent_id}',
                         '{submission_id}',
                         '{subreddit_id}',
                          {creation_date},
                         '{banned_by}',
                          {score},
                         '{guilded}',
                          {likes},
                          {controversiality},
                         '{text}')""".format(comment_id =com_test[0],
                                             author=com_test[1],
                                             parent_id=com_test[2],
                                             submission_id=com_test[3],
                                             subreddit_id=com_test[4],
                                             creation_date=com_test[5],
                                             banned_by=com_test[6],
                                             score=com_test[7],
                                             guilded=com_test[8],
                                             likes=com_test[9],
                                             controversiality=com_test[10],
                                             text=com_test[11])

    print
    print sql_stmt
    print
    pcur.execute(sql_stmt)
    pconn.commit()

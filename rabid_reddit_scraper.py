from postgres_save import to_postgres_comments, parse_comment_data
from postgres_save import to_postgres_submissions, parse_submission_data
import praw


# Scrape Reddit for submissions, and store all comment data
class RabidReddit:
    """Class object for api calls to subreddit"""
    # Initialize PRAW
    # Praw has a build in rate limiter. 1 API call every 2 seconds.

    def __init__(self):
        """root_only will return only top-level comments"""
        self.submissions = None

    def grab_submissions(self, user_agent='no-one', subreddit='explainitlikeim5'):
        """
            store a generator for submissions
        """

        first = True   # bool switch to not check last id
        counter = 1
        r = praw.Reddit(user_agent=user_agent)
        while True and counter < 10000:
            print 'in grab_submissions while loop'
            error_count = 0
            try:
                #submission_generator = self.r.search('url:%s' % self.url, subreddit=subreddit) # to search specific
                if first:
                    submission_generator = r.get_content("https://www.reddit.com/r/%s" %subreddit, limit=0)
                    self.submissions = [sub_obj for sub_obj in submission_generator]
                    first = False
                else:
                    print 'counter in grab_submissions of page ', counter
                    r = praw.Reddit(user_agent=user_agent)
                    submission_generator = r.get_content("https://www.reddit.com/r/%s" %subreddit, \
                                                         limit=0, params = {'before': str(self.submissions[-1].id)})
                    self.submissions.extend([sub_obj for sub_obj in submission_generator])
                    counter += 1
            except praw.errors.APIException:
                print 'API Exception caught... trying again'
                error_count += 1
                continue
            except praw.errors.ClientException:
                print 'Client Exception caught... trying again'
                error_count +=1
                continue
            if error_count > 5:
                print '5 errors found, moving on'
                break
        return self.submissions



    def get_comments(self, submission):
        """
            Replaces MoreComment objects with Comment objects.
            Returns list of objects.
        """

        while True:
            error_count = 0
            try:
                submission.replace_more_comments(limit=5000)
                commentobjs = praw.helpers.flatten_tree(submission.comments)
                # print 'expanding %d comments' %(sum(1 for _ in commentobjs))
                count = 0
                for com in commentobjs:
                    to_postgres_comments(parse_comment_data(com))
                    count+=1
                    print 'finished %d of comments' %count
                break
            except praw.errors.APIException:
                print 'API Exception caught... trying again'
                error_count += 1
                continue
            except praw.errors.ClientException:
                print 'Client Exception caught... trying again'
                error_count +=1
                continue
            if error_count > 5:
                print '5 errors found, moving on'
                break


if __name__ == "__main__":

    user_agent = ("no-one-123-one")
    subreddit = "Games"
    rr = RabidReddit()
    submissions_generators = rr.grab_submissions(user_agent,subreddit)

    count = 0
    for sub in submissions_generators:
        count+=1
        print 'working on sub %d' %(count)
        to_postgres_submissions(parse_submission_data(sub))
        print
        print 'added submission....moving to comments'
        rr.get_comments(sub)
        # loop_through_comments(sub)
        # Iterate through root comments replacing MoreComment objects
        # API is the rate limiter, not loop time

        print
        print 'finished adding comments...moving to next submission'








# while praw.objects.MoreComments in submission.comments:
#     for i, comment in enumerate(submission.comments):
#         if type(comment) == praw.objects.MoreComments:
#             submission.comments.extend(submission.comments.pop[i].comments())

#
# def loop_through_comments(submission):
#     try:
#         submission.replace_more_comments(limit=1000)
#         count = 0
#         stack = submission.comments[:]
#         # print 'stack', stack
#         count = 0
#         while stack:
#             item = stack.pop(0)
#             print 'item ', item
#             count+=1
#             if not getattr(item, 'replies', None):
#                 print 'adding root comment ,comment %d to table' %(count)
#                 to_postgres_coms(parse_comment_data(item))
#     except praw.errors.APIException:
#         print 'API Exception caught... trying again'
#     except praw.errors.ClientException:
#         print 'Client Exception caught... trying again'
#
#
    # print len(sub.comments)

    # com_table = []
    # for i in sub.comments:
    # #     print i
    #     com_table.append(parse_comment_data(i))
    # print com_table


    # def parse_submission_data(self, submission):
    #     """Returns a tuple of relevant submission data"""
    #     subreddit_name = str(submission.subreddit)
    #     subreddit_id = submission.subreddit_id
    #     subscriber_count = submission.subreddit.subscribers # number of redditers subscribed to subreddit
    #     sub_id = submission.id
    #     title = submission.title  # submission title
    #     author = submission.author # account name of poster, null if promotional link
    #     creation_date = submission.created_utc # date utc
    #     comment_count = submission.num_comments
    #     score = submission.score # The net-score of the link.
    #     nsfw = submission.over_18 # Subreddit marked NSFW
    #
    #     return (subreddit_id,
    #             subreddit_name,
    #             subscriber_count,
    #             sub_id,
    #             title,
    #             author,
    #             creation_date,
    #             comment_count,
    #             score,
    #             nsfw)
    #
    # def build_submission_table(self):
    #     """Returns a list of all submission data for the patch"""
    #     submission_table = []
    #     for submission in self.submissions:
    #         data = self.parse_submission_data(submission)
    #         submission_table.append(data)
    #     return submission_table


#
# def parse_comment_data(comment):
#     """Returns a tuple of relevant comment data"""
#     comment_id = comment.id
#     author = comment.author
#     parent_id = comment.parent_id
#     submission_id = comment.submission.id
#     subreddit_id = str(comment.subreddit)
#     creation_date = comment.created_utc
#     banned_by = comment.banned_by
#     if banned_by is None:
#         banned_by = 'NA'
#     score = comment.score
#     if score is None:
#         score = 0
#     gilded = comment.gilded # number of times comment received reddit gold
#     if gilded is None:
#         gilded = 0
#     likes = comment.likes
#     if likes is None:
#         likes = 0
#     controversiality = comment.controversiality
#     if controversiality is None:
#         controversiality = 'NA'
#     text = comment.body.encode('ascii','ignore')
#     text = str(text).translate(string.maketrans("",""), string.punctuation)
#     if text is None:
#         text = 'NA'
#     return [
#         str(comment_id),
#         str(author),
#         str(parent_id),
#         str(submission_id),
#         subreddit_id,
#         creation_date,
#         str(banned_by),
#         score,
#         str(gilded),
#         likes,
#         controversiality,
#         text]
#
# def parse_submission_data(submission):
#     """Returns a tuple of relevant submission data"""
#     subreddit_name = str(submission.subreddit)
#     subreddit_id = submission.subreddit_id
#     subscriber_count = submission.subreddit.subscribers # number of redditers subscribed to subreddit
#     sub_id = submission.id
#     title = submission.title.encode('ascii','ignore')
#     title = str(title).translate(string.maketrans("",""), string.punctuation)
#   # submission title
#     author = str(submission.author) # account name of poster, null if promotional link
#     creation_date = submission.created_utc # date utc
#     comment_count = submission.num_comments
#     score = submission.score # The net-score of the link.
#     nsfw = submission.over_18 # Subreddit marked NSFW
#
#     return [str(subreddit_id),
#             subreddit_name,
#             subscriber_count,
#             str(sub_id),
#             title,
#             author,
#             creation_date,
#             comment_count,
#             score,
#             nsfw]
#
#
#
# #######################################
# import psycopg2
#
# def to_postgres_subs(sub_test):
#     # pconn = psycopg2.connect(dbname='submissions', user='postgres', password='', host='localhost')
#     pconn = psycopg2.connect(dbname='reddit_scrape', user="roboto", password='', host='localhost')
#     pcur = pconn.cursor()
#
#     sql_stmt ="""INSERT INTO submissions (subreddit_id,
#                                            subreddit_name,
#                                            subscriber_count,
#                                            sub_id,
#                                            title,
#                                            author,
#                                            creation_date,
#                                            comment_count,
#                                            score,
#                                            nsfw) VALUES ('{subreddit_id}',
#                                                             '{subreddit_name}',
#                                                             {subscriber_count},
#                                                             '{sub_id}',
#                                                             '{title}',
#                                                             '{author}',
#                                                             {creation_date},
#                                                             {comment_count},
#                                                             {score},
#                                                             '{nsfw}')""".format(subreddit_id =sub_test[0],
#                                                                                 subreddit_name=sub_test[1],
#                                                                                 subscriber_count=sub_test[2],
#                                                                                 sub_id=sub_test[3],
#                                                                                 title=sub_test[4],
#                                                                                 author=sub_test[5],
#                                                                                 creation_date=sub_test[6],
#                                                                                 comment_count=sub_test[7],
#                                                                                 score=sub_test[8],
#                                                                                 nsfw=sub_test[9])
#
#     print sql_stmt
#     pcur.execute(sql_stmt)
#     pconn.commit()
#
# ##############################
# def to_postgres_coms(com_test):
#     pconn = psycopg2.connect(dbname='reddit_scrape', user="roboto", password='', host='localhost')
#     pcur = pconn.cursor()
#
#     sql_stmt ="""INSERT INTO comments  (comment_id,
#                                        author,
#                                        parent_id,
#                                        submission_id,
#                                        subreddit_id,
#                                        creation_date,
#                                        banned_by,
#                                        score,
#                                        guilded,
#                                        likes,
#                                        controversiality,
#                                        text) VALUES ('{comment_id}',
#                                                     '{author}',
#                                                     '{parent_id}',
#                                                     '{submission_id}',
#                                                     '{subreddit_id}',
#                                                     {creation_date},
#                                                     '{banned_by}',
#                                                     {score},
#                                                     '{guilded}',
#                                                     {likes},
#                                                     {controversiality},
#                                                     '{text}')""".format(comment_id =com_test[0],
#                                                                         author=com_test[1],
#                                                                         parent_id=com_test[2],
#                                                                         submission_id=com_test[3],
#                                                                         subreddit_id=com_test[4],
#                                                                         creation_date=com_test[5],
#                                                                         banned_by=com_test[6],
#                                                                         score=com_test[7],
#                                                                         guilded=com_test[8],
#                                                                         likes=com_test[9],
#                                                                         controversiality=com_test[10],
#                                                                         text=com_test[11])
#
#
#     print sql_stmt
#     pcur.execute(sql_stmt)
#     pconn.commit()


    #
    #
    # def parse_comment_data(self, comment):
    #     """Returns a tuple of relevant comment data"""
    #     comment_id = comment.id
    #     author = comment.author
    #     parent_id = comment.parent_id
    #     submission_id = comment.submission.id
    #     subreddit_id = str(comment.subreddit)
    #     creation_date = comment.created_utc
    #     banned_by = comment.banned_by
    #     score = comment.score
    #     gilded = comment.gilded # number of times comment received reddit gold
    #     likes = comment.likes
    #     controversiality = comment.controversiality
    #     text = comment.body
    #     return (
    #         comment_id,
    #         author,
    #         parent_id,
    #         submission_id,
    #         subreddit_id,
    #         creation_date,
    #         banned_by,
    #         score,
    #         gilded,
    #         likes,
    #         controversiality,
    #         text)
    #
    # def build_comment_table(self):
    #     comment_table = []
    #     for submission in self.submissions:
    #         comments = self.get_comments(submission)
    #         for comment in comments:
    #             data = self.parse_comment_data(comment)
    #             comment_table.append(data)
    #     return comment_table
    #
    # def collect_all(self):
    #     """collect all data and return submission and comment table"""
    #
    #     self.grab_submissions()
    #     submission_table = self.build_submission_table()
    #     print 'Submission table appended to run %d.'
    #     print 'Collecting comments...'
    #     comment_table = self.build_comment_table()
    #
    #     return submission_table, comment_table
    #
    # def collect_submissions(self, subreddit):
    #     """collect all data and return submission table"""
    #     self.grab_submissions(subreddit)
    #     submission_table = self.build_submission_table()
    #     print 'Submission table appended to run...'
    # #     print 'Collecting comments...'
    # #     comment_table = self.build_comment_table()
    #
    #     return submission_table #, comment_table

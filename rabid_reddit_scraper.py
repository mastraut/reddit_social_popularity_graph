from save_to_postgres import to_postgres_comments, parse_comment_data
from save_to_postgres import to_postgres_submissions, parse_submission_data
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

    user_agent = [Insert str here]
    subreddit = [Insert str here] # i.e. "Games"
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

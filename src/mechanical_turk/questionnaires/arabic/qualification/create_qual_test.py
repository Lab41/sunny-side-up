import os
from boto.mturk.connection import MTurkConnection
from boto.mturk.question import ExternalQuestion
from boto.mturk.qualification import Qualifications, PercentAssignmentsApprovedRequirement

def test():
    mturk = open_mturk_connection()
    print mturk.get_account_balance()



def open_mturk_connection(host='mechanicalturk.sandbox.amazonaws.com'):
      return MTurkConnection(aws_access_key_id=os.environ.get("MTURK_ACCESS_ID"),
                            aws_secret_access_key=os.environ.get("MTURK_SECRET_KEY"),
                            host=host)



def get_qualification_test_and_answers( dir_root="/data/questionnaires/arabic/qualification",
                                        file_test="sentiment-qualification-arabic-test.xml",
                                        file_answers="sentiment-qualification-arabic-answers.xml"):
    """
    load the XML qualification test and answers
    """

    # read test XML
    with open(os.path.join(dir_root, file_test), "rb") as fh:
        qual_test = fh.read()

    # read answers XML
    with open(os.path.join(dir_root, file_answers), "rb") as fh:
        qual_answers = fh.read()

    # return
    return (qual_test, qual_answers)



def create_qualification():

    # params for the qualification test
    qual_name = "Arabic Tweet Sentiment Qualification Test"
    qual_description = "A qualification test in which you read Arabic Tweets and categorize the overall sentiment of the Twitter user"
    keywords = ["arabic", "twitter", "tweet", "tweets", "sentiment"]
    duration = 30*60


    # get the test and answers
    qual_test, qual_answers = get_qualification_test_and_answers()

    # connect to MTurk account
    mturk = open_mturk_connection()

    # create the test with the given parameters
    mturk.create_qualification_type(name=qual_name,
                                    description=qual_description,
                                    status="Active",
                                    retry_delay=60*5,
                                    keywords=keywords,
                                    test=qual_test,
                                    answer_key=qual_answers,
                                    test_duration=duration)




def create_hit():

    # connect to MTurk account
    mturk = open_mturk_connection()

    # params
    title = "Arabic Tweet Sentiment Test"
    annotation = "Categorize the sentiment of the Twitter user"
    keywords = ["arabic", "twitter", "tweet", "tweets", "sentiment"]
    duration = 60*6
    approval_delay = 60*60
    lifetime = 60*65
    max_assignments = 2
    reward = 0.05

    # add qualification
    qualifications = Qualifications()
    qualifications.add(PercentAssignmentsApprovedRequirement(comparator="GreaterThan", integer_value="95"))

    # create HIT
    create_hit_rs = mturk.create_hit(question=q,
                                      lifetime=lifetime,
                                      max_assignments=max_assignments,
                                      title=title,
                                      keywords=keywords,
                                      reward=reward,
                                      duration=duration,
                                      approval_delay=approval_delay,
                                      annotation=annotation,
                                      qualifications=qualifications)

    # ensure created
    assert(create_hit_rs.status == True)
    print create_hit_rs.HITTypeId



if __name__ == '__main__':
    test()
    #create_qualification()
    #create_hit()

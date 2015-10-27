from IPython.html.utils import url_path_join as ujoin
from tornado.web import RequestHandler
import os, json, git, urllib, requests
from git import Repo, GitCommandError
from subprocess import check_output

class GitCommitHandler(RequestHandler):
    def initialize(self, log=None):
        self.log = log


    def error_and_return(self, dirname, reason):

        # send error
        self.send_error(500, reason=reason)

        # return to directory
        os.chdir(dirname)

    def put(self):

        # git parameters
        notebook_path_relative = '/ipython-notebooks'
        git_branch = git_remote = 'gitbot-jupyter-updates'
        git_url = '<FIXME>'
        git_user = '<FIXME>'
        git_repo_upstream = '<FIXME>'

        # obtain filename and msg for commit
        data = json.loads(self.request.body.decode('utf-8'))
        filename = urllib.unquote(data['filename']).decode('utf8')
        msg = data['msg']

        # select branch within repo
        try:
            cwd = os.getcwd()
            os.chdir(cwd + notebook_path_relative)
            dir_repo = check_output(['git','rev-parse','--show-toplevel']).strip()
            repo = Repo(dir_repo)
        except GitCommandError as e:
            self.error_and_return(cwd, "Could not checkout repo: {}".format(dir_repo))
            return

        # create new branch
        try:
            print(repo.git.checkout('HEAD', b=git_branch))
        except GitCommandError:
            print("Switching to {}".format(repo.heads[git_branch].checkout()))

        # commit current notebook
        try:
            print(repo.git.add(cwd + notebook_path_relative + filename))
            print(repo.git.commit( a=True, m="{}\n\nUpdated {}".format(msg, filename) ))
        except GitCommandError as e:
            self.error_and_return(cwd, "Could not commit changes to notebook: {}".format(cwd + notebook_path_relative + filename))
            return

        # create or switch to remote
        try:
            remote = repo.create_remote(git_remote, git_url)
        except GitCommandError:
            print("Remote {} already exists...".format(git_remote))
            remote = repo.remote(git_remote)

        # push changes
        try:
            pushed = remote.push(git_branch)
            assert len(pushed)>0
            assert pushed[0].flags in [git.remote.PushInfo.UP_TO_DATE, git.remote.PushInfo.FAST_FORWARD, git.remote.PushInfo.NEW_HEAD, git.remote.PushInfo.NEW_TAG]
        except GitCommandError as e:
            self.error_and_return(cwd, "Could not push to remote {}".format(git_remote))
            return
        except AssertionError as e:
            self.error_and_return(cwd, "Could not push to remote {}: {}".format(git_remote, pushed[0].summary))
            return

        # open pull request
        try:
          github_url = "https://api.github.com/repos/{}/pulls".format(git_repo_upstream)
          github_pr = {
              "title":"{} Notebooks".format(git_user),
              "body":"IPython notebooks submitted by {}".format(git_user),
              "head":"{}:{}".format(git_user, git_remote),
              "base":"master"
          }
          github_headers = {"Authorization": "token {}".format(os.environ.get('GITHUB_ACCESS_TOKEN'))}
          r = requests.post(github_url, data=json.dumps(github_pr), headers=github_headers)
          if r.status_code != 201:
            print("Error submitting Pull Request to {}".format(git_repo_upstream))
        except:
            print("Error submitting Pull Request to {}".format(git_repo_upstream))

        # return to directory
        os.chdir(cwd)

        # close connection
        self.write({'status': 200, 'statusText': 'Success!  Changes to {} captured on branch {} at {}'.format(filename, git_branch, git_url)})


def load_jupyter_server_extension(nbapp):
    nbapp.log.info('Loaded Jupyter extension: Git Commit and Push')

    webapp = nbapp.web_app
    base_url = webapp.settings['base_url']
    webapp.add_handlers(".*$", [
        (ujoin(base_url, r"/git/commit"), GitCommitHandler,
            {'log': nbapp.log}),
    ])

define(['base/js/namespace','base/js/dialog','jquery'],function(IPython, dialog, $, mc){

    // we will define an action here that should happen when we ask to clear and restart the kernel.
    var git_commit_push  = {
        help: 'Commit current notebook and push to GitHub',
        icon : 'fa-github',
        help_index : '',
        handler : function (env) {
            var on_success = undefined;
            var on_error = undefined;

            var p = $('<p/>').text("Please enter your commit message. Only this notebook will be committed.")
            var input = $('<textarea rows="4" cols="72"></textarea>')
            var div = $('<div/>')

            div.append(p)
               .append(input)

            // get the canvas for user feedback
            var container = $('#notebook-container');

            function on_ok(){
                var re = /^\/notebooks(.*?)$/;
                var filepath = window.location.pathname.match(re)[1];
                var payload = {
                             'filename': filepath,
                             'msg': input.val()
                           };
                var settings = {
                    url : '/git/commit',
                    processData : false,
                    type : "PUT",
                    dataType: "json",
                    data: JSON.stringify(payload),
                    contentType: 'application/json',
                    success: function(data) {

                        // display feedback to user
                        var container = $('#notebook-container');
                        var feedback = '<div class="commit-feedback alert alert-success alert-dismissible" role="alert"> \
                                          <button type="button" class="close" data-dismiss="alert" aria-label="Close"><span aria-hidden="true">&times;</span></button> \
                                          '+data.statusText+' \
                                           \
                                        </div>';

                        // display feedback
                        $('.commit-feedback').remove();
                        container.prepend(feedback);
                    },
                    error: function(data) {

                        // display feedback to user
                        var feedback = '<div class="commit-feedback alert alert-danger alert-dismissible" role="alert"> \
                                          <button type="button" class="close" data-dismiss="alert" aria-label="Close"><span aria-hidden="true">&times;</span></button> \
                                          <strong>Warning!</strong> Something went wrong. \
                                          <div>'+data.statusText+'</div> \
                                        </div>';

                        // display feedback
                        $('.commit-feedback').remove();
                        container.prepend(feedback);
                    }
                };

                // display preloader during commit and push
                var preloader = '<img class="commit-feedback" src="https://cdnjs.cloudflare.com/ajax/libs/slick-carousel/1.5.8/ajax-loader.gif">';
                container.prepend(preloader);

                // commit and push
                $.ajax(settings);
            }


            dialog.modal({
                body: div ,
                title: 'Commit and Push Notebook',
                buttons: {'Commit and Push':
                            { class:'btn-primary btn-large',
                              click:on_ok
                            },
                          'Cancel':{}
                    },
                notebook:env.notebook,
                keyboard_manager: env.notebook.keyboard_manager,
            })
        }
    }

    function _on_load(){

        // log to console
        console.info('Loaded Jupyter extension: Git Commit and Push')

        // register new action
        var action_name = IPython.keyboard_manager.actions.register(git_commit_push, 'commit-push', 'jupyter-git')

        // add button for new action
        IPython.toolbar.add_buttons_group(['jupyter-git.commit-push'])

    }

    return {load_ipython_extension: _on_load };
})

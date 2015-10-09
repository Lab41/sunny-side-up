#!/bin/bash

# create qualification
__response=$(/aws-mturk-clt-1.3.1/bin/createQualificationType.sh \
                            -noretry \
                            -properties sentiment-qualification-arabic.properties \
                            -question sentiment-qualification-arabic-test.xml \
                            -answer sentiment-qualification-arabic-answers.xml)
echo "$__response"

# continue if successful
__qualification_id=$(echo $__response | cut -d ' ' -f4)
if [ -z $__qualification_id ]; then
  echo "FAILURE!  Could not create qualification test"
else
  echo "Created qualification ${__qualification_id}"
fi

#!/bin/bash
# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
set -e
# THIS FILE ASSUMES IT IS RUN INSIDE THE tests DIRECTORY

# Get all the tests marked with standalone marker
TEST_FILE="standalone_tests.txt"

test_path=$1
pytest_arg=$2  # use `-m standalone`
printf "source path: $test_path\n"
printf "pytest arg: $pytest_arg\n"


python -um pytest $test_path -q --collect-only $pytest_arg --pythonwarnings ignore 2>&1 > $TEST_FILE

# if any command in a shell script returns a non-zero exit status,
#  the script will immediately terminate and the remaining commands will not be executed
#set -e

# removes the last line of the file
sed -i '$d' $TEST_FILE

# Get test list and run each test individually
tests=$(grep -oP '\S+::test_\S+' "$TEST_FILE")
printf "collected tests:\n----------------\n$tests\n================\n"

status=0
for test in $tests; do
  python -um pytest -sv "$test" --pythonwarnings ignore --junitxml="$test-results.xml" 2>&1 > "$test-output.txt"
  pytest_status=$?
  if [ $pytest_status -eq 0 ]; then
    echo "$test PASSED"
  else
    status=$pytest_status
    echo "$test returned status $pytest_status"
    echo "================TEST OUTPUT BEGIN================"
    echo "$test-output.txt"
    echo "================TEST OUTPUT END=================="
  fi
done

#find . -name "*.xml" -exec cp -a -t . --parents {} +
rm $TEST_FILE

printf "Exiting with status: $status\n"
exit $status

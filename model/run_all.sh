set -e
python3 -W ignore run_sensitivity.py > sens.log 2>&1
python3 -W ignore bench.py > bench.log 2>&1
python3 -W ignore final_test.py > final.log 2>&1
echo ALLDONE >> final.log

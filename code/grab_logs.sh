#!/bin/bash
mkdir -p 03_EVALUATION/logs/T4/test 03_EVALUATION/logs/E1/test 03_EVALUATION/logs/T5/test 03_EVALUATION/logs/E2/test

# T4 training
cat 00_LOGS/T4/**/411888*.out > 03_EVALUATION/logs/T4/01.txt
cat 00_LOGS/T4/**/411926*.out > 03_EVALUATION/logs/T4/02.txt
cat 00_LOGS/T4/**/420232*.out > 03_EVALUATION/logs/T4/03.txt
cat 00_LOGS/T4/**/420238*.out > 03_EVALUATION/logs/T4/04.txt
cat 00_LOGS/T4/**/428535*.out > 03_EVALUATION/logs/T4/05.txt
cat 00_LOGS/T4/**/444352*.out > 03_EVALUATION/logs/T4/06.txt
cat 00_LOGS/T4/**/471699*.out > 03_EVALUATION/logs/T4/07.txt
cat 00_LOGS/T4/**/471700*.out >> 03_EVALUATION/logs/T4/07.txt
cat 00_LOGS/T4/**/471717*.out > 03_EVALUATION/logs/T4/08.txt
cat 00_LOGS/T4/**/480923*.out > 03_EVALUATION/logs/T4/09.txt
cat 00_LOGS/T4/**/480955*.out > 03_EVALUATION/logs/T4/10.txt
cat 00_LOGS/T4/**/511759*.out > 03_EVALUATION/logs/T4/11.txt

# T4 testing
cat 00_LOGS/T4/**/496030*.out > 03_EVALUATION/logs/T4/test/01.txt

# T5 training
cat 00_LOGS/T5/**/517074*.out > 03_EVALUATION/logs/T5/01.txt

# E1 training
cat 00_LOGS/E1/**/480964*.out > 03_EVALUATION/logs/E1/01.txt
cat 00_LOGS/E1/**/4848432.out > 03_EVALUATION/logs/E1/02.txt
cat 00_LOGS/E1/**/496023*.out > 03_EVALUATION/logs/E1/03.txt
cat 00_LOGS/E1/**/511758*.out > 03_EVALUATION/logs/E1/04.txt
cat 00_LOGS/E1/**/511770*.out > 03_EVALUATION/logs/E1/05.txt

# E1 testing
cat 00_LOGS/E1/**/496211*.out > 03_EVALUATION/logs/E1/test/01.txt
cat 00_LOGS/E1/**/496212*.out >> 03_EVALUATION/logs/E1/test/01.txt

# E2 training
cat 00_LOGS/E2/**/517074*.out > 03_EVALUATION/logs/E2/01.txt

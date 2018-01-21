import paramiko, re, pathlib, io
import pandas as pd

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(
    'gru.ifi.uzh.ch',
    username='rschlaefli',
    key_filename=str(pathlib.Path('C:/Users/roland/.ssh/gru')))

stdin, stdout, _stderr = ssh.exec_command('squeue')
for line in stdout.readlines():
    match = re.search(
        '([0-9_]+) +([a-z_]+) +([a-z-0-9]+) +rschlaef +([A-Z]) +([0-9:-]+) +([0-9]+) +([a-z0-9]+)',
        line)
    if match:
        # get the model version (e.g. T5 or E2)
        model_version = match.group(3).split('-')[1].upper()
        model_index = match.group(1).split('_')[1]

        # calculate the log path
        log_path = f'~/thesis/code/03_EVALUATION/histories/lstm_{model_version}_{model_index}.csv'
        print(f'> Reading {log_path}...')

        stdin2, stdout2, _stderr2 = ssh.exec_command(f'cat {log_path}')

        with io.StringIO(stdout2) as csv:
            df = pd.read_csv(csv, sep=';')
            print(df)

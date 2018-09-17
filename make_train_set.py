# 학습용 데이터 생성(자소)
import os
import csv
import character_util as c_util

tr_input = []
tr_input_view = []
tr_output = []
tr_output_view = []

ts_input = []
ts_input_view = []
ts_output = []
ts_output_view = []

train_cut = 0
with open(os.getcwd()+'/sample_data/transcript_after_watson.csv', encoding='utf-8') as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    next(reader, None)
    train_cut = round(len(list(reader))*0.7)

with open(os.getcwd()+'/sample_data/transcript_after_watson.csv', encoding='utf-8') as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    next(reader, None)
    sos_token = [key for key, value in c_util.special.items() if value == '<sos>']
    eos_token = [key for key, value in c_util.special.items() if value == '<eos>']
    for idx, row in enumerate(reader):
        output_char, output_idx = c_util.decompose_sentence(row[1])
        input_char, input_idx = c_util.decompose_sentence(row[3])
        if idx <= train_cut:
            tr_output.append(sos_token+output_idx+eos_token)
            tr_output_view.append(output_char)
            tr_input.append(sos_token+input_idx+eos_token)
            tr_input_view.append(input_char)
        else:
            ts_output.append(sos_token + output_idx + eos_token)
            ts_output_view.append(output_char)
            ts_input.append(sos_token + input_idx + eos_token)
            ts_input_view.append(input_char)


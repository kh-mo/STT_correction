# 학습용 데이터 생성(음절)
import os
import csv
import syllable_util as s_util

data = {}
with open(os.getcwd()+'/sample_data/transcript_after_watson_v2.csv', encoding='utf-8') as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    next(reader, None)
    for row in reader:
        data[row[3]] = row[1]

data_input = sorted(data, key=len)
data_output = [data[i] for i in data_input]
dataset_cut = round(len(data)*0.3)

tr_input = []
tr_input_view = []
tr_output = []
tr_output_view = []

ts_input = []
ts_input_view = []
ts_output = []
ts_output_view = []

sos_token = [s_util.chr2idx['<sos>']]
eos_token = [s_util.chr2idx['<sos>']]

for idx, row in enumerate(data_input):
    input_char, input_idx = s_util.decompose_sentence(row)

    if idx >= dataset_cut:
        tr_input.append(sos_token + input_idx + eos_token)
        tr_input_view.append(input_char)
    else:
        ts_input.append(sos_token + input_idx + eos_token)
        ts_input_view.append(input_char)


for idx, row in enumerate(data_output):
    output_char, output_idx = s_util.decompose_sentence(row)

    if idx >= dataset_cut:
        tr_output.append(sos_token + output_idx + eos_token)
        tr_output_view.append(output_char)
    else:
        ts_output.append(sos_token + output_idx + eos_token)
        ts_output_view.append(output_char)


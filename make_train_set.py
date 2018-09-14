# 학습용 데이터 생성(자소)
import os
import csv
import character_util as c_util

input = []
input_view = []
output = []
output_view = []

with open(os.getcwd()+'/sample_data/transcript_after_watson.csv', encoding='utf-8') as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    next(reader, None)
    sos_token = [key for key, value in c_util.special.items() if value == '<sos>']
    eos_token = [key for key, value in c_util.special.items() if value == '<eos>']
    for row in reader:
        output_char, output_idx = c_util.decompose_sentence(row[1])
        input_char, input_idx = c_util.decompose_sentence(row[3])

        output.append(sos_token+output_idx+eos_token)
        output_view.append(output_char)
        input.append(sos_token+input_idx+eos_token)
        input_view.append(input_char)

# 음절 단위 데이터 셋 생성
import os
import csv
import math
import syllable.util as util

# key : stt 문장, value : 정답 문장
data = {}
with open(os.getcwd()+'/sample_data/transcript_after_watson_v2.csv', encoding='utf-8') as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    next(reader, None)
    for row in reader:
        data[row[3]] = row[1]

# key를 글자수 오름차순으로 정렬
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

sos_token = [util.syl2idx['<sos>']]
eos_token = [util.syl2idx['<eos>']]

print("build encoder input")
for idx, row in enumerate(data_input):
    input_char, input_idx = util.decompose_sentence(row)

    # 글자수를 오름차순으로 정렬했으므로 idx가 크면 긴 문장
    # 짧은 문장을 ts, 긴 문장을 tr로 배분
    if idx >= dataset_cut:
        tr_input.append(sos_token + input_idx + eos_token)
        tr_input_view.append(input_char)
    else:
        ts_input.append(sos_token + input_idx + eos_token)
        ts_input_view.append(input_char)

    # 10% 마다 데이터 생성 완료 print
    if idx % math.floor(len(data_input)*0.1) == 0:
        print(str(int(round(idx / len(data_input),1) * 100)) + " % 완료")

print("complete encoder input")
print("**********************")
print("build decoder answer")

for idx, row in enumerate(data_output):
    output_char, output_idx = util.decompose_sentence(row)

    if idx >= dataset_cut:
        tr_output.append(sos_token + output_idx + eos_token)
        tr_output_view.append(output_char)
    else:
        ts_output.append(sos_token + output_idx + eos_token)
        ts_output_view.append(output_char)

    # 10% 마다 데이터 생성 완료 print
    if idx % math.floor(len(data_output)*0.1) == 0:
        print(str(int(round(idx / len(data_output), 1) * 100)) + " % 완료")

print("complete decoder answer")

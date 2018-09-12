import csv
import character_util

input = []
input_view = []
output = []
output_view = []

with open('C:/Users/Administrator/PycharmProjects/STT_correction/sample_data/transcript_after_watson.csv', encoding='utf-8') as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    next(reader, None)
    for row in reader:
        output_char, output_idx = character_util.decompose_sentence(row[1])
        input_char, input_idx = character_util.decompose_sentence(row[3])

        output.append([106]+output_idx+[107])
        output_view.append(output_char)
        input.append([106]+input_idx+[107])
        input_view.append(input_char)



code = list(range(44032,55204))
special = ['.', ' ', "1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
           "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O",
           "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "<sos>", "<eos>", "<pad>"]
all_char = code + special
idx2chr = {}
chr2idx = {}

for idx, num in enumerate(all_char):
    if type(num) == int:
        idx2chr[idx] = chr(num)
        chr2idx[chr(num)] = idx
    else:
        idx2chr[idx] = num
        chr2idx[num] = idx

chr_key = list(chr2idx.keys())
voca_size = len(chr2idx)

def decompose_sentence(text):
    result = ""
    result_idx = []
    for char in text:
        if char in chr_key:
            result += (char+" ")
            result_idx.append(chr2idx[char])
        else:
            pass
    return result, result_idx

def compose_sentence(char_text_idx):
    result = ""
    for char in char_text_idx:
        result += idx2chr[char]
    return result

'''
text = "그 애 전화번호 알아?"
char_text, char_text_idx = decompose_sentence(text)
char_text
char_text_idx
compose_sentence(char_text_idx)
'''
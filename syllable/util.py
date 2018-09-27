'''
한국어 음절 유니코드
{가-힣 : AC00-D7AF : }
chr(unicode) : unicode를 받아 한글 return
'''

unicode = list(range(44032,55204))
special = ['.', ' ', "1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
           "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O",
           "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "<sos>", "<eos>", "<pad>"]
all_syl = unicode + special
idx2syl = {}
syl2idx = {}

for idx, num in enumerate(all_syl):
    # 입력 값이 유니코드일 경우
    if type(num) == int:
        idx2syl[idx] = chr(num)
        syl2idx[chr(num)] = idx
    # 입력 값이 유니코드가 아닌 경우
    else:
        idx2syl[idx] = num
        syl2idx[num] = idx

syl_key = list(syl2idx.keys())
voca_size = len(syl_key)

def decompose_sentence(text):
    result = ""
    result_idx = []
    for char in text.strip():
        if char in syl_key:
            result += (char+" ")
            result_idx.append(syl2idx[char])
        else:
            pass
    return result.strip(), result_idx

def compose_sentence(char_text_idx):
    result = ""
    for char in char_text_idx:
        result += idx2syl[char]
    return result

'''
text = "그 애 전화번호 알아? "
char_text, char_text_idx = decompose_sentence(text)
char_text
char_text_idx
compose_sentence(char_text_idx)
'''
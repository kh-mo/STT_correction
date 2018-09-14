from six import unichr

# Code = 0xAC00 + (Chosung_index * NUM_JOONG * NUM_JONG) + (Joongsung_index * NUM_JONG) + (Jongsung_index)
# ord() 유니코드 코드포인트 반환
cho = {0: 'ㄱ', 1: 'ㄲ', 2: 'ㄴ', 3: 'ㄷ', 4: 'ㄸ', 5: 'ㄹ', 6: 'ㅁ', 7: 'ㅂ', 8: 'ㅃ', 9: 'ㅅ',
       10: 'ㅆ', 11: 'ㅇ', 12: 'ㅈ', 13: 'ㅉ', 14: 'ㅊ', 15: 'ㅋ', 16: 'ㅌ', 17: 'ㅍ', 18: 'ㅎ'}
cho_list = list(cho.keys())

joong = {19: 'ㅏ', 20: 'ㅐ', 21: 'ㅑ', 22: 'ㅒ', 23: 'ㅓ', 24: 'ㅔ', 25: 'ㅕ', 26: 'ㅖ', 27: 'ㅗ', 28: 'ㅘ',
         29: 'ㅙ', 30: 'ㅚ', 31: 'ㅛ', 32: 'ㅜ', 33: 'ㅝ', 34: 'ㅞ', 35: 'ㅟ', 36: 'ㅠ', 37: 'ㅡ', 38: 'ㅢ', 39: 'ㅣ'}
joong_list = list(joong.keys())

jong = {40: '', 41: 'ㄱ', 42: 'ㄲ', 43: 'ㄳ', 44: 'ㄴ', 45: 'ㄵ', 46: 'ㄶ', 47: 'ㄷ', 48: 'ㄹ', 49: 'ㄺ',
        50: 'ㄻ', 51: 'ㄼ', 52: 'ㄽ', 53: 'ㄾ', 54: 'ㄿ', 55: 'ㅀ', 56: 'ㅁ', 57: 'ㅂ', 58: 'ㅄ', 59: 'ㅅ',
        60: 'ㅆ', 61: 'ㅇ', 62: 'ㅈ', 63: 'ㅊ', 64: 'ㅋ', 65: 'ㅌ', 66: 'ㅍ', 67: 'ㅎ'}
jong_list = list(jong.keys())

special = {68: '.', 69: ' ', 70: "1", 71: "2", 72: "3", 73: "4", 74: "5", 75: "6", 76: "7", 77: "8", 78: "9", 79: "0",
           80: "A", 81: "B", 82: "C", 83: "D", 84: "E", 85: "F", 86: "G", 87: "H", 88: "I", 89: "J", 90: "K", 91: "L",
           92: "M", 93: "N", 94: "O", 95: "P", 96: "Q", 97: "R", 98: "S", 99: "T", 100: "U", 101: "V", 102: "W",
           103: "X", 104: "Y", 105: "Z", 106: "<sos>", 107: "<eos>", 108: "<pad>"}

jamo = list(cho.values()) + list(joong.values()) + list(jong.values())[1:]
jamo_index = list(cho.keys()) + list(joong.keys()) + list(jong.keys())

num_cho = len(cho)
num_joong = len(joong)
num_jong = len(jong)
voca_size = len(cho)+len(joong)+len(jong)+len(special)

first_hangul_unicode = 0xAC00
last_hangul_unicode = 0xD7A3


def is_hangul(letter):
    # 한 글자를 유니코드로 변환
    # 유니코드 값이 한글 유니코드 범위내에 있거나 단일 자소일 경우 true 반환 == 한글임을 의미
    code = ord(letter)
    if (code >= first_hangul_unicode and code <= last_hangul_unicode) or (letter in jamo):
        return True
    return False


def decompose_letter(letter):
    # Code = 0xAC00 + (Chosung_index * NUM_JOONG * NUM_JONG) + (Joongsung_index * NUM_JONG) + (Jongsung_index)
    code = ord(letter) - first_hangul_unicode
    jong_idx = int(code % num_jong)
    code /= num_jong
    joong_idx = int(code % num_joong)
    code /= num_joong
    cho_idx = int(code)
    return list(cho.values())[cho_idx], list(joong.values())[joong_idx], list(jong.values())[jong_idx], \
           list(cho.keys())[cho_idx], list(joong.keys())[joong_idx], list(jong.keys())[jong_idx]


def decompose_sentence(text):
    # 문장(text)을 받아 초중성 분해와 인덱스 반환
    result = ""
    result_idx = []
    for char in text:
        if is_hangul(char):
            cho, jung, jong, cho_idx, jung_idx, jong_idx = decompose_letter(char)
            result = result + cho + jung + jong
            result_idx.append(cho_idx)
            result_idx.append(jung_idx)
            result_idx.append(jong_idx)
        elif char in list(special.values()):
            result += char
            [result_idx.append(key) for key, value in special.items() if value == char]
        else:
            # 미리 지정한 사전에 없는 자소는 pass 시켜서 제거
            pass
    return result, result_idx


def compose_letter(cho, joong, jong):
    # 초중성 리스트의 인덱스값을 이용해서 글자 복원
    cho_index = cho_list.index(cho)
    joongsung_index = joong_list.index(joong)
    jongsung_index = jong_list.index(jong)
    return unichr(0xAC00 + cho_index * num_joong * num_jong + joongsung_index * num_jong + jongsung_index)


def compose_sentence(char_text_idx):
    # input : char_text_idx,    output : 복원 문장
    tmp_CHO = "null"
    tmp_JUNG = "null"
    tmp_JONG = "null"
    result = ""
    for idx, char in enumerate(char_text_idx):
        if char in jamo_index:
            if char in list(cho.keys()):
                tmp_CHO = char
            elif char in list(joong.keys()):
                tmp_JUNG = char
            elif char in list(jong.keys()):
                tmp_JONG = char
        else:
            result += special[char]

        if (tmp_CHO != "null") and (tmp_JUNG != "null") and (tmp_JONG != "null"):
            result += compose_letter(tmp_CHO, tmp_JUNG, tmp_JONG)
            tmp_CHO = "null"
            tmp_JUNG = "null"
            tmp_JONG = "null"
    return result

'''
decompose_letter("이")
text = "그 애 전화번호 알아?"
text = "아?"
char_text, char_text_idx = decompose_sentence(text)
char_text
char_text_idx
compose_letter(char_text_idx[0], char_text_idx[1], char_text_idx[2])
compose_sentence(char_text_idx)
'''


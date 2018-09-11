from six import unichr

# Code = 0xAC00 + (Chosung_index * NUM_JOONG * NUM_JONG) + (Joongsung_index * NUM_JONG) + (Jongsung_index)
# ord() 유니코드 코드포인트 반환
CHO = {0: 'ㄱ', 1: 'ㄲ', 2: 'ㄴ', 3: 'ㄷ', 4: 'ㄸ', 5: 'ㄹ', 6: 'ㅁ', 7: 'ㅂ', 8: 'ㅃ', 9: 'ㅅ',
       10: 'ㅆ', 11: 'ㅇ', 12: 'ㅈ', 13: 'ㅉ', 14: 'ㅊ', 15: 'ㅋ', 16: 'ㅌ', 17: 'ㅍ', 18: 'ㅎ'}
CHO_list = [value for item, value in enumerate(CHO)]

JOONG = {19: 'ㅏ', 20: 'ㅐ', 21: 'ㅑ', 22: 'ㅒ', 23: 'ㅓ', 24: 'ㅔ', 25: 'ㅕ', 26: 'ㅖ', 27: 'ㅗ', 28: 'ㅘ',
         29: 'ㅙ', 30: 'ㅚ', 31: 'ㅛ', 32: 'ㅜ', 33: 'ㅝ', 34: 'ㅞ', 35: 'ㅟ', 36: 'ㅠ', 37: 'ㅡ', 38: 'ㅢ', 39: 'ㅣ'}
JOONG_list = [value for item, value in enumerate(JOONG)]

JONG = {40: '', 41: 'ㄱ', 42: 'ㄲ', 43: 'ㄳ', 44: 'ㄴ', 45: 'ㄵ', 46: 'ㄶ', 47: 'ㄷ', 48: 'ㄹ', 49: 'ㄺ',
        50: 'ㄻ', 51: 'ㄼ', 52: 'ㄽ', 53: 'ㄾ', 54: 'ㄿ', 55: 'ㅀ', 56: 'ㅁ', 57: 'ㅂ', 58: 'ㅄ', 59: 'ㅅ',
        60: 'ㅆ', 61: 'ㅇ', 62: 'ㅈ', 63: 'ㅊ', 64: 'ㅋ', 65: 'ㅌ', 66: 'ㅍ', 67: 'ㅎ'}
JONG_list = [value for item, value in enumerate(JONG)]

SPECIAL = {68: '.', 69: ' ', 70: "1", 71: "2", 72: "3", 73: "4", 74: "5", 75: "6", 76: "7", 77: "8", 78: "9", 79: "0",
           80: "A", 81: "B", 82: "C", 83: "D", 84: "E", 85: "F", 86: "G", 87: "H", 88: "I", 89: "J", 90: "K", 91: "L",
           92: "M",
           93: "N", 94: "O", 95: "P", 96: "Q", 97: "R", 98: "S", 99: "T", 100: "U", 101: "V", 102: "W", 103: "X",
           104: "Y", 105: "Z"}

JAMO = list(CHO.values()) + list(JOONG.values()) + list(JONG.values())[1:]
JAMO_index = list(CHO.keys()) + list(JOONG.keys()) + list(JONG.keys())

NUM_CHO = 19
NUM_JOONG = 21
NUM_JONG = 28

FIRST_HANGUL_UNICODE = 44032
LAST_HANGUL_UNICODE = 0xD7A3


def is_jamo(letter):
    return letter in JAMO


def is_hangul(letter):
    code = ord(letter)
    if (code >= FIRST_HANGUL_UNICODE and code <= LAST_HANGUL_UNICODE) or is_jamo(letter):
        return True
    return False


def decompose_letter(letter):
    code = ord(letter) - FIRST_HANGUL_UNICODE
    jong = int(code % NUM_JONG)
    code /= NUM_JONG
    joong = int(code % NUM_JOONG)
    code /= NUM_JOONG
    cho = int(code)
    return list(CHO.values())[cho], list(JOONG.values())[joong], list(JONG.values())[jong], list(CHO.keys())[cho], \
           list(JOONG.keys())[joong], list(JONG.keys())[jong]


def decompose_sentence(text):
    result = ""
    result_idx = []
    char = " "
    for char in text:
        # print(char)
        if is_hangul(char):
            cho, jung, jong, cho_idx, jung_idx, jong_idx = decompose_letter(char)
            result = result + cho + jung + jong
            result_idx.append(cho_idx)
            result_idx.append(jung_idx)
            result_idx.append(jong_idx)
        elif char in list(SPECIAL.values()):
            spec_idx = list(SPECIAL.values()).index(char)
            result += list(SPECIAL.values())[spec_idx]
            result_idx.append(list(SPECIAL.keys())[spec_idx])
        else:
            pass
    return result, result_idx


def compose_letter(cho, joong, jong):
    cho_index = CHO_list.index(cho)
    joongsung_index = JOONG_list.index(joong)
    jongsung_index = JONG_list.index(jong)

    return unichr(0xAC00 + cho_index * NUM_JOONG * NUM_JONG + joongsung_index * NUM_JONG + jongsung_index)


def compose_sentence(char_text_idx):
    # input : char_text_idx
    # output : 완성형 string
    tmp_CHO = ""
    tmp_JUNG = ""
    tmp_JONG = ""
    result = ""
    for idx, char in enumerate(char_text_idx):
        if char in JAMO_index:
            if char in list(CHO.keys()):
                tmp_CHO = char
            elif char in list(JOONG.keys()):
                tmp_JUNG = char
            elif char in list(JONG.keys()):
                tmp_JONG = char
        else:
            result += SPECIAL[char]

        if (tmp_CHO != "") and (tmp_JUNG != "") and (tmp_JONG != ""):
            result += compose_letter(tmp_CHO, tmp_JUNG, tmp_JONG)
            tmp_CHO = "";
            tmp_JUNG = "";
            tmp_JONG = ""
    return result


decompose_letter("이")
text = "그 애 전화번호 알아?"
char_text, char_text_idx = decompose_sentence(text)
char_text
char_text_idx
compose_letter(char_text_idx[0], char_text_idx[1], char_text_idx[2])
compose_sentence(char_text_idx)



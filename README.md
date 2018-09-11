# STT_correction

## description
한국어 STT를 통해 얻은 문장은 약간의 오류가 존재하기에 이를 보정하여 정확한 문장을 생성하는 딥러닝 네트워크를 구성하고자 함.

## To do
1. 정확한 문장을 자소단위로 분해
2. 자소 단위 입력 일부를 변형하여 오류 문장 생성
3. input : 오류문장, output : 정상문장
4. seq2seq 네트워크를 구성하여 test
5. 자소 output을 원래 문장으로 복원

## reference

[1] [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
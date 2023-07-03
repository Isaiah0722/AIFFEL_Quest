d# AIFFEL Campus Online 4th Code Peer Review Templete
- 코더 : 이규상
- 리뷰어 : 이윤상


# PRT(PeerReviewTemplate) 
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.

- [X] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  
- [X] 주석을 보고 작성자의 코드가 이해되었나요?
  > 지정된 변수나 명령어 마다 주석이 달려있어 이해하기 쉬웠습니다.
- [X] 코드가 에러를 유발할 가능성이 없나요?
  > 논리적으로 문제가 없어 텍스트파일이 바뀌어도 에러를 유발할 가능성은 없어보입니다.
- [X] 코드 작성자가 코드를 제대로 이해하고 작성했나요?
  > 잘 이해하고 작성했다고 생각합니다. 코드의 여러 부분에 근거를 들어가면서 설명하였습니다.
- [X] 코드가 간결한가요?
  > 짧고 간결하게 작성했다고 생각합니다. 코드의 전체적인 길이가 짧고 알아보기 쉬웠습니다.

# 예시
import re
from google.colab import drive
from collections import Counter

drive.mount('/content/drive')

def remove_symbols(text):
    cleaned_text = re.sub(r'[^\w\s]', ' ', text)  # 정규표현식 패턴을 사용하여 기호를 제거하고 ' '대체합니다.
    cleaned_text = cleaned_text.replace('\n',' ') #\n을 ' '으로 대체합니다.

    return cleaned_text

file = open('/content/drive/MyDrive/Avengers.txt') #Avengers파일 불러오기
contents = file.read()
file.close()
f = remove_symbols(contents) #기호를 제거한 문자열을 임의의 변수에 저장합니다.
new_contents = f.lower() #대문자를 소문자로 변환

words = new_contents.split() #word 단위로 나누어 list에 저장
two_grams = [' '.join(words[i:i+2]) for i in range(len(words)-1)]

ngrams = zip(words, words[1:]) # 2-grams

counter = Counter(ngrams)

most_common = counter.most_common(1)
print(most_common)


# 참고 링크 및 코드 개선
import re
from google.colab import drive
from collections import Counter

drive.mount('/content/drive')

def remove_symbols(text):
    cleaned_text = re.sub(r'[^\w\s]', ' ', text)  
    cleaned_text = cleaned_text.replace('\n',' ') 

    return cleaned_text

file = open('/content/drive/MyDrive/Avengers.txt') 
contents = file.read()
file.close()
f = remove_symbols(contents) 
new_contents = f.lower() 

words = new_contents.split() 
two_grams = [' '.join(words[i:i+2]) for i in range(len(words)-1)]                       #two_grams 변수를 지정하고 사용되지 않았습니다.

ngrams = zip(words, words[1:]) # 2-grams

counter = Counter(ngrams)

most_common = counter.most_common(1)
print(most_common)


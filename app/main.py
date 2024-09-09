import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from fastapi import FastAPI

df = pd.read_csv('./data/dataset.csv')
titles = df['넘버 제목'].tolist()
textData = df['정제'].tolist()


def cosineCalcul(titles, textData, inputTitle):
    # 모든 곡의 가사를 하나의 리스트로 묶어서 벡터화
    all_lyrics = [' '.join(lyrics) for lyrics in textData]

    vectorizer = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False, token_pattern=None)
    tfidf_matrix = vectorizer.fit_transform(all_lyrics)

    # 입력 곡의 인덱스 찾기
    inputIndex = titles.index(inputTitle)

    # 입력 곡의 TF-IDF 벡터 추출
    input_vector = tfidf_matrix[inputIndex]

    # 모든 곡과 입력 곡 간의 코사인 유사도 계산
    cosine = cosine_similarity(input_vector, tfidf_matrix).flatten()

    # 입력 곡 제외하고 유사도 높은 순서로 정렬
    similarSongs = [(titles[i], cosine[i]) for i in range(len(titles)) if i != inputIndex]
    similarSongs = sorted(similarSongs, key=lambda x: x[1], reverse=True)

    return similarSongs[:10]

# def __init():
#         return cosineCalcul(titles,textData,inputTitle)

app = FastAPI()

@app.get('/')
def root():
    return {"message" : "This is Main Page"}

@app.get('/recommand/{numberTitle}')
async def recommand(numberTitle : str):
    if numberTitle not in titles:
        return {"message" : "다시 한 번 넘버(뮤지컬 곡의 제목)을 확인 부탁드립니다."}
    return cosineCalcul(titles, textData, numberTitle)
    
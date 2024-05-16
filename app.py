# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
from konlpy.tag import Okt
from gensim.models import FastText
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

app = Flask(__name__)
okt = Okt()

def recommendations(title, filtered_titles):

    # FastText model 학습
    model = FastText(vector_size=80, window=5, min_count=2, workers=-1)
    model.build_vocab(corpus_iterable=filtered_titles)
    model.train(corpus_iterable=filtered_titles, total_examples=len(filtered_titles), epochs=15)
   
    title_tokens = okt.morphs(title)
    title_vector = model.wv[title_tokens]
    title_vector_padded = np.pad(title_vector, ((0,1520 - len(title_vector)), (0,0)), mode='constant')
    title_vector_padded = title_vector_padded.reshape(1,-1)

    similarities = []
    for filtered_title in filtered_titles:

        filtered_title_tokens = okt.morphs(filtered_title)
        filtered_title_vector = model.wv[filtered_title_tokens]
        filtered_title_vector_padded = np.pad(filtered_title_vector, ((0, 1520 - len(filtered_title_vector)), (0, 0)), mode='constant')
        filtered_title_vector_padded = filtered_title_vector_padded.reshape(1,-1)

        similarity = cosine_similarity(title_vector_padded, filtered_title_vector_padded)[0][0]
        similarities.append(similarity)
        print("----")

    # 유사도가 높은 순서로 데이터 정렬
    sorted_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
    print("sorted_indices: ", sorted_indices)

    # 상위 n개 데이터 선택
    top_n = 5
    recommend = [filtered_titles[i] for i in sorted_indices[:top_n]]
    print("recommend: ", recommend)

    return recommend

@app.route('/')
def default():
    return "<p>This is Hi-sujung Recommendation Page</p>"

@app.route("/recommend/univ", methods=['POST'])
def univAct():
    data = request.json
    # 현재 페이지 공지명
    title = data.get('title')
    # 현재 페이지의 학과와 동일한 공지명들
    filtered_title = data.get('filtered_title', [])
    id_filtered_title = data.get('id', [])
    print("title: ", title)
    df = pd.DataFrame({'id': id_filtered_title, 'filtered_title': filtered_title})
    print("df: ", df)

    if title is None:
        return jsonify({"error": "Notice title is required."}), 400

    try:
        recommend_list = recommendations(title, filtered_title)
        recommend_id_list = df[df['filtered_title'].isin(recommend_list)]['id'].tolist()
        return jsonify({'recommend': recommend_id_list})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/recommend/external", methods=['POST'])
def externalAct():
    data = request.json
    title = data.get('title')
    filtered_title = data.get('filtered_title', [])

    if title is None:
        return jsonify({"error": "Notice title is required."}), 400

    try:
        recommend_list = recommendations(title, filtered_title)
        return jsonify({'recommend': recommend_list})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from numpy import dot
from numpy.linalg import norm
from transformers import T5Config
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify
import matplotlib.pyplot as plt
from gensim.models import FastText
from konlpy.tag import Okt
from sklearn.metrics.pairwise import cosine_similarity
import requests
import gensim
import urllib.request
from google.cloud.sql.connector import Connector
import sqlalchemy
import pymysql.cursors
import csv
import config

app = Flask(__name__)

# 서비스계정 key JSON 파일
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./rising-woods-388317-8a18f2625ba5.json"

##1. 대외활동
connector = Connector()
def getconn() -> pymysql.connections.Connection:
    conn1: pymysql.connections.Connection = connector.connect(
        config.DB_CONFIG['host'],
        "pymysql",
        user=config.DB_CONFIG['user'],
        password=config.DB_CONFIG['password'],
        db=config.DB_CONFIG['db']
    )
    return conn1

pool = sqlalchemy.create_engine(
    "mysql+pymysql://",
    creator=getconn,
)

conn1 = pool.raw_connection()
cursor = conn1.cursor()

try:
    cursor = conn1.cursor()
    sql = "SELECT DISTINCT title FROM external_act LIMIT 70;"
    cursor.execute(sql)
    result = cursor.fetchall()

    # Writing the data to a CSV file locally on the client-side
    with open('contest.csv', 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([i[0] for i in cursor.description])  # Write column headers
        csv_writer.writerows(result)  # Write data rows

    #print("Data exported to contest.csv successfully.")
except Exception as e:
    print("Error:", e)
finally:
    conn1.close()
    
## 2. 교내활동
connector = Connector()
def getconn() -> pymysql.connections.Connection:
    conn2: pymysql.connections.Connection = connector.connect(
        config.DB_CONFIG['host'],
        "pymysql",
        user=config.DB_CONFIG['user'],
        password=config.DB_CONFIG['password'],
        db=config.DB_CONFIG['db']
    )
    return conn2

pool = sqlalchemy.create_engine(
    "mysql+pymysql://",
    creator=getconn,
)

conn2 = pool.raw_connection()
cursor = conn2.cursor()

try:
    cursor2 = conn2.cursor()
    sql2 = "SELECT DISTINCT title FROM univ_activity LIMIT 70;"
    cursor2.execute(sql2)
    result2 = cursor2.fetchall()

    # Writing the data to a CSV file locally on the client-side
    with open('univ.csv', 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([i[0] for i in cursor2.description])  # Write column headers
        csv_writer.writerows(result2)  # Write data rows

    #print("Data exported to univ.csv successfully.")
except Exception as e:
    print("Error:", e)
finally:
    conn2.close()
    
def reccomendations(filename, univ_activity_id):
    # 데이터를 데이터프레임으로 로드
    df = pd.read_csv(filename)
    okt = Okt()
    tokenized_data = []
    for sentence in df['title']:
        tokenized_sentence = okt.morphs(sentence, stem=True)
        tokenized_data.append(tokenized_sentence)

    # FastText 모델 학습
    model = FastText(vector_size=80, window=5, min_count=2, workers=-1)
    model.build_vocab(corpus_iterable=tokenized_data)
    model.train(corpus_iterable=tokenized_data, total_examples=len(tokenized_data), epochs=15)
    #word_vectors = model.wv
    activities = df[['title']]

    # 문장 벡터 계산
    sentence_vectors = [model.wv[tokenized_sentence].mean(axis=0) for tokenized_sentence in tokenized_data]
    cosine_similarities = cosine_similarity(sentence_vectors, sentence_vectors)
    
    # 입력한 ID에 해당하는 제목 찾기
    title =df['title'][univ_activity_id]
    
    # 유사한 활동 5개 선정
    indices = pd.Series(df.index, index=df['title']).drop_duplicates(keep='first')
    if title in indices:
        idx = indices[title]
        sim_scores = list(enumerate(cosine_similarities[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:6]
        activity_indices = [i[0] for i in sim_scores]
        recommend = activities.iloc[activity_indices].reset_index(drop=True)
        return recommend
    

@app.route("/recommend/univ", methods=['GET'])
def univAct():
        act = request.args.get('activity_name', type=int, default='')
        recommend_df = reccomendations("univ.csv", act)  # Assuming 'recommend' is the result of the recommendations

        # Extract the titles from the recommendation dataframe
        recommend_titles = recommend_df['title'].tolist()

        # Create placeholders for the SQL query
        placeholders = ', '.join(['%s'] * len(recommend_titles))

        # SQL query to fetch information based on titles
        sql = "SELECT univ_activity_id, title, link FROM univ_activity WHERE title IN ({});".format(placeholders)

        connector = Connector()
        def getconn() -> pymysql.connections.Connection:
            conn: pymysql.connections.Connection = connector.connect(
                config.DB_CONFIG['host'],
                "pymysql",
                user=config.DB_CONFIG['user'],
                password=config.DB_CONFIG['password'],
                db=config.DB_CONFIG['db']
            )
            return conn

        pool = sqlalchemy.create_engine(
            "mysql+pymysql://",
            creator=getconn,
        )

        conn = pool.raw_connection()
        cursor = conn.cursor()

        try:
            with conn.cursor() as cursor:
                cursor.execute(sql, recommend_titles)
                result = cursor.fetchall()
        finally:
            conn.close()
            # Create a dataframe from the SQL query result
        result_df = pd.DataFrame(result, columns=['univ_activity_id', 'title', 'link'])
        return jsonify({'reccomend': result_df.to_dict(orient='records')})
    
@app.route("/recommend/external", methods=['GET'])
def externalAct():
        act = request.args.get('activity_name', type=int, default='')
        recommend_df = reccomendations("contest.csv", act)  # Assuming 'recommend' is the result of the recommendations

        # Extract the titles from the recommendation dataframe
        recommend_titles = recommend_df['title'].tolist()

        # Create placeholders for the SQL query
        placeholders = ', '.join(['%s'] * len(recommend_titles))

        # SQL query to fetch information based on titles
        sql = "SELECT external_act_id, title, link FROM external_act WHERE title IN ({});".format(placeholders)

        connector = Connector()
        def getconn() -> pymysql.connections.Connection:
            conn: pymysql.connections.Connection = connector.connect(
                config.DB_CONFIG['host'],
                "pymysql",
                user=config.DB_CONFIG['user'],
                password=config.DB_CONFIG['password'],
                db=config.DB_CONFIG['db']
            )
            return conn

        pool = sqlalchemy.create_engine(
            "mysql+pymysql://",
            creator=getconn,
        )

        conn = pool.raw_connection()
        cursor = conn.cursor()

        try:
            with conn.cursor() as cursor:
                cursor.execute(sql, recommend_titles)
                result = cursor.fetchall()
        finally:
            conn.close()
            # Create a dataframe from the SQL query result
        result_df = pd.DataFrame(result, columns=['external_act_id', 'title', 'link'])

        return jsonify({'reccomend': result_df.to_dict(orient='records')})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False,host="0.0.0.0",port=5000)

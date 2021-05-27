"""Import required libraries"""
import pandas as pd
import numpy as np
import json
from compress_fasttext.models import CompressedFastTextKeyedVectors

"""Import data"""
df = pd.read_csv("../Task_2/train.csv", index_col=0)
df['title_cleared'] = df['title_cleared'].astype(str)

"""Preprocessing the date"""
df["date"] = pd.to_datetime(df['date'])
df["HOUR"] = df["date"].map(lambda x: x.hour)
df["MONTH"] = df["date"].map(lambda x: x.month)
df["WEEKDAY"] = df["date"].map(lambda x: x.weekday())

df = df[['title_cleared', 'score', 'cleared_text', 'num_comments', 'target', 'HOUR', 'MONTH', 'WEEKDAY']]

"""Encoding a time characteristic as a continuous variable"""


def make_cos_list(list_, period=24):
    def make_cos(value, period=period):
        return np.cos(value * 2 * np.pi / period)

    return [make_cos(x) for x in list_]


def make_sin_list(list_, period=24):
    def make_sin(value, period=period):
        return np.sin(value * 2 * np.pi / period)

    return [make_sin(x) for x in list_]


df['sin_hour'] = make_sin_list(df['HOUR'])
df['cos_hour'] = make_cos_list(df['HOUR'])

df['cos_month'] = make_cos_list(df['MONTH'], 12)
df['sin_month'] = make_sin_list(df["MONTH"], 12)

df['cos_weekday'] = make_cos_list(df['WEEKDAY'], 7)
df['sin_weekday'] = make_sin_list(df['WEEKDAY'], 7)

df.drop(columns=['HOUR', 'MONTH', 'WEEKDAY'], inplace=True)

"""Loading a compressed model"""
embs = "compressed.cc.en.300.bin"
embeddings = CompressedFastTextKeyedVectors.load(str(embs))

"""Averaging the vector by the words included in the text"""


def embed(tokens, default_size=100):
    if not tokens:
        return np.zeros(default_size)
    embs = [embeddings[x] for x in tokens]
    return sum(embs) / len(tokens)


def process_record(record):
    return embed(record.split())


"""Generating features"""
df['emb_title'] = df['title_cleared'].map(process_record)
df['emb_text'] = df['cleared_text'].map(str).map(process_record)
df.drop(columns=['title_cleared', 'cleared_text'], inplace=True)

"""Data formatting"""
df['emb_title'] = df['emb_title'].map(lambda x: json.dumps([float(y) for y in x]))
df['emb_text'] = df['emb_text'].map(lambda x: json.dumps([float(y) for y in x]))

emb_text = pd.DataFrame(df['emb_text'].map(json.loads).to_list(),
                        columns=[f"emb_text_{i}" for i in range(300)])

emb_title = pd.DataFrame(np.array(df['emb_title'].map(json.loads).to_list()),
                         columns=[f"emb_title_{i}" for i in range(300)])

df_embedded = pd.concat([df.drop(columns=['emb_title', 'emb_text']).reset_index(drop=True),
                         emb_text, emb_title], axis=1)

"""Export data"""
df_embedded.to_csv("final_embedded.csv")


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 14:10:14 2019

@author: kenneth
"""
#--dependencies for unsupervised modeling and GUI
import pandas as pd
import dash
import os 
import nltk
import random
import pickle
import re
import math
import random 
import json
import numpy as np
import statistics as stats
import collections
from kernelkmeans import kkMeans
from KPCA import kPCA
from nltk.stem.snowball import FrenchStemmer, PorterStemmer
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from os.path import join
from flask import Flask
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from sklearn.manifold import TSNE
#--dependencies for graph mining
import networkx as nx 
from networkx.algorithms.distance_measures import diameter
from networkx.algorithms.community import community_utils
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community import k_clique_communities
from networkx.algorithms.centrality import edge_betweenness_centrality
from networkx.algorithms.community.centrality import girvan_newman

#from PyDictionary import PyDictionary

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
server = Flask(__name__)
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, server = server)

#-config tools
config={
        "displaylogo": False,
        'modeBarButtonsToRemove': ['pan2d','lasso2d', 'hoverClosestCartesian',
                                   'hoverCompareCartesian', 'toggleSpikelines',
                                   ]
    }

#%% File imports
path = '/home/kenneth/Documents/MLDM M2/DATA MINING FOR BIG DATA/PROJECT/'+\
        'Social-Mining-Recommendation-System/DATASET'

#--book with properties
#booknames = pd.read_csv(os.path.join(path, 'pdf_names.csv'), sep = '\t', error_bad_lines=False)

#English books
#english = pd.read_csv(os.path.join(path, 'english_articles_translated.csv'), sep = ',', error_bad_lines=False)
#english.rename(columns = {'Unnamed: 0': 'pdf_names', 'content_translated': 'sentence'}, inplace = True)
#english['pdf_names'] = english['pdf_names'].apply(lambda x: f"pdf_{x}.txt")

#--complete book features
#booknamescomplete = pd.read_csv(os.path.join(path, 'processed/pcomplete.csv'), sep = ',', error_bad_lines=False).iloc[:, 1:]

#---update english versions with french translations
def updatetranslation(path, old, translated):
    '''
    :param:
        path: dataset path
        old: csv of file containing all documents
        translated: csv file containing translated documents
    '''
    pdfs_t = [x for x in translated.pdf_names]
    sentence_t = [x for x in translated.sentence]
    pdfs_o = [x for x in old.pdf_names]
    sentence_o = [x for x in old.sentence]
    for m, (ii, ij) in enumerate(zip(pdfs_t, sentence_t)):
        for w, (p, q) in enumerate(zip(pdfs_o, sentence_o)):
            if ii == p:
                old.loc[w, 'sentence'] = ij
    pdfs_o = [x for x in old.pdf_names]
    sentence_o = [x for x in old.sentence]
    for ii, ij in zip(pdfs_o, sentence_o):
        if not os.path.exists(os.path.join(path, f'books/{ii}')):
            with open(os.path.join(path, f'books/{ii}'), 'w+') as wr:
                if type(ij) == np.float:
                    wr.writelines('null')
                else:
                    wr.writelines(ij)
        else:
            with open(os.path.join(path, f'books/{ii}'), 'w+') as wr:
                if type(ij) == np.float:
                    wr.writelines('null')
                else:
                    wr.writelines(ij)
    return old
    
#load stopwords from drive
with open(os.path.join(path, 'stopwords'), 'r+') as st:
    stopwords = [x for x in st.read().split()]
    
#%%

#booknamescomplete = updatetranslation(path, booknamescomplete, english)

#%%
    
def tokStemmer(booknames):
    '''
    :params: 
        :booknames: original csv with authors and paper description
        :token_len: threshhold for length of tokens to filter.
        :toks: tokens
    :Return type:
        :Updated booknames with filtered book content
    '''
    toks = np.arange(5, 11, 1)
    book_path = sorted(os.listdir(join(path, 'books')), key = lambda x: int(x[4:].replace('.txt', '')))
    with open(os.path.join(path, 'stopwords'), 'r+') as st:
        stopwords = [x for x in st.read().split()]
    sentence = []
    for ii in book_path:
        with open(os.path.join(path, f'books/{ii}'), 'r+') as file:
            file_dt = file.read()
            #tokenize and stem
            fr_stem = FrenchStemmer()
            tokenizer = RegexpTokenizer(r'\w+')
            up_text = tokenizer.tokenize(file_dt)
            stemmed = []
            for ij in up_text:
                stemmed.append(fr_stem.stem(ij))
            file.close()
            final = ''
            new_token = []
            for each_word in stemmed:
                each_word = each_word.lower()
                if each_word not in stopwords:
                    new_token.append(each_word)
            final = ' '.join(new_token)
            sentence.append(final)
            #--processed tokens
            for tk in toks:
                word_freq = Counter()
                new_words = [ii for ii in new_token if len(ii) <= int(tk)]
                if not os.path.exists(os.path.join(path, f'counter/{tk}')):
                    os.makedirs(os.path.join(path, f'counter/{tk}'))
                    if not os.path.exists(os.path.join(path, f'counter/{tk}/{ii}')):
                        word = []
                        freq = []
                        for wd in new_words:
                            word_freq.update([wd])
                        for x, y in word_freq.most_common(30): #select frequent words
                            word.append(x)
                            freq.append(y)
                        most_frequent_words = pd.DataFrame({'word': word, 'freq': freq})
                        most_frequent_words.to_csv(os.path.join(path, f'counter/{tk}/{ii}'), mode='w')
                    else:
                        pass
                else:
                    if not os.path.exists(os.path.join(path, f'counter/{tk}/{ii}')):
                        word = []
                        freq = []
                        for wd in new_token:
                            word_freq.update([wd])
                        for x, y in word_freq.most_common(30):
                            word.append(x)
                            freq.append(y)
                        most_frequent_words = pd.DataFrame({'word': word, 'freq': freq})
                        most_frequent_words.to_csv(os.path.join(path, f'counter/{tk}/{ii}'), mode='w')
                    else:
                        pass
            if not os.path.exists(os.path.join(path, f'pbooks/{ii}')):
                with open(os.path.join(path, f'pbooks/{ii}'), 'w') as wr:
                    wr.writelines(final)
            else:
                with open(os.path.join(path, f'pbooks/{ii}'), 'w') as wr:
                    wr.writelines(final)
    complete = booknames.copy(deep = True)
    complete['sentence'] = sentence
    #--save files to directory
    if not os.path.exists(os.path.join(path, 'processed')):
        complete.to_csv(os.path.join(path, 'processed/pcomplete.csv'), mode='w')
    else:
        complete.to_csv(os.path.join(path, 'processed/pcomplete.csv'), mode='w')
    return complete
        
#complete = tokStemmer(booknamescomplete)
#%% Detect Language  
from textblob import TextBlob
def detect(path):
    language = []
    book_path = sorted(os.listdir(join(path, 'books')), key = lambda x: int(x[4:].replace('.txt', '')))
    for ii in book_path:
        with open(os.path.join(path, f'pbooks/{ii}'), 'r+') as file:
            file_dt = file.read()
            language.append(TextBlob(file_dt[:50]).detect_language())
    return language

#language = detect(path)

#%% Translate

def translate(path):
    transl = []
    book_path = sorted(os.listdir(join(path, 'books')), key = lambda x: int(x[4:].replace('.txt', '')))
    for ii in book_path:
        with open(os.path.join(path, f'books/{ii}'), 'r+') as file:
            file_dt = file.read()
            transl.append(TextBlob(file_dt).translate(to = 'en'))
    return transl

#translated = translate(path)

#%% Complete dataset csv
#complete['language'] = 'fr'
#complete.to_csv(os.path.join(path, 'processed/pcomplete.csv'))


#%% Similarity Score

from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity 
from sklearn.cluster import KMeans 
#from sklearn.decomposition import PCA

def similarity_score(df):
    '''
    :params:
        df: dataframe or list of documents
    '''
    tv = TfidfVectorizer(min_df = 1, use_idf = True) 
    tv_matrix = tv.fit_transform(df)
    tv_matrix = tv_matrix.toarray()
    vocab = tv.get_feature_names()
    similarity_matrix = cosine_similarity(tv_matrix)
    similarity_df = pd.DataFrame(similarity_matrix)
    return vocab, similarity_df

#_, simscore = similarity_score(complete['sentence'])
#pd.DataFrame(simscore).to_csv(os.path.join(path, 'processed/similarityscore.csv'))
    

    
#%%
data = pd.read_csv(os.path.join(path, 'processed/pcomplete.csv'), sep = ',',).iloc[:, 1:]
sort_dataset = data.sort_values(by=['year'])
simscore = pd.read_csv(os.path.join(path, 'processed/similarityscore.csv')).iloc[:, 1:]


#%%Optimum number of clusters
def k_optimal():
    import matplotlib.pyplot as plt
    sse = []
    for k in np.arange(2,40):
        model = KMeans(n_clusters = k)
        model.fit(simscore)
        sse.append(model.inertia_)
    plt.figure(figsize=(16,8))
    plt.plot(np.arange(2,40), sse, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()

#k_optimal()

#%%TSNE

def tsne(score, c_size):
    '''
    param:
        score: similarity score
        c_size: TSNE components
        
    return TSNE RESULT
    '''
    import pandas as pd
    for cs in c_size:
        ts = TSNE(random_state = cs, perplexity = 50).fit_transform(score)
        if not os.path.exists(os.path.join(path, 'tsne')):
            os.makedirs(os.path.join(path, 'tsne'))
            print(f'*******Saving TSNE_{cs}*******')
            ts = pd.DataFrame(ts)
            ts['pdf_names'] = np.array(data['pdf_names'])
            ts['year'] = np.array(data['year'])
            ts['language'] = np.array(data['language'])
            ts['authors'] = np.array(data['authors'])
            ts['title'] = np.array(data['title'])
            ts.to_csv(os.path.join(path, f'tsne/tsne_{cs}.csv'))
        else:
            print(f'*******Saving TSNE_{cs}*******')
            ts = pd.DataFrame(ts)
            ts['pdf_names'] = np.array(data['pdf_names'])
            ts['year'] = np.array(data['year'])
            ts['language'] = np.array(data['language'])
            ts['authors'] = np.array(data['authors'])
            ts['title'] = np.array(data['title'])
            ts.to_csv(os.path.join(path, f'tsne/tsne_{cs}.csv'))
                
#tsne(simscore, np.arange(32, 38, 1))


#%% Graph Mining Application



#%% Application

app.layout = html.Div([
    html.Div([
        #--header section
        html.Div([
                html.H3('GRAPH MINING FOR BIG DATA\nRecommender system for social mining'),
                ], style={'text-align': 'left','width': '49%', 'display': 'inline-block','vertical-align': 'middle'}),
        html.Div([
                html.H4('Orhan M., Ekpo E., Jayani G.V, Ezukwoke K.I'),
                html.Label('Recommender system using Graph Mining and Unsupervised Machine Learning\
                           for paper recommendation. This is done in partial fulfilment of the course\
                           BIG DATA')
                ], style= {'width': '49%', 'display': 'inline-block','vertical-align': 'middle', 'font-size': '12px'})
                ], style={'background-color': 'white', 'box-shadow': 'black 0px 1px 0px 0px'}),
    #--scaling section
    html.Div([
            html.Div([
                    #---Graph mode returns either network or Clustering result
                    html.Label('Graph mode'),                    
                    dcc.RadioItems(
                            #---
                            id='g_mode',
                            options = [{'label': i, 'value': i} for i in ['Network', 'Cluster']],
                            value = "Cluster",
                            labelStyle={'display': 'inline-block'}
                            ), 
                    ], style = {'display': 'inline-block', 'width': '14%'}),
                    
            html.Div([
                    #---Cluster size
                    html.Label('Cluster size: Default is optimum'),                    
                    dcc.RadioItems(
                            #---
                            id='cluster',
                            options = [{'label': i, 'value': i} for i in [str(x) for x in np.arange(32, 38, 1)]],
                            value = "33",
                            labelStyle={'display': 'inline-block'}
                            ), 
                    ], style = {'display': 'inline-block', 'width': '14%'}),
            html.Div([
                    #---Cluster size
                    html.Label('Kernel'),                    
                    dcc.RadioItems(
                            #---
                            id='kernel',
                            options = [{'label': i, 'value': i} for i in ['linear', 'laplace', 'cosine', 'rbf']],
                            value = "linear",
                            labelStyle={'display': 'inline-block'}
                            ), 
                    ], style = {'display': 'inline-block', 'width': '14%'}),
            html.Div([
                    #---Number of Topics
                    html.Label('Number of Topics:'),                    
                    dcc.RadioItems(
                            #---
                            id='topic-number',
                            options = [{'label': i, 'value': i} for i in [str(x) for x in np.arange(5, 11, 1)]],
                            value = "5",
                            labelStyle={'display': 'inline-block'}
                            ), 
                    ], style = {'display': 'inline-block', 'width': '14%'}),
            html.Div([#---Cluster size
                    html.Label('y-scale:'),                    
                    dcc.RadioItems(
                            #---
                            id='y-items',
                            options = [{'label': i, 'value': i} for i in ['Linear', 'Log']],
                            value = "Linear",
                            labelStyle={'display': 'inline-block'}
                            ), 
                    ], style = {'display': 'inline-block', 'width': '14%'}),
            #--- Token length
            html.Div([
                    html.Label('Token length:'),                    
                    dcc.RadioItems(
                            #---
                            id='tokens',
                            options = [{'label': i, 'value': i} for i in [str(x) for x in np.arange(5, 11, 1)]],
                            value = "5",
                            labelStyle={'display': 'inline-block'}
                            ), 
                    ], style = {'display': 'inline-block', 'width': '14%'}),
            #--- Sort Tags
            html.Div([
                    html.Label('Sort Tags'),                    
                    dcc.RadioItems(
                            #---
                            id='Sort-Tags',
                            options = [{'label': i, 'value': i} for i in ['A-z', 'Most Tags']],
                            value = "Most Tags",
                            labelStyle={'display': 'inline-block'}
                            ), 
                    ], style = {'display': 'inline-block', 'width': '14%'})
            ], style={'background-color': 'rgb(204, 230, 244)', 'padding': '1rem 0px', 'margin-top': '2px','box-shadow': 'black 0px 0px 1px 0px','vertical-align': 'middle'}),
    #-- Graphs
    html.Div([
            html.Div([
                    dcc.Dropdown(
                        #---dropdown for categories
                        id='dd',
                        options =[{'label': i, 'value': i} for i in [str(x) for x in np.arange(data.year.min(), data.year.max(), 1)]],
                        value = [],
                        placeholder = 'Select a year',
                        multi = True,
                        ),
                   dcc.Graph(id = 'scatter_plot',
#                              style={'width': '690px', 'height': '395px'},
                      config = config,
                      hoverData={'points': [{'customdata': ["pdf_0", "2004", "fr", "Hicham Hajji, Nourdine Badji, Jean-Pierre Asté", "Vers un entrepôt de données pour la gestion des risques naturels"]}]}
                      ),
#                    ], style = {'display': 'inline-block', 'width': '65%','background-color': 'white'}),
        
            ],style = {'display': 'inline-block', 'background-color': 'white', 'width': '65%', 'padding': '0 20','vertical-align': 'middle'}),
    #--horizontal dynamic barplot
    html.Div([
            dcc.Graph(id = 'bar_plot',
                      config = config,
                      )
            ],style = {'display': 'inline-block', 'background-color': 'white', 'width': '35%','vertical-align': 'middle'}),
    html.Div([
            dcc.RangeSlider( #----year slider
                    id = 'year-slider',
                    min = data.year.min(),
                    max = data.year.max(),
                    updatemode = 'drag',
                    value = [data.year.min(), data.year.max()],
                    marks={str(year): str(year) for year in range(data.year.min(), data.year.max(), 2)}
                ),
            ], style = {'background-color': 'white', 'display': 'inline-block', 'width': '65%', 'padding': '0px 20px 20px 20px','vertical-align': 'middle'}),
            ], style = {'background-color': 'white','margin': 'auto', 'width': '100%', 'display': 'inline-block'}),
    
    #-- Footer section
    html.Div([
        #--footer section
        html.Div([
                html.Div([
                        html.H4(id = 'topic')], style = {'color':' rgb(35, 87, 137)'}),
                html.Div([
                        html.Label(id = 'date')], style = {'color':' black', 'font-weight': 'bold', 'display': 'inline-block'}),
                html.Div([
                        html.Label(id = 'author')], style = {'color':' black', 'font-weight': 'bold', 'display': 'inline-block', 'padding': '0px 0px 10px 35px'}),
                html.Div([
                        html.Label(id = 'cat')], style = {'color':' black', 'font-weight': 'bold', 'display': 'inline-block', 'padding': '0px 0px 10px 35px'}),
                html.Label(id = 'label'),
                ], style= {'width': '74%', 'display': 'inline-block','vertical-align': 'middle', 'font-size': '15px'}),
        html.Div([
                html.H2('Topics'),
#                html.Label(id = 'topic-tags'),
                html.Label(id = 'topic-tags', style={'text-align': 'center', 'margin': 'auto', 'vertical-align': 'middle'})
                ], style={'text-align': 'center','width': '25%', 'display': 'inline-block','vertical-align': 'middle'}),
                ], style={'background-color': 'rgb(204, 230, 244)', 'margin': 'auto', 'width': '100%', 'max-width': '1200px', 'box-sizing': 'border-box', 'height': '30vh'}),
    #---
    #main div ends here
    ],style = {'background-color': 'rgb(204, 230, 244)','margin': 'auto', 'width': '100%', 'display': 'block'})

#--Main chart
@app.callback(
        Output('scatter_plot', 'figure'),
        [Input('year-slider', 'value'),
         Input('g_mode', 'value'),
         Input('kernel', 'value'),
         Input('dd', 'value'),
         Input('y-items', 'value'),
         Input('cluster', 'value'),
         ])
def update_figure(make_selection, g_m, knl, drop, yaxis, clust):
#    data_places = data[(data.year_edited >= make_selection[0]) & (data.year_edited <= make_selection[1])]
    ts = pd.read_csv(os.path.join(path, f'tsne/tsne_{int(clust)}.csv')).iloc[:, 1:]
    ts = ts.sort_values(by=['year'])
    data_places = ts[(ts.year >= make_selection[0]) & (ts.year <= make_selection[1])] 
    if g_m == 'Cluster':
        if drop != []:
            traces = []
            for val in drop:
                traces.append(go.Scattergl(
                        x = np.array(data_places.loc[data_places.year == int(val), '0']),
                        y = np.array(data_places.loc[data_places.year == int(val), '1']),
                        text = [(x, y, z, w, p) for (x, y, z, w, p) in zip(\
                                 data_places.loc[data_places['year'] == int(val), 'pdf_names'].apply(lambda x: x.split('.')[0]),\
                                 data_places.loc[data_places['year'] == int(val), 'year'],\
                                 data_places.loc[data_places['year'] == int(val), 'language'],\
                                 data_places.loc[data_places['year'] == int(val), 'authors'],\
                                data_places.loc[data_places['year'] == int(val), 'title'])],
                        customdata = [(x, y, z, w, p) for (x, y, z, w, p) in zip(\
                                 data_places.loc[data_places['year'] == int(val), 'pdf_names'].apply(lambda x: x.split('.')[0]),\
                                 data_places.loc[data_places['year'] == int(val), 'year'],\
                                 data_places.loc[data_places['year'] == int(val), 'language'],\
                                 data_places.loc[data_places['year'] == int(val), 'authors'],\
                                data_places.loc[data_places['year'] == int(val), 'title'])],
                        mode = 'markers',
                        opacity = 0.6,
                        marker = {'size': 15, 
                                  'line': {'width': 0.5, 'color': 'white'}},
                        name = val,
                        ))
            
            return {'data': traces,
                    'layout': go.Layout(
                            xaxis={'title': 'tsne-2'},
                            yaxis={'type': 'linear' if yaxis == 'Linear' else 'log','title': 'tsne-1'},
                            margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                            legend={'x': 1, 'y': 1},
                            hovermode='closest')
                            }
        else:
            pca = kPCA(k = int(clust), kernel = knl).fit(np.array(simscore))
            pca = pca.components_.T
            km = KMeans(n_clusters = int(clust))
            km = kkMeans(k = int(clust), kernel = knl, gamma = 1).fit_predict(pca)
            cluster_labels = km.clusters
            ts = pd.read_csv(os.path.join(path, f'tsne/tsne_{int(clust)}.csv')).iloc[:, 1:]
            ts = ts[(ts.year >= make_selection[0]) & (ts.year <= make_selection[1])] 
            traces = go.Scattergl(
                    x = np.array(ts)[:, 0],
                    y = np.array(ts)[:, 1],
                    text = [(x, y, z, w, p) for (x, y, z, w, p) in zip(\
                                 ts['pdf_names'].apply(lambda x: x.split('.')[0]),\
                                 ts['year'],\
                                 ts['language'],\
                                 ts['authors'],\
                                 ts['title'])],
                    customdata = [(x, y, z, w, p) for (x, y, z, w, p) in zip(\
                                  ts['pdf_names'].apply(lambda x: x.split('.')[0]),\
                                  ts['year'],\
                                 ts['language'],\
                                 ts['authors'],\
                                ts['title'])],
                    mode = 'markers',
                    opacity = 0.7,
                    marker = {'size': 15, 
    #                          'opacity': 0.9,
                              'color': cluster_labels,
                              'colorscale':'Viridis',
                              'line': {'width': .5, 'color': 'white'}},
                    )
            
            return {'data': [traces],
                    'layout': go.Layout(
                            height = 600,
                            xaxis={'title': 'tsne-2'},
                            yaxis={'type': 'linear' if yaxis == 'Linear' else 'log','title': 'tsne-1'},
                            margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                            legend={'x': 1, 'y': 1},
                            hovermode='closest')
                            }
    else:
        ss = np.array(simscore)
        m, n = ss.shape
        G = nx.Graph()
        for n in range(m):
            G.add_node(n)
        for i in range(m):
            for j in range(n):
                if ss[i,j] != 0 and i != j:
                    G.add_edge(i,j)
        E = [edg for edg in G.edges]
        pos = nx.fruchterman_reingold_layout(G)
        Xv = [pos[k][0] for k in range(n)]
        Yv = [pos[k][1] for k in range(n)]
        Xed = []
        Yed = []
        for edge in E:
            Xed += [pos[edge[0]][0], pos[edge[1]][0], None]
            Yed += [pos[edge[0]][1], pos[edge[1]][1], None]
            
        etrace = go.Scattergl(x = Xed,
                       y = Yed,
                       mode = 'lines',
                       line = dict(color='rgb(210,210,210)', width = .5),
                       hoverinfo = 'none'
                       )
        
        vtrace = go.Scattergl(x = Xv,
                       y = Yv,
                       mode = 'markers',
                       name = 'net',
                       marker = dict(symbol='circle-dot',
                                     size = 5,
                                     color='#6959CD',
                                     line=dict(color='rgb(50,50,50)', width=0.5)
                                     ),
#                       text = labels,
                       hoverinfo='text'
                       )

        return {'data': [etrace, vtrace],
                    'layout': go.Layout(
                            height = 600,
                            xaxis={'title': 'year'},
                            yaxis={'type': 'linear' if yaxis == 'Linear' else 'log','title': 'Similarity score'},
                            margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                            legend={'x': 1, 'y': 1},
                            hovermode='closest')
                            }
        
@app.callback(
        Output('topic', 'children'),
        [Input('scatter_plot', 'hoverData')]
        )
def update_bookheader(hoverData):
    book_number = hoverData['points'][0]['customdata'][0]
    dirlis = list(data['pdf_names'].apply(lambda x: x.split('.')[0]))
    for ii in dirlis:
        if ii == book_number:
            subject = data[data.pdf_names == f'{ii}.txt']['title'].values[0]
    return subject
#
@app.callback(
        Output('date', 'children'),
        [Input('scatter_plot', 'hoverData')]
        )
def update_bookyear(hoverData):
    book_number = hoverData['points'][0]['customdata'][0]
    dirlis = list(data['pdf_names'].apply(lambda x: x.split('.')[0]))
    for ii in dirlis:
        if ii == book_number:
            date = data[data.pdf_names == f'{ii}.txt']['year'].values[0]
    return f'YEAR: {date}'
#
@app.callback(
        Output('author', 'children'),
        [Input('scatter_plot', 'hoverData')]
        )
def update_bookauthor(hoverData):
    
    book_number = hoverData['points'][0]['customdata'][0]
    dirlis = list(data['pdf_names'].apply(lambda x: x.split('.')[0]))
    for ii in dirlis:
        if ii == book_number:
            authors = data[data.pdf_names == f'{ii}.txt']['authors'].values[0]
    return f'Authors: {authors}'
#
#@app.callback(
#        Output('cat', 'children'),
#        [Input('scatter_plot', 'hoverData')]
#        )
#def update_category(hoverData):
#    book_number = hoverData['points'][0]['customdata'][0]
#    dirlis = list(data['pdf_names'].apply(lambda x: x.split('.')[0]))
#    for ii in dirlis:
#        if ii == book_number:
#            abstract = data[data.pdf_names == f'{ii}.txt']['abstract'].values[0]
#    return f'Abstract: {abstract}'[:100]
#    
@app.callback(
        Output('label', 'children'),
        [Input('scatter_plot', 'hoverData')]
        )
def update_abstract(hoverData):
    #--
    book_number = hoverData['points'][0]['customdata'][0]
    dirlis = list(data['pdf_names'].apply(lambda x: x.split('.')[0]))
    for ii in dirlis:
        if ii == book_number:
            abstract = data[data.pdf_names == f'{ii}.txt']['abstract'].values[0]
    return f'Abstract: {abstract}'


@app.callback(
        Output('topic-tags', 'children'),
        [Input('scatter_plot', 'hoverData'),
         Input('tokens', 'value'),
         Input('topic-number', 'value')]
        )
def topic_tags(hoverData, token, topic):
    #--
#    import random
    book_number = hoverData['points'][0]['customdata'][0]
    dirlis = os.listdir(join(path, f'counter/{token}'))
    dirlis = [(lambda x: x.split('.')[0])(x) for x in dirlis]
    for ii in dirlis:
        if ii == book_number:
            #-open csv file and extract content
            trac_x = random.sample(list(pd.read_csv(os.path.join(path, f'counter/{token}/{ii}.txt'))['word']), int(topic))
            result = ', '.join(trac_x)
    return result


@app.callback(
        Output('bar_plot', 'figure'),
        [Input('scatter_plot', 'hoverData'),
         Input('Sort-Tags', 'value'),
         Input('tokens', 'value')]
        )
def bar_plot(hoverData, sort, token):
    #--locate book and extract data from drive
    book_number = hoverData['points'][0]['customdata'][0]
    #--set extract.directory
    dirlis = os.listdir(join(path, f'counter/{token}'))
    dirlis = [(lambda x: x.split('.')[0])(x) for x in dirlis]
    for ii in dirlis:
        if ii == book_number:
            #-open csv file and extract content
            trac_x = list(pd.read_csv(os.path.join(path, f'counter/{token}/{ii}.txt'))['word'])[:15]
            trac_y = list(pd.read_csv(os.path.join(path, f'counter/{token}/{ii}.txt'))['freq'])[:15]
            if sort == 'A-z':
                trace = go.Bar(
                        x = trac_y,
                        y = trac_x,
                        marker = dict(
                            color='rgba(50, 171, 96, 0.6)',
                            line=dict(
                                color='rgba(50, 171, 96, 1.0)',
                                width=2),
                        ),
                        orientation = 'h',
                        )
                return {'data': [trace],
                    'layout': go.Layout(
                            autosize  =False,
                            width = 500,
                            height = 600,
                            margin=go.layout.Margin(
                                    l=100,
                                    r=50,
                                    b=100,
                                    t=0,
                                    pad=4
                                    ),
                            yaxis = {'categoryorder': 'array',
                                     'categoryarray': [x[0] for x in sorted(zip(trac_x, trac_y))],
                                     'autorange': 'reversed'}
                            )
                            
                    }
            else:
                trace = go.Bar(
                        x = trac_y,
                        y = trac_x,
                        marker = dict(
                            color='rgba(50, 171, 96, 0.6)',
                            line=dict(
                                color='rgba(50, 171, 96, 1.0)',
                                width=2),
                        ),
                        orientation = 'h',
                        )
                return {'data': [trace],
                        'layout': go.Layout(
                                autosize  =False,
                                width = 500,
                                height = 600,
                                margin=go.layout.Margin(
                                        l=100,
                                        r=50,
                                        b=100,
                                        t=0,
                                        pad=4
                                        ),
                                yaxis = {'autorange': 'reversed'}
                                )
                                
                        }
#
#
if __name__ == '__main__':
  app.run_server(debug = True)
  
    

    
    
    
    
    
    
    
    
    
    
    
    
    
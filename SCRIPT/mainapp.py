#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 14:10:14 2019

@author: kenneth
"""
import pandas as pd
import dash
import os 
import nltk
import random
import pickle
import numpy as np
from collections import Counter
from nltk.stem.snowball import FrenchStemmer, PorterStemmer
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from os.path import join
from flask import Flask
from collections import Counter
from nltk.tokenize import RegexpTokenizer
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
            tokenizer = RegexpTokenizer(r'\w+')
            up_text = tokenizer.tokenize(file_dt)
            file.close()
            #--processed tokens
            for tk in toks:
                new_token = []
                word_freq = Counter()
                final = ''
                new_words = [ii for ii in up_text if len(ii) >= int(tk)]
                for each_word in new_words:
                    each_word = each_word.lower()
                    if each_word not in stopwords:
                        new_token.append(each_word)
                final = ' '.join(new_words)
                sentence.append(str(final.lower()))
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
                        most_frequent_words.to_csv(os.path.join(path, f'counter/{tk}/{ii}'))
                    else:
                        pass
                else:
                    if not os.path.exists(os.path.join(path, f'counter/{tk}/{ii}')):
                        word = []
                        freq = []
                        for wd in new_words:
                            word_freq.update([wd])
                        for x, y in word_freq.most_common(30):
                            word.append(x)
                            freq.append(y)
                        most_frequent_words = pd.DataFrame({'word': word, 'freq': freq})
                        most_frequent_words.to_csv(os.path.join(path, f'counter/{tk}/{ii}'))
                    else:
                        pass
            if not os.path.exists(os.path.join(path, f'pbooks/{ii}')):
                with open(os.path.join(path, f'pbooks/{ii}'), 'w') as wr:
                    wr.writelines(final)
            else:
                with open(os.path.join(path, f'pbooks/{ii}'), 'w+') as wr:
                    wr.writelines(final)
    try:
        
        complete = booknames.copy(deep = True)
        complete['sentence'] = sentence
        #--save files to directory
        if not os.path.exists(os.path.join(path, 'processed')):
            complete.to_csv(os.path.join(path, 'processed/pcomplete.csv'))
        else:
            complete.to_csv(os.path.join(path, 'processed/pcomplete.csv'))
    except (ValueError, RuntimeError, TypeError, NameError, OSError):
        pass
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
from sklearn.decomposition import PCA

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
                    #---Cluster size
                    html.Label('Cluster size: Default is optimum'),                    
                    dcc.RadioItems(
                            #---
                            id='cluster',
                            options = [{'label': i, 'value': i} for i in [str(x) for x in np.arange(2, 7, 1)]],
                            value = "3",
                            labelStyle={'display': 'inline-block'}
                            ), 
                    ], style = {'display': 'inline-block', 'width': '20%'}),
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
                    ], style = {'display': 'inline-block', 'width': '20%'}),
            html.Div([#---Cluster size
                    html.Label('y-scale:'),                    
                    dcc.RadioItems(
                            #---
                            id='y-items',
                            options = [{'label': i, 'value': i} for i in ['Linear', 'Log']],
                            value = "Linear",
                            labelStyle={'display': 'inline-block'}
                            ), 
                    ], style = {'display': 'inline-block', 'width': '20%'}),
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
                    ], style = {'display': 'inline-block', 'width': '20%'}),
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
                    ], style = {'display': 'inline-block', 'width': '20%'})
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
         Input('dd', 'value'),
         Input('y-items', 'value'),
         Input('cluster', 'value'),
         ])
def update_figure(make_selection, drop, yaxis, clust):
#    data_places = data[(data.year_edited >= make_selection[0]) & (data.year_edited <= make_selection[1])]
    data_places = sort_dataset[(sort_dataset.year >= make_selection[0]) & (sort_dataset.year <= make_selection[1])] 
    if drop != []:
        traces = []
        for val in drop:
            traces.append(go.Scattergl(
                    x = data_places.loc[data_places['year'] == int(val), 'year'],
                    y = simscore.iloc[:, 0].values,
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
#                              'color': 'rgba(50, 171, 96, 0.6)',
                              'line': {'width': 0.5, 'color': 'white'}},
                    name = val,
                    ))
        
        return {'data': traces,
                'layout': go.Layout(
#                        height = 600,
                        xaxis={'title': 'year'},
                        yaxis={'type': 'linear' if yaxis == 'Linear' else 'log','title': 'Similarity score'},
                        margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                        legend={'x': 1, 'y': 1},
                        hovermode='closest')
                        }
    else:
        pca = PCA(n_components = int(clust)).fit(simscore)
        km = KMeans(n_clusters = int(clust), init = pca.components_, n_init = 1)
        km.fit_transform(simscore)
        cluster_labels = km.labels_
        traces = go.Scattergl(
                x = data_places['year'],
                y = simscore.iloc[:, 0].values,
                text = [(x, y, z, w, p) for (x, y, z, w, p) in zip(\
                             data_places['pdf_names'].apply(lambda x: x.split('.')[0]),\
                             data_places['year'],\
                             data_places['language'],\
                             data_places['authors'],\
                             data_places['title'])],
                customdata = [(x, y, z, w, p) for (x, y, z, w, p) in zip(\
                              data_places['pdf_names'].apply(lambda x: x.split('.')[0]),\
                              data_places['year'],\
                             data_places['language'],\
                             data_places['authors'],\
                            data_places['title'])],
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
  
    

    
    
    
    
    
    
    
    
    
    
    
    
    
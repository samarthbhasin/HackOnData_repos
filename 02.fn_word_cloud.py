#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 23:29:13 2017

@author: samarth bhasin
"""
import nltk
import pandas as pd

df = train[train.brand=='Ikea']

def fn_word_cloud(df,train_data_features=train_data_features,feature_df=feature_df,vocab=vocab):
    cloud = pd.DataFrame()
    for locI in df.index:
        print '--- Review %i --- rating: %i' %(locI,df.loc[locI].overall) 
        #print df.loc[locI].reviewText
        #print '-----------------'
        
        count0 = []
        word0 = []
        for word_,count_ in zip(vocab,train_data_features[locI]):
            count0.append(count_)
            word0.append(word_)
            
        df0 = pd.DataFrame({'word':word0,'count':count0})
        
        df0['tag'] = df0['word'].apply(lambda x: nltk.pos_tag([x]))
        df0['tag'] = df0['tag'].apply(lambda x: x[0][1])
        
        
        
        df0w = df0.set_index('word').join(feature_df.set_index('word'))
        df0w = df0w.reset_index()
        
        
        # --- Conditional for JJ, NN -------------------
        #print df0w[(df0w['count']>0) & ((df0w.tag=='JJ') | (df0w.tag=='NN'))].sort_values('weight',ascending=False).head(20)
        
        # All
        #df0w = df0w[(df0w['count']>0) & ((df0w['tag']!='MD') & (df0w['tag']!='CD'))].sort_values('weight',ascending=False)
        df0w = df0w[(df0w['count']>0) & (df0w['tag']=='JJ')].sort_values('weight',ascending=False)
        df0w = df0w.reset_index()
        #print df0w
        cloud = pd.concat([cloud,df0w.loc[0:10]])
        
    return cloud

# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 15:12:27 2018

Data:Movies data and ratings
Packages: Pandas, numpy, matplotlib
Task : Determine Top 10 and Bottom 10 actors from movie ratings data. 

@author: Swetha
"""

import pickle
import hashlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib notebook

moviedata=pd.read_csv("movies.csv")
movieactors=pd.read_csv("movie_actors.csv")

# Exploring moviedata
moviedata=moviedata[["title","id","rtAllCriticsRating","rtAllCriticsNumReviews"]]
moviedata.dtypes
moviedata[['title']].head()

moviedata[['rtAllCriticsRating']].head()

moviedata['rtAllCriticsRating'] = pd.to_numeric(moviedata['rtAllCriticsRating'].str.replace(' ',''), errors='coerce')
moviedata['rtAllCriticsNumReviews'] = pd.to_numeric(moviedata['rtAllCriticsNumReviews'].str.replace(' ',''), errors='coerce')

moviedata = moviedata.dropna(subset=['rtAllCriticsRating','rtAllCriticsNumReviews'])
print moviedata[['rtAllCriticsRating']].isnull().sum()
print moviedata[['rtAllCriticsNumReviews']].isnull().sum()

movieNumA =  moviedata[moviedata.rtAllCriticsNumReviews > 0]

movieNumA.nlargest(10, 'rtAllCriticsRating')
movieNumA.nsmallest(10, 'rtAllCriticsRating')
movieNumA.nsmallest(10, 'rtAllCriticsNumReviews')

# 
movieactors.nlargest(4,'movieID')
movieactors.nsmallest(4,'movieID')
movieactors.head()
movieactors.shape

movieNumA = movieNumA.rename(columns={'id':'movieID'})
movieNumA.head()

moviemerged = pd.merge(movieNumA, movieactors, on='movieID', how='inner')
moviemerged.head()
print moviemerged.shape
print moviemerged[['rtAllCriticsRating']].isnull().sum()


# BEst Actors
actorsaverage = moviemerged[['actorID','rtAllCriticsRating']].groupby('actorID').mean()
actorsaverage.nlargest(10,'rtAllCriticsRating')

#Poplar Actors
BIGACTORS=moviemerged.actorID.value_counts()
type(BIGACTORS)

BIGACTORS = BIGACTORS.to_frame().reset_index()
print BIGACTORS.shape
BIGACTORS.head()

BIGACTORS = BIGACTORS.rename(columns= {'index': 'actorID','actorID':'NumberMovies'})
BIGACTORS = BIGACTORS[BIGACTORS.NumberMovies >= 37]
BIGACTORS.head()

movieBIGACTORS = pd.merge(moviemerged, BIGACTORS[['actorID']], on='actorID', how='inner')
movieBIGACTORS.shape

# Movies wise

movieNumActors =movieBIGACTORS[['actorID','movieID']].groupby('movieID').agg(['count'])
print movieNumActors.dtypes
movieNumActors = movieNumActors.reset_index()
print movieNumActors.shape

movieNumActors.columns = ['movieID','count_movies']
movieNumActors = movieNumActors[movieNumActors.count_movies >= 2] 
print movieNumActors.shape

movieTwoBIGACTORS = pd.merge(movieBIGACTORS,movieNumActors[['movieID']], on='movieID',how='inner')
print movieTwoBIGACTORS.shape

movieTwoBIGACTORS.head()

movieTwoBIGACTORS[['movieID','actorID']].groupby('actorID').agg(['count']).min()

# Top 10 and Bottom 10 actors

Top10Actors = movieTwoBIGACTORS[['actorID','rtAllCriticsRating']].groupby('actorID').agg(['mean']).reset_index()
Top10Actors.columns = ['actorID','MeanMovieRatings']
Top10Actors = Top10Actors.nlargest(10,'MeanMovieRatings')
Top10Actors

Bottom10Actors = movieTwoBIGACTORS[['actorID','rtAllCriticsRating']].groupby('actorID').agg(['mean']).reset_index()
Bottom10Actors.columns = ['actorID','MeanMovieRatings']
Bottom10Actors = Bottom10Actors.nsmallest(10,'MeanMovieRatings')
Bottom10Actors

# Rating by Least Squares

# Indexed matrix
df_ref = movieTwoBIGACTORS[['movieID','actorID']]
df1 = movieTwoBIGACTORS[['movieID']].drop_duplicates(['movieID'])
df2 = movieTwoBIGACTORS[['actorID']].drop_duplicates(['actorID'])

df_ref['combo'] = df_ref['movieID'].astype(str) + df_ref['actorID'].astype(str)

df = pd.DataFrame(np.zeros((1032, 101)),columns=df2.actorID)
df = df.set_index(df1.movieID)
#dfprint df.shape
df.iloc[1:5,2:5]

for j in range(101):
        for i in range(1032):
            df.iloc[i,j] = df.index[i].astype(str) + df.columns[j]
        df.iloc[:,j] = df.iloc[:,j].isin(df_ref.combo)
df = df.astype(int)

score = movieTwoBIGACTORS[['movieID','actorID']].groupby(['movieID']).count()
print score.shape
score = score.reindex(df.index)
score.columns =['NumActor']
score['InvActorScore'] = 1/score.NumActor
score = score['InvActorScore']
score.head()
#I = np.identity(1032)
S = pd.DataFrame(np.diag(score),index=score.index,columns=score.index)
#MAtrix Multiplication of df x S gives the required matrix
df_final = S.dot(df)
print df_final.shape
df_final.head()

# Indexed movie ratings vector
movieratings = movieTwoBIGACTORS[['movieID','rtAllCriticsRating']].drop_duplicates(['movieID'])
movieratings = movieratings.set_index(['movieID'])
movieratings = movieratings.reindex(df_final.index)

# Solve the least squares for ratings of actors.
A = df_final
b = movieratings
x=np.linalg.inv(A.T.dot(A)).dot(A.T.dot(b))
#MSE
AR = df2
AR['ActorRatings'] = x
AR = AR.set_index('actorID')
Ax = A.dot(AR)
Ax.columns = ['MovieRatings']
b.columns = ['MovieRatings']
print Ax.shape
print b.shape
SE =(Ax-b)**2
MSE = SE.sum()/len(b)
print MSE

# 10 best and worst actors accroding to the above ratings
AR.nlargest(10,'ActorRatings')
AR.nsmallest(10,'ActorRatings')

# Train & Test Sets
def is_train(movie_id):
    hash_object = hashlib.md5(netid + str(movie_id))
    hex_hash = hash_object.hexdigest()
    return int(hex_hash[0],16) < 14 
is_train('1')

movies=movieTwoBIGACTORS[['movieID','rtAllCriticsRating']].drop_duplicates(['movieID'])
movies['t'] =0

for i in range(1032):
    movies.iloc[i,2]=is_train(movies.iloc[i,0])
    
movies.head()

movies_train = movies[movies.t==True]
print movies_train.shape
movies_test =  movies[movies.t==False]
perc_train = (100*len(movies_train.index))/len(movies.index)
print("%.2f" % perc_train)

# compute least square on train data

A_train = A.reindex(movies_train.movieID)
b_train = movieratings.reindex(movies_train.movieID)
print A_train.shape
x_train=np.linalg.inv(A_train.T.dot(A_train)).dot(A_train.T.dot(b_train))
print x_train.shape
AR1 = df2
AR1['ActorRatings'] = x_train

AR1.nlargest(10,'ActorRatings')
AR1.nsmallest(10,'ActorRatings')

# Evaluate least square solution on test

A_test = A.reindex(movies_test.movieID)
b_test = movieratings.reindex(movies_test.movieID)
AR1 = AR1.set_index('actorID')
print A_test.shape
print b_test.shape
print AR1.shape

#y1 = A_test.dot(AR1) - b_test
Ax = A_test.dot(AR1)
Ax.columns = ['MovieRatings']
b_test.columns = ['MovieRatings']
SE =(Ax-b_test)**2
MSE_test = SE.sum()/len(movies_test)
print 'MSE for the entire datasets is %f' %MSE
print 'MSE for the test datasets is %f' %MSE_test

# Compare results of using the whole data for top/bottom 10  VS using train and test data

Top10Actors_match = AR.nlargest(10,'ActorRatings').index.isin(AR1.nlargest(10,'ActorRatings').index)
print Top10Actors_match
print 'No.of matches of Top 10 actors between Whole data and Train data is %d' %Top10Actors_match.sum()

Bottom10Actors_match = AR.nsmallest(10,'ActorRatings').index.isin(AR1.nsmallest(10,'ActorRatings').index)
print Bottom10Actors_match
print 'No.of matches of Bottom 10 actors between Whole data and Train data is %d' %Bottom10Actors_match.sum()

# Conclusion: The train dataset is clipped from the whole dataset. Hence there is a mismatch (30% and 20%) between the top10 actors and bottom10 actors in both datasets. However if larger dataset (with 4500 movieBIGACTOR observations) was considered without pre-processing, this mismatch might be smaller.

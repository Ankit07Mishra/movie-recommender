import numpy as np
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer 
from nltk.stem.porter import PorterStemmer 
from sklearn.metrics.pairwise import cosine_similarity
import pickle
def convert(obj):
    l=[]
    for i in ast.literal_eval(obj):
        l.append(i['name'])
    return l
def convert_cast(obj):
    l=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter<3:
            l.append(i['name'])
            counter=counter+1
        else :
            break 
    return l

def convert_director(obj):
    l=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            l.append(i['name'])
            break 
    return l

def remove_space(obj):
    l=[]
    for i in obj:
        i=i.replace(" ","")
        l.append(i)
    return l


ps=PorterStemmer()

def stem(txt):
    l=[]
    for i in txt.split():
        l.append(ps.stem(i))
    return " ".join(l)

    

credits=pd.read_csv("resources/tmdb_5000_credits.csv")
movies=pd.read_csv("resources/tmdb_5000_movies.csv")
movies=movies.merge(credits,on='title')


movies=movies[['id','title','overview','genres','keywords','production_companies','cast','crew']]
movies.dropna(inplace=True)

movies['overview'] = movies['overview'].apply(lambda x: x.split())

movies['genres']=movies['genres'].apply(convert)
movies['keywords']=movies['keywords'].apply(convert)
movies['cast']=movies['cast'].apply(convert_cast)
movies['crew']=movies['crew'].apply(convert_director)
movies['genres']=movies['genres'].apply(remove_space)
movies['keywords']=movies['keywords'].apply(remove_space)
movies['cast']=movies['cast'].apply(remove_space)
movies['crew']=movies['crew'].apply(remove_space)


movies['tags']=movies['overview']+movies['genres']+movies['cast']+movies['crew']

movies=movies[ ['id','title','tags']]
movies['tags']=movies['tags'].apply(lambda x:" ".join(x))
movies['tags']=movies['tags'].apply(lambda x:x.lower())

#stemming 
movies['tags']=movies['tags'].apply(stem)

#vectorization
counter=CountVectorizer(max_features=5000,stop_words='english')
vectors=counter.fit_transform(movies['tags'])

similarity=cosine_similarity(vectors)
print(similarity)



def recommend(movie):
    movie_index=movies[movies['title']==movie].index[0]
    distances=similarity[movie_index]
    movies_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])
    for i in movies_list[1:6]:
        print(movies.iloc[i[0]].title)


pickle.dump(movies.to_dict(),open('movies.pkl','wb'))
pickle.dump(similarity,open("similarity.pkl","wb"))
    












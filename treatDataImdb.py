import sparse
import numpy as np
import json
import pandas as pd
from ast import literal_eval


def jsonToDic(entry, returnstr=False):
    entry = str(entry)
    entry = entry.replace("]", "").replace("[", "")
    entry = entry.replace("{", "").replace("}", "")
    entry = entry.replace("\"", "\'")
    entry = entry.replace("\\\'", "\'").replace("\'\'", "\'")
    entry = entry.replace("\'}", "}").replace("O\'", "O").replace("D\'", "D").replace("a\'A", "aA").replace("L\'", "L").replace("n\'a", "na")
    entry = entry.replace("None\'", "None")
    entry = "{" + entry + "}"
    if returnstr: return entry
    d = literal_eval(entry)
    #d = eval(entry)

    return d

def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan
def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names

    #Return empty list in case of missing/malformed data
    return []



IDtoCast, occCast = {}, {}
#df = pd.read_csv('Data/Imdb/credits.csv')
df1=pd.read_csv('Data/Imdb/Big/credits.csv')
df2=pd.read_csv('Data/Imdb/Big/movies_metadata.csv')
df2 = df2[df2["id"] != "1997-08-20"]
df2 = df2[df2["id"] != "2012-09-29"]
df2 = df2[df2["id"] != "2014-01-01"]
df2['id'] = df2['id'].astype('int64')
metadata = df1.merge(df2, on='id')
print(metadata.keys())

features = ['cast', 'crew']
for feature in features:
    metadata[feature] = metadata[feature].apply(literal_eval)
metadata['director'] = metadata['crew'].apply(get_director)
metadata['cast'] = metadata['cast'].apply(get_list)
metadata['genres'] = metadata['genres'].apply(get_list)
print(metadata)
metadata = metadata[metadata["vote_count"]>1000]


ratings=pd.read_csv('Data/Imdb/Big/ratings.csv')
cnt = ratings["userId"].value_counts()
usrs = cnt.index[cnt>1000]
ratings = ratings.loc[ratings["userId"].isin(usrs)]
print(ratings)


setArtists, setMovies, setVoters, setDirectors, setGenres, nbVotes = set(), set(), set(), set(), set(), 0
with open("Data/Imdb/outcome.txt", "a", encoding="utf-8") as o:
    o.truncate(0)
    with open("Data/Imdb/feature_0.txt", "a", encoding="utf-8") as f1:
        f1.truncate(0)
        with open("Data/Imdb/feature_1.txt", "a", encoding="utf-8") as f2:
            f2.truncate(0)
            with open("Data/Imdb/feature_2.txt", "a", encoding="utf-8") as f3:
                f3.truncate(0)
                with open("Data/Imdb/feature_3.txt", "a", encoding="utf-8") as f4:
                    f4.truncate(0)
                    with open("Data/Imdb/feature_4.txt", "a", encoding="utf-8") as f5:
                        f5.truncate(0)
                        size = len(ratings)
                        lim = size
                        #lim = 100000
                        for j, line in enumerate(ratings.values):
                            if j%1000==0:
                                print(j*100./lim, "%")
                            if j>lim:
                                break
                            userId, movieId, rating = str(int(line[0])), str(int(line[1])), str(line[2])
                            movieId = int(movieId)
                            if movieId not in metadata["id"].values: continue
                            nbVotes += 1

                            try:
                                title = metadata.loc[metadata['id'] == movieId, 'title'].values[0].replace(" ", "_")
                                director = metadata.loc[metadata['id'] == movieId, 'director'].values[0].replace(" ", "_")
                                cast = metadata.loc[metadata['id'] == movieId, 'cast'].values[0]
                                genres = metadata.loc[metadata['id'] == movieId, 'genres'].values[0]
                            except:
                                continue
                            #print(genres)
                            #print(title, director, cast)

                            o.write(str(j)+"\t"+rating+"\n")
                            f1.write(str(j)+"\t"+title+"\n")
                            setMovies.add(title)
                            f2.write(str(j)+"\t"+userId+"\n")
                            setVoters.add(userId)
                            f3.write(str(j) + "\t" + director+"\n")
                            setDirectors.add(director)
                            f4.write(str(j) + "\t")
                            for act in cast:
                                setArtists.add(act.replace(" ", "_")+" ")
                                f4.write(act.replace(" ", "_")+" ")
                            f4.write("\n")
                            f5.write(str(j) + "\t")
                            for g in genres:
                                setGenres.add(g.replace(" ", "_")+" ")
                                f5.write(g.replace(" ", "_")+" ")
                            f5.write("\n")

print(nbVotes, len(setMovies), len(setArtists), len(setVoters), len(setDirectors), len(setGenres))



import sys
import spotipy
import spotipy.util as util
from spotipy.oauth2 import SpotifyClientCredentials


cid = '0d90192fa6d74cacabfed3550b454c83'
secret = 'e05850cb685a49fd82b6ae5038bd2052'
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)



def saveData(folder, g, nounsPost):
    thres = 500

    print("Counting")
    vals = [item for sublist in list(nounsPost.values()) for item in sublist]

    toCons = []
    for a in set(vals):
        if vals.count(a)>thres:
            toCons.append(a)
    print(thres, len(toCons))

    print("Saving")
    while True:
        try:
            with open(folder + "/outcome.txt", "a", encoding="utf-8") as o:
                o.truncate(0)
                with open(folder + "/feature_0.txt", "a", encoding="utf-8") as f:
                    f.truncate(0)
                    for u in nounsPost:
                        if u<0:
                            continue

                        txt = str(u) + "\t"
                        premPass = True
                        for v in nounsPost[u]: # Pos = feature
                            if v not in toCons:
                                continue
                            if not premPass:
                                txt += " "
                            txt += str(v)
                            premPass = False
                        txt += "\n"

                        if premPass: continue


                        txto = str(u) + "\t"
                        premPass = True
                        for v in nounsPost[-u]: # Neg = output
                            if v not in toCons:
                                pass
                                #continue
                            if not premPass:
                                txto += " "
                            txto += str(v)
                            premPass = False
                        txto += "\n"

                        if premPass: continue

                        f.write(txt)
                        o.write(txto)

            break

        except Exception as e:
            print("Retrying to write file :", e)
            pass



def retreatCorpus(songs):
    nbPlaylists=2000
    tags = "english,rock"
    lgWindow=4

    g={}
    contentPosts={}

    borneSup=nbPlaylists//50 + 1
    iter=0
    tracksTot=0
    for page in range(0, borneSup):
        try:
            query=sp.search(q=tags, type="playlist", limit=50, offset=50*page)
        except Exception as ex:
            print(ex)
            continue
        for playlist in query["playlists"]["items"]:
            if iter>nbPlaylists:
                break
            iter+=1
            if iter%5==0:
                print("Playlist", iter, "/", nbPlaylists)
            print(playlist["name"], playlist["tracks"], playlist["id"])
            playlistUsr = sp.user_playlist(playlist["owner"]["id"], playlist["id"])["tracks"]["items"]

            if len(playlistUsr)<lgWindow:
                continue  # On doit definir une fenetre dans laquelle on considere que les musiques ont une influence
            nbTrack=0
            for track_i in range(len(playlistUsr)):
                try:
                    track=playlistUsr[track_i]
                    nameSongs = track["track"]["name"].replace(" ", "_").lower()
                    artistsSongs = []
                    for art in track["track"]["artists"]:
                        artistsSongs.append(art["name"].replace(" ", "_").lower())

                    time = track["added_at"]
                    nbTrack+=1
                    tracksTot+=1
                    
                    if songs:
                        artists=[nameSongs]
                    else:
                        artists=artistsSongs

                    for name in artists:
                        try:
                            contentPosts[-tracksTot].append(name)
                        except:
                            contentPosts[-tracksTot] = [name]

                    try:  # Les +- tracktot c'est un moyen de simuler les couples message-reponse ; +tracksTot=message ; -tracksTot=reponse
                        g[tracksTot].append(-tracksTot)
                    except:
                        g[tracksTot] = [-tracksTot]


                    if nbTrack>lgWindow:
                        fen=range(1, lgWindow+1)
                    else:
                        fen=range(1, nbTrack+1)
                    for w in fen:
                        txtPar = playlistUsr[track_i-w]["track"]["name"].replace(" ", "_").lower()
                        artistsPar=[]
                        for art in playlistUsr[track_i-w]["track"]["artists"]:
                            artistsPar.append(art["name"].replace(" ", "_").lower())
                            
                        if songs:
                            artistsPar=[txtPar]
                        else:
                            artistsPar=artistsPar

                        for name in artistsPar:
                            try:
                                contentPosts[tracksTot].append(name)
                            except:
                                contentPosts[tracksTot]=[name]

                except Exception as e:
                    print(e)
                    pass

    return g, contentPosts



def retreat(folder, listFiles=None, songs=False):
    g, contentPost = retreatCorpus(songs=songs)

    folder = "Data/" + folder + "/"
    print("Saving data")
    saveData(folder, g, contentPost)

retreat("Spotify")
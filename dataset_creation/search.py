import requests
import json
import os
client_id = 'YOUR JAMENDO CLIENT ID'

json_keys = {}
last_offset = 1
counter = 0
n_pages = 100
for i in range(n_pages):
    offset = (i * 200)
    print(counter, '/', n_pages * 200)
    r = requests.get(
        'https://api.jamendo.com/v3.0/tracks/?client_id=' +
        client_id +
        '&format=json&order=popularity_total&audiodlformat='
        'flac&limit=200&offset=' + str(offset) + '&include=musicinfo'
        ''
    )
    songs = json.loads(r.text)['results']
    for song in songs:
        if(song['audiodownload_allowed']):
            id = song['id']
            json_keys[id] = song
            last_offset += 1
            counter += 1

if not os.path.isdir('../Jamendo'):
    os.mkdir('../Jamendo')

file_name = '../Jamendo/metadata.json'
with open(file_name, 'w') as outfile:
    json.dump(json_keys, outfile)

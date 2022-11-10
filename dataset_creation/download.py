import json
import os.path
from urllib.request import urlopen


if not os.path.isdir('../Jamendo/music'):
    os.mkdir('../Jamendo/music')

with open('../Jamendo/metadata.json') as file:
    json_file = json.load(file)

errors = {}
counter = 0
for song_id in json_file.keys():
    print(counter, '/', len(json_file.keys()))
    song = json_file.get(song_id)
    file_name = song_id + '.flac'
    url = song['audiodownload']
    if not os.path.isfile('../Jamendo/music/' + file_name):
        try:
            with urlopen(url) as file:
                content = file.read()
            # Save to file
            with open('../Jamendo/music/' + file_name, 'wb') as download:
                download.write(content)
        except Exception as e:
            print(e)
            errors[song['id']] = song
    counter += 1
with open('errors.json', 'w') as outfile:
    json.dump(errors, outfile)

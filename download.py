import json
import os.path
import wget

with open('metadata.json') as file:
  json_file = json.load(file)

errors = {}
counter = 0
for song_id in json_file:
  song = json_file.get(song_id)
  file_name = song_id + '.mp3'
  url = song['audiodownload']
  if not os.path.isfile('music/' + file_name):
    try:
      filename = wget.download(url, out='music/' + file_name)
    except:
      errors[song['id']]=song
  counter +=1

with open('errors.json', 'w') as outfile:
  json.dump(errors, outfile)

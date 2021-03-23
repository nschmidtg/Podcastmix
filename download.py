import json
import os.path

import sys
import wget

if not os.path.exists('Jamendo'):
  os.makedirs('Jamendo')

if not os.path.exists('Jamendo/music'):
  os.makedirs('Jamendo/music')

with open('Jamendo/metadata.json') as file:
  json_file = json.load(file)

errors = {}
counter = 0
for song_id in json_file:
  print("hola")
  song = json_file.get(song_id)
  file_name = song_id + '.flac'
  url = song['audiodownload']
  if not os.path.isfile('Jamendo/music/' + file_name):
    try:
      filename = wget.download(url, out='Jamendo/music/' + file_name)
    except:
      errors[song['id']]=song
  counter +=1

with open('errors.json', 'w') as outfile:
  json.dump(errors, outfile)

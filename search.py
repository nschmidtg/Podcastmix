import json
import os.path
import wget
import requests
"""
  Using your token from Jamendo, it queries the API to get the metadata
  from the most popular & featured songs. The metadata.json generated file will
  be used to download the files later. Files are downloaded in mp3, otherwise the 
  dataset get too big.

  n_pages = 80
"""
# your Jamendo API token here. You can get your own from https://devportal.jamendo.com/admin
client_id = '08bac555'

if not os.path.exists('Jamendo'):
  os.makedirs('Jamendo')

json_keys = {}
last_offset = 1
counter = 0
for i in range(80):
  offset = (i * 200)
  r = requests.get('https://api.jamendo.com/v3.0/tracks/?client_id='+client_id+'&format=json&boost=popularity_total&audiodlformat=flac&featured=true&limit=200&offset='+str(offset)+'&include=licenses')
  songs = json.loads(r.text)['results']
  for song in songs:
    if(song['audiodownload_allowed']):
      id = song['id']
      json_keys[id] = song
      last_offset += 1

file_name = 'Jamendo/metadata.json'
with open(file_name, 'w') as outfile:
    json.dump(json_keys, outfile)
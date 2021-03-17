import requests
import json

client_id = '08bac555'

json_keys = {}
last_offset = 1
counter = 0
for i in range(80):
  offset = (i * 200)
  r =requests.get('https://api.jamendo.com/v3.0/tracks/?client_id='+client_id+'&format=json&boost=downloads_total&limit=200&offset='+str(offset)+'&include=licenses')
  songs = json.loads(r.text)['results']
  for song in songs:
    if(song['audiodownload_allowed']):
      print('last_offset',last_offset)
      print('i',i)
      id = song['id']
      json_keys[id] = song
      last_offset += 1

file_name = 'metadata.json'
with open(file_name, 'w') as outfile:
    json.dump(json_keys, outfile)

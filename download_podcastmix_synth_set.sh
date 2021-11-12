# create directory if test-set download script (download_podcastmix.sh)
# has not been run before
mkdir -p podcastmix
cd podcastmix
# delete podcastmix-synth directory with its metadata
rm -rf podcastmix-synth
# download compressed file
curl -H "Authorization: Bearer $1" https://www.googleapis.com/drive/v3/files/1jouTryUzC9u3SNzwHiMN7kjQigXt-PPG?alt=media -o podcastmix-synth.tar.gz
# uncompress it
cat podcastmix-synth.tar.gz | tar xzvf -
# # delete compressed files to save space
# rm podcastmix-synth.tar.gz -rf

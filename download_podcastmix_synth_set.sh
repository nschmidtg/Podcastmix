cd podcastmix
# delete podcastmix-synth directory with its metadata
rm podcastmix-synth -rf
# download all compressed files
declare -A name_id=(["podcastmix-synth.tar.gzaa"]="1KNax5A4MyitNQn7xCka8IVVnVE-XdqdM" ["podcastmix-synth.tar.gzab"]="1PnmLUT6GcmI5GAHr-NDWlqY61OqBk_Ny" ["podcastmix-synth.tar.gzac"]="1RhB1Sg4jxHruHMBTa9yhtLXcdpliaxWP" ["podcastmix-synth.tar.gzad"]="1jSiJIg4N0yFJ3BVPeAtjfBiSAGE-2Krm" ["podcastmix-synth.tar.gzae"]="1aNo_owP_6egNgTiNMBSMMaW-K3hx3_9d" ["podcastmix-synth.tar.gzaf"]="1R-95Ua4L-_vcU_1OsfIpjfcyfuqZdtdi" ["podcastmix-synth.tar.gzag"]="1_cPlUL7JNcwK2J1SuITrMEf9NdOjkxPW" ["podcastmix-synth.tar.gzah"]="1uAcjx3GdIsnNaaTZTDUPZv5OHXHZMCW0" ["podcastmix-synth.tar.gzai"]="1gU6Nmc8e51jzOoNvMo0mrrzsknAim3je" ["podcastmix-synth.tar.gzaj"]="1n4ZfKRNAUosdFbuAesvWYH_tuVWm3Diu" ["podcastmix-synth.tar.gzak"]="1beRuZufDBWbCoJbcZlcZOh38SoIOpyE0" ["podcastmix-synth.tar.gzal"]="1E-wNX0q0P0LRMsLwmyt-bX2gzMG0YhLB" ["podcastmix-synth.tar.gzam"]="1GOk8R6yrWKdkJMBz9PEOfdAbHmVM5tQr" ["podcastmix-synth.tar.gzan"]="1hnFl8FOrDaoF2TH-QtREPemtFKbCHffF" ["podcastmix-synth.tar.gzao"]="1T03DuHqp8mnIkfclb4RaP0qu-6X6EEWq" ["podcastmix-synth.tar.gzap"]="1OemS2S2aru5A_U2jghrT6C8GKfMGXsoh" ["podcastmix-synth.tar.gzaq"]="1WvLG0ChXg9z_b33WOIgc7nGMiQjiQ52c" ["podcastmix-synth.tar.gzar"]="1ZLcLZBE-KslHBVVYgW4-IaNXZjMKiym0" ["podcastmix-synth.tar.gzas"]="1nUWKXNSJT-eGnUs7fAqgLDMNo-3oane7")
for key in "${!name_id[@]}" ; do
    KEY="${key}"
    VALUE="${name_id[${key}]}"
    # echo "$KEY"
    # echo "$VALUE"
    if [ -f "$KEY" ]; then
        echo "$KEY exists."
        ERROR=$(sed -n '7p' $KEY)
        echo $ERROR
        if [[ $ERROR == *"Invalid Credentials"* ]]; then
            echo "Token has expired, please click on Refresh access token, copy the new code and re-run the command"
            rm $KEY
        else 
            echo "$KEY is valid, wont download again."
        fi
    else 
        echo "$KEY does not exist. Proceed to download"
        curl -H "Authorization: Bearer $1" https://www.googleapis.com/drive/v3/files/${VALUE}?alt=media -o ${KEY}
    fi
done
# uncompress them
gunzip podcastmix-synth.tar.gzaa
# # delete compressed files to save space
# rm podcastmix-synth.tar.gz* -rf
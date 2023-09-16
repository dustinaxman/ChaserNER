#!/bin/bash

# URL and API key settings
URL="https://aqq6khoi8b.execute-api.us-east-1.amazonaws.com/prod"
API_KEY="v6onllhw5i2wUdxND2BOi69bFUjn75LOebmSTe26"


send_request() {
    local line=$1
    curl -X POST "$URL" \
         -H "Content-Type: application/json" \
         -H "x-api-key: $API_KEY" \
         -d "{\"text\": \"$line\"}"
    echo ""
}

process_file() {
    local filename=$1
    local mode=$2

    while IFS= read -r line; do
        if [ "$mode" == "parallel" ]; then
            send_request "$line" &
        else
            send_request "$line"
        fi
    done < "$filename"
}

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <path_to_text_file> <mode: sequential|parallel>"
    exit 1
fi

process_file "$1" "$2"


I made this script for running each line of a text file through the api, either all at once time (as if all of the people sent them at once), or one by one.
If you put the above into some file like ~/Downloads/send_requests.sh and save it, you should be able to run:

chmod +x ~/Downloads/send_requests.sh

Then run something like:
~/Downloads/send_requests.sh ~/Downloads/test_input.txt sequential
or
~/Downloads/send_requests.sh ~/Downloads/test_text.txt parallel

The parallel version will be much faster but the script above kind of mushes the output of both calls so it's more for testing speed than accuracy.  I can look into the issues with the script above after work today.

If you don't want to bother with that, you can just run this, and swap in your text:
curl -X POST ${URL} -H "Content-Type: application/json" -H "x-api-key: ${API_KEY}" -d '{"text": "dustin please finish the report on profit by 10/21"}'




rm ~/Downloads/test_input.txt;for i in {1..32}; do echo "dustin to do report by 27th" >> ~/Downloads/test_input.txt; done



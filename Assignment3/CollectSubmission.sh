files="./models_captioning/*
./models_char_rnn/*
Assignment3_Part1_Implementing_RNN.ipynb
rnn_layers.py
Assignment3_Part2_ImageCaptioning.ipynb
coco_utils.py
captioning.py
Assignment3_Part3_CharRNN.ipynb
utils.py
char_rnn.py
"


for file in $files
do
    if [ ! -f $file ]; then
        echo "Required $file not found."
        exit 0
    fi
done

if [ -z "$1" ]; then
    echo "Team number is required.
Usage: ./CollectSubmission team_#"
    exit 0
fi


rm -f $1.zip
zip -r $1.zip $files

files="Assignment4_1.ipynb
Assignment4_2.ipynb
Assignment4_3.ipynb"

for file in $files
do
    if [ ! -f $file ]; then
        echo "Required $file not found."
        exit 0
    fi
done

if [ -z "$1" ]; then
    echo "Team number is required.
Usage: ./CollectSubmission Team_#"
    exit 0
fi


rm -f $1.zip
zip -r $1.zip ./*.ipynb 

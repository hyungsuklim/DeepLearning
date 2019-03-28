#!/bin/bash

usage(){
    echo "Usage: $0 student-number"
    exit 1
}

[[ $# -eq 0 ]] && usage

tar czvf ${1}.tar.gz *.ipynb utils

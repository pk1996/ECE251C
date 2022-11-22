#!/bin/bash

'''
Script to convert .wv* files to .sph files
./script.sh <path to txt file containing paths> <target dir>
'''

filename="$1"
base_path="$2"
k=0
while read -r line; do
    file_name="$(basename "$line")"
    k=$((k + 1))
    file_name=$(printf "%06d.sph" $k)
    target="$base_path/$file_name"
    source=$line
    sph2pipe $source $target
    echo $source $target
done < "$filename"

#     file_name="${file_name%.wv*}.sph"
#     target="$base_path/$file_name"
#     target="$base_path/$k"
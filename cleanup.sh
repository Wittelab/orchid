#!/bin/bash
echo -e "\033[1;37mThis will delete all nextflow run files and potentially other *.log, *.txt, and *.html file in this and the work/ directories.\033[0m"
echo "This should be OK if you've not modified the workflow/ directory."
read -r -p "Are you sure? [y/N] " response
case $response in
    [yY][eE][sS]|[yY]) 
        rm -rf work
        rm -f  .nextflow.*
	rm -rf .nextflow 
        rm -f  *timeline.html*
        rm -f  *trace.txt*
        rm -rf workflow/work
        rm -f  workflow/.nextflow.*
        rm -f  workflow/*timeline.html*
        rm -f  workflow/*trace.txt*
        ;;
    *)
        ;;
esac






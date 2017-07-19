#!/bin/bash
echo -e "This \033[1;37mWILL DELETE all nextflow run files AND POTENTIALLY other *.log, *.txt, and *.html file\033[0m in this and the work/ directories."
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






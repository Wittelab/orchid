#!/bin/bash
rootPath="http://hgdownload.cse.ucsc.edu/goldenpath/hg19/phyloP46way/primates/"
for i in `seq 1 22` X Y; \

    do \
        echo "Processing phyloP data for chromosome chr${i}..."; \
        wigFn="chr${i}.phyloP46way.primate.wigFix"; \
        url="${rootPath}/${wigFn}.gz"; \
        wget -qO- ${url} | gunzip -c - | wig2bed - > ${wigFn}.bed; \
    done
echo "Combining..."
ls chr* | sort | xargs cat >> phyloP46way.bed_
mv phyloP46way.bed_ phyloP46way.bed
bgzip phyloP46way.bed > phyloP46way.bed.gz
tabix -p bed phyloP46way.bed.gz

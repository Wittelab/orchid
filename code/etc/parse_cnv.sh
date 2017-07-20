#!/bin/bash
awk 'BEGIN{OFS="\t"; FS="\t"}NR==1{next;}!$10{next;}{print $1"_chr"$12, $13-1, $14, $1, $10}' $1 | sort-bed - > $2
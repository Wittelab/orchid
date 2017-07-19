#!/bin/bash
## You can specify custom code locations in a machine specific manner when building a database. 
## Simply replace 'local' or 'cluster' with the machine's hostname and the the location of orchid's workflow directory.
## When running this script within the orchid directory (recommented), you don't need to do anything.
case `hostname` in
  (local) WF_DIR=/workmachine/pathto/orchid;;
  (cluster) WF_DIR=/cluster/pathto/orchid;;
  (*) WF_DIR=`pwd`/workflow; echo "Using the default nextflow code directory for this host...";;
esac

time $WF_DIR/nextflow run $WF_DIR/reset.nf && \
time $WF_DIR/nextflow run $WF_DIR/populate.nf  -with-trace -with-timeline populate_timeline.html && mv trace.txt populate_trace.txt  && \
time $WF_DIR/nextflow run $WF_DIR/annotate.nf  -with-trace -with-timeline annotate_timeline.html && mv trace.txt annotate_trace.txt



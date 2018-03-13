#!/bin/bash
## You can specify custom code locations in a machine specific manner when building a database. 
## Simply replace 'local' or 'cluster' with the machine's hostname and the the location of orchid's workflow directory.
## When running this script within the orchid directory (recommented), you don't need to do anything.
case `hostname` in
  (local) WF_DIR=`pwd`/workflow;;
  (cluster) WF_DIR=/cluster/pathto/orchid;;
  (*) WF_DIR=`pwd`/workflow; echo "Using the default nextflow code directory for this host...";;
esac

# If DB parameters are defined as environment variables (such as through docker) use them
if [[ "$ORCHID_DB_USED" == "true" ]]; then
  time $WF_DIR/nextflow run $WF_DIR/reset.nf \
      --database_ip $ORCHID_DB_HOST \
      --database_port $ORCHID_DB_PORT \
      --database_username $ORCHID_DB_USER \
      --database_password $ORCHID_DB_PASS && \
  time $WF_DIR/nextflow run $WF_DIR/populate.nf  \
      -with-trace \
      -with-timeline populate_timeline.html 
      --database_ip $ORCHID_DB_HOST \
      --database_port $ORCHID_DB_PORT \
      --database_username $ORCHID_DB_USER \
      --database_password $ORCHID_DB_PASS && \
  mv trace.txt populate_trace.txt  && \
  time $WF_DIR/nextflow run $WF_DIR/annotate.nf \
      -with-trace \
      -with-timeline annotate_timeline.html \
      --database_ip $ORCHID_DB_HOST \
      --database_port $ORCHID_DB_PORT \
      --database_username $ORCHID_DB_USER \
      --database_password $ORCHID_DB_PASS && \
   mv trace.txt annotate_trace.txt
# Otherwise, go with the defaults
else
  time $WF_DIR/nextflow run $WF_DIR/reset.nf && \
  time $WF_DIR/nextflow run $WF_DIR/populate.nf  -with-trace -with-timeline populate_timeline.html && mv trace.txt populate_trace.txt  && \
  time $WF_DIR/nextflow run $WF_DIR/annotate.nf  -with-trace -with-timeline annotate_timeline.html && mv trace.txt annotate_trace.txt
fi
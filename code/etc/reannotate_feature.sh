#!/bin/bash
pushd `dirname $0` > /dev/null
SCRIPTPATH=`pwd`
popd > /dev/null
DIRS=`cat $SCRIPTPATH/../../annotate_trace.txt | grep -w $1 | cut -f2`

for i in $DIRS
do
   PATTERN=`basename $i`"*"
   echo "Working on [$i]";
   PREFIX=`find $SCRIPTPATH/../../work -maxdepth 2 -type d -name $PATTERN -print -quit`;
   cd $PREFIX;
   source .command.env;
   sh .command.sh;
   cd ../../..;
done

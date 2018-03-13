#!/usr/bin/env nextflow
/* 
 * Clinton Cario 
 *    01/24/2016 Initial Version
 *       --      Many updates
 *
 * This script resets the database
 * 
 * Notes:
 *     DATA_DIR, CODE_DIR, DATABASE (mysql connection string), and other variables are defined in nextflow.config, which nextflow automatically loads.
 *     Individual MySQL parameters are prefixed with MYSQL_ and include USER, PASS, IP, PORT, DB
 */

println "Database:        $DATABASE"

process resetDB {
    echo true
    executor  'local'
    errorStrategy 'ignore'

    shell:
    '''
    # Create the orchid user on a MemSQL database
    mysql --user='root' --password='' --host=$MYSQL_IP --port=$MYSQL_PORT -e "GRANT ALL ON $MYSQL_DB.* TO '$MYSQL_USER'@'%' IDENTIFIED BY '$MYSQL_PWD';" || echo "No DB root access, attempting to continue..."
    # Other database tweaks, may have to be done by hand if the root password is set
    mysql --user=$MYSQL_USER --password=$MYSQL_PWD --host=$MYSQL_IP --port=$MYSQL_PORT -e "SET GLOBAL connect_timeout = 600;" || echo "No DB root access, can not set connect_timeout..."
    mysql --user=$MYSQL_USER --password=$MYSQL_PWD --host=$MYSQL_IP --port=$MYSQL_PORT -e "SET GLOBAL lock_wait_timeout = 600;" || echo "No DB root access, can not set lock_wait_timeout..."
    mysql --user=$MYSQL_USER --password=$MYSQL_PWD --host=$MYSQL_IP --port=$MYSQL_PORT -e "SET GLOBAL multistatement_transactions = off;" || echo "Skipping memsql multistatement_transactions..."
    mysql --user=$MYSQL_USER --password=$MYSQL_PWD --host=$MYSQL_IP --port=$MYSQL_PORT -e "SET GLOBAL max_allowed_packet = 500000000;" || echo "No DB root access, can not set max_allowed_packet..."
    mysql --user=$MYSQL_USER --password=$MYSQL_PWD --host=$MYSQL_IP --port=$MYSQL_PORT -e "trigger full gc flush;" || echo "Skipping memsql flush..." || echo "Skipping db flush..."
    # Create the database
    mysql --user=$MYSQL_USER --password=$MYSQL_PWD --host=$MYSQL_IP --port=$MYSQL_PORT -e "DROP DATABASE IF EXISTS $MYSQL_DB; CREATE DATABASE $MYSQL_DB;"
    '''
}


// TODO: mysql/memsql specific commands-- 
//mysql -BN -u root --password='' --host=$MYSQL_IP -P $MYSQL_PORT -e "SELECT count(SCHEMA_NAME) FROM INFORMATION_SCHEMA.SCHEMATA WHERE SCHEMA_NAME = 'memsql'"
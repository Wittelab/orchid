#!/usr/bin/env nextflow
/* 
 * Clinton Cario 11/5/2015
 *
 * Notes:
 *     DATA_DIR, CODE_DIR, and DATABASE (mysql connection string) are all defined in nextflow.config, which nextflow automatically loads.
 *     Individual MySQL parameters are prefixed with MYSQL_ and include USER, PASS, IP, PORT, DB
 */


cmd           = "mysql -u$MYSQL_USER -p$MYSQL_PWD -h$MYSQL_IP -P$MYSQL_PORT -D$MYSQL_DB -NB -e 'SHOW TABLE STATUS LIKE \"${params.mutation_table_name}\"' | cut -f5"
num_mutations = [ '/bin/sh', '-c', cmd ].execute().text.trim()
chunk_num     = Math.ceil((num_mutations.toInteger()/params.chunk_size)).toInteger()

println "======================== Run Info =================================================="
println "Database:                    $DATABASE                                              "
println "Mutations:                   ${num_mutations}                                       "
println "Number of chunks / process:  ${chunk_num}                                           "
println "===================================================================================="

// Get chunks
chunks = Channel.from(1..chunk_num).map{ "${it}/${chunk_num}" }

/* 
 * Generates a table for all categorical features (see nextflow.config for what they are)
 */
process featurize {
    tag { "${feature['column']} | ${chunk}" }
    maxForks 10

    input:
    val chunk from chunks
    each feature from params.features

    shell:
    '''
    python !{CODE_DIR}/orchid_db.py featurize -k !{chunk} -c !{feature['column']} -j '!{feature['params']}'
    '''
}


#!/usr/bin/env nextflow
/* 
 * Clinton Cario 
 *    01/18/2016 Initial Version
 *       --      Many updates
 *
 * This script imports real and simulated mutation data
 * 
 * Notes:
 *     DATA_DIR, CODE_DIR, DATABASE (mysql connection string), and other variables are defined in nextflow.config, which nextflow automatically loads.
 *     Individual MySQL parameters are prefixed with MYSQL_ and include USER, PASS, IP, PORT, DB
 */
import groovy.json.*
import groovy.transform.Field



filename   = "echo $INPUT_FILE".execute().text.trim()
basename   = filename.lastIndexOf('/').with {it != -1 ? filename[it..-1] : filename}
extension  = filename.lastIndexOf('.').with {it != -1 ? filename[it+1..-1] : 'vcf'}
// Build json object of acceptable consequences for mutation import 
acceptable_consequences = new JsonBuilder(params.acceptable_consequences)

println "======================== Run Info =================================================="
println "Database:            $DATABASE                                                      "
println "Mutation file:       data/mutations${basename}                                     "
println "Simulating Variants? ${params.add_simulated}                                        "
println "Simulating Cache:    ${params.use_simulated_cache}                                  "
println "===================================================================================="


// For an initial icgc file
icgc = Channel.create()
// For an initial vcf file 
vcf  = Channel.create()

// Direct the file to the appropriate channel for processing
input = Channel.value([extension, file(filename)])
input.route ( tsv: icgc, vcf: vcf) { e, f -> e }



// =============================
// Real Mutations 
// ---------
// Import the (downsampled) SSM mutations
process icgcToVcf {
    input: 
    set val(extension), file('icgc_file') from icgc

    output:
    set val(extension), file('vcf_file') into vcf_converted

    shell:
    '''
    $CODE_DIR/icgc_to_vcf.awk < icgc_file > vcf_file
    '''
}



// Create a channel for the ready vcf file (either converted or given initially)
vcf.mix(vcf_converted).set { real }
// For a initial or converted vcf file
process countVcf {
    input: 
    set val(extension), file('vcf_file') from real

    output:
    set val('real'), file('vcf_file') into real_ready
    stdout simulated_number

    shell:
    '''
    # Find number of mutations
    [[ ! -f $COUNT_FILE ]] && cat vcf_file | sed '/^#/d' | wc -l | sed -e 's/^[ \\n\\t]*//' > $COUNT_FILE
    if [ !{params.num_mutations} = -1 ]; then
        cat $COUNT_FILE
    else
        echo "!{params.num_mutations}"
    fi
    '''
}



// =============================
// Simulated Mutations
// ---------
// Begin by simulating a given number of mutations 
process simulateMutations {
    echo true

    input:
    val num_mutations from simulated_number

    output:
    set val('sim'), file('simulated') into simulated_ready

    when:
    params.add_simulated

    shell:
    '''
    # Check to see if the cached version was was requested and exists
    if [ -f $SIM_CACHE/simulated.vcf ] && [ "!{params.use_simulated_cache}" = "true" ]; then
        # Use it
        echo "simulateMutations: Using simulated.vcf cache"
        #head -n !{num_mutations.trim()} $SIM_CACHE/simulated.vcf > simulated.vcf
        TOTAL=`wc < $SIM_CACHE/simulated.vcf | awk '{print $1}'`;
        NEEDED=!{num_mutations.trim()}
        CUTOFF=`bc -l <<< $NEEDED/$TOTAL`
        awk -v cutoff=$CUTOFF 'BEGIN{FS="\\t";OFS="\\t";srand();}/^#/{print $0}{if(rand()<=cutoff){print $0}}' $SIM_CACHE/simulated.vcf > simulated.vcf
    else
        # Otherwise generate new variants
        suffix=_`date +%F`
        echo "simulateMutations: Generating simulated.vcf variants..."
        python $SIM_CMD -n !{num_mutations.trim()} -i $HG19_DIR/reference_genome.fa -o simulated${suffix}.vcf 2>&1
        cp simulated${suffix}.vcf $SIM_CACHE/
        ln -sf $SIM_CACHE/simulated${suffix}.vcf $SIM_CACHE/simulated.vcf
        mv simulated${suffix}.vcf simulated.vcf
    fi
    # Could cache this scripts results as well, but breaks compatibility with old sim caches
    $CODE_DIR/sim_to_vcf.awk < simulated.vcf > simulated
    '''
}



// =============================
// Annotate and Import Mutations 
// ---------
// Mix all inputs and split into chunks for downstream processing
real_ready.mix(simulated_ready).splitText(by: params.chunk_size , file: true, elem: 1).set { mutation_splits }
// Process each split with snpEff
process snpEff { 
    tag { "${type}" }

    input:
    set val(type), file('vcf_split') from mutation_splits

    output:
    set val(type), file('annotated_split') into annotated_splits

    shell:
    '''
    $SNPEFF_CMD vcf_split > annotated_split
    '''
}



//
process importMutations {
    tag { "${type}" }

    input: 
    set val(type), file('annotated_split') from annotated_splits
    
    output:
    val "" into import_flags // Flag for when import has finished

    shell:
    '''
    python $CODE_DIR/orchid_db.py populate -i annotated_split -c !{params.consequence_table_name} -m !{params.mutation_table_name} !{type=='real'? '':'--simulated'} -A '!{acceptable_consequences}'
    '''
}
import_flags.last().set{ mutations_populated } // Flag for when import has finished


// =============================
// Add gene pathway information from KEGG (preprocessed or generated on-the-fly)
// ---------
process populateKegg {
    
    output:
    val "" into kegg_populated

    echo true

    when:
    params.use_kegg

    shell:
    '''
    # Try to import kegg tables from a pregenerated mysql dump before populating these tables
    # This will save many hours requerying kegg directly
    if [ -f $DATA_DIR/kegg/memsql_kegg_tables.sql ]; then
        mysql --user=$MYSQL_USER --password=$MYSQL_PWD --host=$MYSQL_IP --port=$MYSQL_PORT --database=$MYSQL_DB < $DATA_DIR/kegg/memsql_kegg_tables.sql && exit 0;
    else
        echo "Kegg tables not found, querying Kegg (this will take a while)..."
        python $CODE_DIR/kegg_populator.py
        mysqldump --user=$MYSQL_USER --password=$MYSQL_PWD --host=$MYSQL_IP --port=$MYSQL_PORT --database=$MYSQL_DB kegg_gene kegg_gene_alias kegg_gene_pathway kegg_pathway > $DATA_DIR/kegg/memsql_kegg_tables.sql
    fi
    '''
}


// =============================
// Add is_cancer_gene from kegg
// ---------
// Make an 'is_cancer_gene' column in the consequence table based on kegg info 
// May take some time with memSQL, but runs independently of other processes 
process finishKEGG {
    
    input: 
    val ok1 from kegg_populated
    val ok2 from mutations_populated
    
    when:
    params.use_kegg
    
    shell:
    '''
    mysql --user=$MYSQL_USER --password=$MYSQL_PWD --host=$MYSQL_IP --port=$MYSQL_PORT --database=$MYSQL_DB -e "DROP TABLE IF EXISTS kegg_cancer_gene_pathway;" && \
    mysql --user=$MYSQL_USER --password=$MYSQL_PWD --host=$MYSQL_IP --port=$MYSQL_PORT --database=$MYSQL_DB -e "
        CREATE TABLE kegg_cancer_gene_pathway (
          kegg_cancer_gene_pathway_id         INT unsigned NOT NULL AUTO_INCREMENT,
          mutation_id                         CHAR(32) DEFAULT NULL, 
          gene_id                             CHAR(64) NOT NULL,
          kegg_pathway_id                     CHAR(16) DEFAULT NULL,
          pathway_name                        CHAR(128) DEFAULT NULL,
          PRIMARY KEY                         (kegg_cancer_gene_pathway_id),
          KEY                                 (mutation_id),
          KEY                                 (gene_id),
          KEY                                 (kegg_pathway_id),
          KEY                                 (pathway_name)
        );" && \
    mysql --user=$MYSQL_USER --password=$MYSQL_PWD --host=$MYSQL_IP --port=$MYSQL_PORT --database=$MYSQL_DB -e "
        INSERT INTO kegg_cancer_gene_pathway (mutation_id, gene_id, kegg_pathway_id, pathway_name) 
        SELECT g.mutation_id, g.gene_id, k.kegg_pathway_id, n.pathway_name
        FROM 
        (
            SELECT DISTINCT mutation_id, gene_id
            FROM !{params.consequence_table_name}
        ) g
        LEFT JOIN kegg_gene_pathway AS k ON g.gene_id = k.ensembl_id
        LEFT JOIN kegg_pathway AS n ON k.kegg_pathway_id = n.kegg_pathway_id
        WHERE g.gene_id IS NOT NULL
        AND k.is_cancer=1
    "
    '''
}


// =============================
// Convert wig files
// ---------
def ext(file) {
  file = new File(file)
  name = file.name
  ext = name.drop(name.lastIndexOf('.'))
  return ext
}
if (params.to_convert!=[]){
    process convert2bed {
        echo true

        input: 
        each feature from params.to_convert

        """
        if [ ! -f ${feature['new_file']} ]
        then
            convert2bed --input=${ext(feature['old_file'])[1..-1]} --output=bed < ${feature['old_file']} > ${feature['new_file']}
        fi
        """
    }
}
















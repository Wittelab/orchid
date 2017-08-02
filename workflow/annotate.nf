#!/usr/bin/env nextflow
/* 
 * Clinton Cario 
 *    10/07/2015 
 *
 * This script populates annotation columns from various datasets.
 *
 * Notes:
 *     DATA_DIR, CODE_DIR, DATABASE (mysql connection string), and other variables are defined in nextflow.config, which nextflow automatically loads.
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


// Channels can be broken into chromosomes
// bed_chromosomes  = Channel.from(params.chromosomes)
// tab_chromosomes  = Channel.from(params.chromosomes)
tabixes           = Channel.create()
context_tabixes   = Channel.create()
numeric_beds      = Channel.create()
categorical_beds  = Channel.create()



// Update metadata
process updateMetadata{
    tag { "saving feature info" }
    echo true

    shell:
    '''
    mysql -u$MYSQL_USER -h$MYSQL_IP -P$MYSQL_PORT -D$MYSQL_DB -NB -e "
        INSERT INTO metadata (metakey, metavalue)
        VALUES (\'params\',\'!{params.meta_params}\');
        INSERT INTO metadata (metakey, metavalue)
        VALUES (\'features\',\'!{params.meta_features}\');
    "
    '''
}

// Creates tabix files to be fed into several processes for downstream population
process makeTabixes {
    tag { "splitting data" }

    output:
    file "sorted_variants.tabix" into split_tabixes

    shell:
    '''
    mysql -u$MYSQL_USER -h$MYSQL_IP -P$MYSQL_PORT -D$MYSQL_DB -NB -e "SELECT chromosome, start, end, reference_genome_allele, mutated_to_allele, !{params.mutation_table_name}_id, '' FROM !{params.mutation_table_name}" > variants.tabix
    sort -k1,1 -k2,2 -n -o sorted_variants.tabix variants.tabix
    '''
}
split_tabixes.splitText(by: params.chunk_size , file: true).into { tabixes; context_tabixes }

// Creates bed files to be fed into several processes for downstream population
process makeBeds {
    tag { "splitting data" }

    output:
    file "sorted_variants.bed" into split_beds

    shell:
    '''
    mysql -u$MYSQL_USER -h$MYSQL_IP -P$MYSQL_PORT -D$MYSQL_DB -NB -e "SELECT CONCAT('chr',chromosome), start-1, end, !{params.mutation_table_name}_id, '' FROM !{params.mutation_table_name}" > variants.bed
    sort-bed variants.bed > sorted_variants.bed
    '''
}
split_beds.splitText(by: params.chunk_size , file: true).into { numeric_beds; categorical_beds; cnv_beds }



// ======= Tabix Formatted Files ===========
process annotateTabixFeatures {
    tag { "${feature['column']} | ${tabix_file.name}/${chunk_num}".replaceFirst(/sorted_variants\.(\d+)\.\w+/, "\$1") }
    //beforeScript 'time sleep $[ ( $RANDOM % 5 ) + 1 ]s'

    input:
    file tabix_file from tabixes
    each feature from params.annotation['tabix']

    shell:
    '''
    if [[ -s !{tabix_file} ]]; # tabix freaks out with empty files, so only process files with data
    then
        # Preprocess if required
        if !{ feature['preprocessor']!=null? "true" : "false" }; 
        then
            !{feature['preprocessor']} !{tabix_file} > !{tabix_file}.preprocessed;
        else
            mv !{tabix_file} !{tabix_file}.preprocessed;
        fi

        # Get the score for each variant for this tabix file
        tabix -R !{tabix_file}.preprocessed !{feature['file']} > !{tabix_file}.processed
        
        # Post process if required
        # Usually consists of building a dictionary from the preprocessed file: 
        #    ID=>value where ID is a multi-key index from the first X columns
        # Then for each entry in the tabix file, lookup its value
        if !{ feature['postprocessor']!=null? "true" : "false" }; 
        then
            !{feature['postprocessor']} !{tabix_file}.processed !{tabix_file}.preprocessed > !{tabix_file}.postprocessed;
        else
            mv !{tabix_file}.processed !{tabix_file}.postprocessed;
        fi

        # Have orchid_db populate the database with the results
        python $CODE_DIR/orchid_db.py annotate -i !{tabix_file}.postprocessed -d !{feature['table']} -c !{feature['column']} -j '!{feature['params']}'
    fi
    '''
}



// ======= Numeric Bed Features ===========
//          feature overlap of >=1 bp   show feature values            variant file       reference feature file                   capture the id & score                                        result file    
//                          |                       |                       |                     |                                        |                                                           |        
//   bedmap --faster --bp-ovr 1  --skip-unmapped --echo --${strategy} !{bed_file} ${DATA_DIR}/!{feature}/!{feature}.bed | awk 'BEGIN{FS="\\t"; OFS="\\t"}{split($0,s,"|"); print $4,s[2]}' > !{bed_file}.results
//               |                     |                       |                                                                                                                                                
//           speedup option     (self-explanatory)    show the (mean, median, max (or whatever strategy)) value                                                                                                 
process annotateBed {
    tag { "${feature['column']} | ${bed_file.name}/${chunk_num}".replaceFirst(/sorted_variants\.(\d+)\.\w+/, "\$1") }
    //beforeScript 'time sleep $[ ( $RANDOM % 5 ) + 1 ]s'
    errorStrategy 'retry'
    maxRetries 3

    input:
    file bed_file from numeric_beds
    each feature from params.annotation['bed']
    
    shell:
    '''
    # Use bedmap to subset database entries relative to this chromosome, then map each entry to a database entry, taking the max if multiple mappings exist
    bedmap --faster --bp-ovr 1  --skip-unmapped --echo !{feature['strategy']} !{bed_file} !{feature['file']} | awk 'BEGIN{FS="\\t"; OFS="\\t"}{split($0,s,"|"); if(s[2]==""){s[2]=1;} print $4,s[2]}' > !{bed_file}.results
    # Have orchid_db populate the database with the results
    python $CODE_DIR/orchid_db.py annotate -i !{bed_file}.results -d !{feature['table']} -c !{feature['column']} -j '!{feature['params']}'
    '''
}



// ======= Categorical Bed Features ===========
//   Use bedmap to subset database entries relative to this chromosome, then map each query entry to a database entry, taking the max if multiple database entries exist (other options like min and mean are available)
//    
//              overlap of >=1 bp (SNP)   show query entry               variant file           feature file                       print the id & score                                        result file       
//                           |                       |                       |                       |                                      |                                                       |            
//   bedmap  --faster --bp-ovr 1  --skip-unmapped --echo --echo-map-id !{bed_file} ${DATA_DIR}/!{feature}/!{feature}.bed | awk 'BEGIN{FS="\\t"; OFS="\\t"}{split($0,s,"|"); print $4,s[2]}' > !{bed_file}.results
//                |                     |                        |                                                                                                                                               
//             speedup option     (self-explanatory)         show the feature value                                                                                                                              
process annotateCategoricalFeatures {
    tag { "${feature['column']} | ${bed_file.name}/${chunk_num}".replaceFirst(/sorted_variants\.(\d+)\.\w+/, "\$1") }
    //beforeScript 'time sleep $[ ( $RANDOM % 5 ) + 1 ]s'
    errorStrategy 'retry'
    maxRetries 3


    input:
    file bed_file from categorical_beds
    each feature from params.annotation['categorical']
    
    shell:
    '''
    # Use bedmap to subset database entries relative to this chromosome, then map each entry to a database entry, taking the max if multiple mappings exist
    bedmap --faster --bp-ovr 1  --skip-unmapped --echo --echo-map-id !{bed_file} !{feature['file']} | awk 'BEGIN{FS="\\t"; OFS="\\t"}{split($0,s,"|"); print $4,s[2]}' > !{bed_file}.results
    # Have orchid_db populate the database with the results
    python $CODE_DIR/orchid_db.py annotate -i !{bed_file}.results -d !{feature['table']} -c !{feature['column']} -j '!{feature['params']}'
    '''
}



// ======= Database features ===========
// Data is already in a database, make a feature table for it
process annotateDatabasedFeatures {
    tag { "${feature['column']}" }
    errorStrategy 'retry'
    maxRetries 3

    input:
    each feature from params.annotation['database']
    
    shell:
    '''
    # Create a feature table from another table and column already in the database
    #### NOTE: A bug is preventing the enum values from populating a derived table directly, so created a temporary view
    mysql -u$MYSQL_USER -h$MYSQL_IP -P$MYSQL_PORT -D$MYSQL_DB -NB -e "
        CREATE VIEW !{feature['table']}_temp AS
        SELECT m.!{params.mutation_table_name}_id, c.!{feature['source_column']} AS !{feature['column']}
        FROM !{params.mutation_table_name} AS m
        LEFT JOIN !{feature['source_table']} AS c USING(mutation_id)
        WHERE !{feature['source_column']} !{feature['acceptable']};
        CREATE TABLE !{feature['table']} AS SELECT DISTINCT * FROM !{feature['table']}_temp;
        DROP VIEW !{feature['table']}_temp;
        ALTER TABLE !{feature['table']} ADD INDEX !{params.mutation_table_name}_id (!{params.mutation_table_name}_id);
    "
    '''
}



// ======= CNV Feature ===========
// Annotate variants with donor CNV data
process annotateCNV {
    tag { "${feature['column']} | ${bed_file.name}/${chunk_num}".replaceFirst(/sorted_variants\.(\d+)\.\w+/, "\$1") }
    //beforeScript 'time sleep $[ ( $RANDOM % 5 ) + 1 ]s'
    errorStrategy 'retry'
    maxRetries 3

    input:
    file bed_file from cnv_beds
    file feature from params.annotation['cnv']
    
    when:
    feature != ''

    shell:
    '''
    # Get the donor ids for each variant (need to sort by id for proper file alignment)
    sort -n -k4,4 !{bed_file} > variants.tsv
    cut -f4 variants.tsv > ids.tsv
    IDS=`cat ids.tsv | tr "\\n" "," | sed "s/.$//"`;
    QUERY="
        SELECT !{params.mutation_table_name}_id, donor_id 
        FROM !{params.mutation_table_name} 
        WHERE !{params.mutation_table_name}_id IN(${IDS})
    ";
    echo $QUERY | mysql -u$MYSQL_USER -h$MYSQL_IP -P$MYSQL_PORT -D$MYSQL_DB -NB | sort -n -k1,1 - > query_results.tsv
    cut -f2 query_results.tsv > donors.tsv
    # Create a donor-chromosome to lookup in the correspondingly formatted CNV file
    paste -d_ donors.tsv variants.tsv | sort-bed - > lookup.bed
    # Realign the ids
    cut -f4 lookup.bed > ids.tsv
    # Get CNV scores for each variant
    bedmap --faster --bp-ovr 1 --mean --echo-map-id --echo-map-score lookup.bed !{feature['file']} > scores.tsv
    paste ids.tsv scores.tsv | awk 'BEGIN{FS="\\t"; OFS="\\t"}{split($2,s,"|"); if(s[1]!="NAN"){print $1,s[1]}}' > results.tsv
    # Have orchid_db populate the database with the results
    python $CODE_DIR/orchid_db.py annotate -i results.tsv -d !{feature['table']} -c !{feature['column']} -j '!{feature['params']}'
    '''
}



// ======= Trinucleotide Context ===========
// Populates the trinucleotide context for each mutation
process annotateContext {
    tag { "context | ${tabix_file.name}/${chunk_num}".replaceFirst(/sorted_variants\.(\d+)\.\w+/, "\$1") }
    //beforeScript 'time sleep $[ ( $RANDOM % 5 ) + 1 ]s'
    errorStrategy 'retry'
    maxRetries 3

    input:
    file tabix_file from context_tabixes
    each feature from params.annotation['context']
    
    shell:
    '''
    awk 'BEGIN{FS="\\t";OFS="\\t"}{$2=$2-2; $3=$2+3; $4=$6"|["$4">"$5"]"; print $0}' !{tabix_file} > flank.tab
    bedtools getfasta -name -fi $HG19_DIR/reference_genome.fa -bed flank.tab -fo flank.fasta
    awk 'BEGIN{FS="\\t";OFS="\\t"; mut="";}{split($1,l,"|"); print l[1]; getline nextline; print substr(nextline,1,1) l[2] substr(nextline,3,1)}' flank.fasta > mutflank.fasta
    python $CODE_DIR/orchid_db.py annotate -i mutflank.fasta -d !{feature['table']} -c !{feature['column']} -j '!{feature['params']}'
    #
    # Also get the reverse complement context (only C and T from alleles are considered)
    tr 'ACGT' 'TGCA' < flank.fasta > rev_flank.fasta
    awk 'BEGIN{FS="\\t";OFS="\\t"; mut="";}{split($1,l,"|"); print l[1]; getline nextline; print substr(nextline,1,1) l[2] substr(nextline,3,1)}' rev_flank.fasta > rev_mutflank.fasta
    python $CODE_DIR/orchid_db.py annotate -i rev_mutflank.fasta -d !{feature['table']} -c !{feature['column']} -j '!{feature['params']}'
    '''
}



// ======= The Kegg Feature ===========
// Make an 'is_cancer_gene' column in the consequence table based on kegg info 
// May take some time with memSQL, but runs independently of other processes 
process annotateKEGG {
    tag { "${feature['column']}" }
    
    input: 
    each feature from params.annotation['kegg-cancer']
    
    when:
    params.use_kegg
    
    shell:
    '''
    mysql -u$MYSQL_USER -h$MYSQL_IP -P$MYSQL_PORT -D$MYSQL_DB -NB -e "
        SELECT !{params.consequence_table_name}_id, t.cancer 
        FROM !{params.consequence_table_name}
        JOIN (
            SELECT gene_id, '1' AS cancer 
            FROM  kegg_cancer_gene_pathway 
            GROUP BY gene_id) AS t 
        USING(gene_id);" > cancer_gene.results && \
    python $CODE_DIR/orchid_db.py annotate -i cancer_gene.results -d !{feature['table']} -c !{feature['column']} -j '!{feature['params']}'
    '''
}



// ======= The Frequency Feature ===========
// Mutation frequency within the input data
process annotateFrequency {
    tag { "${feature['column']}" }

    input:
    each feature from params.annotation['frequency']

    shell:
    '''
    mysql -u$MYSQL_USER -h$MYSQL_IP -P$MYSQL_PORT -D$MYSQL_DB -NB -e "
        SELECT s.!{params.mutation_table_name}_id, t.frequency
        FROM !{params.mutation_table_name} as s
        JOIN
            (
                SELECT !{params.mutation_table_name}_id, mutation_id, count(*) as frequency
                FROM !{params.mutation_table_name}
                GROUP BY mutation_id
            ) AS t
        ON t.mutation_id=s.mutation_id
        WHERE t.frequency>1
        ORDER BY frequency DESC" > frequency.results && \
    python $CODE_DIR/orchid_db.py annotate -i frequency.results -d !{feature['table']} -c !{feature['column']} -j '!{feature['params']}'
    '''
}

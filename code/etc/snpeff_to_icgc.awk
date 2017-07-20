#!/usr/bin/awk -f

# Clinton Cario 04/14/2016
#  04/14/2016 -- Initial version (or thereabouts)
#  05/06/2016 -- Added hex key prefix to prevent ID collision when dividing snpeff output into chunks for // processing
#  08/06/2016 -- Fixed bug that swapped gene/transcript columns (bad!)

BEGIN { 
    # Start mutation id suffix at one for this batch
    id=1; 
    # Seed random, and generate a random 4 digit hex prefix
    "echo $RANDOM"|getline seed;
    #print seed;
    srand(seed);
    prefix=toupper(sprintf("%04x",rand()*65536)); 
    # Define field seperators
    FS"\t"; 
    OFS="\t"
}

/^[^#]/{
    split($8,effects,"EFF=")
    split(effects[2],consequences,",");
    for (i=1; i<=length(consequences); i++)
    {   
        # Set the mutation id to the id field, unless it is empty, then make one up
        mutation_id = $3
        if ($3==".") 
            mutation_id = sprintf("S%s%04d",prefix,id)

        c = consequences[i];
        ## Consequence fields are: 
        #  Consequence, Protein_position, Amino_acid_change, CDS_position, Gene, Feature (transcript/regulatoryid)
        # Like:
        #  missense_variant(MODERATE|MISSENSE|Aca/Gca|T141A|305|ENSG00000186092|protein_coding|CODING|ENST00000335137|1|1)
        #  missense_variant|MODERATE|MISSENSE|Aca/Gca|T141A|305|ENSG00000186092|protein_coding|CODING|ENST00000335137|1|1   after first two gsubs
        gsub(/\(/,"|",c)      ## Replace opening '(' with | for easier parsing
        gsub(/\)/,"",c)       ## Remove closing ')'
        split(c,f,"|");       ## Split fields on the pipe
        split(f[4],aa,"\/");  ## Aca/Gca into aa[1] = 'Aca'; aa[2] = 'Gca'

        ## Figure out the mutation type from the REF/ALT fields of the vcf file, using ICGC categories
        if (length($4) > length($5))
        {
            type="deletion of <=200bp"
            
            $2++;                          ## Chromosome start includes base before, so increment
            gsub(/^.{1}/,"",$4);           ## Remove base before the starting base
            chr_end = $2+length($4)-1;     ## Calculate the chromosome_end position
            $5="-";                        ## Make the mutated to column a deletion (dash)
        }
        else if (length($4) < length($5))
        {
            type="insertion of <=200bp"    ## Chromosome start includes base before, so increment
            $2++;                          ## Start/stop positions are the same
            chr_end=$2;                    
            $4="-"                         ## Make the reference allele column an insertion point (dash)
        }
        else if (length($4) == length($5))
        {
            type = "single base substitution"
            chr_end = $2;
        }
        #         1           2                        9  10   11    12   13     14  15  16 17                        26    27  28            29   30                                     
        #      mutation     donor                     chr start end     version      REF    Mutated to        Consequence       CDS REF>ALT   Gene                                        
        #         |           |                         |  |    |  strand  |    type  |  Mutated from                  |   AA1#AA2 (R23L)     |    Transcript                             
        print mutation_id,"DOXXXXXX","","","","","","",$1,$2,chr_end,1,"GRCh37",type,$4,$4,$5,"","","","","","","","",f[1],f[5],f[6]$4">"$5,f[7],f[10],"","","","","","","","","","","",""
    }
    id++;
    # Get a new random prefix if this id space has been exhausted
    if (id>=9999) { prefix=toupper(sprintf("%04x",rand()*65536)); }
}
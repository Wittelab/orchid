#!/usr/bin/awk -f

# Clinton Cario 
#  02/23/2017 -- Initial version
#  05/05/2017 -- VCF output now follows specification 

## Reference this: http://docs.icgc.org/submission/guide/icgc-simple-somatic-mutation-format/#vcf-like-vs-icgc-like-mutation-format
BEGIN { 
	last_id = "";
	FS="\t"; 
	OFS="\t";
	print "##fileformat=VCFv4.0";
	print "##INFO=<ID=ICGC_DONOR_ID,Number=1,Type=String,Description=\"The ICGC donor ID\">";
	print "##INFO=<ID=TYPE,Number=1,Type=String,Description=\"The type of mutation\">";
	print "##INFO=<ID=REF_ALLELE,Number=1,Type=String,Description=\"The reference allele\">";
	print "#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT";
}

# Ignore the header
NR==1 {next}

# For all other non-comment lines
/^[^#]/{
	mutation_id   = $1;
	donor_id      = $2;
	chromosome    = $9;
	start         = $10;
	end           = $11;
	ref_genome    = $13;
	mutation_type = $14;
	ref_allele    = $15;
	from_allele   = $16;
	to_allele     = $17;
	quality       = $18;
	tumor_allele  = "";
	alt_allele    = "";


	# ICGC puts seperate consequences on different lines, all we need is a mutation/chromosome/position for snpEff reannotation
	if (mutation_id==last_id) {next;}

	# Check the genome reference build
	if (ref_genome!="GRCh37")
	{
		print "This ICGC file genome refrence is not GRCh37 (hg19). Please check the 13th column of this input file." > "/dev/stderr";
		exit 1;
	}

	# Replace quality with "." if missing
	if (quality=="") {quality=".";}
	
	# Rename the mutations
	gsub(/single base substitution/, "SBS", mutation_type);
	gsub(/insertion of <=200bp/, "INS", mutation_type);
	gsub(/deletion of <=200bp/, "DEL", mutation_type);
	gsub(/multiple base sibstitution/, "MBS", mutation_type);

	if (mutation_type=="SBS" || mutation_type=="MBS")
	{
		tumor_allele = from_allele;
		alt_allele   = to_allele;
	}
	if (mutation_type=="INS")
	{
		start        = start-1;
		tumor_allele = "N";
		ref_allele   = "N";
		alt_allele   = "N"to_allele;
	}
	if (mutation_type=="DEL")
	{
		start        = start-1;
		tumor_allele = "N"from_allele;
		ref_allele   = "N"ref_allele;
		alt_allele   = "N";
	}

	# Write out the line
	print chromosome, start, mutation_id, tumor_allele, alt_allele, quality, "." , ".", sprintf("DONOR_ID=%s;TYPE=%s;REF_ALLELE=%s;", donor_id, mutation_type, ref_allele);
	last_id = mutation_id;
}
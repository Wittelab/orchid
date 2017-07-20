#!/usr/bin/awk -f

# Clinton Cario 
#  02/23/2017 -- Initial version


BEGIN { 
	last_id = "";
	FS="\t"; 
	OFS="\t";
}

# Ignore the header
NR==1 {next}

# For all other non-comment lines
/^[^#]/{
	mutation_id   = $1;
	donor_id      = $2;
	chromosome    = $9;
	position      = $10;
	ref_genome    = $13;
	mutation_type = $14;
	ref_allele    = $15;
	from_allele   = $16;
	to_allele     = $17;
	quality       = $18;

	# ICGC puts seperate consequences on different lines, all we need is a mutation/chromosome/position for snpEff reannotation
	if (mutation_id==last_id) {next;}

	# Check the genome reference build
	if (ref_genome!="GRCh37") 
	{
		print "This ICGC file genome refrence is not GRCh37 (hg19). Please check the 13th column of this input file." > "/dev/stderr";
		exit 1;
	}
	#if (mutation_type!="single base substitution") {next;}

	# Replace quality with "." if missing
	if (quality=="") {quality="."}
	# Remove whitespace from mutation_type
	gsub(/ /,"_",mutation_type);

	# Write out the line
	print chromosome, position, position+1, mutation_id;
	last_id = mutation_id;
}
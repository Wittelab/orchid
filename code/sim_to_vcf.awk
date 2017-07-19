#!/usr/bin/awk -f

# Clinton Cario 
#  03/08/2017 -- Initial version


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
	print "##fileformat=VCFv4.0";
	print "##INFO=<ID=ICGC_DONOR_ID,Number=1,Type=String,Description=\"The ICGC donor ID\">";
	print "##INFO=<ID=TYPE,Number=1,Type=String,Description=\"The type of mutation\">";
	print "##INFO=<ID=REF_ALLELE,Number=1,Type=String,Description=\"The reference allele\">";
	print "#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT";
}

/^##fileformat/{ next; }

# For all non-comment lines
/^[^#]/{
	chromosome    = $1;
	position      = $2;
	mutation_id   = $3;
	from_allele   = $4;
	to_allele     = $5;
	quality       = $6;
	filter        = $7;
	info          = $8;

	# The ref_allele is unknown, but OK to assume the from_allele
	ref_allele = from_allele;
	donor_id = "DOXXXXXX"

	if (mutation_id=="" || mutation_id==".")
		mutation_id = sprintf("S%s%04d",prefix,id)

	# Replace quality with "." if missing
	if (quality=="") {quality="."}
	# Replace filter with "." if missing
	if (filter=="") {filter="."}
	# Remove CpG annotation
	gsub(/CpG;/,"",info);

	# Rename the mutations
	if (length(from_allele)==1 && length(to_allele)==1) { mutation_type="SBS" }
	if (length(from_allele)>1 && length(to_allele)==length(from_allele)) { mutation_type="MBS" }
	if (length(from_allele) < length(to_allele)) { mutation_type="INS" }
	if (length(from_allele) > length(to_allele)) { mutation_type="DEL" }


	# Write out the line
	print chromosome, position, mutation_id, from_allele, to_allele, quality, "." , info, sprintf("DONOR_ID=%s;TYPE=%s;REF_ALLELE=%s;", donor_id, mutation_type, ref_allele);
	id++;
	# Get a new random prefix if this id space has been exhausted
	if (id>=9999) { prefix=toupper(sprintf("%04x",rand()*65536)); }
}
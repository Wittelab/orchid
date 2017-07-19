#!/usr/bin/env python -u
# Clinton Cario 
#    02/05/2016
#      Rewritten based on SSM_populator for memsql, no dependency on peewee and cleaner more efficient tables
#    02/09/2016
#      Fixed bug where single quotes ruin syntax by incorporating a strip_invalid function that sanitizes syntax inputs. (featurizer)
#    02/10/2016
#      Speed improvements in insert_data with multiple updates instead of case updates
#      Made verbose a class variable for more consistent usage
#      Added key_col as a special parameter so that keys other than the primary can be used to populate/annotate tables
#    02/17/2016
#      Fixed issue with pri_key in featurizer function 
#    02/26/2016
#      Added featurizer_continuous code
#    03/23/2016
#      Added command line verbose argument
#    03/23/2016
#      Modified DB populators a bit.
#    03/28/2016
#      Added retry to DB connect
#    03/30/2016
#      Faster row number query in _chunkify
#    04/04/2016
#      Fixed featurizer bug where multiple columns weren't being flagged
#    04/20/2016 ===~~
#      Added self.create_column scrub parameter to strip invalid characters and spaces
#    04/28/2016
#      Added safe_query/safe_get functions to prevent script failure on DB connection interruptions
#    05/04/2016
#      Modified some command line argument names
#    05/05/2016
#      Changed show_sql to be a formal parameter, added show_all_sql to for explicit insert/update queries, modified __init__ and populate functions
#    08/03/2016 
#      Added more verbose reporting
#    11/15/2016
#      Added pri_key_type and sensible defaults to generate BIGINT(10) sbs_id's for feature tables that don't use mutation_id... results in faster lookup using these keys
#    03/25/2017
#      Added metadata table population to facilitate downstream database lookups
#    03/25/2017
#      Merged populator functionality, renamed mutation_db
#      Faster featurize_category function using cached metadata and mutation key ranges instead of lookups
#    05/03/2017 
#      Renamed orchid_db
#    05/06/2017
#      Modified database table columns a bit
#    05/10/2017
#      Major changes to annotate
#      Removed featurize code
#    05/23/2017
#      Removed redundant safe_get() function
#    05/30/2017
#      All created feature tables are now lowercase (to be consistent between mysql and memsql databases)

# System libraries
import os, sys, re, argparse
from memsql.common import database
from urlparse import urlparse
from time import sleep
import json

from pprint import pprint


# Define Amino Acid Classes
aa_classes = {
    'G': 'Aliphatic',
    'A': 'Aliphatic',
    'V': 'Aliphatic',
    'L': 'Aliphatic',
    'M': 'Aliphatic',
    'I': 'Aliphatic',
    'S': 'Polar',
    'T': 'Polar',
    'C': 'Polar',
    'P': 'Polar',
    'N': 'Polar',
    'Q': 'Polar',
    'K': 'Positive',
    'R': 'Positive',
    'H': 'Positive',
    'D': 'Negative',
    'E': 'Negative',
    'F': 'Aromatic',
    'Y': 'Aromatic',
    'W': 'Aromatic',
    '*': 'Stop'
}

aa_three2one = {
    'Ala':'A',
    'Arg':'R',
    'Asn':'N',
    'Asp':'D',
    'Asx':'B',
    'Cys':'C',
    'Glu':'E',
    'Gln':'Q',
    'Glx':'Z',
    'Gly':'G',
    'His':'H',
    'Ile':'I',
    'Leu':'L',
    'Lys':'K',
    'Met':'M',
    'Phe':'F',
    'Pro':'P',
    'Ser':'S',
    'Thr':'T',
    'Trp':'W',
    'Tyr':'Y',
    'Val':'V',
    '*'  :'*',
    '?'  :'?'
}


class Manager():
  insert_size  = 5000
  max_attempts = 10
  host         = None
  port         = None
  user         = None
  password     = None
  database     = None
  verbose      = None
  show_sql     = None
  show_all_sql = None


  def __init__(self, db_uri, verbose=True, show_sql=False, show_all_sql=False):
    db_info           = urlparse(db_uri)
    self.host         = db_info.hostname
    self.port         = db_info.port
    self.user         = db_info.username
    self.password     = db_info.password
    self.database     = db_info.path.strip('/')
    self.verbose      = verbose
    self.show_sql     = show_sql
    self.show_all_sql = show_all_sql

  def create_tables(self, mut_table, cons_table):
    # Create the mutation table (DEFAULT NULLs make for muuuuch faster updates without crazy syntax)
    syntax = """
    CREATE TABLE IF NOT EXISTS `%s` (
        `%s_id`                       INT unsigned NOT NULL AUTO_INCREMENT, 
        `is_simulated`                BOOL DEFAULT NULL, 
        `mutation_id`                 VARCHAR(32) DEFAULT NULL, 
        `donor_id`                    VARCHAR(16) DEFAULT NULL, 

        `chromosome`                  ENUM('1','10','11','12','13','14','15','16','17','18','19','2','20','21','22','3','4','5','6','7','8','9','X','Y') DEFAULT NULL, 
        `start`                       INT unsigned DEFAULT NULL, 
        `end`                         INT unsigned DEFAULT NULL,
        `mutation_type`               ENUM('SBS','MBS','INS','DEL') DEFAULT NULL, 
        `reference_genome_allele`     VARCHAR(200) DEFAULT NULL, 
        `mutated_from_allele`         VARCHAR(200) DEFAULT NULL, 
        `mutated_to_allele`           VARCHAR(200) DEFAULT NULL, 

        PRIMARY KEY                 (`%s_id`), 
        INDEX                       (`mutation_id`), 
        INDEX                       (`donor_id`), 
        INDEX                       (`chromosome`, `start`), 
        INDEX                       (`is_simulated`) 
    );
    """ % tuple([mut_table]*3)
    with self.get_connection() as db:
        db.execute(syntax)


    # Create the consequence table
    syntax = """
    CREATE TABLE IF NOT EXISTS `%s` (
        `%s_id`                       INT unsigned NOT NULL AUTO_INCREMENT, 
        `mutation_id`                 VARCHAR(32) DEFAULT NULL, 
        `impact`                      ENUM('HIGH','MODERATE','LOW','MODIFIER') DEFAULT NULL, 
        `gene_id`                     VARCHAR(64) DEFAULT NULL, 
        `gene_name`                   VARCHAR(64) DEFAULT NULL, 
        #`feature_type`                VARCHAR(64) DEFAULT NULL, 
        #`feature_id                   VARCHAR(64) DEFAULT NULL, 
        `transcript_biotype`          VARCHAR(64) DEFAULT NULL,
        `consequence_type`            VARCHAR(64) DEFAULT NULL, 
        `cds_position`                INT unsigned DEFAULT NULL, 
        `aa_position`                 INT unsigned DEFAULT NULL, 

        `aa_from_allele`              CHAR(1) DEFAULT NULL, 
        `aa_to_allele`                CHAR(1) DEFAULT NULL, 
        `aa_from_class`               VARCHAR(10) DEFAULT NULL, 
        `aa_to_class`                 VARCHAR(10) DEFAULT NULL, 
        `aa_class_change`             VARCHAR(24) DEFAULT NULL, 

        PRIMARY KEY                 (`%s_id`), 
        INDEX                       (`mutation_id`), 
        INDEX                       (`gene_id`) 
    );
    """ % tuple([cons_table]*3)
    with self.get_connection() as db:
        db.execute(syntax)


    # Create the consequence table
    syntax = """
    CREATE TABLE IF NOT EXISTS `metadata` (
        `metadata_id`       INT unsigned NOT NULL AUTO_INCREMENT,
        `metakey`           VARCHAR(32) DEFAULT NULL, 
        `metavalue`         TEXT DEFAULT NULL, 
        PRIMARY KEY       (`metadata_id`)
    );
    """
    with self.get_connection() as db:
        db.execute(syntax)

  ## To draw a progress bar
  def progress_bar(self, cur,total,message="Parsing..."):
      progress = min(int(cur*100/total),100)
      sys.stdout.write("\rProgress: [{0:100s}] {1:3d}% [{2:d}/{3:d}] {4:50s}".format('=' * progress, progress, cur, total, message))
      sys.stdout.flush()

  def safe_query(self, query, ignore_errors=False, ignore_codes=None):
    for attempt in xrange(1,self.max_attempts+1):
      try:
        with self.get_connection() as db:
          return db.query(query)
        break;
      except Exception as e:
        print "Error: ", e.message, e.args
        if ignore_errors or e.args[0] in ignore_codes: 
          print "(Ignored)"
          return
        print "\tTrouble querying the database, retrying... (attempt: %d/%d)" % (attempt, self.max_attempts)
        sleep(attempt)
        continue
    sys.exit('Quering the database failed after repeated attempts, giving up.')

  def get_connection(self):
    for attempt in xrange(1,self.max_attempts+1):
      try:
        return database.connect(host=self.host, port=self.port, user=self.user, password=self.password, database=self.database)
        break;
      except Exception as e:
        print e.message, e.args
        if self.verbose: print "\tTrouble establishing a database connection, retrying... (attempt: %d/%d)" % (attempt, self.max_attempts)
        sleep(attempt)
        continue
    sys.exit('Establishing a database connection failed after 5 attempts, giving up.')

  def run_sql(self, syntax, success_msg="(OK)", error_msg="(Failed)", ignore_errors=False, ignore_codes=None):
    try:
      if self.show_sql: print syntax
      self.safe_query(syntax, ignore_errors, ignore_codes)
      if self.verbose and success_msg!=None: print success_msg
    except Exception as e:
      if self.verbose and error_msg!=None: print error_msg
      print e.message, e.args

  def strip_invalid(self, instr):
    return re.sub('[^0-9a-zA-Z]+', '_', instr)

  def create_table(self, table, pri_key, pri_key_type="VARCHAR(255)"):
    syntax  = "CREATE TABLE IF NOT EXISTS %s (%s %s NOT NULL UNIQUE, PRIMARY KEY (%s))" % (table.lower(), pri_key, pri_key_type, pri_key)
    success = "The '%s' table was created" % table
    error   = "Creation failed. Please check table parameter and database connection"
    self.run_sql(syntax, success, error)

  def create_column(self, table, column, sql_type="VARCHAR(255) DEFAULT NULL", scrub=True):
    if scrub:
      column = self.strip_invalid(column).replace(' ','_').lower()
    syntax  = "ALTER TABLE %s ADD `%s` %s" % (table, column, sql_type)
    success = "The '%s' column was created" % column
    error   = "Column exists or creation failed. Please check table and column parameters and database connection"
    self.run_sql(syntax, success, error, ignore_codes=[1060]) # Ignore column exists error code

  def delete_table(self, table):
    syntax  = "DROP TABLE %s" % table
    success = "The '%s' table was dropped." % table
    error   = "Table deletion failed. Please check table name and database connection"
    self.run_sql(syntax, success, error)


  def delete_column(self, table, column):
    syntax  = "ALTER TABLE %s DROP COLUMN %s" % (table, column)
    success = "The '%s' column was dropped." % column
    error   = "Column deletion failed. Please table and column names and database connection"
    self.run_sql(syntax, success, error)


  def reset_table(self, table, pri_key):
    self.delete_table(table)
    self.create_table(table, pri_key)


  def reset_column(self, table, column, sql_type):
    self.delete_column(table, column)
    self.create_column(table, column, sql_type)

  def create_if_needed(self, table, pri_key=None, pri_key_type="VARCHAR(255)", column=None, sql_type="VARCHAR(255)"):
    # Try to figure out the primary key if not provided
    pri_key = pri_key if pri_key!=None else "%s_id" % (table)

    self.create_table(table, pri_key, pri_key_type)
    self.create_column(table, column, sql_type=sql_type)

  # Define a populator function 
  # data is [{key: value}] where key is a column name and value is the row entry
  # Multiple columns can be specified, and values should have correspondence in order
  def populate_table(self, data, table, pri_key, verbose=None, show_sql=None, show_all_sql=None):
    if verbose      == None: verbose = self.verbose
    if show_sql     == None: show_sql = self.show_sql
    if show_all_sql == None: show_all_sql = self.show_all_sql

    if verbose: print "Populating [ %s ]" % table
    if len(data)==0: print "No data, skipping"; return

    columns = data[0].keys()
    non_key_columns = [ col for col in columns if col != pri_key ]

    batch = 0
    for i in xrange(0, len(data), self.insert_size):
      batch += 1
      for attempt in xrange(1,self.max_attempts+1):
        try:
          if verbose: print "Inserting batch [%d], attempt [%d]..." % (batch, attempt),
          # Subset the insert values for this chunk
          values  = [ "('%s')" % ("','".join([ str(entry[col]) for col in columns ])) for entry in data[i:i+self.insert_size] ]
          syntax =  "INSERT INTO `%s` (`%s`) VALUES %s " % (table, "`,`".join(columns), ",".join(values))
          syntax += "ON DUPLICATE KEY UPDATE " + ", ".join(["`%s`=VALUES(`%s`)" %(col, col) for col in non_key_columns]) + ";"
          syntax = re.sub("'None'",'NULL',syntax, flags=re.MULTILINE)
          #if show_sql: print "\t"+syntax[0:200]+"    ......    \t"+syntax[-50:]
          if show_all_sql: print syntax
          #if verbose: print "\tAttempting to execute"
          self.safe_query(syntax)
          if verbose: print "(OK)"
          break
        except Exception as e:
          if verbose: print "(Failed)"
          if attempt >= self.max_attempts: 
            print e.message, e.args
            sys.exit(0)
          continue

  # metakey: 'counts', 'distincts', or other
  # metavalue: JSONified string of values
  def update_metadata(self, metakey, metavalue):
    # Check if an entry exists
    syntax = "SELECT * FROM metadata WHERE `metakey`='%s';" % (metakey)
    result = self.safe_query(syntax)
    if len(result)>0:
      # Update
      id_ = result[0]['metadata_id']
      if metakey=='count':
        value = json.dumps(int(result[0]['metavalue']) + value)
      data = [{'metadata_id':id_, 'metakey':metakey, 'metavalue':metavalue}]
      self.populate_table(data, 'metadata', 'metadata_id', verbose=False)
    else:
      # Insert
      data = [{'metakey':metakey, 'metavalue':metavalue}]
      self.populate_table(data, 'metadata', 'metadata_id', verbose=False)


  # Fasta file should look like:
  ## >primary_key
  ## sequence
  # The 'processor_*' arguments are optional but otherwise should be lambda functions to apply to the description (primary key) and sequence fields
  # Note that the '>' of the description field is parsed out
  def parse_fasta(self, source_file, acceptable=None, id_processor=None, seq_processor=None):
    if self.verbose: print "Parsing fasta...",
    if source_file == None:
      sys.exit("You must specify a source file to parse as the first argument!")

    # Get the data
    data = []
    with open(source_file, 'r') as ifh:
      for line in ifh:
        pri_key = line.replace(">","").strip()
        pri_key = id_processor(pri_key) if (id_processor!=None) else pri_key
        seq     = ifh.next().strip()
        seq     = seq_processor(seq) if (seq_processor!=None) else seq
        if acceptable and (seq not in acceptable): continue
        data.append({'key':pri_key, 'val':seq})
        #print pri_key, seq
    if self.verbose: print "done."
    return data


  def parse_flatfile(self, source_file, acceptable=None, id_col=0, data_col=1, delimiter="\t", id_processor=None, data_processor=None):
    if self.verbose: print "Parsing flatfile...",
    if source_file == None:
      sys.exit("You must specify a source file to parse as the first argument!")
    
    data = []
    with open(source_file, 'r') as ifh:
      for line in ifh:
        line = line.strip().split(delimiter)
        pri_key = line[id_col]
        pri_key = id_processor(pri_key) if (id_processor!=None) else pri_key
        val     = line[data_col]
        val     = data_processor(val) if (data_processor!=None) else val
        if acceptable and (val not in acceptable): continue
        data.append({'key':pri_key, 'val':val})
    if self.verbose: print "done."
    return data


  def populate_annotations(self, data, dst_tbl, dst_val_col, dst_val_sql="VARCHAR(255)", dst_id_col=None, dst_id_sql="VARCHAR(255)", ):
    # Try to figure out the primary key if not provided
    dst_id_col = dst_id_col if dst_id_col!=None else "%s_id" % (dst_tbl)
    # Create the table and column (if needed)
    self.create_if_needed(dst_tbl, dst_id_col, dst_id_sql, dst_val_col, dst_val_sql)

    # Don't do anything if data wasn't provided
    if data==None or len(data)==0:
      print "No data to insert"
      return

    # Determine whether or not to collect distinct categories
    collect = False if dst_val_sql[0:3]=='INT' or dst_val_sql[0:5]=='FLOAT' else True

    # Do the inserts (populator method)
    import_data = []
    distincts = set()
    feature_type = None
    for insert in data:
      # Prepare this data for import 
      import_data.append({ dst_id_col: insert['key'], dst_val_col: insert['val'] })
      # Determine if the value is not numeric
      try:
        float(insert['val'])
        feature_type = 'numeric'
      except:
        # Only for non numeric types
        distincts = distincts.union(set([insert['val']]))
        feature_type = 'categorical'

    self.populate_table(import_data, dst_tbl, dst_id_col)
    if self.verbose: print "Update successful"


  def populate_mutations(self, infile, chunk_size, simulated, mut_types, mut_table, cons_table, acceptable):
    num_lines = sum(1 for line in open(infile,'r'))
    infile = open(infile,'r')

    ## Store all relational data into these three structures, corresponding to the tables
    ## Donor has many mutations (1 to many)
    ## Mutation (SSM) has many consequences (1 to many)
    ## Store the primary key of one side of relationship as
    ##   the foreign key in table on many side of relationship
    mutations          = []
    consequences       = []
    last_donor         = None
    last_mutation      = None
    line_no            = 0
    mutation_count     = 0
    consequence_count  = 0
    distincts          = {}

    for line in infile:
        line_no += 1

        # Ignore header lines
        if line.startswith("#") or line.startswith("\n"):
            if self.verbose: self.progress_bar(line_no,num_lines,"Skipped header line...")
            continue

        # Show the progress bar if in verbose mode
        if (line_no % 100 == 0) and self.verbose:
          self.progress_bar(line_no,num_lines)

        # Give friendly names to all columns, trying to get the etcetera column if it exists
        # The workflow will add this column, which is the FORMAT column in VCF, but use for a custom purpose here
        entry = line.strip("\n").split("\t")
        etcetera = None 
        try:
            chromosome, start, mutation_id, from_allele, to_allele, quality, filt, info, etcetera = entry
        except ValueError:
            chromosome, start, mutation_id, from_allele, to_allele, quality, filt, info = entry

        # Parse etcetera information
        ref_allele, donor_id, mutation_type = (None, None, None)
        if etcetera!=None and etcetera!='.':
            ref_allele = re.match(r'^.*REF_ALLELE=([\w-]+);.*$', etcetera, re.M)
            if ref_allele: ref_allele = ref_allele.group(1)
            donor_id = re.match(r'^.*DONOR_ID=([^;]+);.*$', etcetera, re.M)
            if donor_id: donor_id = donor_id.group(1)
            mutation_type = re.match(r'^.*TYPE=([\w_]+);.*$', etcetera, re.M)
            if mutation_type: mutation_type = mutation_type.group(1)
        # Try sensible defaults if there are any issues
        ref_allele = ref_allele or from_allele
        donor_id   = donor_id or "DOXXXXXX"
        mutation_type = mutation_type or None
        if mutation_type == None:
          if from_allele == '-':
              mutation_type = 'INS'
          elif to_allele == '-':
              mutation_type = 'DEL'
          elif len(from_allele)==1 and len(to_allele)==1:
              mutation_type = 'SBS'
          elif len(from_allele)>1 or len(to_allele)>1:
              mutation_type = 'MBS'

        # Skip non-requested mutation types
        if mutation_type and mut_types and mutation_type not in mut_types:
            if self.verbose: self.progress_bar(line_no,num_lines,"Skipped non requested mutation type (%s)..." % mutation_type)
            continue
        # Skip all non standard chromosomes 
        chromosome = chromosome.strip('chr')
        if chromosome not in map(str,range(1,23))+['X','Y']:
            if self.verbose: self.progress_bar(line_no,num_lines,"Skipped invalid chromosome (%s)..." % chromosome)
            continue

        # Calculate the end position (not available from VCF)
        end = start
        if mutation_type in ("MBS", "DEL"):
          end = int(start) + len(ref_allele)-1

        # Reduce indel size to 200 (as per ICGC claims)
        ref_allele  = ref_allele[0:200]
        from_allele = from_allele[0:200]
        to_allele   = to_allele[0:200]

        # Make up a mutation id if needed
        if mutation_id=='.':
            mutation_id = str(chromosome) + '_' + str(start) + '_' + str(end)

        # Add the mutation to the population list!
        if not (mutation_id == last_mutation):
            mutations.append({
                    'is_simulated'                     : (1 if simulated else 0),
                    'mutation_id'                      : mutation_id,
                    'donor_id'                         : donor_id,
                    'chromosome'                       : chromosome,
                    'start'                            : int(start),
                    'end'                              : int(end),
                    'mutation_type'                    : mutation_type,
                    'reference_genome_allele'          : ref_allele,
                    'mutated_from_allele'              : from_allele,
                    'mutated_to_allele'                : to_allele,
            })
            mutation_count += 1
        # Parse the info consequences
        for shard in info.split(";"):
            if shard[0:4]=='ANN=':
                for splinter in shard.split(','):
                    splinter = splinter.replace('ANN=','')
                    allele, consequence, impact, name, gene_id, feature_type, feature_id, \
                    transcript_biotype, rank_total, dna_position, aa_position, cDNA_position_len, \
                    CDS_position_len, protein_position_len, distance, errors = splinter.split('|')

                    # Skip entries with consequences that aren't considered acceptable
                    if acceptable and (consequence not in acceptable): 
                        continue
                    # Truncate the consequence_type if very long (database doesn't like this)
                    if len(consequence) >= 32: consequence=consequence[0:31]
                    if consequence == 'intergenic_region':
                        gene_id = None
                        name = None

                    # Amino acid information 
                    aa_pos, aa_from, aa_to, aa_from_class, aa_to_class, aa_class_change = (None, None, None, None, None, None)
                    aa_info = re.match(r'^p\.([a-zA-Z\*]+)(\d+)([a-zA-Z\*\?]*)$', aa_position, re.M)
                    if aa_info:
                        aa_pos  = aa_info.group(2)
                        aa_from = aa_info.group(1)
                        aa_to   = aa_info.group(3)
                        try:
                            aa_from_class = aa_classes[aa_three2one[aa_from]]
                        except:
                            pass
                        try:
                            aa_to_class = aa_classes[aa_three2one[aa_to]]
                        except:
                            pass
                        try:
                            if aa_from_class != None and aa_to_class != None:
                                aa_class_change = aa_from_class + " => " + aa_to_class
                                if aa_from_class == aa_to_class: aa_class_change = "Unchanged"
                        except:
                            pass

                    # The CDS position
                    cds_position = None if CDS_position_len==None else CDS_position_len.split("/")[0]

                    # Add the consequence to the population list!
                    this = {
                            'mutation_id'                      : mutation_id,
                            'impact'                           : impact or None,
                            'gene_id'                          : gene_id or None,
                            'gene_name'                        : name or None,
                            #'feature_type'                     : feature_type or None,
                            #'feature_id'                       : feature_id or None,
                            'transcript_biotype'               : transcript_biotype or None,
                            'consequence_type'                 : consequence or None,
                            'cds_position'                     : cds_position or None,
                            'aa_position'                      : aa_pos or None,
                            'aa_from_allele'                   : aa_from or None,
                            'aa_to_allele'                     : aa_to or None,
                            'aa_from_class'                    : aa_from_class or None,
                            'aa_to_class'                      : aa_to_class or None,
                            'aa_class_change'                  : aa_class_change or None,
                    }
                    if this not in consequences:
                        consequences.append(this)
                        consequence_count +=1


        # Do a bulk insert if we've collect at least chunk_size lines and have finished a mutation
        if ((last_mutation != mutation_id) and (len(consequences) >= chunk_size)) or (line_no>=num_lines-1):
            #print "Populating... %s" % line_no
            if self.verbose: self.progress_bar(line_no,num_lines,"Populating Mutations...")
            self.populate_table(mutations, mut_table, "%s_id"%mut_table, verbose=False)
            if self.verbose: self.progress_bar(line_no,num_lines,"Populating consequences...")
            self.populate_table(consequences, cons_table, "%s_id"%cons_table, verbose=False)
            # Reset chunk variables
            mutations    = []
            consequences = []

        last_mutation = mutation_id
        last_donor    = donor_id

    if self.verbose: self.progress_bar(num_lines,num_lines)
    print "\nAdding metadata table..."
    data = []

    # Update table counts
    self.update_metadata(
                          metakey = "%s_count" % mut_table, 
                          metavalue = mutation_count
                        )
    self.update_metadata(
                          metakey = "%s_count" % cons_table, 
                          metavalue = consequence_count
                        )

    print "Completed Successfully."
    # Close everything
    infile.close()


if __name__ == "__main__":
  # =============================
  # Parameter Definition
  # ---------
  parser = argparse.ArgumentParser()
  subparsers = parser.add_subparsers(help='commands', dest="command")

  #### The populate command
  populate = subparsers.add_parser('populate',                                                             help='This command takes a SNPEff VCF file and populates a database.')
  populate.add_argument('-x',   '--connection',                 action='store',      dest='db_uri',        help='A database URI connection string (e.g. mysql://user:pass@host:port/DB) if $DATABASE is not defined.')
  populate.add_argument('-i',   '--input-file',                 action='store',      dest='infile',        help='A tab delimited file of SNPEff VCF or simulated mutations.')
  populate.add_argument('-k',   '--chunk-size',                 action='store',      dest='chunk_size',    help='About how many mutation consequences should be collected before a database insert (actual insert depends on when the current donor finishes when this limit is reached.')
  populate.add_argument('-s',   '--simulated',                  action='store_true', dest='simulated',     help='Whether the imported data is simulated (will set simulated flag in database and ignore metadata fields.')
  populate.add_argument('-t',   '--mutation-types',             action='store',      dest='mut_types',     help='The type of mutation(s) to populate (space-delimit list items; default: accept all).', nargs='+')
  populate.add_argument('-m',   '--mutation-table',             action='store',      dest='mut_table',     help='The name of the table to store mutations of type [mutation-type] (default: ssm).')
  populate.add_argument('-c',   '--consequence-table',          action='store',      dest='cons_table',    help='The name of the table to store mutation consequences (default: consequence).')
  populate.add_argument('-A',   '--acceptable-consequences',    action='store',      dest='acceptable',    help='A JSON string containing a list of acceptable consequences.')
  populate.add_argument('-v',   '--verbose',                    action='store_true', dest='verbose',       help='Whether to be verbose and display status on the command line.', default=True)
  populate.add_argument('-vv',  '--show-sql',                   action='store_true', dest='show_sql',      help='Whether to show sql statements, excluding insert/update queries.', default=True)
  populate.add_argument('-vvv', '--show-all-sql',               action='store_true', dest='show_all_sql',  help='Whether to show sql statements, inclusing insert/update queries.', default=False)

  #### The annotate command
  # Input file options
  annotate = subparsers.add_parser('annotate',                                                                   help='Populate the database with feature information.')
  annotate.add_argument('-i',   '--source-file',                 action='store',      dest='src_file',           help='The name of the source file used to annotate the destination table.')
  annotate.add_argument('-t',   '--source-type',                 action='store',      dest='src_type',           help='The source file type: \'flatfile\' or \'fasta\'.', choices=['fasta', 'flatfile'])
  annotate.add_argument('-sic', '--source-id-column',            action='store',      dest='src_id_col',         help='The column in the source file that corresponds to the mutation id. 0-indexed, defaults to 0.')
  annotate.add_argument('-svc', '--source-value-column',         action='store',      dest='src_val_col',        help='The column in the source data file that corresponds to the value to insert. 0-indexed, defaults to 1.')
  annotate.add_argument('-D',   '--delimiter',                   action='store',      dest='delimiter',          help='The field delimiter for the source file. Defaults to tab.')
  # Database table options
  annotate.add_argument('-x',   '--connection',                  action='store',      dest='connection',         help='A database URI connection string (e.g. mysql://user:pass@host:port/DB) if $DATABASE is not defined.')
  annotate.add_argument('-d',   '--destination-table',           action='store',      dest='dst_tbl',            help='The name of the destination table. Defaults to \'ssm\'.')
  annotate.add_argument('-dic', '--destination-id-column',       action='store',      dest='dst_id_col',         help='The primary key column name of [destination-table]. Defaults to [destination-table]_id.')
  annotate.add_argument('-dis', '--destination-id-sql',          action='store',      dest='dst_id_sql',         help='The SQL type of id (primary key). Defaults to \'BIGINT\'.')
  annotate.add_argument('-c',   '--destination-value-column',    action='store',      dest='dst_val_col',        help='The column name in [destination-table] for the inserted values. Defaults to \'values\'.')
  annotate.add_argument('-dvs', '--destination-value-sql',       action='store',      dest='dst_val_sql',        help='The SQL type of inserted values. Defaults to \'VARCHAR(255)\'.')
  # Processing options
  annotate.add_argument('-A',   '--acceptable',                  action='store',      dest='acceptable',         help='A list of acceptable values for the value column in the destination table (space-delimit list items; default: accept all).', nargs='+')
  annotate.add_argument('-I',   '--id-processor',                action='store',      dest='id_processor',       help='Python lambda function as a string that will be applied to the id column before inserting into the destination table. Defaults to None.')
  annotate.add_argument('-S',   '--sequence-processor',          action='store',      dest='seq_processor',      help='Python lambda function as a string that will be applied to fasta sequence entries before inserting into the destination table. Defaults to None.')
  annotate.add_argument('-V',   '--value-processor',             action='store',      dest='val_processor',      help='Python lambda function as a string that will be applied to value column entries before inserting into the destination table. Defaults to None.')
  # Debug options 
  annotate.add_argument('-v',   '--verbose',                     action='store_true', dest='verbose',            help='Whether to be verbose and display status on the command line.')
  annotate.add_argument('-vv',  '--show-sql',                    action='store_true', dest='show_sql',           help='Whether to show sql statements, excluding insert/update queries.')
  annotate.add_argument('-vvv', '--show-all-sql',                action='store_true', dest='show_all_sql',       help='Whether to show sql statements, inclusing insert/update queries.')
  # JSON list of options 
  annotate.add_argument('-j',   '--json',                        action='store',      dest='json',               help='A JSON string of command arguments as a map')

  args = parser.parse_args()


  # =============================
  # POPULATE MUTATIONS
  # ---------
  if args.command=='populate':
    # Parse parameters from the command line or as given in the passed JSON string (preferring command line arguments)
    db_uri       = os.environ.get('DATABASE') or args.db_uri or extra.get('connection')
    infile       = args.infile
    chunk_size   = args.chunk_size or 20000
    simulated    = args.simulated or False
    mut_types    = args.mut_types or None
    mut_table    = args.mut_table or 'ssm'
    cons_table   = args.cons_table or 'consequence'
    acceptable   = args.acceptable
    verbose      = args.verbose
    show_sql     = args.show_sql
    show_all_sql = args.show_all_sql


    # Create a manager to parse the data ...
    manager = Manager(db_uri, verbose, show_sql, show_all_sql)
    print "Creating database tables..."
    manager.create_tables(mut_table, cons_table)
    print "Importing mutations..."
    manager.populate_mutations(infile, chunk_size, simulated, mut_types, mut_table, cons_table, acceptable)



  # =============================
  # ANNOTATE MUTATIONS
  # ---------
  if args.command=='annotate':
    extra = json.loads(args.json) if args.json else {}

    # Input file options
    src_file         = args.src_file or extra.get('source_file')
    src_type         = args.src_type or extra.get('source_type') or 'flatfile'
    src_id_col       = args.src_id_col or extra.get('source_id_column') or 0
    src_val_col      = args.src_val_col or extra.get('source_value_column') or 1
    delimiter        = args.delimiter or extra.get('delimiter') or "\t"
    # Database table options
    db_uri           = os.environ.get('DATABASE') or args.connection or extra.get('connection')
    dst_tbl          = args.dst_tbl or extra.get('destination_table') or 'ssm'
    dst_id_col       = args.dst_id_col or extra.get('destination_id_column') or dst_tbl+'_id'
    dst_id_sql       = args.dst_id_sql or extra.get('destination_id_sql') or 'BIGINT'
    dst_val_col      = args.dst_val_col or extra.get('destination_value_column') or 'values'
    dst_val_sql      = args.dst_val_sql or extra.get('destination_value_sql') or 'VARCHAR(255)'
    # Processing options
    acceptable       = args.acceptable or extra.get('acceptable') or None
    id_processor     = args.id_processor or extra.get('id_processor') or None
    seq_processor    = args.seq_processor or extra.get('sequence_processor') or None
    val_processor    = args.val_processor or extra.get('value_processor') or None
    # Debug options
    verbose          = args.verbose or extra.get('verbose') or False
    show_all_sql     = args.show_all_sql or extra.get('show_all_sql') or False
    show_sql         = args.show_sql or extra.get('show_sql') or False or show_all_sql

    # Convert data processor functions into python functions proper
    # These processors are not sanitized. 
    # ASSUMES CODE PASSED ON THE COMMAND LINE IS TRUSTED!
    try:
      seq_processor = eval(seq_processor)
    except:
      pass
    try:
      id_processor  = eval(id_processor)
    except:
      pass
    try:
      val_processor = eval(val_processor)
    except:
      pass

    # Create a manager to parse the data ...
    manager = Manager(db_uri, verbose, show_sql, show_all_sql)
    data = []
    if src_type=="fasta":
      print "Parsing fasta file: %s..." % (src_file)
      data = manager.parse_fasta(src_file, acceptable, id_processor, seq_processor)
    elif src_type=="flatfile":
      print "Parsing flatfile: %s..." % (src_file)
      data = manager.parse_flatfile(src_file, acceptable, src_id_col, src_val_col, delimiter, id_processor, val_processor)
    # ... and then insert it into the database
    print "Importing data to: %s.%s" % (dst_tbl, dst_val_col)
    manager.populate_annotations(data, dst_tbl, dst_val_col, dst_val_sql, dst_id_col, dst_id_sql)


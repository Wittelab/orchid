#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Clinton Cario 9/11/2015
#    02/08/2016
#      Rewritten based on SSM_populator for memsql, no dependency on peewee and cleaner more efficient tables
#    02/09/2016
#      Fixed bug where single quotes ruin syntax by incorperating a strip_invalid function that sanitizes syntax inputs

# Using Peewee DB_models.py, creates the following tables:
#
# KEGG_genes      KEGG_gene_alias      KEGG_pathway      KEGG_gene_path
# ----------      ---------------      ------------      --------------
# ensembl_id      ensembl_id           pathway_id        ensembl_id
# gene_name       gene_name            pathway_name      pathway_id
#                                      is_cancer

import os, sys, re, argparse
from restkit import Resource
from memsql.common import database
from random import random # To generate a fake Ensembl ID if none are found
from urlparse import urlparse
from time import sleep

## Get the command line arguments
parser = argparse.ArgumentParser(description='This script populates gene pathway membership tables in the specified database using information from KEGG (http://www.genome.jp/kegg/)')
parser.add_argument('-x', '--connection',   action='store',                       dest='db_uri',    help='A database URI connection string (e.g. mysql://user:pass@host:port/DB) if $DATABASE is not defined')
parser.add_argument('-v', '--verbose',      action='store_true', default=False,   dest='verbose',   help='Whether to be verbose and display status on the command line')
options = parser.parse_args()

# Define database parameters
db_uri     = options.db_uri or os.environ.get('DATABASE')
db_info    = urlparse(db_uri)

# Define the connection interface
def get_connection(host=db_info.hostname, port=db_info.port, user=db_info.username, password=db_info.password, db=db_info.path.strip('/'), verbose=options.verbose):
  for attempt in xrange(1,21):
    try:
      return database.connect(host=host, port=port, user=user, password=password, database=db)
    except:
      if verbose: print "\tTrouble establishing a database connection, retrying... (attempt: %d/20)" % attempt
      sleep(attempt*2)
      continue

def run_sql(sql, verbose=options.verbose):
    for attempt in xrange(0,21):
        try:
            with get_connection() as db:
                return db.execute(syntax)
        except:
          if verbose: print "\tTrouble running a query, retrying... (attempt: %d/20)" % attempt
          sleep(attempt*2)
          continue

def try_api(url, api, verbose=options.verbose):
    for attempt in xrange(0,21):
        try:
            with get_connection() as db:
                return api.get(url).body_string()
        except:
          if verbose: print "\tTrouble with the rest api, retrying... (attempt: %d/20)" % attempt
          sleep(attempt*2)
          continue



def strip_invalid(instr):
    return re.sub('[^0-9a-zA-Z ]+', '', instr)

# =============================
# Create the tables
# ---------
if options.verbose: print "Verbose mode on.\n(Re)creating tables..."

syntax = """
CREATE TABLE IF NOT EXISTS `kegg_gene` (
  kegg_gene_id    INT unsigned NOT NULL AUTO_INCREMENT,
  ensembl_id      CHAR(16) DEFAULT NULL,
  gene_name       CHAR(63) NOT NULL,
  PRIMARY KEY     (kegg_gene_id),
  KEY             (ensembl_id)
);
"""
run_sql(syntax)

syntax = """
CREATE TABLE IF NOT EXISTS `kegg_gene_alias` (
  kegg_gene_alias_id   INT unsigned NOT NULL AUTO_INCREMENT,
  ensembl_id           CHAR(16) DEFAULT NULL,
  gene_alias           CHAR(63) NOT NULL,
  PRIMARY KEY          (kegg_gene_alias_id),
  KEY                  (ensembl_id)
);
"""
run_sql(syntax)

syntax = """
CREATE TABLE IF NOT EXISTS `kegg_pathway` (
  kegg_pathway_id    CHAR(16) DEFAULT NULL, 
  pathway_name       CHAR(128) DEFAULT NULL, 
  PRIMARY KEY        (kegg_pathway_id)
);
"""
run_sql(syntax)

syntax = """
CREATE TABLE IF NOT EXISTS `kegg_gene_pathway` (
  kegg_gene_pathway_id  INT unsigned NOT NULL AUTO_INCREMENT,
  kegg_pathway_id       CHAR(16) DEFAULT NULL, 
  ensembl_id            CHAR(16) DEFAULT NULL,
  is_cancer             BOOL NOT NULL, 
  
  PRIMARY KEY     (kegg_gene_pathway_id), 
  KEY             (ensembl_id), 
  KEY             (kegg_pathway_id)
);
"""
run_sql(syntax)



if options.verbose: print "Querying KEGG REST API, please wait..."
api = Resource('http://rest.kegg.jp')

kegg_cancers = {
    'hsa05200': 'Pathways in cancer [PATH:ko05200]',
    'hsa05230': 'Central carbon metabolism in cancer [PATH:ko05230]',
    'hsa05231': 'Choline metabolism in cancer [PATH:ko05231]',
    'hsa05202': 'Transcriptional misregulation in cancers [PATH:ko05202]',
    'hsa05206': 'MicroRNAs in cancer [PATH:ko05206]',
    'hsa05205': 'Proteoglycans in cancer [PATH:ko05205]',
    'hsa05204': 'Chemical carcinogenesis [PATH:ko05204]',
    'hsa05203': 'Viral carcinogenesis [PATH:ko05203]',
    'hsa05210': 'Colorectal cancer [PATH:ko05210]',
    'hsa05212': 'Pancreatic cancer [PATH:ko05212]',
    'hsa05214': 'Glioma [PATH:ko05214]',
    'hsa05216': 'Thyroid cancer [PATH:ko05216]',
    'hsa05221': 'Acute myeloid leukemia [PATH:ko05221]',
    'hsa05220': 'Chronic myeloid leukemia [PATH:ko05220]',
    'hsa05217': 'Basal cell carcinoma [PATH:ko05217]',
    'hsa05218': 'Melanoma [PATH:ko05218]',
    'hsa05211': 'Renal cell carcinoma [PATH:ko05211]',
    'hsa05219': 'Bladder cancer [PATH:ko05219]',
    'hsa05215': 'Prostate cancer [PATH:ko05215]',
    'hsa05213': 'Endometrial cancer [PATH:ko05213]',
    'hsa05222': 'Small cell lung cancer [PATH:ko05222]',
    'hsa05223': 'Non-small cell lung cancer [PATH:ko05223]',
}


entry = 0
# Get all human pathways
results = try_api('/link/pathway/hsa', api).split('\n')
for result in results:
    entry = entry + 1
    if result == '': continue  # Skip the final blank entry
    
    # =============================
    # Get the gene and pathway information for this result
    # ---------
    gene, pathway = result.split('\t')
    pathway = pathway.replace('path:','')
    if options.verbose: print "Pathway:       %s\nGene:          %s" % (pathway, gene)
    gene_info = try_api('/get/'+gene, api) #.split('\n')
    #if options.verbose: print gene_info
    #if options.verbose: print gene_info[1].split('        ')[1:]

    # =============================
    # Get the pathway name
    # ---------
    path_name = "NOPATH" + str(entry)
    path_info = try_api('/get/'+pathway, api) #.split('\n')
    m = re.search("NAME (.*)\n", path_info)
    if m:
        path_name = m.groups()[0].lstrip(' ')
        path_name = path_name.replace(' - Homo sapiens (human)','')
        if options.verbose: print "Pathway Name:  %s" % (path_name)
    is_cancer = True if pathway in kegg_cancers.keys() else False

    # =============================
    # Get the gene name, aliases, and ensembl ID
    # ---------
    gene_names = ['NOGENE']
    ensembl_id = None
    m = re.search("NAME (.*)\n", gene_info)
    if m:
        gene_names = m.groups()[0].replace(' ','').split(',')
        if options.verbose: print "Gene Names:    %s" % (gene_names)
    # And ensembl ID
    m = re.search("Ensembl: (.*)\n", gene_info)
    if m:
        ensembl_id = m.groups()[0].split(' ')[0]
        if options.verbose: print "Ensembl ID:    %s" % (ensembl_id)
    gene_name = gene_names[0]
    # Attempt to fix bad ensembl ids by creating a dummy ID
    if ensembl_id == None:
        ensembl_id = "NOID_" + gene_name
    
    # =============================
    # Try to save this entry 
    # ---------
    #try:
    # Save gene name and ensembl ID
    syntax = "INSERT IGNORE INTO `kegg_gene` (ensembl_id, gene_name) VALUE ('%s', '%s');" % (strip_invalid(ensembl_id), strip_invalid(gene_names[0]))
    run_sql(syntax)
    # Save aliases (if any)
    if len(gene_names)>1:
        for alias in gene_names[1:]:
            syntax = "INSERT IGNORE INTO `kegg_gene_alias` (ensembl_id, gene_alias) VALUE ('%s', '%s');" % (strip_invalid(ensembl_id), strip_invalid(alias))
            run_sql(syntax)
    # Create pathway if it doesn't exist
    syntax = "INSERT IGNORE INTO `kegg_pathway` (kegg_pathway_id, pathway_name) VALUE ('%s', '%s');" % (strip_invalid(pathway), strip_invalid(path_name))
    #syntax = re.sub('^\s+','',syntax, flags=re.MULTILINE).replace('\n','')
    run_sql(syntax)
    # Link the gene to the pathway
    syntax = "INSERT IGNORE INTO `kegg_gene_pathway` (ensembl_id, kegg_pathway_id, is_cancer) VALUE ('%s', '%s', %d);" % (strip_invalid(ensembl_id), strip_invalid(pathway), 1 if is_cancer else 0)
    #syntax = re.sub('^\s+','',syntax, flags=re.MULTILINE).replace('\n','')
    run_sql(syntax)
    if options.verbose: print "ENTRY SAVED"
    if options.verbose: print "============================================"














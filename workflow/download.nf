#!/usr/bin/env nextflow
/* 
 * Clinton Cario 11/11/2015
 *
 * Notes:
 *    This file is used to download and install some software and data prerequisites. 
 *    By executing this file, you agree to the software licenses for each of these packages, which are listed below.
 */


// =============================
// Required Software and Feature Data
// ---------

// Anaconda2 v4.4.0 2017 © Continuum Analytics, Inc. All Rights Reserved.
// https://www.continuum.io/downloads
// Anaconda is BSD licensed which gives you permission to use Anaconda commercially and for redistribution.
process pythonPackages {
    echo true

    '''
    set -x
    if ! [ -x "$(command -v conda)" ]; then
        echo "Installing Anaconda"
        wget https://repo.continuum.io/archive/Anaconda2-4.4.0-Linux-x86_64.sh
        bash Anaconda2-4.4.0-Linux-x86_64.sh -b -p $HOME/anaconda2
        echo \'PATH=$PATH:$HOME/anaconda2/bin\' >> ~/.bash_profile
        source ~/.bash_profile
    fi
    echo "Installing python modules"
    conda clean --index-cache
    conda config --add channels conda-forge
    conda install -y IPython numpy  pandas  seaborn  matplotlib scikit-learn scipy
    conda install -y dill category_encoders python-snappy mysql-connector-python
    pip install mysql-python
    pip install memsql
    pip install restkit
    echo "Python modules installed"
    '''
}

// SnpEff v3.6 © Pablo Cingolani
// SnpEff is open source, released as "LGPLv3". 
// http://snpeff.sourceforge.net/download.html
// NOTE: version 3.3 was used to call ICGC mutations, but this version does not currently have the ensembl GRCh37 database, so version 3.6 is used here instead.
// NOTE: The data directory is changed to a custom location. 
process getSNPEff {
    echo true

    '''
    # Make the data directory 
    mkdir -p $CODE_DIR/external
    cd $CODE_DIR/external
    [ -f snpEff/data/GRCh37.68/snpEffectPredictor.bin ] && echo "snpEFF already installed" && exit 0
    rm -rf snpEff
    echo "Installing SnpEff"
    wget -t 10 -Nc https://sourceforge.net/projects/snpeff/files/snpEff_v3_6_core.zip
    unzip snpEff_v3_6_core.zip
    rm snpEff_v3_6_core.zip
    cd snpEff
    # Change the data directory to the new location
    sed -i -e "s|./data/|$CODE_DIR/external/snpEff/data|" snpEff.config
    rm -f snpEff.config-e
    # Download the GRCh37 annotation database for snpEff
    wget http://downloads.sourceforge.net/project/snpeff/databases/v3_6/snpEff_v3_6_GRCh37.68.zip
    unzip snpEff_v3_6_GRCh37.68.zip
    echo "SnpEff installed"
    '''
}


// The CADD mutation simulator © University of Washington and Hudson-Alpha Institute for Biotechnology
// This software is released under a MIT license (license text available from the ZIP-archive)
// http://cadd.gs.washington.edu/simulator
// NOTE: This is referenced in the publication Kircher, Martin et al. 2014 "A General Framework for estimating the Relative Pathogenicity of Human Genetic Variants." Nature genetics 46(3): 310-15.
process getSimulator {
    echo true

    '''
    mkdir -p $CODE_DIR/external
    cd $CODE_DIR/external
    [ -d simulator ] && echo "Simulator already installed" && exit 0
    echo "Installing simulator"
    rm -rf simulator
    wget -t 10 -Nc http://cadd.gs.washington.edu/static/NG-TR35288_Supp_File1_simulator.zip
    unzip NG-TR35288_Supp_File1_simulator.zip
    rm -f NG-TR35288_Supp_File1_simulator.zip
    echo "Simulator installed"
    '''
}

// The hg19 (GRC37) reference genome
// Public Domain
// hgdownload.cse.ucsc.edu:goldenPath/hg19/chromosomes
process getHg19 {
    echo true

    '''
    echo "Downloading hg19";
    mkdir -p $HG19_DIR
    cd $HG19_DIR
    [ -f reference_genome.fa ] && echo "hg19 reference sequences already installed" && exit 0
    echo "Installing hg19"
    wget -t 10 -Nc -r -nd --no-parent hgdownload.cse.ucsc.edu:goldenPath/hg19/chromosomes
    # Prepare reference genome file:
    wget -t 10 -nc ftp://ftp.ensembl.org/pub/release-68/fasta/homo_sapiens/dna/Homo_sapiens.GRCh37.68.dna.toplevel.fa.gz
    gunzip Homo_sapiens.GRCh37.68.dna.toplevel.fa.gz || true
    # Creating reference genome
    ln -sf Homo_sapiens.GRCh37.68.dna.toplevel.fa reference_genome.fa
    echo "Indexing reference genome"
    samtools faidx reference_genome.fa
    echo "hg19 installed"
    '''
}





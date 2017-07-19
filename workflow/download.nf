#!/usr/bin/env nextflow
/* 
 * Clinton Cario 11/11/2015
 *
 * Notes:
 *    This file is not guarenteed to be functional, but serves as a guide on how software and features were obtained.
 */


process pythonPackages {
    '''
    wget https://repo.continuum.io/archive/Anaconda2-4.4.0-Linux-x86_64.sh
    bash Anaconda2-4.4.0-Linux-x86_64.sh -b -p $HOME/anaconda2
    conda config --add channels conda-forge
    conda install -y dill category_encoders python-snappy mysql-python mysql-connector-python
    '''
}

// =============================
// Required Software and Feature Data
// ---------

// Get the mutation simulator
// Taken from CADD 
// http://cadd.gs.washington.edu/simulator
// https://www.encodeproject.org/publications/3cae12ef-5327-4222-8e46-b14898803c2f/
process getSimulator {

    '''
    cd $CODE_DIR/external
    [ -d simulator ] && echo "Simulator already installed" && exit 0
    wget -t 10 -Nc http://cadd.gs.washington.edu/static/NG-TR35288_Supp_File1_simulator.zip
    unzip NG-TR35288_Supp_File1_simulator.zip
    # The simulator directory is created with all code
    '''
}

// Get the SNPEff software
// NOTE: version 3.3 was used to call ICGC mutations, but this version does not currently have the ensembl GRCh37 database, so version 3.6 is used here instead
process getSNPEff {

    '''
    [ -f $CODE_DIR/external/snpEff/GRCh37.68/snpEffectPredictor.bin ] && echo "snpEFF already installed" && exit 0
    # Make the data directory 
    mkdir -p $CODE_DIR/external/snpEff
    cd $CODE_DIR/external/snpEff
    wget -t 10 -Nc https://sourceforge.net/projects/snpeff/files/snpEff_v3_6_core.zip
    unzip snpEff_v3_6_core.zip
    rm snpEff_v3_6_core.zip
    cd snpEff
    # Change the data directory to the new location
    sed -i -e "s|./data/|$CODE_DIR/external/snpEff/|" snpEff.config
    # Download the GRCh37 annotation database for snpEff
    java -jar snpEff.jar download -v GRCh37.68
    '''
}



// =============================
// Required Input data
// ---------

// Mutations from the ICGC dataset
process getICGC {

    '''
    echo "ICGC data must be installed by hand into $FEATURE_DIR/icgc based on what you\'d like to model. Go to https://dcc.icgc.org/ and download simple somatic and copy number data."
    ## This has to be done by hand. Download all Whole Genome Simple Somatic Substitutions for the cancer of interest
    ## Extract to the /data/icgc directory
    '''
}

// The hg19 (GRC37) reference genome
process getHg19 {

    '''
    mkdir -p $HG19_DIR
    cd $HG19_DIR
    [ -f reference_genome.fa ] && echo "hg19 reference sequences already installed" && exit 0
    wget -t 10 -Nc -r -nd --no-parent hgdownload.cse.ucsc.edu:goldenPath/hg19/chromosomes
    # Prepare reference genome file:
    wget -t 10 -nc ftp://ftp.ensembl.org/pub/release-68/fasta/homo_sapiens/dna/Homo_sapiens.GRCh37.68.dna.toplevel.fa.gz
    gunzip Homo_sapiens.GRCh37.68.dna.toplevel.fa.gz || true
    ln -sf Homo_sapiens.GRCh37.68.dna.toplevel.fa reference_genome.fa
    samtools faidx reference_genome.fa
    '''
}


// =============================
// Get features that are accessed by tabix 
// ---------
// RAW:  "observed" (negative values) vs "simulated" (positive values). 
//       Raw values do have relative meaning, with higher values indicating that a variant is more likely to be simulated (or "not observed") and therefore more likely to have deleterious effects.
// SCALED: "PHRED-scaled" those values by expressing the rank in order of magnitude terms rather than the precise rank itself. 
//         For example, reference genome single nucleotide variants at the 10th-% of CADD scores are assigned to CADD-10, top 1% to CADD-20, top 0.1% to CADD-30, etc.
process getCADD {

    '''
    mkdir -p $FEATURE_DIR/cadd
    cd $FEATURE_DIR/cadd
    [ -f whole_genome_SNVs.tsv.gz.tbi ] && echo "CADD files already installed" && exit 0
    wget -t 10 -Nc http://krishna.gs.washington.edu/download/CADD/v1.3/whole_genome_SNVs.tsv.gz
    wget -t 10 -Nc http://krishna.gs.washington.edu/download/CADD/v1.3/whole_genome_SNVs.tsv.gz.tbi
    '''
}


process getDANN {

    '''
    mkdir -p $FEATURE_DIR/dann
    cd $FEATURE_DIR/dann
    [ -f DANN_whole_genome_SNVs.tsv.bgz.tbi ] && echo "DANN files already installed" && exit 0
    wget -t 10 -Nc https://cbcl.ics.uci.edu/public_data/DANN/data/DANN_whole_genome_SNVs.tsv.bgz
    wget -t 10 -Nc https://cbcl.ics.uci.edu/public_data/DANN/data/DANN_whole_genome_SNVs.tsv.bgz.tbi
    '''
}

process getFunSeq2 {

    '''
    mkdir -p $FEATURE_DIR/funseq2
    cd $FEATURE_DIR/funseq2
    [ -f hg19_wg_score.tsv.gz.tbi ] && echo "FunSeq2 files already installed" && exit 0
    wget -t 10 -Nc http://archive.gersteinlab.org/funseq2/hg19_wg_score.tsv.gz
    wget -t 10 -Nc http://archive.gersteinlab.org/funseq2/hg19_wg_score.tsv.gz.tbi
    '''
}


/*
 * Eigen: A spectral approach integrating functional genomic annotations for coding and noncoding variants
 * http://www.columbia.edu/~ii2135/information_eigen.html
 *
 * Note: Skipped
 *
process getEigen {
    input:
    val chr from 1..22
    
    """
    mkdir -p $FEATURE_DIR/Eigen
    cd $FEATURE_DIR/Eigen
    [ -f Eigen_hg19_0916_chr22.tab.bgz.tbi ] && echo "Eigen files already installed" && exit 0
    wget -t 10 -Nc https://xioniti01.u.hpc.mssm.edu/v1.0/EIGEN/Eigen_hg19_0916_chr${chr}.tab.bgz
    wget -t 10 -Nc https://xioniti01.u.hpc.mssm.edu/v1.0/EIGEN/Eigen_hg19_0916_chr${chr}.tab.bgz.tbi
    """
}
*/


/* 
 * Genomic Evolutionary Rate Profiling: GERP
 * http://mendel.stanford.edu/SidowLab/downloads/gerp/
 * 
 * Note: There is a problem extracting these files
process getGerp {

    '''
    mkdir -p $FEATURE_DIR/Gerp
    cd $FEATURE_DIR/Gerp
    [ -f hg19.GERP_scores.tar.gz ] && echo "Gerp++ files already installed" && exit 0
    wget -t 10 -Nc http://mendel.stanford.edu/SidowLab/downloads/gerp/hg19.GERP_scores.tar.gz
    tar -xzvf hg19.GERP_scores.tar.gz
    '''
}
*/


/*
 * GWAVA - Genome Wide Annotation of VAriants
 * https://www.sanger.ac.uk/sanger/StatGen_Gwava
 *
 * Note: File format requires too much overhead to parse, skipping...
 *
process getGWAVA {
    '''
    mkdir -p $FEATURE_DIR/GWAVA
    cd $FEATURE_DIR/GWAVA
    [ -f gwava_db_csv.tgz ] && echo "GWAVA files already installed" && exit 0
    wget -t 10 -Nc ftp://ftp.sanger.ac.uk/pub/resources/software/gwava/v1.0/annotated/gwava_db_csv.tgz
    tar -xzvf gwava_db_csv.tgz
    mv csv/* .
    rmdir csv
    '''
}
*/



// =============================
// Files that can be processed with bedtools/bedops but are broken into different cell types / annotation tracks
// ---------

// dbSuper Enhancer tracks, for all hg19 cell lines
// http://bioinfo.au.tsinghua.edu.cn/dbsuper/
process getDbSuper {

    '''
    mkdir -p $FEATURE_DIR/dbsuper
    cd $FEATURE_DIR/dbsuper
    [ -f all.bed ] && echo "SuperDB already installed." && exit 0
    ## Download Super Enhancer tracks
    wget -t 10 -Nc http://bioinfo.au.tsinghua.edu.cn/dbsuper/data/bed/hg19/all_hg19_bed.zip
    unzip all_hg19_bed.zip
    mv all_hg19_bed/* .
    rmdir all_hg19_bed
    ## Download and sort the merged file 
    wget -t 10 -Nc http://bioinfo.au.tsinghua.edu.cn/dbsuper/data/bed/hg19/all_hg19_bed.bed
    sort-bed all_hg19_bed.bed > all.bed
    '''
}


// Encode Tracks from the project page, version 2
// https://www.encodeproject.org/data/annotations/v2/
// Distal DNase Peaks:              https://www.encodeproject.org/files/ENCFF751YPI/@@download/ENCFF751YPI.bigBed
// Proximal DNase Peaks:            https://www.encodeproject.org/files/ENCFF860GYD/@@download/ENCFF860GYD.bigBed
// Distal H3K27ac annotations:      https://www.encodeproject.org/files/ENCFF786PWS/@@download/ENCFF786PWS.bigBed
// Distal H3K4me1 annotations:      https://www.encodeproject.org/files/ENCFF076KTT/@@download/ENCFF076KTT.bigBed
// Distal H3K9 annotations:         https://www.encodeproject.org/files/ENCFF690JTO/@@download/ENCFF690JTO.bigBed
// Proximal H3K4me3 annotations:    https://www.encodeproject.org/files/ENCFF140OFS/@@download/ENCFF140OFS.bigBed
// Proximal H3K9ac annotations:     https://www.encodeproject.org/files/ENCFF649LOF/@@download/ENCFF649LOF.bigBed
// Distal TF binding sites:         https://www.encodeproject.org/files/ENCFF787QYS/@@download/ENCFF787QYS.bigBed
// Proximal TF binding sites:       https://www.encodeproject.org/files/ENCFF029ZUJ/@@download/ENCFF029ZUJ.bigBed
def renames = [
            'ENCFF751YPI': 'distal_dnase', 
            'ENCFF860GYD': 'proximal_dnase', 
            'ENCFF786PWS': 'distal_h3k27ac', 
            'ENCFF076KTT': 'distal_h3k4me1', 
            'ENCFF690JTO': 'distal_h3k9', 
            'ENCFF140OFS': 'proximal_h3k4me', 
            'ENCFF649LOF': 'proximal_h3k9ac', 
            'ENCFF787QYS': 'distal_tfbs', 
            'ENCFF029ZUJ': 'proximal_tfbs', 
          ]
process getEncode {
    tag { renames[type_] }

    input:
    val type_ from renames.keySet()

    """
    mkdir -p \$FEATURE_DIR/encode
    cd \$FEATURE_DIR/encode
    [ -f ${renames[type_]}.bed ] && echo "Encode ${renames[type_]} already downloaded." && exit 0
    ## Download, convert, and sort each
    wget -t 10 -Nc https://www.encodeproject.org/files/${type_}/@@download/${type_}.bigBed
    bigBedToBed ${type_}.bigBed ${type_}.bed
    sort-bed ${type_}.bed > ${renames[type_]}.bed
    """
}


// RFECS: human enhancer tracks (Random forest generated enhancer sites for different human cell types)
// https://www.encodeproject.org/publications/3cae12ef-5327-4222-8e46-b14898803c2f/
process getRFECS {
    tag { cell_line }

    input:
    val cell_line from 'Gm12878','H1hesc','Helas3','Hepg2','Hmec','Hsmm','Huvec','K562','Nha','Nhdfad','Nhek','Nhlf'

    """
    mkdir -p \$FEATURE_DIR/rfecs
    cd \$FEATURE_DIR/rfecs
    [ -f ${cell_line}.enhancer.bed ] && echo "RFECS ${cell_line} already downloaded." && exit 0
    ## Download a RFECS enhancer track
    wget -t 10 -Nc http://yuelab.org/mouseENCODE/downloads/human_enhancers/${cell_line}.enhancer.final
    ## Sort the file for faster searching with bedtools
    sort -k1,1 -k2,2n -o ${cell_line}.enhancer.final ${cell_line}.enhancer.final
    ## Add a 1000bp window around the peak of the enhancer (publication is not clear on how large this window should be)
    awk 'BEGIN{FS="\t";OFS="\t"}{print \$1,\$2-500,\$2+500,"${cell_line}",\$3}' ${cell_line}.enhancer.final > ${cell_line}..bed
    ## Remove the original file
    rm ${cell_line}.enhancer.final
    """
}


// Segmentation: Each segment belongs to one of a few specific genomic "states" which is assigned an intuitive label. 
// http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeAwgSegmentation/
//  TSS     Bright Red  Predicted promoter region including TSS
//  PF      Light Red   Predicted promoter flanking region
//  E       Orange      Predicted enhancer
//  WE      Yellow      Predicted weak enhancer or open chromatin cis regulatory element
//  CTCF    Blue        CTCF enriched element
//  T       Dark Green  Predicted transcribed region
//  R       Gray        Predicted Repressed or Low Activity region
process getSegmentation {
    tag { cell_line }

    input:
    val cell_line from 'Gm12878','H1hesc','Helas3','Hepg2','Huvec','K562'
    

    """
    mkdir -p \$FEATURE_DIR/segmentation
    cd \$FEATURE_DIR/segmentation
    [ -f ${cell_line}.bed ] && echo "Segmentation ${cell_line} already downloaded." && exit 0
    ## Download and sort the segmentation tracks
    wget -t 10 -Nc http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeAwgSegmentation/wgEncodeAwgSegmentationCombined${cell_line}.bed.gz
    gunzip wgEncodeAwgSegmentationCombined${cell_line}.bed.gz
    ## Sort the file for faster searching with bedtools
    sort-bed wgEncodeAwgSegmentationCombined${cell_line}.bed > ${cell_line}.bed
    """
}


// =============================
// Files that can be processed with bedtools/bedops (UCSC Genome tracks-- (0-based))
// ---------

// dnase:
// http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeRegDnaseClustered/
process getDnase {

    '''
    mkdir -p $FEATURE_DIR/dnase
    cd $FEATURE_DIR/dnase
    [ -f dnase.bed ] && echo "Dnase already installed." && exit 0
    wget -t 10 -Nc ftp://hgdownload.cse.ucsc.edu/goldenPath/hg19/database/wgEncodeRegDnaseClusteredV3.txt.gz
    gunzip wgEncodeRegDnaseClusteredV3.txt.gz
    awk 'BEGIN{OFS="\t"; FS="\t"}{print $2,$3,$4,$5,$6,$7,$8,$9}' wgEncodeRegDnaseClusteredV3.txt > dnase_.txt
    sort-bed dnase_.txt > dnase.bed
    rm *.txt
    '''
}


// gwas: This track displays SNPS from published GWAS studies, collected in the Catalog of Published Genome-Wide Association Studies at the NNHGRI. 
// http://ucscbrowser.genap.ca/cgi-bin/hgTrackUi?g=gwasCatalog&hgsid=6989_rcRoiqtYoU8R124Q8tUiMyUMADtc
process getGwas {

    '''
    mkdir -p $FEATURE_DIR/gwas
    cd $FEATURE_DIR/gwas
    [ -f gwas.bed ] && echo "gwas already installed." && exit 0
    wget -t 10 -Nc ftp://hgdownload.cse.ucsc.edu/goldenPath/hg19/database/gwasCatalog.txt.gz
    gunzip gwasCatalog.txt.gz
    awk '{if ($4==$3){$4=$4+1}; print $2,$3,$4,$5,$6,$7,$8,$9,$10}' gwasCatalog.txt > gwas_.txt
    sort-bed gwas_.txt > gwas.bed
    rm *.txt
    '''
}


// PhyloP: Compute conservation or acceleration p-values based on an alignment and a model of neutral evolution.
// http://compgen.cshl.edu/phast/help-pages/phyloP.txt
// ftp://hgdownload.cse.ucsc.edu/goldenPath/hg19/phyloP100way/hg19.100way.phyloP100way.bw
process getPhyloP {


    '''
    mkdir -p $FEATURE_DIR/phylop
    cd $FEATURE_DIR/phylop
    [ -f phylop.bed ] && echo "PhyloP already installed." && exit 0
    rootPath="http://hgdownload.cse.ucsc.edu/goldenpath/hg19/phyloP46way/primates/"
    for i in `seq 1 22` X Y; \

        do \
            echo "Processing phyloP data for chromosome chr${i}..."; \
            wigFn="chr${i}.phyloP46way.primate.wigFix"; \
            url="${rootPath}/${wigFn}.gz"; \
            wget -qO- ${url} | gunzip -c - | wig2bed - > ${wigFn}.bed; \
        done
    echo "Combining..."
    ls chr* | sort | xargs cat >> phyloP46way.bed_
    mv phyloP46way.bed_ phyloP46way.bed
    bgzip phyloP46way.bed > phyloP46way.bed.gz
    tabix -p bed phyloP46way.bed.gz
    '''
}

// ReMap: An integrative ChIP-seq analysis of regulatory elements
// http://tagc.univ-mrs.fr/remap/
process getReMap {

    '''
    mkdir -p $FEATURE_DIR/remap
    cd $FEATURE_DIR/remap
    [ -f remap.bed ] && echo "ReMap already installed." && exit 0
    wget -t 10 -Nc http://tagc.univ-mrs.fr/remap/download/All/nrPeaks_all.bed.gz
    gunzip nrPeaks_all.bed.gz
    sort-bed nrPeaks_all.bed > remap.bed
    '''
}


/* 
// tfbs: 
// http://genome.ucsc.edu/cgi-bin/hgTables?hgsid=295670123&clade=mammal&org=Human&db=hg19&hgta_group=regulation&hgta_track=tfbsConsSites&hgta_table=0&hgta_regionType=genome&position=chr21%3A33031597-33041570&hgta_outputType=primaryTable&hgta_outFileName=
//
// Note: Remap used instead
process getTfbs {

    '''
    mkdir -p $FEATURE_DIR/tfbs
    cd $FEATURE_DIR/tfbs
    [ -f tfbs.bed ] && echo "TFBS already installed." && exit 0
    wget -t 10 -Nc ftp://hgdownload.cse.ucsc.edu/goldenPath/hg19/database/tfbsConsSites.txt.gz
    gunzip tfbsConsSites.txt.gz
    awk '{if ($4==$3){$4=$4+1}; print $2,$3,$4,$5,$6,$7,$8}' tfbsConsSites.txt > tfbs_.txt
    sort-bed tfbs_.txt > tfbs.bed
    rm *.txt
    '''
}
*/


// TargetScanS: TTargetScan predicts biological targets of miRNAs by searching for the presence of 8mer, 7mer, and 6mer sites that match the seed region of each miRNA (Lewis et al., 2005). 
// http://www.targetscan.org/
// ftp://hgdownload.cse.ucsc.edu/goldenPath/hg19/database/targetScanS.txt.gz
process getTargetScanS {

    '''
    mkdir -p $FEATURE_DIR/targetscans
    cd $FEATURE_DIR/targetscans
    [ -f targetScanS.bed ] && echo "TargetScanS already installed." && exit 0
    wget -t 10 -Nc ftp://hgdownload.cse.ucsc.edu/goldenPath/hg19/database/targetScanS.txt.gz
    gunzip targetScanS.txt.gz
    awk '{print $2,$3,$4,$5,$6,$7,$8,$9,$10}' targetScanS.txt > targetScanS_.txt
    sort-bed targetScanS_.txt > targetScanS.bed
    rm *.txt
    '''
}


// wgRNA: This track displays positions of four different types of RNA in the human genome:
//    * precursor forms of microRNAs (pre-miRNAs)
//    * C/D box small nucleolar RNAs (C/D box snoRNAs)
//    * H/ACA box snoRNAs
//    * small Cajal body-specific RNAs (scaRNAs)
// https://genome.ucsc.edu/cgi-bin/hgTrackUi?db=hg19&g=wgRna
process getWgrna {

    '''
    mkdir -p $FEATURE_DIR/wgrna
    cd $FEATURE_DIR/wgrna
    [ -f wgrna.bed ] && echo "wgRNA already installed." && exit 0
    wget -t 10 -Nc ftp://hgdownload.cse.ucsc.edu/goldenPath/hg19/database/wgRna.txt.gz
    gunzip wgRna.txt.gz
    awk '{print $2,$3,$4,$5,1,$7,$8,$9,$10}' wgRna.txt > wgrna_.txt
    sort-bed wgrna_.txt > wgrna.bed
    rm *.txt
    '''
}





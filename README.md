**NOTE: The hosted multi25 database is currently being migrated as part of IT infrastructure improvements at UCSF. There may be periodic downtime during the next week.**

<img src="http://wittelab.ucsf.edu/images/orchid.png" alt="Orchid" height=150px; align="right">

# orchid
A management, annotation, and machine learning system for analyzing cancer mutations.  
<br/>  

>NOTE: This code is still an early release and is being actively developed. Please report any issues using the [Issues](https://github.com/Wittelab/orchid/issues) tab and they will be fixed as soon as possible.

# Introduction

Please refer to the following publication for a detailed description of this software:  
Bioinformatics, btx709, [https://doi.org/10.1093/bioinformatics/btx709](https://doi.org/10.1093/bioinformatics/btx709)

or, for a quick and dirty explanation:  
  
<br />  


_What is orchid?_ 
  
The purpose of orchid is to facilitate machine learning of tumor genetic data. For example, you might be interested in sub-typing aggressive vs. non aggressive prostate cancer based on tumor mutational profiles derived from tumor sequence data.
  
<br />  

_What is a 'tumor mutational profile'?_
  
A mutational profile is the annotated set of mutations within a tumor. A typical tumor might contain thousands of mutations, but most are likely irrelevant to disease and arise simply due to an important hallmark of cancer-- an unstable genome. However, some of these mutations (be it one, ten, or even a hundred) may play important roles in carcinogenesis and/or be useful for identifying tumor characteristics, like aggressiveness. There is a growing body of research that concerns itself with finding and focusing only on biologically important mutations, but orchid takes the approach of analyzing all mutations in aggregate with machine learning algorithms to try to tease apart subtle patterns. Afterall, it has been demonstrated that even irrelevant mutations may encode important information about the underlying biology of the tumor (e.g., [trinucleotide signatures](https://goo.gl/6tHS7Q)).

<br />  


_What is meant by an 'annotated set of mutations'?_
  
An annotation is simply a numeric or ordinal value that can be associated with a particular mutation. For example, 'mutation A' may change the amino acid sequence of a protein, so we can annotate it as a 'non-synonymous single nucleotide polymorphism' or 'nsSNP'. On the other hand, 'mutation B' may not change the amino acid sequence, so we annotate it as a 'synonymous SNP'. In machine learning parlance, an annotation is called a _feature_. If we gather many mutations across a tumor (or tumors) and annotate each mutation across many features, we end up with a set of annotated mutations, or _tumor mutational profile_.

At this time, many regulatory and coding features of the human genome have been extensively cataloged, resulting in a wealth of data to mine. If we gather enough biological data, we can increase our understanding of each individual mutation and its possible role in cancer, or at least begin to see if patterns emerge from the data. A list of features used in our publication and available in our public database can be found here: [http://wittelab.ucsf.edu/orchid](http://wittelab.ucsf.edu/orchid).

Here's an example. If we arrange a set of mutations from a tumor in rows and corresponding feature values in columns, a mutational profile can be created and visualized:  
![Mutational Profile](http://wittelab.ucsf.edu/images/mutational_profiles.png)  

Here large feature values (or more 'severe' categories) are shown as more orange, while smaller (less 'severe') feature values are whiter. There is also a final column of sample labels, which is ultimately what we're interested in learning. In other words, this column's values are used to train supervised machine learning algorithms for the purpose of future sample classification. 


# Getting Started
1. Download this code and install prerequisites  
2. Obtain tumor and annotation data  
3. Build the database  
4. Perform machine learning  

Please refer to the [wiki](https://github.com/Wittelab/orchid/wiki) to begin! 


_NOTICE:_
_This software requires the use of other code and/or data that must be obtained with respect to its license or copyright. Generally speaking, this implies orchid's use is restricted to non-commercial activities. Orchid itself is licensed under the MIT license requiring only preservation of copyright and license notices. Please see the LICENSE file for more details._

# MultiscaleSP

How to use

To create a new model place the corpus in a subdirectory and use the name of the subdirectory 
when invoking commands. THe corpus must already be okenized (i.e. have spaces between tokens) 
and must be called "corpus".

Step 1 Create the vocab file

vocab <corpusname>

Step 2 Create the order count matrix

order <corpusname>

Step 3 Create the syntagmatic count matrices (Cowan, Miller, HoneyHasson and Kintsch)

syn <corpusname>

Step 4 Create or reset the parameters file

resetparams <corpusname>

- to see the current values of the parameters 

showparams <corpusname>

Step 5 Train the parameters

learn -i <numiterations> <corpusname>

Step 6 Test the model

- to get the model to complete a sequence given a context string use:

generate <corpusname> <prefix>

e.g. generate small "australia is"

- to get the current perplexity of the corpus

testmodel <corpusname>

- to fill in a specific pattern of slots embedded within a sequence use:

testpattern <corpusname> <pattern>

e.g. testpattern small "australia is _ huge"

will show the probabiities of tokens in the location of the "_" given the rest of the context.


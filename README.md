# runLMM


## Introduction

Here we provide almost all state of art LMM algoritms for user. We will maintain our code and update with more algorithms.

File Structure:

* **models/** -   main method and some general funtions for algoritms
* **utility/** -  other general funtions
* **runLMM.py**  -  main entry point of using the algorithms to work with your own data

## How to Use

### An Example Command

```
python runLMM.py -in data/mice.plink --lam 1
```
this command will run the program and specify the the lambda to be 1. 



Options:
  * **-h, --help**   

```
    show this help message and exit
```



  * **Data Options:**
```
    --choice=FILETYPE             choices of input file type, now we process plink, csv, npy files (plink:bed,fam; csv:.geno, .pheno, .marker; npy:snps,Y,marker)

    --in=FILENAME                 name of the input file
```


  * **Model Options:**
```
    --threshold=THRESHOLD         The threshold to mask the weak genotype relatedness

    --quiet=QUIET                 Run in quiet mode

    --missing=MISSING             Run without missing genotype imputation

    --lam=LAM                     The weight of the penalizer. If neither lambda or discovernum is given, cross validation will be run.

    --gamma=GAMMA                 The weight of the penalizer of GFlasso. If neither lambda or discovernum is given, cross validation will be run.

    --mau=MAU                     a parameter of the penalizer of GFlasso.

    --discovernum=DISCOVERNUM     the number of targeted variables the model selects. If neither lambda or discovernum is given, cross validation will be run.

    --dense=DENSE                 choose the density to run LMM-Select

    --lr=lr                       give the learning rate of all methods
```

  * **LMM Options:**
```
     --lmm=LMM_FLAG               The lmm method we can choose:Linear,Lmm,Lowranklmm,Lmm2,Lmmn,Bolt,Select,Ltmlm
```


  * **Penalty Options:**
```
   --penalty=PENALTY_FLAG         The penalty method we can choose:Mcp,Scad,Lasso,Tree,Group,Linear,Lasso2(Ridge Regression)
```

  * **Experiment Options:**
```
    --real=REAL_FLAG              run the experiment on real dataset or synthetic dataset

    --generation=GENERATION_FLAG  the way to generate synthetic dataset, you can choose normal, tree, group

    --normal=NORMALIZE_FLAG       whether to normalize data after lmm

    --warning=WARNING_FLAG        whether to show the warnings or not

    --seed=NUMBER_OF_SEED         how many random seeds you want to run
```




#### Data Support

* We currently supports NPY, CSV and binary PLINK files.

* Extensions to other data format can be easily implemented through `FileReader` in `utility/dataLoadear`. Feel free to contact us for the support of other data format.

## Installation
First, you need to download this repository by using :

```
git clone git@github.com:lebronlambert/Framework_bibm.git
```

Then, before you run the program, you need to install following package in your Python, that is :

- Numpy
- Scipy
- Pysnptool

You can install the dependency by using:

```
pip install -r requirement.txt
```

## Contact
[Xiang Liu](mailto:xiang.liu1995@gmail.com)

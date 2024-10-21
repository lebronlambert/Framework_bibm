# Cite information: For now, we have not write a paper for it.

__author__='Xiang Liu'


from optparse import OptionParser, OptionGroup

usage = """usage: %prog [options] -in fileName
This program provides the basic usage to runLMM, e.g:
python runLMM.py -in data/mice.plink --lam 1
        """
parser = OptionParser(usage=usage)
dataGroup = OptionGroup(parser, "Data Options")
LmmModelGroup = OptionGroup(parser, "LMM Options")
PenaltyModelGroup = OptionGroup(parser, "Penalty Options")
ModelGroup = OptionGroup(parser, "Model Options")
ExperimentGroup = OptionGroup(parser, "Experiment Options")

## data options
dataGroup.add_option("--choice", dest='fileType', default='plink', help="choices of input file type, now we process plink, csv, npy files (plink:bed,fam; csv:.geno, .pheno, .marker; npy:snps,Y,marker)")
dataGroup.add_option("--in", dest='fileName',default="", help="name of the input file")

## model options
ModelGroup.add_option("--lam", dest="lam",type=float,
                      help="The weight of the penalizer. If neither lambda or discovernum is given, cross validation will be run.")
ModelGroup.add_option("--gamma", dest="gamma", default=0.7,type=float,
                      help="The weight of the penalizer of GFlasso. If neither lambda or discovernum is given, cross validation will be run.")
ModelGroup.add_option("--mau", dest="mau", default=0.1,type=float,
                      help="a parameter of the penalizer of GFlasso.")
ModelGroup.add_option("--discovernum", dest="discovernum",type=int,
                      help="the number of targeted variables the model selects. If neither lambda or discovernum is given, cross validation will be run.")
ModelGroup.add_option("--threshold", dest="threshold", default=1.,type=float,
                      help="The threshold to mask the weak genotype relatedness")
ModelGroup.add_option('--quiet', dest='quiet', default=False, help='Run in quiet mode') #action='store_true',
ModelGroup.add_option('--missing',  dest='missing', default=False,help='Run without missing genotype imputation')
ModelGroup.add_option('--dense',  dest='dense', default=0.05,help='choose the density to run LMM-Select')
ModelGroup.add_option('--lr',  dest='lr', default=1e-6,type=float,help='give the learning rate of all methods')

## experiment options
ExperimentGroup.add_option("--real", dest="real_flag", default="True",help="run the experiment on real dataset or synthetic dataset")
ExperimentGroup.add_option("--generation",dest="generation_flag",default="normal",help="the way to generate synthetic dataset,you can choose normal, tree, group")
ExperimentGroup.add_option("--normal",dest="normalize_flag",default="False",help="whether to normalize data after lmm")
ExperimentGroup.add_option("--warning",dest="warning_flag",default="False",help="whether to show the warnings or not")
ExperimentGroup.add_option("--seed",dest="number_of_seed",default=1,type=int,help="how many random seeds you want to run")

## LMM model
LmmModelGroup.add_option("--lmm",dest="lmm_flag",default="Lmm",help="The lmm method we can choose:Linear,Lmm,Lowranklmm,Lmm2,Lmmn,Bolt,Select,Ltmlm ")

##penaltyMode
PenaltyModelGroup.add_option("--penalty",dest="penalty_flag",default="Lasso",help="The penalty method we can choose:Mcp, Scad, Lasso, Tree, Group, Linear, Lasso2(Ridge regression)")


## advanced options
parser.add_option_group(dataGroup)
parser.add_option_group(ModelGroup)
parser.add_option_group(LmmModelGroup)
parser.add_option_group(PenaltyModelGroup)
parser.add_option_group(ExperimentGroup)



import sys
sys.path.append('../')
import numpy as np
from model.LMM import LMM
from utility.dataLoader import FileReader
from utility.syntheticDataGeneration import  generateData
from utility.simpleFunctions import roc




def print_out_head(out): out.write("\t".join(["RANK", "SNP_ID", "EFFECT_SIZE_ABS"]) + "\n")


def output_result(out, rank, id, beta):
    out.write("\t".join([str(x) for x in [rank, id, beta]]) + "\n")


def run(opt, outFile):
    '''
    There is the main funtion of the whole file
    :param opt:
    :param outFile:
    :return: None
    '''
    numintervals = 500
    ldeltamin = -5
    ldeltamax = 5
    if opt.real_flag=="True":
        if (opt.lam is not None) and (opt.discovernum is not None):
            print 'Invalid options: lambda and discovernum cannot be set together.'
            exit(1)
        cv_flag = ((opt.lam is None) and (opt.discovernum is None))

        #filename: formalized output filename
        outFile+='_Real_'
        if cv_flag:
            outFile+='CvFlag'
        elif (opt.discovernum is not None):
            outFile+='Discover'+str(opt.discovernum)
        else:
            outFile+='Lambda'+str(opt.lam)

        #read file
        reader = FileReader(fileName=opt.fileName, fileType=opt.fileType, imputation=(not opt.missing))
        snps, Y, Xname = reader.readFiles()
        K = np.dot(snps, snps.T)


        if not opt.quiet:
            print "Runing now"
        if opt.normalize_flag=="True":
            normalize_flag=True
        else:
            normalize_flag=False

        if opt.quiet=="True":
            quiet=True
        else:
            quiet=False


        #run our model
        lmm_model = LMM(discoverNum=opt.discovernum, ldeltamin=ldeltamin, ldeltamax=ldeltamax,learningRate=opt.lr,
                        numintervals=numintervals, lam=opt.lam, threshold=opt.threshold, isQuiet=quiet,
                        cv_flag=cv_flag,gamma=opt.gamma,lmm_flag=opt.lmm_flag,penalty_flag=opt.penalty_flag,mau=opt.mau,normalize_flag=normalize_flag,dense=opt.dense)
        beta_model_lmm = lmm_model.train(X=snps, K=K, y=Y)


        if beta_model_lmm is None:
            print 'No legal effect size is found under this setting'
            exit(1)

        if not opt.quiet:
            print "Finished"

        # Output the result to the file
        ind = np.where(beta_model_lmm != 0)[0]
        xname = []
        for i in ind:
            xname.append(i)

        beta_model_lmm=beta_model_lmm.flatten()
        beta_name = zip(beta_model_lmm, Xname)
        bn = sorted(beta_name)
        bn.reverse()
        out = open(outFile + '.output', 'w')
        print_out_head(out)
        for i in range(len(bn)):
            output_result(out, i + 1, bn[i][1], bn[i][0])
        out.close()

        np.save(outFile+'.beta',beta_model_lmm)
        #result
        print 'Computation ends normally, check the output file at', outFile

    else:
        #synthetic
        if opt.quiet == "True":
            quiet = True
        else:
            quiet = False

        for seed in range(opt.number_of_seed):
            np.random.seed(seed)
            snps, Y, Kve, Kva, beta = generateData(n=250, p=500, g=10, d=0.05, k=50, sigX=0.001, sigY=1, we=0.05,tree=opt.generation_flag, test=False, lmmn=opt.lmm_flag,quiet=quiet)
            K = np.dot(snps, snps.T)
            if not opt.quiet:
                print "Running now"
            lmm_model = LMM(discoverNum=None, ldeltamin=ldeltamin, ldeltamax=ldeltamax,learningRate=opt.lr,
                            numintervals=numintervals, lam=opt.lam, threshold=opt.threshold, isQuiet=quiet,
                            cv_flag=False, gamma=opt.gamma, lmm_flag=opt.lmm_flag, penalty_flag=opt.penalty_flag,normalize_flag=False)
            beta_model_lmm = lmm_model.train(X=snps, K=K, y=Y)
            if not opt.quiet:
                print "Finished"

            # filename: formalized output filename
            outFile += '_Synthetic_'+opt.generation_flag+'Lambda'+str(opt.lam)+'Seed'+str(seed)

            #save
            np.save('./result/'+outFile+'_snps',snps)
            np.save('./result/'+outFile+'_Y',Y)
            np.save('./result/'+outFile+'_betaTrue',beta)
            np.save('./result/'+outFile+'_beta',beta_model_lmm)
            roc_auc, fp_prc, tp_prc, fpr, tpr=roc(beta_model_lmm,beta)
            #result
            print "roc_auc:",roc_auc

#run code
(options, args) = parser.parse_args()
# filename: formalized output filename
outFile = options.fileName + options.lmm_flag+ options.penalty_flag
print 'Running ... '
if len(args) != 0:
    parser.print_help()
    sys.exit()
if options.warning_flag=="True":
    import warnings
    warnings.filterwarnings("ignore")
run(options, outFile)



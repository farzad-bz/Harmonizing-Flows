# Harmonizing FLows

The codes for ["Harmonizing Flows: Unsupervised MR harmonization based on normalizing flows"](https://arxiv.org/abs/2301.11551)


### Step 0: Preparing the data and training target task(segmentation)

20 MRIs from each of three different sites (KKI, PITT, NYU) of ABIDE dataset and 19 other MRIs from CALTECH site are selected. Then MRIs are splitted to train(60%)/val(15%)/test(25%) and for each split, coronal 2D slices of the MRIs are saved as numpy array as well as segmentatoin and brain mask and the information are saved at: "./data/ABIDE-slices-{split}-dataframce.csv". You can check sample dataframe in the data folder to check the how it should look like (header of columns are important).
Also, a 2D segmentation network is trained for each of the sites and is saved at "./checkpoints/segmentation_network_{site}.pkl".


### step1: Training the harmonizer
For training the harmonizer network, run "./step1_Harmonizer_network/train-haromnizer.py"
You should be careful of the paths and Visible cuda.
It will pre-train the haromnizer network for each site and save the  best model at "../checkpoints/ABIDE-FLOW-{site}/ABIDE-Guided-Flow-variational/"

### step2: Training the NF model
For training the NF model, run "./step2_NF_model/train_NF_guided_variational.py"
It will train the NF model for each site and save the last model at "../checkpoints/ABIDE-FLOW-{site}/ABIDE-Guided-Flow-variational/"

### step3: Adapting the harmonizer network with NF supervision
This step will adapt the pre-trained harmonizer using the trained NF model. Each time it consider one site as soure domain and after that for each remainin site (target domains), it adapt the harmonizer network which belongs to the source domain using NF model which also belongs to the source domain. for each source domain to target domain, harmonizer network is adapted for 20 iteration and after each iterarion performance of the segmentation network using the harmonized images of adapted harmonizer network is shown. Also, before that, the results of the segmentation network before harmonization and only with pretrained Harmonizer will be showed for comparison. 
To start adapting the harmonizer and runnin the validating harmonizer network, run "./step3_Adapting_Harmonizer_using_NF/validate_HF.py"


# ISIC 2018: Skin Lesion Analysis Towards Melanoma Detection
[Full report](https://s3.amazonaws.com/covalic-prod-assetstore/3b/5c/3b5c3fc89d0a4709bf4ef8331e887261?response-content-disposition=inline%3B%20filename%3D%22isic-2018-journey%282%29.pdf%22&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Expires=3600&X-Amz-Credential=AKIAITHBL3CJMECU3C4A%2F20190224%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-SignedHeaders=host&X-Amz-Date=20190224T163049Z&X-Amz-Signature=3a92eb692a1db1dccca7f20e7f1a25d8a0f6ef27c59c414d4be0d39302acd849)

 Notable performance managed to achieve by combining early and late stage fusion viasnapshot ensembling and D4 group at test time correspondingly. 
 Transfer learning from semantic segmentation task to classification did not outperform networks with ImageNet weights instantiating, 
 although the former brings diversity in final ensemble.
 
 ## Lesion Segmentation
 For this task [UNet-like](https://github.com/ternaus/TernausNet) architecture was employed with an encoder chosen to be [dual path network](https://arxiv.org/abs/1707.01629) `DPN68`.  
 
 ## Lesion Attribute Detection
 During [SLIC](http://www.kev-smith.com/papers/SLIC_Superpixels.pdf) driven annotation many attributes masks become obstacle:
 Central row: a crop of skin image acquired from Task 1-2 training set with bright spots of milia-like cyst.
 Top: predicted spots (small bright spots) overlapped with its annotation (large dim polygons) provided as corresponding ground truth.  
 Bottom: Refined and thresholded enhanced filters segments.  
 ![Segments](https://habrastorage.org/webt/qa/q5/co/qaq5coo99tjfkf3mj-lzarsdtna.png)
 
 ## Lession Classification
 [ResNext101x64](https://arxiv.org/abs/1611.05431) along with [dual path network](https://arxiv.org/abs/1707.01629) `DPN92` has been used.
 
 ### Training
 Prior to training the classifier, model’s weights were instantiated from the model pre-trained over the ImageNet dataset 
 except for the last linear layer, for which it were acquired in accordance with the Xavier uniform initial-ization. 
 The same initialization was performed over fully convolutional networks for semantic segmentation. 
 As an experiment the weights which belongs to the core architecture of FCN was extracted after training over attribute detection task 
 and was used further in classification task. Despite, this did not lead to any notable improvements it still brings diversity in final ensemble.
 The stochastic gradient decent optimization process was selected as training procedure with initial learning rate 5e−4, 
 exponential decay: `lr(i) = lrinit ∗ .95i`, i is epoch number and lies in [0,28] with Nesterov’s momentum.  
 Confusion matrices for ResNext101, computed during cross-validation:
 ![Conf Matrix](https://habrastorage.org/webt/lx/dg/er/lxdgerz7c-q4btiacdtvathq6po.png)
 
 ### [Snapshot Ensembling](https://arxiv.org/abs/1704.00109)
 4-folds class-balanced cross-validation was performed over training dataset for each model architecture (DPN92, ResNex101x64).
 Having 4-folds cross-validation two best scored epochs from each split were then picked out as snapshots. 
 In order to enlarge diversity,the fine tuning was performed for each selected snapshot with rapidly decreasing learning rate.
 This operation was repeated three times, leading to `4×2×3 = 24` modelsper architecture. 
 To reduce that number snapshot ensembling has been employed which is based on the fact that the model on its ith iteration 
 is a weighted sum of its gradient history and initial state, while optimized with SGD. Thus it’s possible to re-assign model’s 
 parameters to re-weighted sum of its snapshots. This operation was done for each forked model, resulting in only 8 models for each architecture.
 
 ### Test Time Augmentation (TTA)
 In this work TTA consists of D4 dihedral group of symmetries, resulting in 8 versions of each sample. 
 The reduction of 8 acquired probabilitiesvectors was performed as geometric average.

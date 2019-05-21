from utils_print import *

### Setup path of saved parameter files
savedir_alex = 'parameters/alex_2/'
dataset_alex_name = 'alexnet'

savedir_alexNorm = 'parameters/alex_normalized/'

savedir_dense = 'parameters/dense_2/'
dataset_dense_name = 'densenet'

savedir_denseNorm = 'parameters/dense_normalized/'


# In[3]:


### Load data from files
alexData = load_per_label(savedir_alex, dataset_alex_name, 'test')
alexTop1, alexTop3 = load_acc(savedir_alex, dataset_alex_name, 'test')
alexLearning = load_learning(savedir_alex, dataset_alex_name, 'validation')

alexNormData = load_per_label(savedir_alexNorm, dataset_alex_name, 'test')
alexNormTop1, alexNormTop3 = load_acc(savedir_alexNorm, dataset_alex_name, 'test')
alexNormLearning = load_learning(savedir_alexNorm, dataset_alex_name, 'validation')

denseData = load_per_label(savedir_dense, dataset_dense_name, 'test')
denseTop1, denseTop3 = load_acc(savedir_dense, dataset_dense_name, 'test')
denseLearning = load_learning(savedir_dense, dataset_dense_name, 'validation')

denseNormData = load_per_label(savedir_denseNorm, dataset_dense_name, 'test')
denseNormTop1, denseNormTop3 = load_acc(savedir_denseNorm, dataset_dense_name, 'test')
denseNormLearning = load_learning(savedir_denseNorm, dataset_dense_name, 'validation')


### Display overall accuracies:
print('Top 1 accuracy: alexnet:' + alexTop1 + ', alexnet normalized:' + alexNormTop1 + ', densenet:' + denseTop1 + ', densenet normalized:' + denseNormTop1)
print('Top 3 accuracy: alexnet:' + alexTop3 + ', alexnet normalized:' + alexNormTop3 + ', densenet:' + denseTop3 + ', densenet normalized:' + denseNormTop3)


### Plot graphs for more in detail comparison of accuracies and training progress over time.
plot(alexData[:,0],alexData[:,1], 'alexnet Top 1' ,denseData[:,1], 'densenet Top 1', 'comp_alex_dense_top1.jpg')
plot(alexData[:,0],alexData[:,2], 'alexnet Top 3' ,denseData[:,2], 'densenet Top 3', 'comp_alex_dense_top3.jpg')
plot(alexNormData[:,0],alexNormData[:,1], 'alexnet Top 1' ,denseNormData[:,1], 'densenet Top 1', 'comp_alex_dense_norm_top1')
plot(alexNormData[:,0],alexNormData[:,2], 'alexnet Top 3' ,denseNormData[:,2], 'densenet Top 3', 'comp_alex_dense_norm_top3')

# Delete the two categories that dont have enought data and were deleted in the normlized dataset
alexData = np.delete(alexData,12,0)
alexData = np.delete(alexData,10,0)

denseData = np.delete(denseData,12,0)
denseData = np.delete(denseData,10,0)

plot(alexData[:,0],alexData[:,1], 'alexnet Top 1' ,alexNormData[:,1], 'alexnet normalized Top 1', 'comp_alex_notnorm_norm.jpg')
plot(denseData[:,0],denseData[:,1], 'densenet Top 1' ,denseNormData[:,1], 'densenet normalized Top 1', 'comp_dense_notnorm_norm.jpg')


plot_learning(alexLearning, 'learning_alex.jpg')
plot_learning(alexNormLearning, 'learning_alex_norm.jpg')
plot_learning(denseLearning, 'learning_dense.jpg')
plot_learning(denseNormLearning, 'learning_dense_norm.jpg')


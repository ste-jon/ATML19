
def save_performance(savedir, dataset_name, performance_measure):
    file = open(savedir + dataset_name +'_performance.txt', 'w+')
    file.write('{} loss: {:.4f}\n'.format(dataset_name, performance_measure[0][-1]))
    file.write('{} accuracy: {:.4f}\n'.format(dataset_name, performance_measure[1][-1]))
    file.write('{} Top3 accuracy: {:.4f}\n'.format(dataset_name, performance_measure[2][-1]))

    
    file2 = open(savedir + dataset_name +'_learning_iterations.txt', 'w+')
    file2.write('loss, acc, top3 acc\n')
    for it, perf in enumerate(performance_measure[0]):
        file2.write('{:.4f}, {:.4f}, {:.4f}\n'.format(performance_measure[0][it], performance_measure[1][it], performance_measure[2][it]))

    if(len(performance_measure)>3):
        accuracy_per_label, accuracy_per_label_top3 = performance_measure[3:5]
        for it, nbr in enumerate(accuracy_per_label):
            file.write('{}: accuracy: {:.4f}, Top3 accuracy: {:.4f}\n'.format(it, accuracy_per_label[it], accuracy_per_label_top3[it]))



def save_test_results(savedir, dataset_name, performance_measure):
    file = open(savedir + dataset_name +'_results.txt', 'w+')
    file.write('{} loss: {:.4f}\n'.format(dataset_name, performance_measure[0]))
    file.write('{} accuracy: {:.4f}\n'.format(dataset_name, performance_measure[1]))
    file.write('{} Top3 accuracy: {:.4f}\n'.format(dataset_name, performance_measure[2]))

    if(len(performance_measure)>3):
        accuracy_per_label, accuracy_per_label_top3 = performance_measure[3:5]
        for it, nbr in enumerate(accuracy_per_label):
            file.write('{}: accuracy: {:.4f}, Top3 accuracy: {:.4f}\n'.format(it, accuracy_per_label[it], accuracy_per_label_top3[it]))


def save_performance(savedir, dataset_name, performance_measure):
    file = open(savedir + dataset_name +'_performance.txt', 'w+')
    file.write('{} loss: {:.4f}\n'.format(dataset_name, performance_measure[2][-1]))
    file.write('{} accuracy: {:.4f}\n'.format(dataset_name, performance_measure[3][-1]))
    file.write('{} Top3 accuracy: {:.4f}\n'.format(dataset_name, performance_measure[4][-1]))

    accuracy_per_label, accuracy_per_label_top3 = performance_measure[5:7]
    for it, nbr in enumerate(accuracy_per_label):
        file.write('{}: accuracy: {:.4f}, Top3 accuracy: {:.4f}\n'.format(it, accuracy_per_label[it], accuracy_per_label_top3[it]))
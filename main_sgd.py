import torch
#from mnist import * 
from mnist import *
from utils_gpu import *
from parallel_apply import *

torch.cuda.manual_seed_all(2020)

""" Initialize models on the server and workers """
net_s = Net().to(device_s)
#print(net_s.state_dict().keys())
net_w = [Net().to(device) for device in devices]
grads = [ None for i in range(num_workers)]
broadcast_models(net_s, net_w, device_s, devices)
#print(device_s, devices)

""" record """
comm_count = [0 for i in range(num_workers)]
comm_iter = 0
comm = []
loss = []
test_acc = []

""" start training """
iter = 0
loss_temp = 0
for epoch in range(num_epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    for zipdata in zip(*subtrainloader):
        broadcast_models(net_s, net_w, device_s, devices)
        grads_w= parallel_apply_SGD(net_w, zipdata, criterion, weights, devices)
        comm_flag = gather_grads(grads_w, grads, device_s)
        update_params_sgd(net_s, grads, lr, num_workers)
        """ keep record of communication rounds """
        comm_iter += sum(comm_flag)
        comm.append(comm_iter)
        for i in range(len(comm_flag)):
            comm_count[i] += comm_flag[i]
        """ keep record of running loss """
        net_s.to(devices[0])
        with torch.no_grad():
            for data in zipdata:
                inputs, labels = data[0].to(devices[0]), data[1].to(devices[0])
                outputs = net_s(inputs)
                loss_temp += criterion(outputs, labels)
        net_s.to(device_s)
            
        """ print and test """
        if iter%print_iter == 0:
            net_s.to(devices[0])
            """ evaluate on training data """
            loss.append(loss_temp/print_iter/num_workers)
            comm.append(comm_iter)
            loss_temp = 0
            print(comm_count)
            print('Epoch %d: training loss %.2f, communication %d' %(epoch, loss[-1], comm[-1]))
            correct = 0
            total = 0
            with torch.no_grad():
                for data in testloader:
                    test_inputs, test_labels = data[0].to(devices[0]), data[1].to(devices[0])
                    outputs = net_s(test_inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += test_labels.size(0)
                    correct += (predicted == test_labels).sum().item()
            print('Accuracy of the network on the 10000 test images: %.2f %%' % (100 * correct / total))
            test_acc.append(100*correct/total)
            net_s.to(device_s)
        iter = iter + 1

print('Finished Training')
np.savetxt(path+'loss_sgd_%d.txt' %seed, np.array(loss))
np.savetxt(path+'comm_sgd_%d.txt' %seed, np.array(comm))
np.savetxt(path+'testacc_sgd_%d.txt' %seed, np.array(test_acc))

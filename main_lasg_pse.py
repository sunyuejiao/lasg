import torch
from mnist import *
#from tiny_imagenet1 import *
from utils_gpu import *
from parallel_apply import *

torch.cuda.manual_seed_all(2020)
""" LASG-PSE """
params = {'method':'pse', 'thrd':0,'delay_bound':50} 
params['delay'] = [0 for i in range(num_workers)]
params['L'] = [torch.tensor(0.1).to(device) for device in devices]
params['flag'] = [True for i in range(num_workers)]

triggerslot = 10
triggerlist = [0 for i in range(triggerslot)]
thrd_scale = 1/num_workers**2 * 1/lr**2 * 1/triggerslot 

""" Initialize models on the server and workers """
net_s = Net().to(device_s)
net_w = [Net().to(device) for device in devices]
net_w_old = [Net().to(device) for device in devices]
grads = [ None for i in range(num_workers)]
broadcast_models(net_s, net_w, device_s, devices)
broadcast_models(net_s, net_w_old, device_s, devices)
print(device_s, devices)

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
        for i in range(num_workers):
            params['delay'][i] += 1
            if params['delay'][i]>= params['delay_bound'] or check_ps(net_w[i], net_w_old[i], params['L'][i], params['thrd']):
                params['flag'][i] = True
                params['delay'][i] = 0
            else:
                params['flag'][i] = False
        grads_w = parallel_apply_LASG_PSE(net_w, net_w_old, zipdata, criterion, weights, params, devices)
        comm_flag = gather_grads(grads_w, grads, device_s)
        diff = update_params_sgd(net_s, grads, lr, num_workers)
        broadcast_models(net_s, net_w, device_s, devices)
        triggerlist.append(diff)
        triggerlist.pop(0)
        params['thrd'] = sum(triggerlist)*thrd_scale
        """ keep record of communication rounds """
        comm_iter += sum(comm_flag)
        #comm.append(comm_iter)
        for i in range(len(comm_flag)):
            comm_count[i] += comm_flag[i]
        """ keep record of running loss"""
        net_s.to(devices[0])
        with torch.no_grad():
            for data in zipdata:
                inputs, labels = data[0].to(devices[0]), data[1].to(devices[0])
                outputs = net_s(inputs)
                loss_temp += criterion(outputs, labels)
        net_s.to(device_s)
        """ print and test """
        if iter%print_iter == 0:
            loss.append(loss_temp/print_iter/num_workers)
            loss_temp = 0
            comm.append(comm_iter)
            print(comm_count)
            print('Epoch %d: training loss %f, communication %d' %(epoch, loss[-1], comm[-1]))
        
            correct = 0
            total = 0
            net_s.to(devices[0])
            with torch.no_grad():
                for data in testloader:
                    test_inputs, test_labels = data[0].to(devices[0]), data[1].to(devices[0])
                    outputs = net_s(test_inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += test_labels.size(0)
                    correct += (predicted == test_labels).sum().item()
            net_s.to(device_s)
            print('Accuracy of the network on the 10000 test images: %f %%' % (100 * correct / total))
            test_acc.append(100*correct/total)
        iter = iter + 1

print('Finished Training')
np.savetxt(path+'loss_'+params['method']+'_%d.txt' %seed, np.array(loss))
np.savetxt(path+'comm_'+params['method']+'_%d.txt' %seed, np.array(comm))
np.savetxt(path+'testacc_'+params['method']+'_%d.txt' %seed, np.array(test_acc))

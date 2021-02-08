import torch
import torch.cuda.comm as comm
import time
#################################################################################################

def average_model(net_list, net_t):
    params_t = net_t.parameters()
    num_net = len(net_list)   
    params_list = []
    for i, net in enumerate(net_list):
        params_list.append(list(net.parameters()))
    for i, param in enumerate(params_t):
        param.data.mul_(0)
        for j in range(num_net):
            param.data.add_(1/num_net, params_list[j][i].data)

    
def copy_params(net1, net2):
    params1 = list(net1.parameters())
    params2 = list(net2.parameters())
    for i in range(len(params1)):
        params2[i].data.copy_(params1[i].data)

def copy_state(net1, net2):
    net2.load_state_dict(net1.state_dict())

def zero_grad(net):
    for param in net.parameters():
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()
            
def get_grad(net, weight):
    grad = []
    params = list(net.parameters())
    for param in params:
        grad.append(param.grad.data*weight)
    return grad

def get_prox_grad(net, prox_net, mu, weight):
    grad = []
    params = list(net.parameters())
    prox_params = list(prox_net.parameters())
    for param, prox_param in zip(params, prox_params):
        grad.append((param.grad.data + mu*(param.data - prox_param.data))*weight)
    return grad

def check_ps(net_new, net_old, L, thrd):
    diff = get_diff(net_new, net_old)*L
    if diff >= thrd:
        return True
    else:
        return False

def check_wk1(grad_new, grad_new_t, delta, thrd):
    diff = 0
    delta_new = []
    if delta is None:
        for i in range(len(grad_new)):
            delta_new.append(grad_new[i]-grad_new_t[i])
            diff += torch.norm(delta_new[i])**2
    else:
        for i in range(len(grad_new)):
            delta_new.append(grad_new[i]-grad_new_t[i])
            diff += torch.dist(delta_new[i], delta[i])**2
    if diff >= thrd:
        return delta_new
    else:
        return None
    
def check_wk2(grad_new, grad_old, thrd):
    diff = 0
    for i in range(len(grad_new)):
        diff += torch.norm(grad_new[i]-grad_old[i])**2
    if diff >= thrd:
        return True
    else:
        return False

def update_params_qsgd(net, grads, lr, num_workers, num_bits):
    params = net.parameters()
    diff = 0
    for i, param in enumerate(params):
        gradsi = 0
        buf = 0
        for j in range(num_workers):
            buf += quantize(grads[j][i], num_bits)
        param.data.add_(-lr, buf)
        diff += (torch.norm(buf)*lr)**2
    return diff

def check_grads(grads, gradt):
    for i, grad in enumerate(gradt):
        grad1 = 0
        for j in range(len(grads)):
            grad1 += grads[j][i]
        #print(torch.norm(grad1)/torch.norm(grad))

def update_params_sgd(net, grads, lr, num_workers):
    params = net.parameters()
    diff = 0
    for i, param in enumerate(params):
        buf = 0
        for j in range(num_workers):
            #print(grads[j][i].device)
            #print(buf.device)
            buf += grads[j][i]
        #print(buf.requires_grad)
        param.data.add_(-lr, buf)
        diff += (torch.norm(buf)*lr)**2
    return diff

def update_params_sgd_momentum(net, grads, v, lr, momentum, weight_decay, num_workers): 
    if lr is  None:
        lr = 1e-2
    if momentum is None:
        momentum = 0.9
    if weight_decay is None:
        weight_decay = 0
    params = list(net.parameters())
    diff = 0
    if v is None:
        v = [0 for i in range(len(params))]
    for i, param in enumerate(params):
        buf = 0
        for j in range(num_workers):
            buf += grads[j][i]
        buf = buf + weight_decay * param.data
        v_old = v[i]
        v[i] = momentum*v[i] - lr * buf
        param.data.add_(v[i]+momentum*(v[i]-v_old))
        diff += (torch.norm(v[i] + momentum*(v[i]-v_old)))**2
    return diff, v
def update_params_svrg(net, grads, grads_t, full_grads_t, lr, num_workers):
    params = net.parameters()
    diff = 0
    for i, param in enumerate(params):
        buf = 0
        for j in range(num_workers):
            buf += grads[j][i]-grads_t[j][i]
        param.data.add_(-lr, buf+full_grads_t[i])
        diff += (torch.norm(buf+full_grads_t[i])*lr)**2
    return diff

def get_diff(net1, net2):
    params1 = list(net1.parameters())
    params2 = list(net2.parameters())
    diff = 0
    for i in range(len(params1)):
        diff += torch.dist(params1[i], params2[i])**2
    return diff
def get_grad_dist(grad1, grad2):
    dist2 = 0
    for i in range(len(grad1)):
        dist2 += torch.dist(grad1[i], grad2[i])**2
    return dist2

def quantization(vlist, num_bits):
    qv = []
    for i in range(len(vlist)):
        v = vlist[i]
        v_norm = torch.norm(v)
        if v_norm < 1e-10:
            qv.append(0)
        else:
            s = 2**(num_bits-1)
            l = torch.floor(torch.abs(v)/v_norm*s)
            p = torch.abs(v)/v_norm-l
            qv.append(v_norm*torch.sign(v)*(l/s + l/s*(torch.rand(v.shape)<p)))
    return qv
def quantize(v, num_bits):
    v_norm = torch.norm(v)
    if v_norm < 1e-10:
        qv = 0
    else:
        s = 2**(num_bits-1)
        l = torch.floor(torch.abs(v)/v_norm*s)
        p = torch.abs(v)/v_norm-l
        qv = v_norm*torch.sign(v)*(l/s + l/s*(torch.rand_like(v)<p).float())
    return qv

def sort_dataset(dataset, num_classes, num_samples):
    sorted = [[] for i in range(num_classes)]
    for i in range(num_samples):
        sorted[dataset[i][1]].append(dataset[i])
    alldata = []
    for i in range(num_classes):
        for data in sorted[i]:
            alldata.append(data)
    return alldata

def average_model(net_list, net_t):
    params_t = net_t.parameters()
    num_net = len(net_list)   
    params_list = []
    for i, net in enumerate(net_list):
        params_list.append(list(net.parameters()))
    for i, param in enumerate(params_t):
        param.data.mul_(0)
        for j in range(num_net):
            param.data.add_(1/num_net, params_list[j][i].data)

def update_delta(delta, net_wk, net_ps, lr, H):
    params_ps_list = list(net_ps.parameters())
    params_wk_list = list(net_wk.parameters())
    for i in range(len(params_ps_list)):
        p_ps = params_ps_list[i]
        p_wk = params_wk_list[i]
        delta[i] += (p_ps.data-p_wk.data)/lr/H
        
""" cuda communication """
def broadcast_models(net_s, net_w, device_s, devices):
    for i, net in enumerate(net_w):
        net.to(device_s)
        #copy_params(net_s, net)
        copy_state(net_s, net)
        net.to(devices[i])

def gather_models(net_w, device_s, comm_t=0):
    comm_flag = []
    for net in net_w:
        net.to(device_s)
        comm_flag.append(1)
        if comm_t>0:
            time.sleep(comm_t)
    return comm_flag

def broadcast_params(net_s, net_w, device_s, devices):
    # device_s must be difference from devices, gpu to mulple gpus
    params_s = list(net_s.parameters())
    params_w = [list(net.parameters()) for net in net_w]
    params_copies = comm.broadcast_coalesced(params_s, [device_s, *devices])
    for i in range(len(net_w)):
        for j in range(len(params_s)):
            params_w[i][j].data.copy_(params_copies[i][j].data)
    del params_copies

def gather_grads(grads_w, grads, device_s, comm_t=0):
    comm_flag = []
    for i in range(len(grads)):
        if grads_w[i] is not None:
            for j in range(len(grads_w[i])):
                grads_w[i][j] = grads_w[i][j].to(device_s)
            grads[i] = grads_w[i]
            comm_flag.append(1)
            if comm_t>0:
                time.sleep(comm_t)
        else:
            comm_flag.append(0)
    return comm_flag
    
def reduce_average_params(net_s, net_w, device_s):
    params_s = list(net_s.parameters())
    params_w = [list(net.parameters()) for net in net_w]
    num_workers = len(net_w)
    for j, param_s in enumerate(params_s):
        param_w_sum = comm.reduce_add([params_w[i][j] for i in range(num_workers)], device_s)
        param_s.data.mul_(0)
        param_s.data.add_(1/num_workers, param_w_sum.data)
    del param_w_sum

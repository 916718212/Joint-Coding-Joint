import torch
fy=torch.load("/root/Joint-Coding/mvx_pre.pth")
pr=torch.load("/root/Joint-Coding/epoch_85.pth")
re=torch.load("/root/Joint-Coding/mvx_revise.pth")

a=fy
b=pr
print(len(re['state_dict'].items() & pr['state_dict'].items()))
print(len(b['state_dict'].items() & a['state_dict'].items())) 
print(len(fy['state_dict'].items() & pr['state_dict'].items()))
for k,v in b['state_dict'].items():
    for k1,v1 in a['state_dict'].items():
        if k == k1:
            b['state_dict'][k]=a['state_dict'][k1]
print(len(b['state_dict'].items() & a['state_dict'].items()))
print(len(fy['state_dict'].items() & pr['state_dict'].items()))
# print(len(b['state_dict']))

torch.save(b,'/root/Joint-Coding/mvx_revise_85.pth')

fusion_list = '4-6-2-4-2-3-4-5-4-4-4-4-0-0-4'
my_fusion_list = [int(x) for x in fusion_list.split('-')]
n = len(my_fusion_list)

view_cnt = n / 2 + 1
fusion = my_fusion_list[:int(view_cnt)]
opt = my_fusion_list[int(view_cnt):]
f = [64, 225, 144, 73, 128,500,1000]
print(fusion)
print(opt)


nb_feats = []

for i in fusion:
    nb_feats.append(f[i])

print(nb_feats)
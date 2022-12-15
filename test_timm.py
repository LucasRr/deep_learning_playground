import timm 
import torch

# l = timm.list_models()
# print([m for m in l if "efficientnet" in m])


model = timm.create_model('efficientnet_b2')

l = [torch.rand((1, 3, 360, 4279)),
torch.rand((1, 3, 360, 360)),
torch.rand((1, 3, 360, 89))]

for x in l:
	y = model.conv_stem(x)
	y = model.blocks(y)

	print(y.shape)
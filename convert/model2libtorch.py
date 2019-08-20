import torch
import torchvision
from model_irse import IR_152

modelname = '/home/ww/work/faceRecognition/libtorch-faceRecognition/model/Backbone_IR_152.pth'
model = IR_152([112, 112])
ckpt = torch.load(modelname)
model.load_state_dict(ckpt, strict=False)
model.eval()
# An instance of your model.
#model = torchvision.models.resnet152()

# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 3, 112, 112)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)

output = traced_script_module(torch.ones(1, 3, 112, 112))
print(output.shape)


traced_script_module.save("my_model_152.pt")


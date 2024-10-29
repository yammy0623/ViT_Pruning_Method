import timm
import tome
from torchinfo import summary
import torch
torch.cuda.empty_cache()
# for i in timm.list_models():
#     print(i)
# a



# Load Model
model_name = "vit_base_patch16_224"
# model_name = "vit_base_patch32_224"
model = timm.create_model(model_name, pretrained=True)

# GPU setting
device = "cuda:0"
runs = 50
batch_size = 256  # Lower this if you don't have that much memory
input_size = model.default_cfg["input_size"]
print(input_size)

# Baseline benchmark
baseline_throughput = tome.utils.benchmark(
    model,
    device=device,
    verbose=True,
    runs=runs,
    batch_size=batch_size,
    input_size=input_size
)
summary(model, input_size=(batch_size, 3, 224, 224))

# Apply ToMe
print("----Apply ToMe----")
tome.patch.timm(model)
# ToMe with r=16
model.r = 16
tome_throughput = tome.utils.benchmark(
    model,
    device=device,
    verbose=True,
    runs=runs,
    batch_size=batch_size,
    input_size=input_size
)
print(f"Throughput improvement: {tome_throughput / baseline_throughput:.2f}x")
summary(model, input_size=(batch_size, 3, 224, 224))
# ToMe with r=16 and a decreasing schedule
print("----Apply ToMe with decreasing schedule----")
model.r = (16, -1.0)
tome_decr_throughput = tome.utils.benchmark(
    model,
    device=device,
    verbose=True,
    runs=runs,
    batch_size=batch_size,
    input_size=input_size
)
print(f"Throughput improvement: {tome_decr_throughput / baseline_throughput:.2f}x")
summary(model, input_size=(batch_size, 3, 224, 224))


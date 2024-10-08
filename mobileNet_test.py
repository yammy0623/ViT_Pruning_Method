import timm
import tome
# Load mobileNet
model_name = "tf_mobilenetv3_large_075"
model = timm.create_model(model_name, pretrained=True)
device = "cuda:0"
runs = 50
batch_size = 256  # Lower this if you don't have that much memory
input_size = model.default_cfg["input_size"]
print(model_name)
baseline_throughput = tome.utils.benchmark(
    model,
    device=device,
    verbose=True,
    runs=runs,
    batch_size=batch_size,
    input_size=input_size
)
import timm
import tome
from torchinfo import summary

device = "cuda:0"
runs = 50
batch_size = 256  
cnn_model_list = ["mobilenetv3_large_100.ra_in1k",
                  "efficientnet_b0.ra_in1k",
                  "mixnet_s.ft_in1k"]
cnn_throughput = []
vit_model_list = [ "deit_tiny_distilled_patch16_224.fb_in1k",
                  "vit_base_patch16_224.orig_in21k_ft_in1k",
                    "cait_xxs24_224.fb_dist_in1k", # https://huggingface.co/timm/cait_xxs24_224.fb_dist_in1k
                    # "resmlp_12_224.fb_in1k", # https://huggingface.co/timm/resmlp_12_224.fb_in1k
                    "vit_small_patch32_224.augreg_in21k_ft_in1k", # https://huggingface.co/timm/vit_small_patch32_224.augreg_in21k_ft_in1k
                    "deit3_small_patch16_384.fb_in22k_ft_in1k", # https://huggingface.co/timm/deit3_small_patch16_384.fb_in22k_ft_in1k
                  #"tiny_vit_5m_224.dist_in22k_ft_in1k",
                  #"fastvit_t8.apple_dist_in1k",
                #"efficientvit_m2.r224_in1k"
              ]
vit_throughput = []
vit_tome_throughput = []
vit_tome_improvement = []
token_reduce_amount = 16

print("ViT \n")
for model_name in vit_model_list:
    # model_name = "tf_mobilenetv3_large_075"
    print(model_name)
    model = timm.create_model(model_name, pretrained=True)
    # summary(model, input_size=(batch_size, 3, 224, 224))
    input_size = model.default_cfg["input_size"]
    baseline_throughput = tome.utils.benchmark(
        model,
        device=device,
        verbose=True,
        runs=runs,
        batch_size=batch_size,
        input_size=input_size
    )

    vit_throughput.append(baseline_throughput)
    print("\n Apply ToMe \n")
    tome.patch.timm(model)
    model.r = token_reduce_amount
    tome_throughput = tome.utils.benchmark(
        model,
        device=device,
        verbose=True,
        runs=runs,
        batch_size=batch_size,
        input_size=input_size
    )
    print(f"Throughput improvement: {tome_throughput / baseline_throughput:.2f}x")
    vit_tome_throughput.append(tome_throughput)
    vit_tome_improvement.append(tome_throughput / baseline_throughput)
    print()

print("\n CNN Throughput \n")
print(cnn_throughput)
print("\n ViT Throughput \n")
print(vit_throughput)
print("\n ViT + ToMe Throughput \n")
print(vit_tome_throughput)
print("\n ViT + ToMe Improvement \n")
print(vit_tome_improvement)
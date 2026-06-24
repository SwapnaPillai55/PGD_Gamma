import torch
import torch.nn as nn
import time
import os
import platform
from ptflops import get_model_complexity_info

#from Networks_Sid.models.convnext_isotropic_x2_f16_3c_for_L1_L2_loss import convnext_isotropic_base
#from Networks_Sid.models.convnext_isotropic_x2_f512_3c import convnext_isotropic_base
from Networks_Sid.models.convnext_isotropic_x2_f8_3c import convnext_isotropic_base



def print_execution_header(res, n_runs):
    print("\n" + "="*60)
    print(f"  CONVNEXT BENCHMARK - INITIALIZING SYSTEM INFO")
    print("-" * 60)
    print(f"  OS/Platform:     {platform.system()} {platform.release()}")
    print(f"  Device:          {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  PyTorch:         {torch.__version__}")
    print(f"  Input Channels:  1 (Grayscale)")
    print(f"  Resolution:      {res}x{res}")
    print(f"  Iterations:      {n_runs} runs (for average throughput)")
    print("="*60 + "\n")

def main():
    # Settings
    res = 160
    n_runs = 100
    n_warmup = 20
    
    print_execution_header(res, n_runs)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 1. Initialize and modify model for 1-channel
    net = convnext_isotropic_base().to(device)
    
    # Surgery: Replace 3-ch stem with 1-ch stem
    # This prevents the 'expected 3 channels' RuntimeError
    found_stem = False
    for name, module in net.named_modules():
        if isinstance(module, nn.Conv2d):
            old_conv = module
            new_conv = nn.Conv2d(
                in_channels=1, 
                out_channels=old_conv.out_channels, 
                kernel_size=old_conv.kernel_size, 
                stride=old_conv.stride, 
                padding=old_conv.padding,
                bias=(old_conv.bias is not None)
            ).to(device)
            
            # Target the specific attribute in ConvNeXt (usually net.stem)
            if hasattr(net, 'stem'):
                net.stem = new_conv
                found_stem = True
            break 

    net.eval()
    input_shape = (1, res, res)

    # 2. Compute FLOPs and Params via ptflops
    try:
        macs, params = get_model_complexity_info(
            net, input_shape, as_strings=True, 
            print_per_layer_stat=False, verbose=False
        )
    except Exception as e:
        print(f"!! Complexity calculation failed: {e}")
        macs, params = "N/A", "N/A"

    # 3. Measure Average Throughput
    dummy_input = torch.randn(1, 1, res, res).to(device)
    
    with torch.no_grad():
        # Warmup phase
        for _ in range(n_warmup):
            _ = net(dummy_input)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        start_time = time.perf_counter()
        for _ in range(n_runs):
            _ = net(dummy_input)
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()

    # 4. Final Output Metrics
    total_time = end_time - start_time
    avg_latency = total_time / n_runs
    fps = 1.0 / avg_latency

    print("-" * 50)
    print(f"{'METRIC (Avg over ' + str(n_runs) + ' runs)':<35} | {'VALUE':<12}")
    print("-" * 50)
    print(f"{'Model Parameters':<35} | {params}")
    print(f"{'GMACs (Complexity)':<35} | {macs}")
    print(f"{'Average Latency':<35} | {avg_latency*1000:.3f} ms")
    print(f"{'Average Throughput':<35} | {fps:.2f} FPS")
    print("-" * 50 + "\n")

if __name__ == '__main__':
    main()
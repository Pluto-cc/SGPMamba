import os
import torch
import cv2
import numpy as np
import time
from tqdm import tqdm
from thop import profile

from prior import SARParameterExtractor, SARImageGenerator
from model import SARRefinementModel 

def run_full_pipeline_test():

    RAW_SAR_DIR = '/home/changqi/王长启/Code/Umamba/dataset/train/train' 
    XML_DIR = '/home/changqi/王长启/Code/Umamba/dataset/Annotations'     
    MODEL_PATH = '/home/changqi/王长启/Code/Umamba/checkH/check7/best_model.pth'
    SAVE_DIR = '/home/changqi/王长启/Code/Umamba/test_results'
    
    DEVICE =  'cuda' if torch.cuda.is_available() else 'cpu'
    IMG_SIZE = (800, 800)
    os.makedirs(SAVE_DIR, exist_ok=True)

    model = SARRefinementModel(in_chans=1, num_classes=1, embed_dim=64, decode_channels=64, use_lsm=True)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    dummy_input = torch.randn(1, 1, *IMG_SIZE).to(DEVICE)
    macs, params = profile(model, inputs=(dummy_input, ), verbose=False)
    print("\n" + "="*50)
    print(f"  MODEL COMPLEXITY (Static Metrics)")
    print(f"   - Params : {params/1e6:.3f} M")
    print(f"   - MACs   : {macs/1e9:.3f} G")
    print(f"   - FLOPs  : {2*macs/1e9:.3f} G (Estimate)")
    print("="*50 + "\n")

    stats = {
        "stage1_extract": [], 
        "stage2_prior": [],  
        "stage3_mamba": [],  
        "total_pipeline": []  
    }

    filenames = [f for f in os.listdir(RAW_SAR_DIR) if f.endswith(('.png', '.jpg', '.tif'))]
    
    print(f"Starting End-to-End Testing on {len(filenames)} images...")

    for name in tqdm(filenames):
        file_id = os.path.splitext(name)[0]
        sar_path = os.path.join(RAW_SAR_DIR, name)
        xml_path = os.path.join(XML_DIR, file_id + '.xml')
        
        if not os.path.exists(xml_path):
            continue

        if DEVICE == 'cuda': torch.cuda.synchronize()
        t_start_all = time.perf_counter()

        t1_start = time.perf_counter()
        extractor = SARParameterExtractor(sar_path, xml_path)
        extractor.preprocess(denoise_sigma=0.6)
        extractor.segment_terrain()

        _ = extractor.extract_parameters()

        param_json_path = os.path.join(SAVE_DIR, f"{file_id}.json")
        extractor.save_parameters(param_json_path) 
        
        masks = extractor.masks
        t1_end = time.perf_counter()
        stats["stage1_extract"].append((t1_end - t1_start) * 1000)

        t2_start = time.perf_counter()

        generator = SARImageGenerator(param_json_path, masks, extractor.land_exit, extractor.sea_exit, extractor.ship_exit)
        prior_img = generator.generate()

        prior_resized = cv2.resize(prior_img, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
        prior_norm = (prior_resized.astype(np.float32) / 127.5) - 1.0
        input_tensor = torch.from_numpy(prior_norm).unsqueeze(0).unsqueeze(0).to(DEVICE)
        t2_end = time.perf_counter()
        stats["stage2_prior"].append((t2_end - t2_start) * 1000)

        t3_start = time.perf_counter()
        with torch.no_grad():
            output = model(input_tensor)
        if DEVICE == 'cuda': torch.cuda.synchronize()
        t3_end = time.perf_counter()
        stats["stage3_mamba"].append((t3_end - t3_start) * 1000)

        t_end_all = time.perf_counter()
        stats["total_pipeline"].append((t_end_all - t_start_all) * 1000)

        final_out = ((output.squeeze().cpu().numpy() + 1.0) * 127.5).astype(np.uint8)
        cv2.imwrite(os.path.join(SAVE_DIR, f"{file_id}_SGPMamba.png"), final_out)

    print("\n" + "🩺 SYSTEM HEALTH REPORT (Resolution: 800x800) " + "="*15)
    
    stages = ["stage1_extract", "stage2_prior", "stage3_mamba", "total_pipeline"]
    titles = ["1. Parameter Extraction", "2. Prior Generation", "3. Mamba Refinement (MO)", "TOTAL END-TO-END"]
    
    for stage, title in zip(stages, titles):
        data = np.array(stats[stage])
        avg, p50, p95 = np.mean(data), np.percentile(data, 50), np.percentile(data, 95)
        print(f"[{title}]")
        print(f"   Avg Latency : {avg:.2f} ms")
        print(f"   P50 Latency : {p50:.2f} ms")
        print(f"   P95 Latency : {p95:.2f} ms")
        
        if stage == "stage3_mamba":
            print(f"    MO FPS    : {1000.0/avg:.2f}")
        if stage == "total_pipeline":
            print(f"    System FPS: {1000.0/avg:.2f}")
    print("="*55 + "\n")

if __name__ == "__main__":
    run_full_pipeline_test()

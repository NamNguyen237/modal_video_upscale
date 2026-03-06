import modal
import os
import subprocess
import shutil
import re
from datetime import datetime

# 1. Môi trường Cloud tối ưu (L40S + FFmpeg Pro)
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "git", "libgl1-mesa-glx", "libglib2.0-0", "unzip", "curl", "wget")
    .pip_install("torch", "torchvision", "numpy", "gdown", "ffmpeg-python")
)

volume = modal.Volume.from_name("video_storage", create_if_missing=True)
app = modal.App("ultra-gameplay-upscaler-v9", image=image)

@app.function(gpu="L40S", cpu=16, memory=65536, volumes={"/data": volume}, timeout=21600)
def super_render(drive_id: str, target_4k: bool = True, force_60fps: bool = True, phone_ratio: bool = True):
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    work_dir, final_dir = "/data/processing", "/data/final_outputs"
    merged_path = f"{work_dir}/merged.mp4"
    
    for d in [work_dir, f"{work_dir}/inputs", final_dir]:
        os.makedirs(d, exist_ok=True)

    # --- 1. DOWNLOAD & FIX LINK ---
    def fix_url(url):
        drive_match = re.search(r'(?:id=|/d/)([\w-]+)', url)
        if "drive.google.com" in url and drive_match: return drive_match.group(1)
        return url

    processed_id = fix_url(drive_id)
    temp_file = f"{work_dir}/temp_download"
    
    import gdown
    print(f"📥 Downloading ID: {processed_id}")
    gdown.download(f'https://drive.google.com/uc?id={processed_id}', temp_file, quiet=False, fuzzy=True)
    if os.path.exists(temp_file): shutil.move(temp_file, merged_path)

    # --- 2. TÍNH TOÁN THÔNG SỐ SIÊU NÉT ---
    # target_w theo logic gameplay "khủng" của bạn
    target_w = (4800 if target_4k else 3200) if phone_ratio else (3840 if target_4k else 2560)
    target_h = 2160 if target_4k else 1440
    bitrate = "85M" if target_4k else "50M"

    # Bộ lọc "Tinh hoa": Sửa lỗi typo 'correct_scaling_colors'
    custom_filters = (
        f"format=yuv420p,"
        f"hqdn3d=1.5:1.5:6:6," 
        f"scale={target_w}:{target_h}:flags=bitexact+lanczos+correct_scaling_colors,"
        f"unsharp=5:5:1.0:5:5:0.0,"
        f"unsharp=3:3:0.8:3:3:0.0"
    )
    if force_60fps: custom_filters += ",fps=fps=60"

    filename = f"ULTRA_GAMEPLAY_{target_w}p_{timestamp}.mp4"
    final_video = f"{final_dir}/{filename}"

    # --- 3. RENDER VỚI GPU L40S (Sử dụng Preset P7 cao nhất) ---
    print(f"🚀 Render khởi động: {target_w}x{target_h} | GPU: L40S")
    
    render_cmd = (
        f"ffmpeg -y -hwaccel cuda -i {merged_path} "
        f"-vf '{custom_filters}' "
        f"-c:v h264_nvenc -preset p7 -rc vbr -cq 18 "
        f"-b:v {bitrate} -maxrate:v 120M -bufsize:v 120M "
        f"-c:a aac -b:a 192k "
        f"-pix_fmt yuv420p {final_video}"
    )
    
    subprocess.run(render_cmd, shell=True, check=True)
    
    volume.commit()
    return filename

@app.local_entrypoint()
def main():
    link_full = "https://drive.google.com/file/d/1upZs6hpJg5uloO7xq8LAU2451Ve3aozE/view?usp=sharing"
    
    remote_filename = super_render.remote(
        drive_id=link_full, 
        target_4k=True, 
        phone_ratio=True
    )
    print(f"✅ HOÀN TẤT! Video tại Modal Volume: /final_outputs/{remote_filename}")
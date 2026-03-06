import modal
import os
import subprocess
import shutil
from datetime import datetime

# 1. Môi trường Cloud tối ưu (L40S + Real-ESRGAN)
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "git", "libgl1-mesa-glx", "libglib2.0-0", "unzip", "curl", "wget")
    .pip_install("torch", "torchvision", "cython", "setuptools<70")
    .pip_install("basicsr", "realesrgan", "numpy", "gdown", "ffmpeg-python")
    .run_commands(
        "git clone https://github.com/xinntao/Real-ESRGAN.git /root/Real-ESRGAN",
        "cd /root/Real-ESRGAN && pip install -r requirements.txt && python setup.py develop",
        "sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/g' /usr/local/lib/python3.10/site-packages/basicsr/data/degradations.py"
    )
)

volume = modal.Volume.from_name("video_storage", create_if_missing=True)
app = modal.App("ultimate-renderer-final-v7", image=image)

LOCAL_DOWNLOAD_PATH = "D:/Rendered_Videos"

@app.function(gpu="L40S", cpu=16, memory=65536, volumes={"/data": volume}, timeout=21600, retries=0)
def super_render(drive_id: str, use_ai: bool = True, phone_ratio: bool = True, keep_aspect: bool = False, 
                 target_4k: bool = False, native_x2: bool = False, force_60fps: bool = True, 
                 zip_password: str = None, force_rebuild: bool = False):
    import torch
    import zipfile

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    work_dir, final_dir = "/data/processing", "/data/final_outputs"
    merged_path = f"{work_dir}/merged.mp4"
    
    if force_rebuild:
        print("♻️ Force Rebuild: Đang dọn dẹp thư mục làm việc...")
        if os.path.exists(work_dir): shutil.rmtree(work_dir)

    for d in [work_dir, f"{work_dir}/inputs", f"{work_dir}/ai_out", final_dir]:
        os.makedirs(d, exist_ok=True)

    def fix_url(url):
        if "pixeldrain.com/u/" in url: return url.replace("/u/", "/api/file/")
        if "dropbox.com" in url and "dl=0" in url: return url.replace("dl=0", "dl=1")
        return url

    # --- TẢI & GIẢI NÉN (HỖ TRỢ PASS) ---
    if not os.path.exists(merged_path):
        temp_file = f"{work_dir}/temp_download"
        if isinstance(drive_id, list):
            for i, url in enumerate(drive_id):
                os.system(f"curl -L -o {work_dir}/inputs/part_{i:04d}.mp4 '{fix_url(url)}'")
        else:
            drive_id = fix_url(drive_id)
            if drive_id.startswith("http"): os.system(f"curl -L -o {temp_file} '{drive_id}'")
            else: 
                import gdown
                gdown.download(f'https://drive.google.com/uc?id={drive_id}', temp_file, quiet=False, fuzzy=True)
            
            if os.path.exists(temp_file):
                if zipfile.is_zipfile(temp_file):
                    print(f"📦 Đang giải nén ZIP {'(có mật khẩu)' if zip_password else ''}...")
                    with zipfile.ZipFile(temp_file, 'r') as z:
                        z.extractall(f"{work_dir}/inputs", pwd=zip_password.encode() if zip_password else None)
                    os.remove(temp_file)
                else: 
                    shutil.move(temp_file, merged_path)

    # --- GHÉP FILE (HỖ TRỢ MP4 + MKV) ---
    if not os.path.exists(merged_path):
        files = sorted([f for f in os.listdir(f"{work_dir}/inputs") if f.lower().endswith((".mp4", ".mkv"))])
        if not files: raise Exception("❌ Không tìm thấy file video nào trong đầu vào!")
        
        list_path = f"{work_dir}/list.txt"
        with open(list_path, "w") as f:
            for file in files: f.write(f"file '{work_dir}/inputs/{file}'\n")
        
        print(f"🧩 Đang ghép {len(files)} file...")
        os.system(f"ffmpeg -y -f concat -safe 0 -i {list_path} -c copy {merged_path}")

    # --- BƯỚC 3: TÍNH TOÁN FILTER ---
    probe_cmd = f"ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 {merged_path}"
    dimensions = subprocess.check_output(probe_cmd, shell=True).decode().strip().split('x')
    orig_w, orig_h = int(dimensions[0]), int(dimensions[1])

    fps_filter = ",fps=fps=60" if force_60fps else ""

    if native_x2:
        target_w, target_h = orig_w * 2, orig_h * 2
        bitrate, vf_scale, label = "45M", f"scale={target_w}:{target_h}:flags=lanczos", "NATIVE_X2"
    else:
        target_w = (4800 if target_4k else 3200) if phone_ratio else (3840 if target_4k else 2560)
        target_h = 2160 if target_4k else 1440
        bitrate, label = ("60M" if target_4k else "35M"), ("4K" if target_4k else "2K")
        if keep_aspect:
            vf_scale = f"scale={target_w}:{target_h}:flags=lanczos:force_original_aspect_ratio=decrease,pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2"
        else:
            vf_scale = f"scale={target_w}:{target_h}:flags=lanczos"

    filename = f"RENDER_{label}_{timestamp}.mp4"
    final_video = f"{final_dir}/{filename}"

    print(f"🚀 Render: {label} | 60FPS: {force_60fps} | AI: {use_ai}")

    # --- BƯỚC 4: RENDER ---
    if use_ai:
        if os.path.exists("/tmp/ai_out"): shutil.rmtree("/tmp/ai_out")
        os.makedirs("/tmp/ai_out", exist_ok=True)
        cmd = f"python /root/Real-ESRGAN/inference_realesrgan_video.py -i {merged_path} -o /tmp/ai_out -n realesr-animevideov3 -s 2 --tile 1024 --num_process_per_gpu 2"
        subprocess.run(cmd, shell=True, check=True)
        shutil.move("/tmp/ai_out/merged_out.mp4", f"{work_dir}/ai_out/merged_out.mp4")
        
        os.system(f"ffmpeg -y -i {work_dir}/ai_out/merged_out.mp4 -i {merged_path} "
                  f"-map 0:v:0 -map 1:a? -vf '{vf_scale}{fps_filter}' -c:a copy -c:v h264_nvenc -preset p4 -b:v {bitrate} {final_video}")
    else:
        os.system(f"ffmpeg -y -hwaccel cuda -i {merged_path} "
                  f"-vf '{vf_scale},unsharp=3:3:1.5{fps_filter}' "
                  f"-c:v h264_nvenc -preset p4 -b:v {bitrate} -pix_fmt yuv420p {final_video}")

    if os.path.exists(f"{work_dir}/inputs"): shutil.rmtree(f"{work_dir}/inputs")
    volume.commit()
    return filename

@app.local_entrypoint()
def main():
    display_id = "1upZs6hpJg5uloO7xq8LAU2451Ve3aozE" # ID file hoặc link ZIP
    
    remote_filename = super_render.remote(
        drive_id=display_id,
        use_ai=False, 
        phone_ratio=True, 
        keep_aspect=True, 
        target_4k=False,
        native_x2=False, 
        force_60fps=True,
        zip_password="", # Đổi mật khẩu ở đây, nếu không có thì để None
        force_rebuild=False
    )

    if not os.path.exists(LOCAL_DOWNLOAD_PATH): os.makedirs(LOCAL_DOWNLOAD_PATH)
    print(f"📥 Đang tải: {remote_filename}...")
    subprocess.run(["modal", "volume", "get", "video_storage", f"/final_outputs/{remote_filename}", LOCAL_DOWNLOAD_PATH])
    print(f"✅ HOÀN TẤT! Video tại: {LOCAL_DOWNLOAD_PATH}/{remote_filename}")
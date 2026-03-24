import modal
import os
import subprocess
import shutil
import re
from datetime import datetime

# --- MÔI TRƯỜNG CLOUD (L40S + Real-ESRGAN + FFmpeg Optimized) ---
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
app = modal.App("ultimate-renderer-v11-pro-plus", image=image)

LOCAL_DOWNLOAD_PATH = "./Rendered_Videos"
#LOCAL_DOWNLOAD_PATH = os.path.expanduser("/mnt/nvme/Rendered_Videos")

@app.function(gpu="L40S", cpu=16, memory=65536, volumes={"/data": volume}, timeout=21600, retries=0)
def super_render(drive_id: str, 
                 use_ai: bool = True,       # True: Dùng Real-ESRGAN (cho Live2D/CG), False: FFmpeg Upscale (cho Gameplay)
                 phone_ratio: bool = True,   # Tỉ lệ 20:9 (4800x2160) hay 16:9 (3840x2160)
                 keep_aspect: bool = False,  # Giữ tỉ lệ gốc, chèn đen hai bên
                 target_4k: bool = False,    # True: 4K, False: 2K
                 native_x2: bool = False,    # Nhân đôi độ phân giải gốc (không ép chuẩn 4K)
                 force_60fps: bool = True,   # Ép về 60fps và fix xé hình
                 zip_password: str = None, 
                 fix_black_pixels: bool = True, 
                 fix_fade: bool = True,      # Safe Mode: Giảm độ nét để tránh artifact khi fade
                 force_rebuild: bool = False):
    
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
        drive_match = re.search(r'(?:id=|/d/)([\w-]+)', url)
        if "drive.google.com" in url and drive_match: return drive_match.group(1)
        if "pixeldrain.com/u/" in url: return url.replace("/u/", "/api/file/")
        if "dropbox.com" in url and "dl=0" in url: return url.replace("dl=0", "dl=1")
        return url

    # --- 1. TẢI & GIẢI NÉN ---
    if not os.path.exists(merged_path):
        temp_file = f"{work_dir}/temp_download"
        processed_id = fix_url(drive_id)
        if processed_id.startswith("http"): os.system(f"curl -L -o {temp_file} '{processed_id}'")
        else:
            import gdown
            gdown.download(f'https://drive.google.com/uc?id={processed_id}', temp_file, quiet=False, fuzzy=True)
            
        if os.path.exists(temp_file):
            if zipfile.is_zipfile(temp_file):
                print(f"📦 Đang giải nén ZIP...")
                with zipfile.ZipFile(temp_file, 'r') as z:
                    z.extractall(f"{work_dir}/inputs", pwd=zip_password.encode() if zip_password else None)
                os.remove(temp_file)
            else: shutil.move(temp_file, merged_path)

    # --- 2. GHÉP FILE ---
    if not os.path.exists(merged_path):
        files = sorted([f for f in os.listdir(f"{work_dir}/inputs") if f.lower().endswith((".mp4", ".mkv", ".webm"))])
        if not files: raise Exception("❌ Không thấy video đầu vào!")
        list_path = f"{work_dir}/list.txt"
        with open(list_path, "w") as f:
            for file in files: f.write(f"file '{work_dir}/inputs/{file}'\n")
        print(f"🔗 Đang ghép {len(files)} file đầu vào...")
        os.system(f"ffmpeg -y -f concat -safe 0 -i {list_path} -c copy {merged_path}")

    # --- 3. LẤY THÔNG TIN GỐC & TÍNH TOÁN TARGET ---
    probe_cmd = f"ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 {merged_path}"
    dimensions = subprocess.check_output(probe_cmd, shell=True).decode().strip().split('x')
    orig_w, orig_h = int(dimensions[0]), int(dimensions[1])

    if native_x2:
        target_w, target_h = orig_w * 2, orig_h * 2
        label = "NATIVE_X2"
    else:
        # 4K (2160p) hoặc 2K (1440p) dựa trên Boolean target_4k
        target_h = 2160 if target_4k else 1440
        # Tính chiều rộng dựa trên phone_ratio (20:9) hoặc mặc định (16:9)
        ratio = (20/9) if phone_ratio else (16/9)
        target_w = int(target_h * ratio)
        # Làm tròn về số chẵn để encoder h264 không lỗi
        target_w = target_w if target_w % 2 == 0 else target_w + 1
        label = "4K" if target_4k else "2K"

    bitrate = "75M" if target_4k else "40M"
    filename = f"RENDER_{label}_{timestamp}.mp4"
    final_video = f"{final_dir}/{filename}"

    # --- 4. XÂY DỰNG FILTER CHAIN (TRỊ XÉ HÌNH + ANIME CLEANUP) ---
    # Fix xé hình (PTS/VFR fix) - Quan trọng nhất cho scrcpy
    vfr_fix = "fps=fps=60:round=near,setpts=N/(60*TB)," if force_60fps else ""
    # Sửa lỗi mkv/black pixels & Khử nhiễu cho mảng màu Anime/Live2D
    cleanup = "format=yuv420p,hqdn3d=1.5:1.5:3:3," if fix_black_pixels else ""
    # Logic Scale & Aspect Ratio
    if keep_aspect:
        vf_scale = f"scale={target_w}:{target_h}:flags=lanczos:force_original_aspect_ratio=decrease,pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2"
    else:
        vf_scale = f"scale={target_w}:{target_h}:flags=lanczos"
    
    # Sharpness cho Anime: Live2D cần sắc nét nhưng gameplay cần mượt
    sharp_val = "0.6" if fix_fade else "1.1"
    sharpen = f",unsharp=3:3:{sharp_val}:3:3:0.0"

    final_vf = f"'{vfr_fix}{cleanup}{vf_scale}{sharpen}'"

    print(f"🚀 Render: {label} | AI: {use_ai} | 20:9: {phone_ratio} | MKV/VFR Fix: On")

    # --- 5. TIẾN HÀNH RENDER ---
    input_source = merged_path
    
    if use_ai:
        print("🤖 AI Mode: Đang chạy Real-ESRGAN (Tối ưu cho Live2D/CG)...")
        if os.path.exists("/tmp/ai_out"): shutil.rmtree("/tmp/ai_out")
        os.makedirs("/tmp/ai_out", exist_ok=True)
        # Dùng model animevideov3 cực hợp cho Nikke/Arknights
        cmd_ai = f"python /root/Real-ESRGAN/inference_realesrgan_video.py -i {merged_path} -o /tmp/ai_out -n realesr-animevideov3 -s 2 --tile 1024"
        subprocess.run(cmd_ai, shell=True, check=True)
        shutil.move("/tmp/ai_out/merged_out.mp4", f"{work_dir}/ai_out/merged_out.mp4")
        input_source = f"{work_dir}/ai_out/merged_out.mp4"

    # Encode cuối cùng với NVENC (L40S)
    codec = "hevc_nvenc" if target_w > 4096 else "h264_nvenc"
    cmd_final = (
        f"ffmpeg -y -hwaccel cuda -i {input_source} "
        f"-map 0:v:0 -map 0:a:0? " # Xử lý track âm thanh MKV linh hoạt
        f"-vf {final_vf} "
        f"-c:v {codec} -preset p7 -rc vbr -cq 18 " # Preset p7 cho chất lượng cao nhất
        f"-b:v {bitrate} -maxrate 100M -bufsize 100M "
        f"-spatial-aq 1 -temporal-aq 1 -nonref_p 1 " # AQ giúp giữ chi tiết Live2D khi chuyển động
        f"-pix_fmt yuv420p -c:a aac -b:a 320k "     # Đổi audio sang AAC 320k chuẩn YouTube
        f"{final_video}"
    )
    
    subprocess.run(cmd_final, shell=True, check=True)

    # Dọn dẹp
    if os.path.exists(f"{work_dir}/inputs"): shutil.rmtree(f"{work_dir}/inputs")
    volume.commit()
    return filename

@app.local_entrypoint()
def main():
    # Dán link Drive file MKV của bạn vào đây
    display_id = "https://drive.google.com/file/d/1gVaI7kuXSsm9prvQ7gAFS4uu_qxqLz0r/view?usp=sharing"
    
    remote_filename = super_render.remote(
        drive_id=display_id,
        use_ai=False,           # Gameplay thì False, Live2D/CG thì bật True
        phone_ratio=True,       # Bật True cho màn hình dài 20:9
        keep_aspect=True, 
        target_4k=True,         # Luôn bật True để ép YouTube dùng VP9
        native_x2=False, 
        force_60fps=True,       # Fix xé hình từ scrcpy
        fix_black_pixels=True,
        fix_fade=True,          # Bật Safe Mode (độ nét vừa phải, mượt mà)
        force_rebuild=True
    )

    if not os.path.exists(LOCAL_DOWNLOAD_PATH): os.makedirs(LOCAL_DOWNLOAD_PATH)
    print(f"⏳ Đang tải kết quả về: {LOCAL_DOWNLOAD_PATH}...")
    subprocess.run(["modal", "volume", "get", "video_storage", f"/final_outputs/{remote_filename}", LOCAL_DOWNLOAD_PATH])
    print(f"✅ XONG! Video tại: {LOCAL_DOWNLOAD_PATH}/{remote_filename}")

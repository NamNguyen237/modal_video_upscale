import modal
import os
import subprocess
import shutil
import re
from datetime import datetime

# --- 1. MÔI TRƯỜNG CLOUD TỐI ƯU (L40S + Real-ESRGAN + FFmpeg) ---
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "git", "libgl1-mesa-glx", "libglib2.0-0", "unzip", "curl", "wget")
    .pip_install("torch", "torchvision", "cython", "setuptools<70")
    .pip_install("basicsr", "realesrgan", "numpy", "gdown", "ffmpeg-python")
    .run_commands(
        "git clone https://github.com/xinntao/Real-ESRGAN.git /root/Real-ESRGAN",
        "cd /root/Real-ESRGAN && pip install -r requirements.txt && python setup.py develop",
        # Fix lỗi tương thích thư viện Basicsr với Torchvision mới
        "sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/g' /usr/local/lib/python3.10/site-packages/basicsr/data/degradations.py"
    )
)

volume = modal.Volume.from_name("video_storage", create_if_missing=True)
app = modal.App("ultimate-renderer-v12-final", image=image)

LOCAL_DOWNLOAD_PATH = "D:/Rendered_Videos"

@app.function(gpu="L40S", cpu=16, memory=65536, volumes={"/data": volume}, timeout=21600, retries=0)
def super_render(drive_id: str, 
                 use_ai: bool = False,       # True cho Live2D/CG (Real-ESRGAN), False cho Gameplay (Fast Upscale)
                 phone_ratio: bool = True,   # True: ép về 20:9 (4800x2160), False: 16:9 (3840x2160)
                 keep_aspect: bool = False,  # True: giữ tỉ lệ gốc và thêm padding đen
                 target_4k: bool = True,     # True: Upscale lên 4K (Ưu tiên để lấy VP9 YouTube)
                 native_x2: bool = False,    # True: Chỉ nhân đôi độ phân giải gốc, bỏ qua chuẩn 4K
                 force_60fps: bool = True,   # Cực kỳ quan trọng để fix lỗi 0.00fps và xé hình từ scrcpy
                 zip_password: str = None, 
                 fix_black_pixels: bool = True, 
                 fix_fade: bool = True,      # Safe Mode: Giảm độ nét (0.7) để tránh artifact khi chuyển cảnh
                 force_rebuild: bool = False):
    
    import torch
    import zipfile

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    work_dir, final_dir = "/data/processing", "/data/final_outputs"
    merged_path = f"{work_dir}/merged.mp4"
    
    # Dọn dẹp nếu người dùng yêu cầu làm mới hoàn toàn
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

    # --- 2. TẢI & GIẢI NÉN ---
    if not os.path.exists(merged_path):
        temp_file = f"{work_dir}/temp_download"
        processed_id = fix_url(drive_id)
        if processed_id.startswith("http"): os.system(f"curl -L -o {temp_file} '{processed_id}'")
        else:
            import gdown
            gdown.download(f'https://drive.google.com/uc?id={processed_id}', temp_file, quiet=False, fuzzy=True)
            
        if os.path.exists(temp_file):
            if zipfile.is_zipfile(temp_file):
                print(f"📦 Đang giải nén file ZIP đầu vào...")
                with zipfile.ZipFile(temp_file, 'r') as z:
                    z.extractall(f"{work_dir}/inputs", pwd=zip_password.encode() if zip_password else None)
                os.remove(temp_file)
            else: 
                shutil.move(temp_file, merged_path)

    # --- 3. GHÉP FILE (NẾU CÓ NHIỀU FILE TRONG ZIP) ---
    if not os.path.exists(merged_path):
        files = sorted([f for f in os.listdir(f"{work_dir}/inputs") if f.lower().endswith((".mp4", ".mkv"))])
        if not files: raise Exception("❌ Không tìm thấy video hợp lệ!")
        list_path = f"{work_dir}/list.txt"
        with open(list_path, "w") as f:
            for file in files: f.write(f"file '{work_dir}/inputs/{file}'\n")
        print(f"🔗 Đang ghép {len(files)} video thành một bản duy nhất...")
        os.system(f"ffmpeg -y -f concat -safe 0 -i {list_path} -c copy {merged_path}")

    # --- 4. TÍNH TOÁN KÍCH THƯỚC ĐẦU RA ---
    probe_cmd = f"ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 {merged_path}"
    dimensions = subprocess.check_output(probe_cmd, shell=True).decode().strip().split('x')
    orig_w, orig_h = int(dimensions[0]), int(dimensions[1])

    if native_x2:
        target_w, target_h = orig_w * 2, orig_h * 2
        label = "NATIVE_X2"
    else:
        target_h = 2160 if target_4k else 1440
        ratio = (20/9) if phone_ratio else (16/9)
        target_w = int(target_h * ratio)
        target_w = target_w if target_w % 2 == 0 else target_w + 1
        label = "4K" if target_4k else "2K"

    bitrate = "80M" if target_4k else "45M"
    filename = f"RENDER_{label}_{timestamp}.mp4"
    final_video = f"{final_dir}/{filename}"

    # --- 5. XÂY DỰNG CHUỖI FILTER TỐI ƯU CHO SCRCPY & ANIME ---
    # 5.1. Sửa lỗi 0.00fps và Tearing: Ép về 60fps chuẩn và gán lại thời gian (PTS)
    vfr_fix = "fps=fps=60:round=near,setpts=N/(60*TB)," if force_60fps else ""
    # 5.2. Format & Khử nhiễu: Giúp mảng màu da nhân vật anime mịn màng, ít noise
    cleanup = "format=yuv420p,hqdn3d=1.5:1.5:3:3," if fix_black_pixels else ""
    # 5.3. Scale: Sử dụng Lanczos để giữ độ chi tiết cao nhất
    if keep_aspect:
        vf_scale = f"scale={target_w}:{target_h}:flags=lanczos:force_original_aspect_ratio=decrease,pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2"
    else:
        vf_scale = f"scale={target_w}:{target_h}:flags=lanczos"
    # 5.4. Làm nét & Màu sắc: Tăng nhẹ saturation cho game anime gacha rực rỡ
    sharp_val = "0.7" if fix_fade else "1.1"
    color_and_sharp = f",eq=saturation=1.1,unsharp=3:3:{sharp_val}:3:3:0.0"

    final_vf = f"'{vfr_fix}{cleanup}{vf_scale}{color_and_sharp}'"

    print(f"🚀 Render: {label} | AI: {use_ai} | Mode: {'Safe' if fix_fade else 'Normal'}")

    # --- 6. QUY TRÌNH XỬ LÝ CHÍNH ---
    input_source = merged_path
    if use_ai:
        print("🤖 AI Mode: Đang chạy Real-ESRGAN (AnimeVideoV3)...")
        if os.path.exists("/tmp/ai_out"): shutil.rmtree("/tmp/ai_out")
        os.makedirs("/tmp/ai_out", exist_ok=True)
        # Model v3 cực tốt cho game gacha và Live2D
        cmd_ai = f"python /root/Real-ESRGAN/inference_realesrgan_video.py -i {merged_path} -o /tmp/ai_out -n realesr-animevideov3 -s 2 --tile 1024"
        subprocess.run(cmd_ai, shell=True, check=True)
        shutil.move("/tmp/ai_out/merged_out.mp4", f"{work_dir}/ai_out/merged_out.mp4")
        input_source = f"{work_dir}/ai_out/merged_out.mp4"

    # Encode Final Video bằng NVENC (Hardware Accel trên L40S)
    # Tối ưu cho YouTube: Codec H.264, Preset P7, AQ Enabled
    cmd_final = (
        f"ffmpeg -y -hwaccel cuda -hwaccel_output_format cuda -i {input_source} "
        f"-map 0:v:0 -map 0:a:0? " # Trích xuất đúng 1 track video và 1 track audio
        f"-vf {final_vf} "
        f"-c:v h264_nvenc -preset p7 -rc vbr -cq 18 "
        f"-b:v {bitrate} -maxrate 110M -bufsize 110M "
        f"-spatial-aq 1 -temporal-aq 1 -nonref_p 1 " # Giữ chi tiết chuyển động combat
        f"-pix_fmt yuv420p -c:a aac -b:a 320k "     # Chuyển Opus sang AAC 320k chuẩn YouTube
        f"{final_video}"
    )
    
    subprocess.run(cmd_final, shell=True, check=True)

    # Commit dữ liệu vào Volume để lưu trữ lâu dài
    volume.commit()
    return filename

@app.local_entrypoint()
def main():
    # LINK DRIVE FILE MKV CỦA BẠN (Dán link full thoải mái)
    display_id = "https://drive.google.com/file/d/1J5QhuzmvxzsRSxFMlqCnlwhi9klTZD8X/view?usp=sharing"
    
    remote_filename = super_render.remote(
        drive_id=display_id,
        use_ai=False,           # Gameplay -> False (Nhanh), Live2D -> True (Nét)
        phone_ratio=True,       # 20:9 (Phù hợp file 2800x1264 của bạn)
        target_4k=True,         # Đẩy lên 4K để YouTube cấp codec VP9
        force_60fps=True,       # Bắt buộc để fix lỗi 0.00fps từ scrcpy
        fix_fade=True,          # Bật Safe Mode tránh vỡ hình khi chuyển cảnh
        force_rebuild=True
    )

    if not os.path.exists(LOCAL_DOWNLOAD_PATH): os.makedirs(LOCAL_DOWNLOAD_PATH)
    print(f"⏳ Đang tải video thành phẩm về: {LOCAL_DOWNLOAD_PATH}...")
    subprocess.run(["modal", "volume", "get", "video_storage", f"/final_outputs/{remote_filename}", LOCAL_DOWNLOAD_PATH])
    print(f"✅ HOÀN TẤT! Video chất lượng cao tại: {LOCAL_DOWNLOAD_PATH}/{remote_filename}")
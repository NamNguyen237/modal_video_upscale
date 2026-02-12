import modal
import os
import subprocess

# 1. Môi trường Cloud
# 1. Môi trường Cloud
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "git", "libgl1-mesa-glx", "libglib2.0-0", "unzip", "curl")
    .pip_install("torch", "torchvision", "cython", "setuptools<70")
    .pip_install("basicsr", "realesrgan", "numpy", "gdown", "ffmpeg-python")
    .run_commands(
        "git clone https://github.com/xinntao/Real-ESRGAN.git /root/Real-ESRGAN",
        "cd /root/Real-ESRGAN && pip install -r requirements.txt && python setup.py develop",
        "sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/g' /usr/local/lib/python3.10/site-packages/basicsr/data/degradations.py"
    )
)

volume = modal.Volume.from_name("video_storage")
app = modal.App("ultimate-renderer", image=image)

# ĐƯỜNG DẪN TRÊN MÁY DELL CỦA BẠN (Sửa lại cho đúng ý bạn)
LOCAL_DOWNLOAD_PATH = "D:/Rendered_Videos"

@app.function(gpu="L40S", volumes={"/data": volume}, timeout=11000, retries=0)
def super_render(drive_id: str, use_ai: bool = True, auto_upload_drive: bool = False):
    import gdown
    import zipfile
    import shutil

    # Thiết lập thư mục làm việc trên Cloud
    work_dir = "/data/processing"
    final_dir = "/data/final_outputs"
    for d in [work_dir, f"{work_dir}/inputs", f"{work_dir}/ai_out", final_dir]:
        if not os.path.exists(d): os.makedirs(d)

    # --- BƯỚC 1: TẢI & GIẢI NÉN ---
    print("🚀 Đang kéo file ZIP từ Drive...")
    zip_path = f"{work_dir}/temp.zip"
    gdown.download(f'https://drive.google.com/uc?id={drive_id}', zip_path, quiet=False)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(f"{work_dir}/inputs")
    
    # --- BƯỚC 2: GHÉP FILE ---
    files = sorted([f for f in os.listdir(f"{work_dir}/inputs") if f.endswith(".mp4")])
    list_path = f"{work_dir}/list.txt"
    with open(list_path, "w") as f:
        for file in files: f.write(f"file '{work_dir}/inputs/{file}'\n")
    
    merged = f"{work_dir}/merged.mp4"
    os.system(f"ffmpeg -y -f concat -safe 0 -i {list_path} -c copy {merged}")

    # --- BƯỚC 3: XỬ LÝ ĐỘ NÉT ---
    final_video = f"{final_dir}/FINAL_2K_20_9.mp4"
    if use_ai:
        print("🎨 Chế độ: AI UPSCALE (Real-ESRGAN)...")
        # Sử dụng inference_realesrgan_video.py từ repo đã clone
        cmd = (
            f"cd /root/Real-ESRGAN && python inference_realesrgan_video.py "
            f"-i {merged} "
            f"-n RealESRGAN_x4plus_anime_6B "
            f"-s 2 "
            f"--outscale 2 "
            f"--tile 400 "
            f"--fp32 "
            f"-o {work_dir}/ai_out"
        )
        ret = os.system(cmd)
        if ret != 0:
            raise RuntimeError(f"Real-ESRGAN failed with exit code {ret}")

        os.system(f"ffmpeg -y -i {work_dir}/ai_out/merged_out.mp4 -vf 'scale=3200:1440' -c:v h264_nvenc -b:v 25M -pix_fmt yuv420p {final_video}")
    else:
        print("⚡ Chế độ: FAST UPSCALE...")
        os.system(f"ffmpeg -y -hwaccel cuda -i {merged} -vf 'scale=3200:1440' -c:v h264_nvenc -b:v 25M -pix_fmt yuv420p {final_video}")

    # --- BƯỚC 4: TỰ ĐỘNG XỬ LÝ ĐẦU RA ---
    volume.commit()
    
    if auto_upload_drive:
        print("☁️ Đang đẩy ngược lên Drive (Yêu cầu cấu hình API)...")
        # Lưu ý: Upload lên Drive từ Cloud cần Token/API key. 
        # Để đơn giản, mình sẽ trả về đường dẫn để máy Dell tự tải.
    
    return final_video

@app.local_entrypoint()
def main():
    # 1. Chạy trên Cloud
    display_id = "1oSWVfM4V-bAGVysVXtWtP5tpjqjTjxgg"
    MY_ID = display_id # <--- THAY ID CỦA BẠN VÀO ĐÂY (Ví dụ: "1A2b3C...")
    
    if MY_ID == "ID_FILE_ZIP_CUA_BAN":
        raise ValueError("❌ BẠN CHƯA NHẬP ID FILE GOOGLE DRIVE! Vui lòng sửa dòng 'MY_ID' trong code.")

    IS_AI = True
    AUTO_DRIVE = False # Nếu để False, máy Dell sẽ tự tải về sau khi xong

    print("🎬 Bắt đầu quy trình Render Cloud...")
    result_path = super_render.remote(MY_ID, use_ai=IS_AI, auto_upload_drive=AUTO_DRIVE)

    # 2. TỰ ĐỘNG TẢI VỀ MÁY DELL KHI XONG (Nếu không upload Drive)
    if not AUTO_DRIVE:
        print(f"📥 Cloud đã xong! Đang tự động tải về: {LOCAL_DOWNLOAD_PATH}")
        if not os.path.exists(LOCAL_DOWNLOAD_PATH): os.makedirs(LOCAL_DOWNLOAD_PATH)
        
        # Lệnh tải file từ Modal Volume về thư mục chỉ định trên Dell
        subprocess.run(["modal", "volume", "get", "video_storage", "/final_outputs/FINAL_2K_20_9.mp4", LOCAL_DOWNLOAD_PATH])
        print(f"✅ ĐÃ TẢI XONG! Bạn kiểm tra tại: {LOCAL_DOWNLOAD_PATH}")

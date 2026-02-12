import modal
import os
import subprocess

# 1. Môi trường Cloud (Giữ nguyên cấu trúc của bạn)
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "git", "libgl1-mesa-glx", "libglib2.0-0", "unzip", "curl")
    .pip_install("torch", "torchvision", "cython", "setuptools<70")
    .pip_install("basicsr", "realesrgan", "numpy", "gdown", "ffmpeg-python")
    .run_commands(
        "git clone https://github.com/xinntao/Real-ESRGAN.git /root/Real-ESRGAN",
        "cd /root/Real-ESRGAN && pip install -r requirements.txt && python setup.py develop",
        # Fix lỗi thư viện quan trọng để chạy được trên GPU
        "sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/g' /usr/local/lib/python3.10/site-packages/basicsr/data/degradations.py"
    )
)

volume = modal.Volume.from_name("video_storage")
app = modal.App("ultimate-renderer", image=image)

LOCAL_DOWNLOAD_PATH = "D:/Rendered_Videos"

# --- ĐÂY LÀ HÀM XỬ LÝ CHÍNH ĐÃ CẢI TIẾN ---
@app.function(gpu="L40S", cpu=16, memory=32768, volumes={"/data": volume}, timeout=21600, retries=0)
def super_render(drive_id: str, use_ai: bool = True, auto_upload_drive: bool = False, force_rebuild: bool = False):
    import gdown
    import zipfile
    import shutil

    # Kiểm tra GPU
    import torch
    print(f"🔍 CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🔍 Current Device: {torch.cuda.get_device_name(0)}")
        
    # Thiết lập thư mục làm việc trên Cloud
    work_dir = "/data/processing"
    final_dir = "/data/final_outputs"
    
    # XÓA DỮ LIỆU CŨ NẾU CẦN (force_rebuild=True)
    if force_rebuild:
        print("♻️ Đang xóa dữ liệu cũ để chạy lại từ đầu...")
        if os.path.exists(f"{work_dir}/inputs"): shutil.rmtree(f"{work_dir}/inputs")
        if os.path.exists(f"{work_dir}/ai_out"): shutil.rmtree(f"{work_dir}/ai_out")
        if os.path.exists(f"{work_dir}/merged.mp4"): os.remove(f"{work_dir}/merged.mp4")

    for d in [work_dir, f"{work_dir}/inputs", f"{work_dir}/ai_out", final_dir]:
        if not os.path.exists(d): os.makedirs(d, exist_ok=True)

    # --- BƯỚC 1: TẢI & KIỂM TRA LOẠI FILE ---
    print("🚀 Đang kéo file từ nguồn...")
    temp_file = f"{work_dir}/temp_download"
    merged_path = f"{work_dir}/merged.mp4"
    
    # Helper để fix link Pixeldrain
    def fix_url(url):
        if "pixeldrain.com/u/" in url:
            print("🔧 Auto-fix Pixeldrain Link...")
            return url.replace("/u/", "/api/file/")
        return url

    # Check kỹ xem file đã có chưa để skip download
    if not os.path.exists(merged_path):
        # Tạo thư mục inputs nếu chưa có
        if not os.path.exists(f"{work_dir}/inputs"): os.makedirs(f"{work_dir}/inputs", exist_ok=True)
        
        # Chỉ tải nếu chưa có file input hoặc buộc tải lại
        if not os.listdir(f"{work_dir}/inputs") and not os.path.exists(merged_path):
            
            # --- TRƯỜNG HỢP 1: DANH SÁCH URL (NHIỀU PART) ---
            if isinstance(drive_id, list):
                print(f"📥 Phát hiện danh sách {len(drive_id)} file. Đang tải từng phần...")
                for i, url in enumerate(drive_id):
                    url = fix_url(url) # Fix link
                    part_path = f"{work_dir}/inputs/part_{i:04d}.mp4"
                    print(f"  ⬇️ Đang tải phần {i+1}: {url}")
                    try:
                        if url.startswith("http"):
                            os.system(f"curl -L -o {part_path} '{url}'")
                        else:
                            gdown.download(f'https://drive.google.com/uc?id={url}', part_path, quiet=False, fuzzy=True)
                        
                        # Check size
                        if os.path.exists(part_path) and os.path.getsize(part_path) < 1024*1024:
                             raise ValueError(f"File {part_path} quá nhỏ (<1MB). Kiểm tra lại link (có thể là file HTML lỗi)!")

                    except Exception as e:
                        print(f"⚠️ Lỗi tải phần {i+1}: {e}")
                        raise e
                print("✅ Đã tải xong tất cả các phần.")

            # --- TRƯỜNG HỢP 2: MỘT URL/ID DUY NHẤT ---
            else:
                drive_id = fix_url(drive_id) # Fix link
                print(f"📥 Đang tải file đơn từ: {drive_id}")
                try:
                    if drive_id.startswith("http"):
                        print("🔗 Phát hiện Direct Link/URL, sử dụng CURL...")
                        os.system(f"curl -L -o {temp_file} '{drive_id}'")
                    else:
                        print("🔗 Phát hiện Google Drive ID, sử dụng GDOWN...")
                        gdown.download(f'https://drive.google.com/uc?id={drive_id}', temp_file, quiet=False, fuzzy=True)
                    
                    # Check size
                    if os.path.exists(temp_file) and os.path.getsize(temp_file) < 1024*1024:
                         with open(temp_file, 'r', errors='ignore') as f: preview = f.read(500)
                         print(f"📄 Nội dung file tải về (500 chars): {preview}")
                         raise ValueError("File quá nhỏ (<1MB). Có thể link sai hoặc là file HTML lỗi!")

                    # Kiểm tra xem là ZIP hay VIDEO
                    if zipfile.is_zipfile(temp_file):
                        print("📦 File là dạng ZIP. Tiến hành giải nén...")
                        with zipfile.ZipFile(temp_file, 'r') as z:
                            z.extractall(f"{work_dir}/inputs")
                        os.remove(temp_file) # Dọn dẹp
                    else:
                        print("🎥 File là dạng VIDEO đơn (không phải ZIP). Bỏ qua bước giải nén & ghép.")
                        shutil.move(temp_file, merged_path)
                        
                except Exception as e:
                    print(f"⚠️ Lỗi tải file: {e}")
                    raise e

    # --- BƯỚC 2: GHÉP FILE (Chỉ chạy nếu là ZIP/LIST và chưa có merged.mp4) ---
    if os.path.exists(merged_path):
        print("✅ Đã có file merged.mp4 (Video đơn hoặc đã ghép xong). Bỏ qua bước ghép.")
    else:
        # Trường hợp này là ZIP giải nén hoặc LIST URL
        print("🧩 Đang ghép các file video...")
        files = sorted([f for f in os.listdir(f"{work_dir}/inputs") if f.endswith(".mp4")])
        if not files:
            raise ValueError("❌ Không tìm thấy file .mp4 nào trong thư mục inputs!")
            
        list_path = f"{work_dir}/list.txt"
        with open(list_path, "w") as f:
            for file in files: f.write(f"file '{work_dir}/inputs/{file}'\n")
        
        os.system(f"ffmpeg -y -f concat -safe 0 -i {list_path} -c copy {merged_path}")
        print("✅ Ghép file hoàn tất!")

    # --- BƯỚC 3: XỬ LÝ ĐỘ NÉT ---
    final_video = f"{final_dir}/FINAL_2K_20_9.mp4"
    
    if use_ai:
        print("🎨 Chế độ: AI UPSCALE (realesr-animevideov3) on L40S...")
        # Kiểm tra file đầu vào có hợp lệ không
        if os.path.getsize(merged_path) < 1000:
            raise ValueError("❌ File video lỗi (quá nhỏ). Kiểm tra lại link tải!")
        # Sử dụng model chuyên dụng cho Video - Nhanh hơn & Mượt hơn
        cmd = (
            f"cd /root/Real-ESRGAN && python inference_realesrgan_video.py "
            f"-i {merged_path} "
            f"-n realesr-animevideov3 " # Model video xịn
            f"-s 2 " # Scale x2
            f"--suffix _out "
            f"--tile 640 "  # Tile nhỏ để xử lý nhanh hơn
            f"--pre_pad 0 "
            f"--num_process_per_gpu 3 " # Chạy song song 3 luồng để tận dụng hết L40S
            f"-o {work_dir}/ai_out"
        )
        
        ret = os.system(cmd)
        if ret != 0:
            raise RuntimeError(f"Real-ESRGAN failed với lỗi: {ret}")

        # Ghép audio và scale chuẩn cuối cùng
        print("🎬 Đang đóng gói video cuối cùng bằng GPU...")
        os.system(f"ffmpeg -y -i {work_dir}/ai_out/merged_out.mp4 -vf 'scale=3200:1440' -c:v h264_nvenc -b:v 25M -pix_fmt yuv420p {final_video}")
    else:
        print("⚡ Chế độ: FAST UPSCALE...")
        os.system(f"ffmpeg -y -hwaccel cuda -i {merged_path} -vf 'scale=3200:1440' -c:v h264_nvenc -b:v 25M -pix_fmt yuv420p {final_video}")

    # --- BƯỚC 4: TỰ ĐỘNG XỬ LÝ ĐẦU RA (Giữ nguyên) ---
    volume.commit()
    return final_video

@app.local_entrypoint()
def main():
    #drive_id = [
    #    "ID_FILE_CHINH",
    #    "ID_FILE_NGOAC_DON_1",
    #    "ID_FILE_NGOAC_DON_2"
    #]
    display_id = "https://pixeldrain.com/u/ekrwj8xa"
    MY_ID = display_id 
    IS_AI = True # Bật AI Upscale
    AUTO_DRIVE = False 
    FORCE_REBUILD = True # <--- Đặt thành TRUE nếu muốn xóa cũ tải mới

    print("🎬 Bắt đầu quy trình Render Cloud...")
    result_path = super_render.remote(MY_ID, use_ai=IS_AI, auto_upload_drive=AUTO_DRIVE, force_rebuild=FORCE_REBUILD)

    if not AUTO_DRIVE:
        print(f"📥 Cloud đã xong! Đang tự động tải về: {LOCAL_DOWNLOAD_PATH}")
        if not os.path.exists(LOCAL_DOWNLOAD_PATH): os.makedirs(LOCAL_DOWNLOAD_PATH)
        subprocess.run(["modal", "volume", "get", "video_storage", "/final_outputs/FINAL_2K_20_9.mp4", LOCAL_DOWNLOAD_PATH])
        print(f"✅ ĐÃ TẢI XONG! Kiểm tra tại: {LOCAL_DOWNLOAD_PATH}")

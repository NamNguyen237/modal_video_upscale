import modal
import os
import subprocess
import shutil

# 1. Môi trường Cloud (Giữ nguyên cấu trúc của bạn)
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "git", "libgl1-mesa-glx", "libglib2.0-0", "unzip", "curl", "wget")
    .pip_install("torch", "torchvision", "cython", "setuptools<70")
    .pip_install("basicsr", "realesrgan", "numpy", "gdown", "ffmpeg-python")
    .run_commands(
        "git clone https://github.com/xinntao/Real-ESRGAN.git /root/Real-ESRGAN",
        "cd /root/Real-ESRGAN && pip install -r requirements.txt && python setup.py develop",
        # Fix lỗi thư viện quan trọng
        "sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/g' /usr/local/lib/python3.10/site-packages/basicsr/data/degradations.py"
    )
)

volume = modal.Volume.from_name("video_storage")
app = modal.App("ultimate-renderer-v2", image=image)

LOCAL_DOWNLOAD_PATH = "D:/Rendered_Videos"

@app.function(gpu="L40S", cpu=16, memory=32768, volumes={"/data": volume}, timeout=21600, retries=0)
def super_render(drive_id: str, use_ai: bool = True, auto_upload_drive: bool = False, force_rebuild: bool = False):
    import gdown
    import zipfile
    import torch

    # Kiểm tra GPU (Giữ nguyên print của bạn)
    print(f"🔍 CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🔍 Current Device: {torch.cuda.get_device_name(0)}")
        
    work_dir = "/data/processing"
    final_dir = "/data/final_outputs"
    merged_path = f"{work_dir}/merged.mp4"
    final_video = f"{final_dir}/FINAL_2K_20_9.mp4"

    # XÓA DỮ LIỆU CŨ (Giữ nguyên logic của bạn)
    if force_rebuild:
        print("♻️ Đang xóa dữ liệu cũ để chạy lại từ đầu...")
        for path in [f"{work_dir}/inputs", f"{work_dir}/ai_out", f"{work_dir}/frames_in", f"{work_dir}/frames_out", merged_path]:
            if os.path.exists(path):
                if os.path.isdir(path): shutil.rmtree(path)
                else: os.remove(path)

    for d in [work_dir, f"{work_dir}/inputs", f"{work_dir}/ai_out", f"{work_dir}/frames_in", f"{work_dir}/frames_out", final_dir]:
        os.makedirs(d, exist_ok=True)

    # --- BƯỚC 1: TẢI & KIỂM TRA (Giữ nguyên logic của bạn) ---
    def fix_url(url):
        if "pixeldrain.com/u/" in url:
            print("🔧 Auto-fix Pixeldrain Link...")
            return url.replace("/u/", "/api/file/")
        return url

    if not os.path.exists(merged_path):
        temp_file = f"{work_dir}/temp_download"
        if isinstance(drive_id, list):
            print(f"📥 Phát hiện danh sách {len(drive_id)} file. Đang tải từng phần...")
            for i, url in enumerate(drive_id):
                url = fix_url(url)
                part_path = f"{work_dir}/inputs/part_{i:04d}.mp4"
                os.system(f"curl -L -o {part_path} '{url}'")
        else:
            drive_id = fix_url(drive_id)
            if drive_id.startswith("http"):
                os.system(f"curl -L -o {temp_file} '{drive_id}'")
            else:
                gdown.download(f'https://drive.google.com/uc?id={drive_id}', temp_file, quiet=False, fuzzy=True)
            
            if zipfile.is_zipfile(temp_file):
                with zipfile.ZipFile(temp_file, 'r') as z: z.extractall(f"{work_dir}/inputs")
                os.remove(temp_file)
            else:
                shutil.move(temp_file, merged_path)

    # --- BƯỚC 2: GHÉP FILE ---
    if not os.path.exists(merged_path):
        print("🧩 Đang ghép các file video...")
        files = sorted([f for f in os.listdir(f"{work_dir}/inputs") if f.endswith(".mp4")])
        list_path = f"{work_dir}/list.txt"
        with open(list_path, "w") as f:
            for file in files: f.write(f"file '{work_dir}/inputs/{file}'\n")
        os.system(f"ffmpeg -y -f concat -safe 0 -i {list_path} -c copy {merged_path}")

    # --- BƯỚC 3: XỬ LÝ ĐỘ NÉT (CẢI TIẾN 6 LUỒNG) ---
    if use_ai:
        print("🎨 Chế độ: AI UPSCALE (Image Sequence + 6 Luồng) on L40S...")
        
        input_frames = f"{work_dir}/frames_in"
        output_frames = f"{work_dir}/frames_out"

        # Tách khung hình
        print("🎞️ Đang tách khung hình để tối ưu GPU...")
        os.system(f"ffmpeg -y -i {merged_path} -qscale:v 2 {input_frames}/f_%08d.jpg")

        # Chạy inference với 6 luồng song song
        # Dùng bản script VIDEO nhưng truyền input là thư mục ảnh
        os.makedirs("/tmp/ai_out", exist_ok=True)
        cmd = (
            f"python /root/Real-ESRGAN/inference_realesrgan_video.py "
            f"-i {merged_path} " # Chạy trực tiếp từ file video, không tách ảnh nữa
            f"-o /tmp/ai_out "
            f"-n realesr-animevideov3 "
            f"-s 2 "
            f"--tile 1024 " # Tăng tile lên để GPU tính toán tập trung
            f"--num_process_per_gpu 2 " # Chỉ dùng 2 luồng để tránh nghẽn I/O
        )
        
        print("🚀 Đang thực thi AI Inference (2 luồng song song và ghi vào tmp)...")
        subprocess.run(cmd, shell=True, check=True)
        
        # Xong rồi mới copy kết quả về Volume
        shutil.move("/tmp/ai_out/merged_out.mp4", f"{work_dir}/ai_out/merged_out.mp4")
        # Đóng gói video bằng NVENC (GPU)
        print("🎬 Đang đóng gói video cuối cùng bằng GPU...")
        fps_cmd = f"ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1 {merged_path}"
        fps = subprocess.check_output(fps_cmd, shell=True).decode().strip()

        # Kết hợp Upscale + Scale chuẩn + Audio gốc
        os.system(f"ffmpeg -y -r {fps} -i {output_frames}/f_%08d_out.jpg -i {merged_path} "
                  f"-map 0:v:0 -map 1:a? -vf 'scale=3200:1440' -c:a copy -c:v h264_nvenc -preset p4 -b:v 25M -pix_fmt yuv420p {final_video}")
    else:
        print("⚡ Chế độ: FAST UPSCALE...")
        #cũ:

        #os.system(f"ffmpeg -y -hwaccel cuda -i {merged_path} -vf 'scale=3200:1440' -c:v h264_nvenc -b:v 25M -pix_fmt yuv420p {final_video}")
        #os.system(f"ffmpeg -y -hwaccel cuda -i {merged_path} -vf 'hqdn3d,scale=3200:1440:flags=lanczos,unsharp=5:5:1.0:5:5:0.0' -c:v h264_nvenc -b:v 25M {final_video}")

        # Cho chất lượng cực tốt mà tốc độ gần như real-time:
        os.system(f"ffmpeg -y -hwaccel cuda -i {merged_path} "
                  f"-vf 'scale=3200:1440:flags=lanczos,unsharp=3:3:1.5' "
                  f"-c:v h264_nvenc -preset p4 -b:v 30M -pix_fmt yuv420p {final_video}")


    volume.commit()
    return final_video

@app.local_entrypoint()
def main():
    display_id = "https://www.dropbox.com/scl/fi/vsa7y4qjj5tsr1dlgu6zl/belfast-luminousart_1080p.mp4?rlkey=3arhhckihmmtq1gaimjbn6k54&st=xarssrvw&dl=1"
    print("🎬 Bắt đầu quy trình Render Cloud...")
    super_render.remote(display_id, use_ai=False, force_rebuild=False)

    if not os.path.exists(LOCAL_DOWNLOAD_PATH): os.makedirs(LOCAL_DOWNLOAD_PATH)
    subprocess.run(["modal", "volume", "get", "video_storage", "/final_outputs/FINAL_2K_20_9.mp4", LOCAL_DOWNLOAD_PATH])
    print(f"✅ ĐÃ TẢI XONG! Kiểm tra tại: {LOCAL_DOWNLOAD_PATH}")
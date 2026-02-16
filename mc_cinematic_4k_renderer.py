import modal
import subprocess
import os
import shutil
from datetime import datetime

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "curl")
    .pip_install("gdown")
)

volume = modal.Volume.from_name("video_storage", create_if_missing=True)
app = modal.App("mc-cinematic-4k-ultra", image=image)

LOCAL_DOWNLOAD_PATH = "D:/Rendered_Videos"

@app.function(
    gpu="L40S", 
    cpu=16, 
    memory=65536, 
    volumes={"/data": volume}, 
    timeout=14400
)
def cloud_render(video_url: str, audio_url: str, use_sharpen: bool = True, force_rebuild: bool = False):
    import gdown
    # Thư mục lưu trữ cố định trên Volume để tránh tải lại
    storage_dir = "/data/raw_assets"
    final_dir = "/data/final_outputs"
    os.makedirs(storage_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)

    v_path = f"{storage_dir}/input_v.mp4"
    a_path = f"{storage_dir}/input_a.mp3"

    # --- KIỂM TRA VÀ TẢI FILE (ANTI-LIMIT GOOGLE DRIVE) ---
    def smart_download(url, path, label):
        # Nếu không ép rebuild và file đã tồn tại -> Bỏ qua
        if not force_rebuild and os.path.exists(path) and os.path.getsize(path) > 1000:
            print(f"✅ {label} đã tồn tại trên Volume. Bỏ qua tải xuống.")
        else:
            print(f"📥 Đang tải {label} từ Google Drive...")
            if os.path.exists(path): os.remove(path)
            gdown.download(url, path, quiet=False, fuzzy=True)
            # Commit ngay sau khi tải để lưu lại file vào Volume
            volume.commit()

    smart_download(video_url, v_path, "Video gốc")
    smart_download(audio_url, a_path, "Nhạc nền")

    # --- TÍNH TOÁN DURATION ---
    def get_dur(p):
        cmd = f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {p}"
        return float(subprocess.check_output(cmd, shell=True).decode().strip())

    try:
        v_dur = get_dur(v_path)
        a_dur = get_dur(a_path)
    except Exception as e:
        print(f"❌ Lỗi đọc file: {e}. Có thể link Drive bị lỗi hoặc file hỏng.")
        return None

    factor = a_dur / v_dur
    sharp_filter = ",unsharp=3:3:0.8:3:3:0.0" if use_sharpen else ""
    vf_chain = f"setpts={factor}*PTS,scale=3840:2160:flags=lanczos{sharp_filter},fps=60"

    timestamp = datetime.now().strftime('%H%M')
    out_name = f"MC_4K_{'SHARP_' if use_sharpen else ''}{timestamp}.mp4"
    out_path = f"{final_dir}/{out_name}"

    # --- RENDER VỚI GPU L40S ---
    print(f"🚀 Render 4K: Factor={factor:.4f} | Sharpen={use_sharpen}")
    cmd = (
        f'ffmpeg -y -hwaccel cuda -i {v_path} -i {a_path} '
        f'-filter_complex "[0:v]{vf_chain}[v]" '
        f'-map "[v]" -map 1:a '
        f'-c:v h264_nvenc -preset p7 -tune hq -rc vbr -cq 18 '
        f'-b:v 80M -maxrate 120M -bufsize 150M -pix_fmt yuv420p -shortest {out_path}'
    )
    
    subprocess.run(cmd, shell=True, check=True)
    volume.commit() # Lưu file thành phẩm vào Volume
    return out_name

@app.local_entrypoint()
def main():
    VIDEO_DRIVE_LINK = ""
    AUDIO_DRIVE_LINK = ""

    # force_rebuild=False: Nếu file đã có trên Cloud thì không tải lại từ Drive nữa
    remote_filename = cloud_render.remote(
        video_url=VIDEO_DRIVE_LINK, 
        audio_url=AUDIO_DRIVE_LINK, 
        use_sharpen=True,
        force_rebuild=False
    )

    if remote_filename:
        if not os.path.exists(LOCAL_DOWNLOAD_PATH): os.makedirs(LOCAL_DOWNLOAD_PATH)
        print(f"📥 Đang lấy file từ Volume về máy local...")
        subprocess.run(["modal", "volume", "get", "video_storage", f"/final_outputs/{remote_filename}", LOCAL_DOWNLOAD_PATH])
        print(f"✅ HOÀN THÀNH: {LOCAL_DOWNLOAD_PATH}/{remote_filename}")

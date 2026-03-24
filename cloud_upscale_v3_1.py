import modal
import os
import subprocess
import shutil
import re
from datetime import datetime

# --- CẤU HÌNH ĐƯỜNG DẪN LOCAL ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_DOWNLOAD_PATH = os.path.join(BASE_DIR, "Rendered_Videos")

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
app = modal.App("ultimate-parallel-v14-stable", image=image)

# --- HÀM WORKER: XỬ LÝ TỪNG VIDEO TRÊN 1 GPU RIÊNG BIỆT ---
@app.function(gpu="L40S", cpu=8, memory=32768, volumes={"/data": volume}, timeout=7200)
def render_part_worker(v_path, target_w, target_h, use_ai, force_60fps, fix_black_pixels, fix_fade, keep_aspect, bitrate, part_id):
    import os
    import subprocess
    
    print(f"🎬 Đang xử lý file: {os.path.basename(v_path)}")
    part_output = f"/data/processing/parts/part_{part_id:03d}.mp4"
    os.makedirs("/data/processing/parts", exist_ok=True)
    
    current_input = v_path
    
    # 1. AI Upscale (Real-ESRGAN)
    if use_ai:
        cmd_ai = f"python /root/Real-ESRGAN/inference_realesrgan_video.py -i '{v_path}' -o /tmp -n realesr-animevideov3 -s 2 --tile 1024"
        subprocess.run(cmd_ai, shell=True, check=True)
        current_input = f"/tmp/{os.path.basename(v_path).rsplit('.', 1)[0]}_out.mp4"

    # 2. Xây dựng Filter Chain (Đã FIX lỗi dấu cách và cú pháp)
    filters = []
    if force_60fps:
        filters.append("fps=fps=60:round=near,setpts=N/(60*TB)")
    if fix_black_pixels:
        filters.append("format=yuv420p,hqdn3d=1.5:1.5:3:3")
    
    # Scale & Pad
    scale_str = f"scale={target_w}:{target_h}:flags=lanczos"
    if keep_aspect:
        scale_str += f":force_original_aspect_ratio=decrease,pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2"
    filters.append(scale_str)
    
    # Sharpen (FIXED: Không có dấu cách thừa)
    sharp_val = "0.6" if fix_fade else "1.1"
    filters.append(f"unsharp=3:3:{sharp_val}:3:3:0.0")

    final_vf_string = ",".join(filters)

    # 3. Encode (Dùng CPU decode để ổn định, GPU encode để tốc độ)
    cmd_final = [
        "ffmpeg", "-y",
        "-fflags", "+genpts+discardcorrupt", 
        "-i", current_input,
        "-map", "0:v:0", "-map", "0:a:0?",
        "-vf", final_vf_string,
        "-c:v", "hevc_nvenc", "-preset", "p7", "-rc", "vbr", "-cq", "18",
        "-b:v", bitrate, "-maxrate", "100M", "-bufsize", "100M",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "320k",
        part_output
    ]
    
    # Chạy lệnh và in log nếu có lỗi
    result = subprocess.run(cmd_final, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ FFmpeg Error (Part {part_id}): {result.stderr}")
        raise subprocess.CalledProcessError(result.returncode, cmd_final)
        
    return part_output

# --- HÀM ĐIỀU PHỐI CHÍNH ---
@app.function(volumes={"/data": volume}, timeout=21600)
def super_render(drive_id: str, 
                 parallel_render: bool = True, 
                 use_ai: bool = True,       
                 phone_ratio: bool = True,   
                 keep_aspect: bool = False,  
                 target_4k: bool = False,    
                 native_x2: bool = False,    
                 force_60fps: bool = True,   
                 zip_password: str = None, 
                 fix_black_pixels: bool = True, 
                 fix_fade: bool = True,      
                 force_rebuild: bool = False):
    
    import zipfile
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    work_dir = "/data/processing"
    inputs_dir = f"{work_dir}/inputs"
    final_dir = "/data/final_outputs"
    
    if force_rebuild and os.path.exists(work_dir):
        shutil.rmtree(work_dir)

    for d in [work_dir, inputs_dir, f"{work_dir}/parts", final_dir]:
        os.makedirs(d, exist_ok=True)

    # --- TẢI FILE ---
    temp_file = f"{work_dir}/temp_download"
    drive_match = re.search(r'(?:id=|/d/)([\w-]+)', drive_id)
    if "drive.google.com" in drive_id and drive_match:
        import gdown
        gdown.download(f'https://drive.google.com/uc?id={drive_match.group(1)}', temp_file, quiet=False, fuzzy=True)
    else:
        os.system(f"curl -L -o {temp_file} '{drive_id}'")

    # --- GIẢI NÉN ---
    if zipfile.is_zipfile(temp_file):
        with zipfile.ZipFile(temp_file, 'r') as z:
            z.extractall(inputs_dir, pwd=zip_password.encode() if zip_password else None)
        os.remove(temp_file)
    else:
        shutil.move(temp_file, f"{inputs_dir}/video_single.mp4")

    video_files = sorted([os.path.join(r, f) for r, _, fs in os.walk(inputs_dir) 
                         for f in fs if f.lower().endswith((".mp4", ".mkv", ".webm")) and not f.startswith(".")])

    if not video_files: raise Exception("❌ Không tìm thấy video!")

    # --- TÍNH TOÁN TARGET ---
    probe_cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", video_files[0]]
    dims = subprocess.check_output(probe_cmd).decode().strip().split('x')
    orig_w, orig_h = int(dims[0]), int(dims[1])
    
    target_h = 2160 if target_4k else 1440
    if native_x2:
        target_w, target_h = orig_w * 2, orig_h * 2
    else:
        ratio = (20/9) if phone_ratio else (16/9)
        target_w = int(target_h * ratio)
        target_w = target_w if target_w % 2 == 0 else target_w + 1
    
    bitrate = "75M" if target_4k else "40M"

    # --- CHẠY RENDER ---
    render_args = (target_w, target_h, use_ai, force_60fps, fix_black_pixels, fix_fade, keep_aspect, bitrate)
    
    if parallel_render:
        print(f"🔥 Mode: SONG SONG ({len(video_files)} GPUs)")
        processed_paths = list(render_part_worker.starmap(
            [(v, *render_args, i) for i, v in enumerate(video_files)]
        ))
    else:
        print(f"⏳ Mode: TUẦN TỰ")
        processed_paths = [render_part_worker.remote(v, *render_args, i) for i, v in enumerate(video_files)]

    # --- GHÉP FILE CUỐI ---
    final_name = f"FINAL_RENDER_{timestamp}.mp4"
    final_path = f"{final_dir}/{final_name}"
    list_path = f"{work_dir}/list.txt"
    
    with open(list_path, "w", encoding="utf-8") as f:
        for p in processed_paths: f.write(f"file '{p}'\n")
    
    os.system(f"ffmpeg -y -f concat -safe 0 -i {list_path} -c copy {final_path}")
    
    volume.commit()
    return final_name

@app.local_entrypoint()
def main():
    # LINK DRIVE CỦA BẠN
    drive_link = "https://drive.google.com/file/d/1gVaI7kuXSsm9prvQ7gAFS4uu_qxqLz0r/view?usp=sharing"
    
    res = super_render.remote(
        drive_id=drive_link,
        parallel_render=True, # Bật nhiều GPU cùng lúc
        use_ai=False,         # Để False nếu chỉ muốn test tốc độ chuẩn hóa
        target_4k=True,
        force_rebuild=True
    )
    
    if not os.path.exists(LOCAL_DOWNLOAD_PATH): os.makedirs(LOCAL_DOWNLOAD_PATH)
    print(f"⏳ Đang tải kết quả về: {LOCAL_DOWNLOAD_PATH}...")
    subprocess.run(["modal", "volume", "get", "video_storage", f"/final_outputs/{res}", LOCAL_DOWNLOAD_PATH], shell=(os.name=='nt'))
    print(f"✅ HOÀN THÀNH! File tại: {os.path.join(LOCAL_DOWNLOAD_PATH, res)}")
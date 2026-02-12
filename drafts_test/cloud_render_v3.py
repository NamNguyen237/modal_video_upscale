
import modal
import os
import shutil
import subprocess

# 1. Environment
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "git", "libgl1-mesa-glx", "libglib2.0-0", "unzip", "curl", "wget", "libvulkan1", "vulkan-tools")
    .pip_install("gdown", "requests")
)

volume = modal.Volume.from_name("video_storage")
app = modal.App("ultra-fast-upscale-rife", image=image)

@app.function(gpu="L40S", volumes={"/data": volume}, timeout=86400)
def fast_upscale(drive_id: str or list, use_rife: bool = True, force_rebuild: bool = False):
    import gdown
    import zipfile
    
    # Setup working dirs
    root_dir = "/data/upscale_v3"
    inputs_dir = f"{root_dir}/inputs"
    frames_in = f"{root_dir}/frames_in"
    frames_rife = f"{root_dir}/frames_rife" # Sau khi nội suy
    frames_out = f"{root_dir}/frames_out" # Sau khi upscale
    bin_dir = "/root/bin"
    merged_input = f"{root_dir}/merged.mp4"
    final_output = f"{root_dir}/FINAL_OUTPUT.mp4"
    
    if force_rebuild:
        print("♻️ Cleaning workspace...")
        if os.path.exists(root_dir): shutil.rmtree(root_dir)
        
    for d in [root_dir, inputs_dir, bin_dir, frames_in, frames_rife, frames_out]:
        os.makedirs(d, exist_ok=True)

    # 1. Setup Binaries (Real-ESRGAN & RIFE)
    re_exe = f"{bin_dir}/realesrgan-ncnn-vulkan"
    rife_exe = f"{bin_dir}/rife-ncnn-vulkan"
    
    if not os.path.exists(re_exe):
        print("🔧 Downloading Real-ESRGAN Binary...")
        url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip"
        os.system(f"wget {url} -O {bin_dir}/re.zip && unzip {bin_dir}/re.zip -d {bin_dir} && chmod +x {re_exe}")
        
    if use_rife and not os.path.exists(rife_exe):
        print("🔧 Downloading RIFE AI Binary (Frame Interpolation)...")
        # Corrected URL for Linux/Ubuntu release
        url = "https://github.com/nihui/rife-ncnn-vulkan/releases/download/20221029/rife-ncnn-vulkan-20221029-ubuntu.zip"
        os.system(f"wget {url} -O {bin_dir}/rife.zip")
        with zipfile.ZipFile(f"{bin_dir}/rife.zip", 'r') as z: 
            z.extractall(bin_dir)
            
        # Find the extracted folder (it usually starts with rife-ncnn-vulkan-)
        extracted_items = [d for d in os.listdir(bin_dir) if d.startswith("rife-ncnn-vulkan-") and os.path.isdir(os.path.join(bin_dir, d))]
        if extracted_items:
            folder_name = extracted_items[0]
            extracted_path = os.path.join(bin_dir, folder_name)
            
            # Move binary
            shutil.move(f"{extracted_path}/rife-ncnn-vulkan", rife_exe)
            # Move models
            if os.path.exists(f"{bin_dir}/models_rife"): shutil.rmtree(f"{bin_dir}/models_rife")
            shutil.move(f"{extracted_path}/models", f"{bin_dir}/models_rife")
            
            print(f"✅ RIFE setup complete from {folder_name}")
        else:
            raise RuntimeError("Could not find extracted RIFE folder in bin_dir")
            
        os.system(f"chmod +x {rife_exe}")

    # 2. Download Input
    if not os.path.exists(merged_input) or force_rebuild:
        print("🚀 Downloading input...")
        urls = drive_id if isinstance(drive_id, list) else [drive_id]
        downloaded = []
        for i, url in enumerate(urls):
            if "pixeldrain.com/u/" in url: url = url.replace("/u/", "/api/file/")
            dest = f"{inputs_dir}/part_{i:04d}.mp4"
            if not os.path.exists(dest):
                 if url.startswith("http"): os.system(f"curl -L -o {dest} '{url}'")
                 else: gdown.download(f'https://drive.google.com/uc?id={url}', dest, quiet=False, fuzzy=True)
            downloaded.append(dest)
        if len(downloaded) == 1: shutil.move(downloaded[0], merged_input)
        else:
             with open(f"{root_dir}/list.txt", "w") as f:
                 for v in downloaded: f.write(f"file '{v}'\n")
             os.system(f"ffmpeg -y -f concat -safe 0 -i {root_dir}/list.txt -c copy {merged_input}")

    # 3. Probe Info
    fps_out = subprocess.check_output(f"ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 {merged_input}", shell=True).decode().strip()
    n, d = map(int, fps_out.split('/')) if '/' in fps_out else (float(fps_out), 1)
    fps_val = n/d
    print(f"🎥 Video FPS: {fps_val}")

    # 4. EXTRACT
    if not os.listdir(frames_in):
        print("🎞️ Extracting frames...")
        os.system(f"ffmpeg -i {merged_input} -qscale:v 1 -qmin 1 -qmax 1 {frames_in}/frame_%08d.jpg")

    # 5. RIFE AI INTERPOLATION (Nội suy FPS)
    current_frames = frames_in
    target_fps = fps_val
    if use_rife:
        print("🧠 Running RIFE AI: Interpolating frames (2x FPS)...")
        # RIFE mặc định nội suy 2x. 30fps -> 60fps.
        target_fps = fps_val * 2
        cmd = f"{rife_exe} -i {frames_in} -o {frames_rife} -m {bin_dir}/models_rife -n rife-v4"
        ret = os.system(cmd)
        if ret != 0: raise RuntimeError("RIFE Failed")
        current_frames = frames_rife
        print(f"✅ RIFE Finished. New Target FPS: {target_fps}")

    # 6. UPSCALE (Làm nét)
    if not os.listdir(frames_out):
        print("⚡ Starting Real-ESRGAN Upscale (NCNN)...")
        cmd = f"{re_exe} -i {current_frames} -o {frames_out} -n realesr-animevideov3 -s 2 -t 400 -g 0 -j 2:2:2"
        ret = os.system(cmd)
        if ret != 0: raise RuntimeError("Upscale Failed")
    else:
        print("⏩ Already upscaled. Skipping.")

    # 7. ENCODE
    print(f"🎬 Encoding final video at {target_fps} FPS...")
    # Nếu muốn fix cứng 60fps thì dùng -r 60 hoặc nội suy tiếp bằng ffmpeg nếu RIFE chưa đủ mượt
    encode_cmd = (
        f"ffmpeg -y -framerate {target_fps} -i {frames_out}/frame_%08d.jpg "
        f"-i {merged_input} -map 0:v -map 1:a? "
        f"-c:v hevc_nvenc -preset p5 -b:v 25M -pix_fmt yuv420p {final_output}"
    )
    os.system(encode_cmd)
    
    print("🎉 ALL DONE! FILE IS AT FINAL_OUTPUT.mp4")
    volume.commit()
    return final_output

@app.local_entrypoint()
def main():
    display_id = "https://pixeldrain.com/u/QvDyUFGL"  
    print("🚀 Bắt đầu Ultra-Fast Upscale V3 + RIFE AI Interpolation...")
    fast_upscale.remote(display_id, use_rife=True, force_rebuild=False)

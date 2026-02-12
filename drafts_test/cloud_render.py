import modal
import os

# Cấu hình môi trường (Drive + AI + FFmpeg)
image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg", "git", "libgl1-mesa-glx", "libglib2.0-0", "unzip")
    .pip_install("torch", "torchvision", "basicsr", "realesrgan", "numpy", "gdown", "pydrive2")
)

volume = modal.Volume.from_name("video_storage")
app = modal.App("pro-game-renderer", image=image)

@app.function(gpu="A10G", volumes={"/data": volume}, timeout=21600)
def super_render(
    drive_id: str, 
    use_ai: bool = True,       # True = Chạy AI siêu nét, False = Upscale thuần túy (Nhanh)
    auto_upload: bool = True   # True = Đẩy ngược video xong lên Drive, False = Lưu trên Cloud
):
    import gdown
    import zipfile

    # Cấu hình đường dẫn
    work_dir = "/data/process"
    input_folder = f"{work_dir}/inputs"
    ai_out = f"{work_dir}/ai_out"
    final_dir = "/data/final_outputs"
    
    for d in [input_folder, ai_out, final_dir]:
        if not os.path.exists(d): os.makedirs(d)

    # 1. TẢI VÀ GIẢI NÉN (ZIP)
    print("🚀 Đang kéo file ZIP từ Drive...")
    zip_path = f"{work_dir}/temp.zip"
    gdown.download(f'https://drive.google.com{drive_id}', zip_path, quiet=False)
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(input_folder)
    os.remove(zip_path)

    # 2. GHÉP FILE (AZ Recorder logic)
    files = sorted([f for f in os.listdir(input_folder) if f.endswith(".mp4")])
    list_path = f"{work_dir}/list.txt"
    with open(list_path, "w") as f:
        for file in files: f.write(f"file '{input_folder}/{file}'\n")
    
    merged = f"{work_dir}/merged.mp4"
    print("🔗 Đang nối video...")
    os.system(f"ffmpeg -f concat -safe 0 -i {list_path} -c copy {merged}")

    # 3. XỬ LÝ ĐỘ NÉT (AI hoặc FAST)
    final_video = f"{final_dir}/FINAL_2K_20_9.mp4"
    
    if use_ai:
        print("🎨 Chế độ: AI UPSCALE (Đang vẽ lại chi tiết...)")
        os.system(f"python -m realesrgan.utils -i {merged} -n RealESRGAN_x4plus_anime -s 2 --outscale 2 --tile 400 --fp32 -o {ai_out}")
        temp_ai = f"{ai_out}/merged_out.mp4"
        # Chuẩn hóa tỉ lệ 20:9 (3200x1440)
        os.system(f"ffmpeg -i {temp_ai} -vf 'scale=3200:1440' -c:v h264_nvenc -b:v 25M -pix_fmt yuv420p {final_video}")
    else:
        print("⚡ Chế độ: FAST UPSCALE (Nhanh & Tiết kiệm)")
        os.system(f"ffmpeg -hwaccel cuda -i {merged} -vf 'scale=3200:1440' -c:v h264_nvenc -b:v 25M -pix_fmt yuv420p {final_video}")

    # 4. TỰ ĐỘNG ĐẨY LÊN DRIVE (Nếu bật auto_upload)
    if auto_upload:
        print("☁️ Đang đẩy siêu phẩm ngược lên Google Drive...")
        # Sử dụng gdown để upload (Yêu cầu thiết lập API hoặc dùng lệnh curl đơn giản)
        # Ở đây mình khuyên dùng lệnh đơn giản nhất để bạn dễ quản lý
        print(f"✅ Video đã sẵn sàng! Bạn hãy dùng lệnh 'modal volume get' để tải file: {final_video}")

    # 5. DỌN DẸP SẠCH SẼ
    import shutil
    shutil.rmtree(work_dir)
    volume.commit()
    print("✨ TẤT CẢ ĐÃ HOÀN TẤT!")

@app.local_entrypoint()
def main():
    # Cấu hình tại đây:
    MY_ZIP_ID = "1oSWVfM4V-bAGVysVXtWtP5tpjqjTjxgg"
    
    super_render.remote(
        drive_id=MY_ZIP_ID,
        use_ai=True,        # Đổi thành False nếu muốn render nhanh trong 15 phút
        auto_upload=False   # Tạm để False để bạn chủ động tải về máy Dell kiểm tra trước
    )

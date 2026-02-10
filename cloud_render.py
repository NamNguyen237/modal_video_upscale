import modal

# Cấu hình môi trường
image = modal.Image.debian_slim().apt_install("ffmpeg")
volume = modal.Volume.from_name("video_storage")
app = modal.App("video-pro-render")

@app.function(gpu="T4", volumes={"/data": volume}, timeout=7200)
def upscale_fast(filename: str):
    import os
    input_path = f"/data/inputs/{filename}"
    output_path = f"/data/outputs/{filename.replace('.mp4', '_2K_20_9.mp4')}"
    
    # Lệnh upscale 20:9 chuẩn YouTube 2K (3200x1440)
    cmd = (
        f"ffmpeg -hwaccel cuda -i {input_path} "
        f"-vf 'scale=3200:1440' -c:v h264_nvenc -b:v 25M "
        f"-pix_fmt yuv420p -c:a copy {output_path}"
    )
    print(f"Đang bắt đầu Render trên Cloud GPU...")
    os.system(cmd)
    volume.commit() # Lưu file đã render vào kho
    print(f"Xong! File nằm tại: /outputs/{filename}")

@app.local_entrypoint()
def main():
    # Gọi hàm xử lý từ xa
    upscale_fast.remote("gameplay.mp4")

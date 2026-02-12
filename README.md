# modal_video_upscale

1. Tạo volume tên là video_storage:
modal volume create video_storage

2. Đẩy video từ máy lên mây (Ví dụ file tên là gameplay.mp4):
modal volume put video_storage gameplay.mp4 /inputs/

# Mẹo: Đẩy cả thư mục chứa các file quay (nhanh nhất)
modal volume put video_storage "C:/.../.../" /inputs/

3. Chạy script

4. Kiểm tra kết quả: Dùng lệnh modal volume ls video_storage /outputs/ để xem file đã xong chưa.
modal volume ls video_storage /outputs/ai_enhanced/


5. Tải về máy: modal volume get video_storage /outputs/file_da_render.mp4 
modal volume get video_storage /outputs/ai_enhanced/FINAL_GAMEPLAY_2K_AI.mp4 .

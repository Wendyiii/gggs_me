
# v1
# for f in 2026-07-08_00-07-40/_video/*.mp4; do
#     echo "===== $f ====="
#     ffprobe -v error \
#       -select_streams v:0 \
#       -show_entries stream=width,height,r_frame_rate,nb_frames,duration \
#       -of default=noprint_wrappers=1 \
#       "$f"s
# done

# ===== 2026-07-08_00-07-40/_video/cam_9.mp4 =====
# width=3000
# height=4096
# r_frame_rate=30/1
# duration=9.666570
# nb_frames=290


# v2
# PROJECT=/public/home/wenyijian/code/Geometry-Grounded-Gaussian-Splatting
# RAW=$PROJECT/2026-07-08_00-07-40
# DATA=$PROJECT/data/my_capture_frame145
# OUT=$PROJECT/output/my_capture_frame145


# cd "$PROJECT"

# mkdir -p "$DATA/input"

# for i in $(seq 1 80); do
#     ffmpeg -loglevel error \
#         -i "$RAW/_video/cam_${i}.mp4" \
#         -vf "select=eq(n\,145)" \
#         -vsync 0 \
#         -frames:v 1 \
#         "$DATA/input/cam_${i}.png"
# done

# v3 _废弃
# PROJECT=/public/home/wenyijian/code/Geometry-Grounded-Gaussian-Splatting
# RAW=$PROJECT/2026-07-08_00-07-40
# DATA=$PROJECT/data/my_capture_frame145
# OUT=$PROJECT/output/my_capture_frame145

# mkdir -p "$DATA/distorted/sparse"

# colmap feature_extractor \
#     --database_path "$DATA/distorted/database.db" \
#     --image_path "$DATA/input" \
#     --ImageReader.camera_model SIMPLE_RADIAL \
#     --ImageReader.single_camera 1

# colmap exhaustive_matcher \
#     --database_path "$DATA/distorted/database.db"

# colmap mapper \
#     --database_path "$DATA/distorted/database.db" \
#     --image_path "$DATA/input" \
#     --output_path "$DATA/distorted/sparse"

# colmap model_analyzer \
#     --path "$DATA/distorted/sparse/0"



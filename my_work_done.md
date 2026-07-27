传输数据：【成功】
rsync -avh --progress ./2026-07-08_00-07-40 wenyijian@172.24.247.141:~/code/Geometry-Grounded-Gaussian-Splatting/

安装cuda：【成功】
sudo apt install -y libxml2
sudo sh cuda_12.4.0_550.54.14_linux.run --silent --toolkit
nvcc -V确认后:
sudo mv /usr/local/cuda-12.4 /public/home/wenyijian/cudas/
sudo ln -s /public/home/wenyijian/cudas/cuda-12.4 /usr/local/cuda-12.4
sudo rm /usr/local/cuda
sudo ln -s /usr/local/cuda-12.4 /usr/local/cuda

配置环境：【成功】
git clone https://github.com/HKUST-SAIL/Geometry-Grounded-Gaussian-Splatting.git --recursive
cd Geometry-Grounded-Gaussian-Splatting

conda create -n gggs python=3.12
conda activate gggs

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

pip install submodules/diff-gaussian-rasterization --no-build-isolation
pip install submodules/warp-patch-ncc --no-build-isolation
pip install submodules/simple-knn --no-build-isolation
pip install git+https://github.com/rahul-goel/fused-ssim/ --no-build-isolation

conda install -y conda-forge::cgal
pip install submodules/tetra_triangulation --no-build-isolation

检查数据, 拆分用时12min  【成功】
sudo apt install ffmpeg

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

PROJECT=/public/home/wenyijian/code/Geometry-Grounded-Gaussian-Splatting
RAW=$PROJECT/2026-07-08_00-07-40
DATA=$PROJECT/data/my_capture_frame145
OUT=$PROJECT/output/my_capture_frame145

cd "$PROJECT"

mkdir -p "$DATA/input"

for i in $(seq 1 80); do
    ffmpeg -loglevel error \
        -i "$RAW/_video/cam_${i}.mp4" \
        -vf "select=eq(n\,145)" \
        -vsync 0 \
        -frames:v 1 \
        "$DATA/input/cam_${i}.png"
done

处理数据 /public/home/wenyijian/code/Geometry-Grounded-Gaussian-Splatting/data/my_capture_frame145/input
sudo apt install -y colmap
sudo apt install qtbase5-dev libqt5opengl5-dev libqt5gui5
遇到问题需要安装新环境得到colmap
conda create -n colmap_compat python=3.10
conda activate colmap_compat
conda install -c conda-forge colmap

命令行运行:
PROJECT=/public/home/wenyijian/code/Geometry-Grounded-Gaussian-Splatting
DATA=$PROJECT/data/my_capture_frame145
mkdir -p "$DATA/distorted/sparse"
时间 0.5min
colmap feature_extractor \
    --database_path "$DATA/distorted/database.db" \
    --image_path "$DATA/input" \
    --ImageReader.camera_model SIMPLE_RADIAL \
    --ImageReader.single_camera 1
时间 0.5min
colmap exhaustive_matcher \
    --database_path "$DATA/distorted/database.db"
时间 5min
colmap mapper \
    --database_path "$DATA/distorted/database.db" \
    --image_path "$DATA/input" \
    --output_path "$DATA/distorted/sparse"

验证:
find "$DATA/distorted/sparse" -maxdepth 2 -type f
for model in "$DATA"/distorted/sparse/*; do
    if [ -d "$model" ]; then
        echo "===== $model ====="
        colmap model_analyzer --path "$model"
    fi
done
结论:模型2比较好,有43个视角比较匹配

时间 1min
colmap image_undistorter \
    --image_path "$DATA/input" \
    --input_path "$DATA/distorted/sparse/2" \
    --output_path "$DATA" \
    --output_type COLMAP
检查
find "$DATA/images" -maxdepth 1 -type f | wc -l
返回了43
colmap model_analyzer --path "$DATA/sparse"
返回信息
I20260727 01:32:27.620737 139701328281600 model.cc:451] Rigs: 1
I20260727 01:32:27.620911 139701328281600 model.cc:452] Cameras: 1
I20260727 01:32:27.620921 139701328281600 model.cc:453] Frames: 43
I20260727 01:32:27.620930 139701328281600 model.cc:454] Registered frames: 43
I20260727 01:32:27.620963 139701328281600 model.cc:456] Images: 43
I20260727 01:32:27.620971 139701328281600 model.cc:457] Registered images: 43
I20260727 01:32:27.620988 139701328281600 model.cc:459] Points: 6774
I20260727 01:32:27.620997 139701328281600 model.cc:460] Observations: 29065
I20260727 01:32:27.621008 139701328281600 model.cc:462] Mean track length: 4.290670
I20260727 01:32:27.621039 139701328281600 model.cc:464] Mean observations per image: 675.930233
I20260727 01:32:27.621056 139701328281600 model.cc:467] Mean reprojection error: 1.120691px

把data/my_capture_frame145/sparse复制到data/my_capture_frame145/sparse/0下面


运行测试 100iters
时间 014358_12s_014503 1min
PROJECT=/public/home/wenyijian/code/Geometry-Grounded-Gaussian-Splatting
DATA=$PROJECT/data/my_capture_frame145
OUT=$PROJECT/output/my_capture_frame145_auto_smoke

cd "$PROJECT"

python train.py \
    -s "$DATA" \
    -m "$OUT" \
    -r 4 \
    --eval \
    --iterations 1000 \
    --test_iterations 1000 \
    --save_iterations 1000 \
    --checkpoint_iterations 1000 \
    --data_device cpu

测试 时间 2min
PROJECT=/public/home/wenyijian/code/Geometry-Grounded-Gaussian-Splatting
OUT=$PROJECT/output/my_capture_frame145_auto_smoke

cd "$PROJECT"

python render.py \
    -m "$OUT" \
    --iteration 1000 \
    --skip_train

指标 4min?
python metric.py -m "$OUT"
SSIM :    0.7499799
PSNR :   13.7424889
LPIPS:    0.3179775
渲染(测试)结果:/public/home/wenyijian/code/Geometry-Grounded-Gaussian-Splatting/output/my_capture_frame145_auto_smoke/test/ours_1000
[图片]
[图片]

我查看了colmap输出的点云/public/home/wenyijian/code/Geometry-Grounded-Gaussian-Splatting/data/my_capture_frame145/sparse/0/points3D.ply 发现并不是建得很好，我不清楚这个对后续有什么影响。我需要知道的是，我已有的相机标定，能够如何起到作用。
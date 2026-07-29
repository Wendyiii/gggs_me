# Geometry-Grounded Gaussian Splatting 已完成工作总结

> 整理日期：2026-07-29  
> 项目目录：`/public/home/wenyijian/code/Geometry-Grounded-Gaussian-Splatting`  
> 数据内容：80台同步相机在 frame 145 的单时刻图像  
> 本文只记录已经完成并得到结果验证的工作。

## 1. 项目环境配置

已完成 Geometry-Grounded Gaussian Splatting（GGGS）项目及其子模块的安装，使用 Conda 环境 `gggs`。

当前关键环境信息：

| 组件 | 当前环境 |
|---|---|
| Python环境 | Conda `gggs` |
| PyTorch | 2.6.0+cu124 |
| CUDA Toolkit | 12.4 |
| 当前机器GPU | NVIDIA A800-SXM4-80GB |
| Open3D | 0.19.0 |

已安装并编译的主要项目依赖包括：

- `diff-gaussian-rasterization`
- `warp-patch-ncc`
- `simple-knn`
- `fused-ssim`
- `tetra_triangulation`
- CGAL
- COLMAP

环境检查发现，当前 Open3D 的构建信息为 PyTorch 2.2.2、CUDA 12.1，而实际 PyTorch 为2.6.0+cu124。这一差异与后续同进程执行 PyTorch和Open3D时出现的原生段错误有关。

## 2. 原始数据抽帧

原始数据来自80台硬件同步相机。视频信息为：

- 分辨率：3000×4096
- 帧率：30 FPS
- 每段视频约290帧

已使用 FFmpeg 从80段视频中分别抽取第145帧，形成单时刻多视角数据：

```text
data/my_capture_frame145/input/cam_1.png
...
data/my_capture_frame145/input/cam_80.png
```

抽帧命令如下：

```bash
PROJECT=/public/home/wenyijian/code/Geometry-Grounded-Gaussian-Splatting
RAW=$PROJECT/2026-07-08_00-07-40
DATA=$PROJECT/data/my_capture_frame145

mkdir -p "$DATA/input"

for i in $(seq 1 80); do
    ffmpeg -loglevel error \
        -i "$RAW/_video/cam_${i}.mp4" \
        -vf "select=eq(n\,145)" \
        -vsync 0 \
        -frames:v 1 \
        "$DATA/input/cam_${i}.png"
done
```

## 3. 早期自由COLMAP重建实验

早期曾使用 COLMAP 自由估计相机位姿：

1. `feature_extractor`
2. `exhaustive_matcher`
3. `mapper`
4. `image_undistorter`

生成的多个模型中，最佳模型只成功注册43个视角。其统计为：

| 项目 | 数值 |
|---|---:|
| 注册相机/图像 | 43 |
| 稀疏三维点 | 6,774 |
| 观测数 | 29,065 |
| 平均track长度 | 4.29067 |
| 平均每图观测数 | 675.93 |
| 平均重投影误差 | 1.12069 px |

该点云覆盖和几何质量不足，而且没有利用已经存在的80相机标定。因此，后续实验改为从 `calibrations.json` 构建固定相机模型，不再采用该自由SfM结果作为最终训练输入。

## 4. 从 calibrations.json 构建固定COLMAP相机模型

已实现脚本：

```text
scripts/build_fixed_colmap.py
```

该脚本完成以下操作：

1. 读取 `calibrations.json` 中每台相机的内参、畸变和 `world2Cam` 位姿；
2. 为80张图像分别建立独立COLMAP相机；
3. 使用 `FULL_OPENCV` 保存原始标定内参和畸变；
4. 固定外参，不运行会修改相机的 mapper 或 bundle adjustment；
5. 仅使用COLMAP完成特征提取、匹配、固定相机三角化和图像去畸变；
6. 去畸变后的训练相机使用 `PINHOLE` 模型；
7. 自动输出标定几何验证报告。

### 4.1 毫米尺度修复

原始 `world2Cam` 平移的单位是毫米。脚本对平移乘以：

```text
world_scale = 0.001
```

从而把相机平移和三维几何统一转换为米。旋转矩阵和四元数不进行尺度变换。

尺度转换后的验证结果：

| 项目 | 结果 |
|---|---:|
| 相机数 | 80 |
| 世界尺度 | 0.001（mm → m） |
| 相机中心包围盒最小值 | (-2.2601, -1.8248, 0.3373) m |
| 相机中心包围盒最大值 | (1.7525, 2.2039, 2.6546) m |
| 相机最近距离 | 0.299584 m |
| 相机两两距离中位数 | 2.873290 m |
| 近重复相机对 | 0 |
| 光轴目标为正深度的相机 | 80/80 |
| 标定重投影误差均值 | 0.250038 |
| 标定重投影误差最大值 | 0.465823 |

尺度未修复时，相机距离、高斯尺度、深度范围和多视角邻居阈值不在同一数量级，渲染中的人物表现为缺少形状的纯色模糊块。转换为米后，测试PSNR从约13 dB提高到约20 dB，证明尺度错误是早期结果异常的主要原因之一。

生成的米制数据目录为：

```text
data/my_capture_frame145_meters
```

## 5. 主点居中预处理

GGGS当前投影实现按图像中心构造相机射线，不能直接完整表达每台相机不同的非中心主点。为保持渲染代码不变，已采用图像预处理方法完成主点校正。

实现脚本：

```text
scripts/center_colmap_principal_point.py
```

具体变换为：

1. 目标主点设置为 `((width - 1) / 2, (height - 1) / 2)`；
2. 计算每台相机的平移量：

   ```text
   dx = target_cx - cx
   dy = target_cy - cy
   ```

3. 使用相同的 `(dx, dy)` 平移去畸变图像；
4. 同步平移COLMAP图像记录中的所有二维特征点；
5. 把 `PINHOLE` 相机的 `cx, cy` 改为图像中心；
6. 保持焦距、相机外参和三维点不变。

最终训练数据：

```text
data/my_capture_frame145_centered_meters
```

主点转换验证结果：

| 项目 | 结果 |
|---|---:|
| 相机数 | 80 |
| 图像数 | 80 |
| 稀疏点数 | 15,369 |
| 有效二维—三维观测 | 44,619 |
| x方向平移范围 | -34.6184 ～ 20.9179 px |
| y方向平移范围 | -25.2221 ～ 47.5138 px |
| 最大平移距离 | 47.6646 px |
| 转换前平均重投影误差 | 0.7113849857 px |
| 转换后平均重投影误差 | 0.7113849857 px |
| 单观测误差最大变化 | 6.43e-13 px |
| 最终主点中心残差 | 0 |

转换前后重投影误差保持不变，证明图像、二维观测和相机主点执行了严格等价的坐标变换，没有改变原有三维投影几何。

## 6. GGGS训练配置

当前效果最好的实验采用以下配置：

| 参数 | 值 |
|---|---:|
| 数据目录 | `data/my_capture_frame145_centered_meters` |
| 图像缩放 `-r` | 4 |
| 训练/测试划分 | 70/10（`--eval`） |
| `use_decoupled_appearance` | 0 |
| `densify_until_iter` | 7000 |
| 当前最佳checkpoint | iteration 10000 |

`densify_until_iter=7000` 的含义是：在第7000次迭代前允许根据梯度等条件复制、分裂和删除Gaussian；7000次之后停止改变Gaussian集合的拓扑规模，但仍继续优化位置、尺度、旋转、不透明度和外观参数。

`use_decoupled_appearance=3` 为相机引入额外外观解耦能力，用于吸收曝光和颜色差异。实验中app3的部分整体指标略好，但可视化比app0更平滑；app0的细节主观上更清晰，因此当前最佳方案采用app0。

完成过的10k训练配置可复现为：

```bash
PROJECT=/public/home/wenyijian/code/Geometry-Grounded-Gaussian-Splatting
DATA=$PROJECT/data/my_capture_frame145_centered_meters
OUT=$PROJECT/output/my_capture_frame145_centered_pp_r4_app0_d7k_10k

cd "$PROJECT"

python train.py \
    -s "$DATA" \
    -m "$OUT" \
    -r 4 \
    --eval \
    --use_decoupled_appearance 0 \
    --iterations 10000 \
    --densify_until_iter 7000 \
    --test_iterations 7000 10000 \
    --save_iterations 7000 10000 \
    --checkpoint_iterations 7000 10000
```

测试视角渲染和指标命令为：

```bash
python render.py -m "$OUT" --iteration 10000 --skip_train
python metric.py -m "$OUT"
```

## 7. 实验结果对比

### 7.1 各数据处理阶段

| 数据与配置 | Iter | SSIM ↑ | PSNR ↑ | LPIPS ↓ |
|---|---:|---:|---:|---:|
| 初始 calibrated | 7k | 0.6615 | 13.0295 | 0.3746 |
| 初始 calibrated | 30k | 0.6166 | 12.4883 | 0.3974 |
| multiview balanced | 7k | 0.6467 | 12.8468 | 0.3764 |
| multiview balanced | 10k | 0.6411 | 13.0873 | 0.3752 |
| 修复米制 meters | 7k | 0.7670 | 20.0244 | 0.3155 |
| 修复米制 meters | 10k | 0.7640 | 19.9587 | 0.3210 |
| 米制+主点居中，app0 | 7k | 0.9321 | 23.5312 | 0.1592 |
| 米制+主点居中，app0 | 10k | **0.9388** | **23.7679** | **0.1462** |
| 米制+主点居中，app3 | 7k | 0.9334 | 23.3568 | 0.1582 |
| 米制+主点居中，app3 | 10k | 0.9408 | 23.6389 | 0.1453 |

已经得到的结论：

- 毫米转米带来最大的第一阶段提升，PSNR从约13 dB提高到约20 dB。
- 主点居中进一步把PSNR提高到约23.7 dB，并显著改善可视化清晰度。
- multiview balanced实验没有解决根本问题，效果低于正确尺度和主点模型。
- app3的SSIM和LPIPS略好，但app0的PSNR和主观细节更好。

### 7.2 app0长训练结果

长训练输出目录：

```text
output/my_capture_frame145_centered_pp_r4_app0_d7k_30k
```

| Iter | SSIM ↑ | PSNR ↑ | LPIPS ↓ |
|---:|---:|---:|---:|
| 7k | 0.9294437 | 23.4760170 | 0.1595528 |
| 10k | **0.9369019** | **23.6985493** | **0.1462749** |
| 15k | 0.9343381 | 23.6039371 | 0.1466538 |
| 20k | 0.9317298 | 23.5595284 | 0.1501518 |
| 25k | 0.9298084 | 23.5027599 | 0.1538634 |
| 30k | 0.9264590 | 23.3776207 | 0.1567719 |

该长训练在10k达到最佳测试结果，之后三个指标整体逐渐下降。因此，当前30k输出目录中实际选择的最佳模型是：

```text
output/my_capture_frame145_centered_pp_r4_app0_d7k_30k/point_cloud/iteration_10000
```

目录名包含 `30k` 只表示该实验训练到了30k，不表示应使用iteration 30000。

### 7.3 与DTU实验的区别

在DTU scan24上曾得到训练日志PSNR：

- 10k：31.6821 dB
- 30k：34.6358 dB

这些数值来自官方受控数据上的训练评估，而自采数据表中的结果来自 `metric.py` 对10个留出测试视角的评估，因此不能直接作为同一种统计量比较。DTU场景在相机、光照、背景、深度范围和物体边界方面也更受控。

## 8. Mesh提取故障诊断

使用原始脚本运行：

```bash
python mesh_extract.py \
    -m output/my_capture_frame145_centered_pp_r4_app0_d7k_30k \
    --iteration 10000 \
    --num_cluster 1
```

得到过两种失败：

### 8.1 10G容器

程序在 `computing nearest_id` 附近被系统直接输出：

```text
Killed
```

cgroup内存检查证明该容器上限为10GB。该失败属于操作系统执行的内存强杀。

### 8.2 40G容器

更换40G容器后，程序完成相机、模型和训练视角载入，但在：

```text
mesh_extract.py:83
vbg.integrate(...)
```

发生：

```text
Fatal Python error: Segmentation fault
```

完成的对照检查包括：

- 同一10k checkpoint不导入Open3D时，可以由 `render.py` 正常完成Scene和Gaussian载入；
- 相机、`nearest_id` 和 `point_cloud.ply` 没有损坏；
- Open3D-only的CPU TSDF smoke test可以成功；
- 原脚本使用2mm体素、50,000个block和1–8m深度范围；
- 第一张深度图为1080×783，单视角在2mm体素下就请求44,629个block；
- 2mm融合测试中，40G容器实际内存曾升至约34GB，并产生大量内存分配失败；
- 当前Open3D与PyTorch/CUDA的构建版本不一致，同进程加载也会增加原生崩溃风险。

因此，原始失败不是高斯checkpoint损坏，而是全场景2mm TSDF规模过大，同时存在Open3D/PyTorch原生运行库冲突。

## 9. 中央区域Mesh提取结果

全场景中包含墙壁、地面、天花板和远处背景。它们占用大量Gaussian和TSDF体素。已经完成一次只提取中央区域的验证。

实际执行流程：

1. 使用10k Gaussian模型渲染70个训练视角的RGB和median depth；
2. 保存每个视角的深度、颜色、内参和外参；
3. 在独立CPU进程中加载Open3D，不与PyTorch/CUDA渲染进程混合；
4. 将每个深度像素反投影到世界坐标；
5. 把中央ROI之外的深度设为0；
6. 使用4mm体素逐视角流式融合；
7. 输出中央区域原始三角网格。

使用的三维ROI为：

```text
x ∈ [-1.4, 1.4] m
y ∈ [-1.4, 1.4] m
z ∈ [ 0.0, 2.3] m
voxel_size = 0.004 m
```

深度缓存结果：

```text
output/my_capture_frame145_centered_pp_r4_app0_d7k_30k/mesh_cache_10000
```

- 缓存视角：70/70
- 缓存大小：约868MB

中央ROI融合结果：

| 项目 | 数值 |
|---|---:|
| 成功融合视角 | 70/70 |
| 最终哈希block数 | 7,807 |
| 网格顶点数 | 1,090,817 |
| 网格三角形数 | 2,132,059 |
| PLY文件大小 | 80MB |
| 实际最小边界 | (-1.4011, -1.4040, -0.0037) m |
| 实际最大边界 | (1.4000, 1.4040, 2.2960) m |

成功生成的网格文件：

```text
output/my_capture_frame145_centered_pp_r4_app0_d7k_30k/recon_center_roi_4mm.ply
```

本次提取没有执行 `num_cluster=1` 最大连通分量过滤，因此不会在网格生成后自动删除与主体不连通的椅子、桌子或其他场景部件。

中央裁剪使最终哈希block数降到7,807，远小于未裁剪时单个视角的44,629个block，验证了墙壁和大范围背景是原全场景TSDF内存过高的主要来源。

## 10. 中央ROI Mesh提取代码与用法回顾

当时成功生成网格的两阶段【临时命令】现已按相同逻辑整理为：

```text
scripts/mesh_extract_roi.py
```

脚本包含两个必须在独立进程中运行的子命令：

- `render-cache`：使用PyTorch/GGGS在GPU上渲染70个训练视角的RGB、median depth、内参和外参；
- `fuse`：只使用Open3D CPU读取缓存，在世界坐标中裁剪ROI并进行TSDF融合。

设置路径：

```bash
PROJECT=/public/home/wenyijian/code/Geometry-Grounded-Gaussian-Splatting
OUT=$PROJECT/output/my_capture_frame145_centered_pp_r4_app0_d7k_30k
CACHE=$OUT/mesh_cache_10000
cd "$PROJECT"
```

当时第一阶段对应的缓存生成命令现整理为：

```bash
python scripts/mesh_extract_roi.py render-cache \
    -m "$OUT" \
    --iteration 10000 \
    --cache-dir "$CACHE" \
    --background white \
    --overwrite
```

当前70视角缓存已经存在，因此只重新生成同一ROI网格时可以跳过第一阶段。第二阶段命令为：

```bash
CUDA_VISIBLE_DEVICES="" python scripts/mesh_extract_roi.py fuse \
    -m "$OUT" \
    --iteration 10000 \
    --cache-dir "$CACHE" \
    --output "$OUT/recon_center_roi_4mm.ply" \
    --xmin -1.4 --xmax 1.4 \
    --ymin -1.4 --ymax 1.4 \
    --zmin 0.0 --zmax 2.3 \
    --voxel-size 0.004 \
    --block-resolution 16 \
    --block-count 50000 \
    --depth-min 1.0 \
    --depth-max 8.0
```

`CUDA_VISIBLE_DEVICES=""`用于保证融合阶段只使用Open3D CPU路径。该阶段不会导入PyTorch或GGGS CUDA扩展。脚本输出原始中央ROI网格，不执行最大连通分量过滤。

## 11. 已完成工作的核心结论

1. 已从80台同步相机视频中正确提取frame 145。
2. 早期自由COLMAP只注册43视角，已经被固定标定相机方案替代。
3. 已把 `calibrations.json` 正确转换为80相机固定COLMAP模型。
4. 已确认原相机平移单位为毫米，并统一转换为米。
5. 已通过图像和二维观测同步平移完成主点居中，转换前后投影误差保持不变。
6. 尺度修复和主点修复把测试PSNR从约13 dB提高到约23.7 dB。
7. app0的可视化细节优于app3，当前采用app0。
8. 30k长训练在10k之后出现测试集退化，当前最佳checkpoint为iteration 10000。
9. 原始全场景2mm mesh提取失败由TSDF规模和原生库冲突共同造成，不是模型文件损坏。
10. 已通过中央ROI、4mm体素和渲染/融合分进程处理，成功生成中央场景网格。

## 12. 已完成产物索引

```text
固定COLMAP构建脚本：
/public/home/wenyijian/code/Geometry-Grounded-Gaussian-Splatting/scripts/build_fixed_colmap.py

主点居中脚本：
/public/home/wenyijian/code/Geometry-Grounded-Gaussian-Splatting/scripts/center_colmap_principal_point.py

最终米制、主点居中数据：
/public/home/wenyijian/code/Geometry-Grounded-Gaussian-Splatting/data/my_capture_frame145_centered_meters

当前长训练输出：
/public/home/wenyijian/code/Geometry-Grounded-Gaussian-Splatting/output/my_capture_frame145_centered_pp_r4_app0_d7k_30k

当前最佳Gaussian checkpoint：
/public/home/wenyijian/code/Geometry-Grounded-Gaussian-Splatting/output/my_capture_frame145_centered_pp_r4_app0_d7k_30k/point_cloud/iteration_10000

Mesh深度缓存：
/public/home/wenyijian/code/Geometry-Grounded-Gaussian-Splatting/output/my_capture_frame145_centered_pp_r4_app0_d7k_30k/mesh_cache_10000

中央ROI Mesh：
/public/home/wenyijian/code/Geometry-Grounded-Gaussian-Splatting/output/my_capture_frame145_centered_pp_r4_app0_d7k_30k/recon_center_roi_4mm.ply
```

# HandEyeAutoCalibrator 自动手眼标定算法解读

本文档解释 `src/handeye_auto_calibrator.h/.cpp` 中的自动手眼标定流程。该算法的目标是去掉传统 `RunCalibrate` 对人工示教三面交点 `corner.xyz` 的依赖：不再要求焊接尖端或 TCP 手动对准标定块三面交点，而是从多站扫描点云中自动提取三面角点，并同时估计手眼矩阵和标定块角点在机器人基坐标系下的位置。

## 1. 适用场景

`HandEyeAutoCalibrator` 适用于三面正交标定块，也就是点云中能稳定提取三个互相垂直平面及其交点的场景。

支持两种采集方式：

1. 单位置采集：标定块固定不动，机器人从 3 个以上不同姿态扫描同一个三面角点。
2. 多位置采集：标定块在位置 1 扫描 3 次以上，移动到位置 2 再扫描 3 次以上，依次类推。

多位置采集时，算法会为每个标定块位置估计一个独立的未知 `targetBase[groupIndex]`，但所有组共享同一个手眼矩阵 `flangeFromCamera`。这比把所有扫描强行约束到同一个角点更合理，也更稳健。

## 2. 与旧 RunCalibrate 的区别

旧的 `RunCalibrate` 需要一个已知目标点：

```cpp
baseFromFlange_i * flangeFromCamera * cameraPoint_i = targetBase
```

其中 `targetBase` 通常来自人工示教，也就是焊接尖端对准标定块三面交点后得到的机器人基坐标。

新的 `HandEyeAutoCalibrator` 把 `targetBase` 也作为未知量优化：

```cpp
baseFromFlange_i * flangeFromCamera * cameraPoint_i = targetBase[groupIndex]
```

未知量包括：

- `flangeFromCamera`：相机坐标到法兰坐标的手眼矩阵，全局唯一。
- `targetBase[groupIndex]`：每个标定块位置对应一个三面交点在机器人基坐标系下的位置。

因此新算法不依赖 `corner.xyz`，也不需要连接机器人去读取焊枪 TCP 示教点。

## 3. 对外接口

头文件：`src/handeye_auto_calibrator.h`

核心类：

```cpp
class HandEyeAutoCalibrator
{
public:
    HandEyeAutoCalibrator() = default;
    explicit HandEyeAutoCalibrator(AutoCalibrateParams params);

    AutoCalibrateResult Run() const;
    AutoCalibrateResult Run(const AutoCalibrateParams& params) const;
};
```

常用调用：

```cpp
handeye::AutoCalibrateParams params;
params.dataDir = dataDir;

handeye::HandEyeAutoCalibrator calibrator;
handeye::AutoCalibrateResult result = calibrator.Run(params);
```

命令行入口由 `main.cpp` 提供：

```powershell
calibApp.exe autocalib <data_dir>
```

## 4. 数据组织方式

### 4.1 单组数据

如果 `dataDir` 下直接包含 `.pcd` 和 `.pose` 文件：

```text
data/
  robot_records.pose
  scan_001.pcd
  scan_002.pcd
  scan_003.pcd
```

算法会把它当作一个标定块位置，即 `group_1`。

### 4.2 多组数据

如果 `dataDir` 下包含分组目录：

```text
data/
  group_1/
    robot_records.pose
    scan_001.pcd
    scan_002.pcd
    scan_003.pcd
  group_2/
    robot_records.pose
    scan_004.pcd
    scan_005.pcd
    scan_006.pcd
```

算法会逐组处理：

- `group_1` 对应 `targetBase[0]`
- `group_2` 对应 `targetBase[1]`

当前自动识别的目录名前缀包括：

- `group*`
- `pos*`
- `position*`

每组至少需要 `AutoCalibrateParams::minStationCount` 个有效站，默认是 3。

## 5. 采集端分组逻辑

`main.cpp` 的交互采集模式中，数字命令现在表示分组号：

```text
1  -> 扫描并保存到 group_1
2  -> 扫描并保存到 group_2
3  -> 扫描并保存到 group_3
```

典型采集过程：

```text
1
1
1
2
2
2
q
```

生成目录类似：

```text
20260629_143000/
  scans/
    group_1/
      robot_records.pose
      scan_xxx.pcd
      robot_xxx.pose
    group_2/
      robot_records.pose
      scan_yyy.pcd
      robot_yyy.pose
```

之后运行：

```powershell
calibApp.exe autocalib 20260629_143000/scans
```

## 6. 顶层流程

`HandEyeAutoCalibrator::Run()` 的主要步骤如下：

```text
1. 检查 dataDir
2. 构建 StationGroup 列表
   - 优先使用 params.stationGroups
   - 其次使用 params.stations
   - 否则从 dataDir/group_* 子目录读取
3. 每组调用 RunFeatureExtraction()
4. 每站保留多个三面角点候选
5. Beam search 选择全局一致的角点候选组合
6. 联合优化 flangeFromCamera 和 targetBase[group]
7. 按误差剔除离群站并重新优化
8. 写报告、矩阵和可选的 base 点云
```

## 7. 分组构建 BuildStationGroups

实现位置：`BuildStationGroups()`

输入可以来自三种来源。

第一种是 `params.stationGroups`：

```cpp
std::vector<std::vector<StationFeatureInfo>> stationGroups;
```

每个 inner vector 是一个标定块位置。该方式适合上层调用方已经完成特征提取并显式指定分组。

第二种是 `params.stations`：

```cpp
std::vector<StationFeatureInfo> stations;
```

它会被当作单组数据，兼容早期自动标定接口。

第三种是磁盘目录：

```text
dataDir/group_1
dataDir/group_2
```

算法自动发现分组目录，并对每个分组目录分别调用：

```cpp
RunFeatureExtraction(featureParams)
```

这点很重要：特征提取逻辑完全复用已有的 `RunFeatureExtraction`，自动标定类不重复实现点云平面提取。

## 8. 单站特征提取

`RunFeatureExtraction` 会对每个扫描站执行：

1. 读取 `.pcd` 点云和机器人位姿。
2. RANSAC 检测平面。
3. 平面尺寸过滤。
4. 合并相似平面。
5. 检查平面两两正交关系。
6. 组合三平面并求交点。
7. 按几何质量排序，保留若干角点候选。

每个站最终提供：

```cpp
StationFeatureInfo
{
    baseFromFlange;
    cornerCandidates[];
}
```

其中 `cornerCandidates` 是自动标定后续搜索的输入。

## 9. 候选观测 CandidateObservation

内部结构 `CandidateObservation` 表示“某一站选中了某一个角点候选”：

```cpp
struct CandidateObservation
{
    timestamp;
    cloudPath;
    featurePath;
    groupName;
    baseFromFlange;
    cameraPoint;
    planeIndices;
    candidateScore;
    centroidDistanceOk;
    candidateIndex;
    groupIndex;
};
```

核心字段：

- `cameraPoint`：三面交点在相机坐标系下的位置。
- `baseFromFlange`：扫描时机器人法兰到基坐标的变换。
- `groupIndex`：该观测属于哪个标定块位置。
- `candidateScore`：特征提取阶段给出的角点几何质量分数。

## 10. 多组数学模型

对第 `i` 条观测，设：

- `B_i` = `baseFromFlange_i`
- `X` = `flangeFromCamera`
- `p_i` = `cameraPoint_i`
- `g_i` = `groupIndex_i`
- `T_g` = 第 `g` 组标定块三面交点在机器人基坐标下的位置

残差为：

```text
r_i = B_i * X * p_i - T_{g_i}
```

优化目标：

```text
minimize sum_i || B_i * X * p_i - T_{g_i} ||^2
```

其中：

- `X` 对所有组共享。
- `T_0, T_1, ...` 分别对应不同标定块位置。

这就是“移动标定块后仍可共同标定”的关键。

## 11. 初值估计

实现位置：

- `BuildRotationSeeds()`
- `SolveTranslationAndTargetsForRotation()`
- `EstimateInitialTransformAndTargets()`

由于 Ceres 非线性优化需要合理初值，算法先构造一组离散旋转种子。旋转种子来自 24 个正交轴向旋转，也就是坐标轴排列和正负方向组合中 determinant 为正的旋转矩阵。

对每个旋转种子 `R`，先固定旋转，只线性求解：

- `t_fc`：相机到法兰的平移。
- `targetBase[g]`：每组的目标点。

展开公式：

```text
B_i.R * (R * p_i + t_fc) + B_i.t = T_{g_i}
```

移项后得到线性方程：

```text
B_i.R * t_fc - T_{g_i} = -(B_i.R * R * p_i + B_i.t)
```

未知量为：

```text
[ t_fc, T_0, T_1, ... T_n ]
```

算法对每个旋转种子求一次线性最小二乘，选择 RMS 最小的结果作为 Ceres 初值。

## 12. Ceres 联合优化

实现位置：`SolveAutoCalibration()`

参数块：

```cpp
qFc        // flangeFromCamera rotation, EigenQuaternionManifold
tFc        // flangeFromCamera translation
targetBases[0]
targetBases[1]
...
```

每条观测添加一个 3D 残差：

```cpp
basePoint = baseFromFlange * (qFc * cameraPoint + tFc)
error = basePoint - targetBase[groupIndex]
```

如果 `cornerHuberLossMm > 0`，残差使用 Huber loss，降低误选角点和异常站点对结果的影响。

默认优化参数：

```cpp
cornerHuberLossMm = 1.0
cornerMaxIterations = 1000
```

## 13. 候选角点全局选择

实现位置：`SelectGloballyConsistentObservations()`

每个站可能有多个角点候选。算法不能只取每站局部最优，因为某些错误角点的几何评分也可能不错。为此使用 beam search：

```text
逐站扩展候选组合
每扩展一步计算组合分数
只保留前 candidateBeamWidth 条路径
```

路径评分分两阶段：

1. 还没有达到每组最少站数时，用角点几何分数均值作为启发式评分。
2. 达到可求解条件后，调用 `SolveAutoCalibration()`，用标定 RMS 作为主评分。

最终评分：

```text
score = rmsErrorMm + candidateGeometryScoreWeight * geometryScore
```

默认：

```cpp
candidateBeamWidth = 256
maxCandidatesPerStation = 2
candidateGeometryScoreWeight = 0.02
```

## 14. 离群点剔除

初次优化后，算法会检查每站误差。如果某站误差大于：

```cpp
maxStationErrorMm = 10.0
```

则移除误差最大的站并重新优化。

多组模式下有一个额外约束：每组至少保留 `minStationCount` 个站。也就是说，如果某个组只有 3 站，算法不会再从该组删除站点，避免该组不可解。

该过程直到：

- 所有可删除站点误差都小于阈值；或
- 没有任何组还能继续删除站点。

## 15. 输出结果

### 15.1 AutoCalibrateResult

主要字段：

```cpp
flangeFromCamera
cameraCenterInFlange
quaternion
rpyDeg
matrix4x4
targetBases
estimatedTargetBase
estimatedTargetBases
stationErrors
observations
reportText
reportPath
matrixPath
```

说明：

- `estimatedTargetBases`：每组估计出的三面交点基坐标。
- `estimatedTargetBase`：第 0 组目标点，保留用于兼容单组调用。
- `targetBases`：继承自 `CalibrateResult`，内容与 `estimatedTargetBases` 相同。
- `stationErrors[i].targetIndex`：这里表示 `groupIndex`。

### 15.2 报告文件

默认写入：

```text
<outputDir>/auto_handeye_calibration_result.txt
```

报告中包含：

```text
Target group count
target_group_0_base_mm
target_group_1_base_mm
...
camera_center_in_flange_mm
quaternion_wxyz
rpy_xyz_deg
rotation_matrix
ceres cost
每站误差
mean/rms/max error
```

### 15.3 矩阵文件

默认写入：

```text
<outputDir>/handeye_matrix.txt
```

该矩阵为：

```text
T_flange_camera
```

它把相机坐标变换到机器人法兰坐标。

### 15.4 点云导出

如果启用：

```cpp
exportBaseClouds = true
exportFeatureBaseClouds = true
```

则输出：

```text
<dataDir>/base/group_x/*.pcd
<dataDir>/featureBase/group_x/*.pcd
```

## 16. 为什么多位置扫描更稳健

单位置自动标定中，所有扫描都约束同一个未知点：

```text
B_i * X * p_i = T_0
```

如果机器人姿态变化不足，或者某几站角点候选误选，手眼矩阵可能不稳定。

多位置扫描变成：

```text
B_i * X * p_i = T_0   // group_1
B_j * X * p_j = T_1   // group_2
B_k * X * p_k = T_2   // group_3
```

好处：

- 每组内部约束一个固定点。
- 多组共享同一个手眼矩阵。
- 标定块移动后带来更丰富的空间约束。
- 错误候选更容易在全局一致性评分中暴露。
- 每组目标点独立，不会错误地把不同物理位置强行拉到同一点。

建议每组至少 3 站，并且组内机器人姿态要有明显变化。只在完全相同姿态下重复扫描，主要只能降低点云噪声，不能显著提升手眼可观测性。

## 17. 常见失败原因

### 每组有效站少于 3

错误表现：

```text
At least 3 valid stations are required per group
```

解决方式：每个标定块位置至少扫描 3 次，并确保特征提取成功。

### 分组目录未被识别

目录名应使用：

```text
group_1
group_2
pos_1
position_1
```

并且目录中需要有 `robot_records.pose` 或 `robot_*.pose`。

### 点云中三面角点不稳定

表现为：

```text
No globally consistent grouped corner candidate set found
```

可检查：

- 三个平面是否完整进入点云。
- 平面尺寸是否满足 `planeMinExtentMm/planeMaxExtentMm`。
- RANSAC 参数是否适合当前点云密度。
- 每站 `cornerCandidates` 是否为空。

### 姿态变化不足

如果每组扫描时机器人位姿几乎相同，线性初值求解可能秩不足，或最终误差较大。

建议：

- 改变法兰姿态角度。
- 改变相机观察方向。
- 多位置采集时，每个位置都保持 3 个以上有差异的姿态。

## 18. 代码阅读索引

公共接口：

```text
src/handeye_auto_calibrator.h
```

核心实现：

```text
BuildStationGroups
FindGroupDataDirs
SelectGloballyConsistentObservations
EstimateInitialTransformAndTargets
SolveTranslationAndTargetsForRotation
SolveAutoCalibration
PopulatePublicResult
WriteAutoReport
```

采集分组入口：

```text
src/main.cpp
CaptureOnce(...)
交互命令循环中的数字 groupIndex 指令
```

## 19. 一句话总结

`HandEyeAutoCalibrator` 的核心思想是：让点云自动提供每站三面角点观测，把原本需要人工示教的 `targetBase` 变成优化变量；在多位置采集时，为每个标定块位置设置独立的 `targetBase[group]`，同时共享同一个 `flangeFromCamera`，从而提升手眼标定的自动化程度和稳健性。

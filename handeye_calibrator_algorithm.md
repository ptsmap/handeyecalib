# Hand-Eye 标定算法代码解读

> 文件：`src/handeye_calibrator.h` / `src/handeye_calibrator.cpp`
>
> 生成日期：2026-06-27

---

## 目录

1. [整体架构](#1-整体架构)
2. [模块 1 — 扫描采集与保存](#2-模块-1--扫描采集与保存)
3. [模块 2 — 特征提取（三面角点）](#3-模块-2--特征提取三面角点)
   - [3.1 RANSAC 平面检测](#31-ransac-平面检测)
   - [3.2 平面尺寸过滤](#32-平面尺寸过滤)
   - [3.3 平面合并 MergeSimilarPlanes](#33-平面合并-mergesimilarplanes)
   - [3.4 正交伴侣过滤 FilterPlanesWithOrthogonalMate](#34-正交伴侣过滤-filterplaneswithorthogonalmate)
   - [3.5 三面角点候选构建 BuildCornerCandidates](#35-三面角点候选构建-buildcornercandidates)
   - [3.6 单站特征提取总流程 ExtractStationCandidates](#36-单站特征提取总流程-extractstationcandidates)
4. [模块 3 — 标定（三面角点模式）](#4-模块-3--标定三面角点模式)
   - [4.1 全局一致候选选择 SelectGloballyConsistentObservations](#41-全局一致候选选择-selectgloballyconsistentobservations)
   - [4.2 初始位姿估计 EstimateFlangeFromCamera (SVD)](#42-初始位姿估计-estimateflangefromcamera-svd)
   - [4.3 非线性优化 Calibrate](#43-非线性优化-calibrate)
   - [4.4 三面束平差优化（可选）](#44-三面束平差优化可选)
   - [4.5 迭代离群点剔除](#45-迭代离群点剔除)
   - [4.6 RunCalibrate 顶层流程](#46-runcalibrate-顶层流程)
5. [模块 4 — 多块平面标定（MultiBlock 模式）](#5-模块-4--多块平面标定multiblock-模式)
   - [5.1 多块平面提取](#51-多块平面提取)
   - [5.2 跨站匹配策略](#52-跨站匹配策略)
   - [5.3 多块平面优化](#53-多块平面优化)
6. [核心数据结构速查](#6-核心数据结构速查)
7. [误差评估指标](#7-误差评估指标)
8. [输出文件说明](#8-输出文件说明)
9. [依赖库与坐标约定](#9-依赖库与坐标约定)
10. [完整调用流程图](#10-完整调用流程图)

---

## 1. 整体架构

```
handeye_calibrator
├── Module 1: SaveScanStation / BeginScanCollection
│       采集机器人位姿 + 点云，写入 .pcd / .pose 文件
│
├── Module 2: RunFeatureExtraction / ExtractCornerFromPcd
│       点云 → RANSAC 检测平面 → 合并 → 正交过滤 → 三面角点候选
│
├── Module 3: RunCalibrate
│       候选角点 → 全局束搜索选最优 → SVD 初始估计 →
│       Ceres 非线性优化 → （可选）三面束平差 → 离群点剔除
│
└── Module 4: RunMultiBlockCalibrate
        多块平面 → 跨站匹配（初始矩阵 or GICP）→ 平面约束优化
```

整个系统的目标是求解 **法兰坐标系（flange）到相机坐标系（camera）的刚体变换**，记为：

```
flangeFromCamera ∈ SE(3)
```

即满足以下关系：

```
basePoint = baseFromFlange * flangeFromCamera * cameraPoint
```

其中 `baseFromFlange` 由机器人关节角正运动学给出（每个扫描站不同），`cameraPoint` 是在相机坐标系下测量的目标点（三面角点），`basePoint` 是已知的固定标定靶坐标。

---

## 2. 模块 1 — 扫描采集与保存

### 公开 API

| 函数 | 说明 |
|------|------|
| `SaveScanStation(params)` | 将单帧扫描（点云 + 机器人位姿）写入磁盘 |
| `BeginScanCollection(params)` | 初始化采集目录，返回索引文件路径 |

### 关键数据结构 `ScanSaveParams`

```cpp
struct ScanSaveParams {
    std::string timestamp;             // 时间戳字符串，如 "20260612_143022"
    std::filesystem::path outputDir;   // .pcd 和 .pose 文件输出目录
    std::string fileStem;              // 文件名前缀

    // 点云数据（来自相机原始输出）
    const void* cloudData;             // 点云缓冲区指针
    unsigned int cloudPointCount;      // 点数量
    int cloudDataType;                 // 数据格式类型

    // 机器人位姿（法兰坐标系）
    Eigen::Vector3d flangeTranslation; // 平移量，单位 mm
    Eigen::Quaterniond flangeQuaternion; // 旋转四元数
    Eigen::Vector3d tcpTranslation;    // TCP 平移

    // 可选：若已知手眼矩阵，同步输出基坐标系下的点云
    std::filesystem::path baseCloudOutputDir;
    Eigen::Isometry3d flangeFromCamera;
};
```

### 写入格式

- **`.pcd` 文件**：PCL 标准点云格式（XYZ）
- **`.pose` 文件**：包含机器人 flange 的 4×4 变换矩阵（base←flange），以及 TCP 位移
- **`robot_records.pose`**：所有扫描站的汇总索引文件

---

## 3. 模块 2 — 特征提取（三面角点）

### 总体流程

```
读取 .pcd 文件
    ↓
去除 NaN 点
    ↓
VoxelGrid 降采样（1mm leaf）
    ↓
RANSAC 多平面检测
    ↓
平面尺寸过滤（[planeMinExtentMm, planeMaxExtentMm]）
    ↓
法向归一化 + 确保 d > 0
    ↓
MergeSimilarPlanes（合并相似平面）
    ↓
BuildCornerCandidates（合并后平面）
BuildCornerCandidates（原始平面）
    ↓
候选排序（centroidDistanceOk 优先，score 次之）
    ↓
去重（空间距离 < 1mm）
    ↓
保留 maxCandidatesPerStation 个候选
```

---

### 3.1 RANSAC 平面检测

使用 **RANSAC Shape Detector**（`RansacShapeDetector`）库进行多平面同时检测：

```cpp
doDetection(cloud_sample, detectedPlanes, pcl::PointXYZ(),
    params.ransacMinPoints,         // 最少内点数（默认 100）
    params.ransacEpsilon,           // 点到平面距离阈值，mm（默认 1.0）
    params.ransacBitmapEpsilon,     // 位图分辨率，mm（默认 2.0）
    params.planeAngleTolerance,     // 平面法向量与参考方向最大夹角，deg（默认 25°）
    0.01);                          // 概率参数
```

**关键参数含义：**

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `ransacMinPoints` | 100 | 一个平面至少需要多少个内点 |
| `ransacEpsilon` | 1.0 mm | 点到平面的最大允许距离（判定为内点的阈值） |
| `ransacBitmapEpsilon` | 2.0 mm | RANSAC 位图格分辨率 |
| `planeAngleTolerance` | 25° | 平面法向量与重力方向（或参考方向）的最大偏差 |

检测完成后，`PlaneFitResult` 结构体记录每个检测平面的：
- `a, b, c, d`：平面方程系数（`ax + by + cz + d = 0`）
- `dX, dY`：平面上点集的包围矩形尺寸（mm）
- `centroid`：内点质心
- `pointCount`：内点数量

---

### 3.2 平面尺寸过滤

```cpp
if (fabs(shape.dX) > planeMaxExtentMm || fabs(shape.dY) > planeMaxExtentMm ||
    fabs(shape.dX) < planeMinExtentMm || fabs(shape.dY) < planeMinExtentMm)
{
    continue;  // 过滤掉尺寸不符合的平面
}
```

过滤目的：排除太小（噪声）或太大（背景墙、地板）的平面，只保留与标定靶尺寸匹配的面片。

---

### 3.3 平面合并 `MergeSimilarPlanes`

**算法思路：** 按照贪心策略，将法向量接近、距离接近的平面合并为一组，以点数为权重进行加权平均。

```cpp
std::vector<PlaneObservation> MergeSimilarPlanes(
    const std::vector<PlaneObservation>& planes,
    double mergeAngleDeg = 5.0,    // 法向量夹角阈值
    double mergeDistanceMm = 12.0) // 平面距离阈值
```

**合并判断条件（同时满足）：**

```
groupNormal · planeNormal ≥ cos(mergeAngleDeg)   ← 方向相似
|d_group - d_plane| ≤ mergeDistanceMm             ← 距离相近
```

**合并计算（加权平均）：**

```cpp
// 以点数为权重进行加权平均
group.normalSum     += normal * pointCount;   // 法向累积（归一化方向）
group.weightedCentroid += centroid * pointCount;
group.weightedD     += d * pointCount;
group.pointCount    += pointCount;

// 最终合并结果
mergedPlane.normal   = normalSum.normalized();
mergedPlane.centroid = weightedCentroid / totalPointCount;
mergedPlane.d        = weightedD / totalPointCount;
```

**合并后按点数降序排列**，优先保留支持点多的平面。

**为何需要合并：** 扫描同一个物理平面时，RANSAC 可能因点云噪声、遮挡等原因将同一平面分割为多个片段，合并步骤将它们整合为一个更准确的观测。

---

### 3.4 正交伴侣过滤 `FilterPlanesWithOrthogonalMate`

```cpp
std::vector<PlaneObservation> FilterPlanesWithOrthogonalMate(
    const std::vector<PlaneObservation>& planes,
    double orthogonalToleranceDeg = 8.0,   // 正交检查角度容差
    double maxCentroidDistanceMm = 200.0)  // 质心距离上限
```

**过滤逻辑：** 保留那些在平面集合中至少能找到一个"正交伴侣"（法向量夹角约 90°）的平面。

```cpp
// 判断两个平面是否正交
IsOrthogonal(n1, n2, tol) = |n1·n2| < sin(tol)  →  约等于 0
// 额外约束质心距离，避免不同区域的平面被误判为正交
(centroid_i - centroid_j).norm() < maxCentroidDistanceMm
```

**物理含义：** 三面体角（trihedral corner）由三个两两正交的平面构成。任何一个平面必须和其他平面正交才有意义。该过滤步骤去除孤立的非正交平面，减少后续组合搜索的计算量。

---

### 3.5 三面角点候选构建 `BuildCornerCandidates`

这是特征提取的核心函数，通过三重循环枚举所有平面三元组，找到满足正交约束的角点候选。

```cpp
std::vector<CornerCandidate> BuildCornerCandidates(
    const std::vector<PlaneObservation>& planes,
    bool usesMergedPlanes,
    double orthoToleranceDeg = 8.0,
    double expectedCentroidDistanceMm = 100.0,  // 期望质心到角点距离，mm
    double centroidDistanceToleranceMm = 15.0,  // 距离容差，mm
    size_t maxCandidates = 120)
```

**算法流程：**

```
for i, j, k in combinations(planes, 3):
    // Step 1: 正交检查（三对平面两两正交）
    if not IsOrthogonal(p0, p1) or not IsOrthogonal(p0, p2) or not IsOrthogonal(p1, p2):
        continue

    // Step 2: 计算三平面交点（角点坐标）
    intersection = IntersectPlanes(p0, p1, p2)  // 求解线性方程组 Ax=b

    // Step 3: 质心-角点距离约束检验
    pairDistances   = [|c0-c1|, |c0-c2|, |c1-c2|]   // 两两质心距
    cornerDistances = [|c0-corner|, |c1-corner|, |c2-corner|] // 质心到角点距

    centroidDistanceOk = (所有 pairDistances 接近 expectedCentroidDistanceMm)
                      OR (所有 cornerDistances 接近 expectedCentroidDistanceMm)

    // Step 4: 计算候选评分（越低越好）
    angleResidualDeg   = Σ |angle(ni, nj) - 90°|  // 三对正交角残差
    centroidResidualMm = Σ ||dist| - expected|      // 质心距离残差
    score = angleResidualDeg + 0.02 * centroidResidualMm
    if not centroidDistanceOk:
        score += 10.0  // 距离约束不满足，加惩罚项
```

**三平面求交点（`IntersectPlanes`）：**

设三个平面方程为：

```
n0·x = d0
n1·x = d1
n2·x = d2
```

组成线性方程组：

```
A = [n0; n1; n2]  (3×3 矩阵，每行为法向量)
b = [d0; d1; d2]  (右端项)
x = A⁻¹ · b       (三平面交点)
```

**候选排序规则：**

1. `centroidDistanceOk == true` 的候选优先
2. `score` 从小到大排序

---

### 3.6 单站特征提取总流程 `ExtractStationCandidates`

```cpp
StationCandidates ExtractStationCandidates(
    const PoseRecord& record,
    const FeatureExtractionParams& params)
```

完整流程：

```
1. 加载 .pcd 点云
2. 去除 NaN 点 (pcl::removeNaNFromPointCloud)
3. VoxelGrid 降采样 (leafSize=1mm)
4. RANSAC 多平面检测（doDetection）
5. 平面尺寸过滤（[planeMinExtentMm, planeMaxExtentMm]）
6. 法向量归一化 + 确保 d > 0（统一法向方向朝外）
7. MergeSimilarPlanes → mergedPlanes
8. BuildCornerCandidates(mergedPlanes) → mergedCandidates
9. BuildCornerCandidates(rawPlanes)    → rawCandidates
10. 合并两组候选，去重（空间距离 < 1mm），排序，截取 maxCandidatesPerStation 个
```

**为何同时用合并前后的平面构建候选：** 合并后平面参数更稳定（加权平均），但合并可能引入误差；原始平面精度高但数量多。两者取优，取最终评分最好的候选。

---

## 4. 模块 3 — 标定（三面角点模式）

### 4.1 全局一致候选选择 `SelectGloballyConsistentObservations`

这是整个标定流程中**最关键的组合优化步骤**。在每个扫描站上，特征提取（模块 2）可能产生多达数十个三面角点候选——其中绝大多数是**错误的匹配**（由点云噪声、与标定块无关的墙面/地面、遮挡等因素导致）。目标是从每一站中选出**一个正确的候选**，使所有站之间构成一个几何上自洽（globally consistent）的组合，从而得到误差最小的手眼标定结果。

#### 4.1.1 问题形式化

设有 N 个扫描站，每个站 i 上有备选角点集合：

$$\mathrm{Candidates}_i = \{c_{i,1}, c_{i,2}, \dots, c_{i,m_i}\}$$

每个候选 $c_{i,j}$ 携带：
- `point`：相机坐标系下的三平面交点坐标
- `score`：从 BuildCornerCandidates 产生的几何质量评分（角度残差 + 质心约束残差）

目标是选择一组索引序列 $\mathbf{s} = [s_0, s_1, \dots, s_{N-1}]$，使得标定误差最小：

$$\mathbf{s}^* = \arg\min_{\mathbf{s}} \quad \mathrm{RMSError}\big(\mathrm{Calibrate}(\{ \mathrm{Obs}(i, s_i) \}_{i=0}^{N-1})\big)$$

但全局枚举不可行：若每站有 120 个候选、10 个站，则有 $120^{10} \approx 6.2 \times 10^{20}$ 种组合。

#### 4.1.2 束搜索（Beam Search）核心思想

束搜索是一种**宽度受限的树搜索策略**：逐站扩展路径，每步只保留得分最优的 $W$（束宽）条路径。其背后的基本假设是：

> *如果一个候选组合的前缀（部分站序列）在局部标定中表现很差，那么它不可能成为全局最优解的一部分。*

通过每步对 $W$ 条路径"竞价排名"，束搜索在不探索全树的情况下逼近全局最优点。

#### 4.1.3 数据结构：`BeamState`

```cpp
struct BeamState {
    std::vector<int> candidateIndices;  // 每站的候选索引序列，长度 = 已处理站数
    double rmsError = INF;              // 标定 RMS 误差（mm）
    double score = INF;                 // 路径评分（排序依据）
};
```

一个 `BeamState` 代表"到目前为止对前 k 个站各选了一个候选"这条路径。`candidateIndices[i]` 指向第 i 个站的哪个候选被选中。

#### 4.1.4 算法超参数

```cpp
constexpr size_t kBeamWidth = 10000;         // 束宽：每轮保留的最优路径数
constexpr size_t kCandidatesPerStation = 2;  // 每站最多展开候选数（取评分最好的前2个）
```

| 参数 | 值 | 设计考量 |
|------|-----|---------|
| `kBeamWidth` | 10000 | 足够大以保证搜索覆盖度，同时保证单轮排序 O(W log W) 可行 |
| `kCandidatesPerStation` | 2 | 每个站展开的候选数。仅取每站 top-2 是因为：评分最好的候选大概率是正确角点；取更多候选会使束膨胀过快。实际上 2 个候选下，N=10 的树总共只有 $2^{10} = 1024$ 个叶节点，束宽 10000 意味着**完全不剪枝**即可覆盖全树 |

**关键观察：** 当 `kCandidatesPerStation = 2` 且 $2^N \ll W$ 时，束搜索退化为**完全枚举**（不剪枝）。对于典型场景 N=10，$2^{10} < 10000$，束宽远超全树大小，搜索是**精确的全局最优**。

当 N > 14 时（$2^{14} = 16384 > 10000$），束才开始剪枝。

#### 4.1.5 评分函数：分阶段策略

束搜索的核心难点在于：**前几个站无法计算有意义的标定误差**。6-DOF 的手眼矩阵至少需要 3 个非共线观测才能确定。算法对路径评分采用了两阶段策略：

##### 阶段一：几何质量评分（观测数 < 3）

```cpp
double geometryPenalty = 0.0;
for (const auto& obs : observations) {
    geometryPenalty += obs.candidateScore;
}
geometryPenalty /= observations.size();
state.score = geometryPenalty;
```

使用 `CornerCandidate::score`（来自 BuildCornerCandidates 的几何评分）作为初始路径启发式指标。该评分已经编码了：
- 三对平面的正交角度残差（angleResidualDeg）
- 质心到角点的距离偏差（centroidResidual）
- 质心距离约束是否满足（+10 惩罚项）

> **物理直觉：** 一个好的三面角点候选，它的三个平面必然接近两两正交、质心位于合理的距离范围。这些几何属性**与手眼矩阵无关**——即使只有 1 个站也可以评估。因此在前 2 站的盲选阶段，用几何评分筛选质量高的候选是合理的启发式策略。

##### 阶段二：标定 RMS 评分（观测数 ≥ 3）

```cpp
const auto result = Calibrate(observations, targetBase, /*enableTrihedral=*/false);
if (!result) {
    continue;  // 丢弃无解路径
}
state.score = result->rmsError;
```

从第 3 站起，观测集合已足够求出手眼矩阵。此时每个路径状态调用 `Calibrate()`（包含 SVD 初始估计 + Ceres 非线性优化，**不含**三面束平差以节省计算），将 RMS 重投影误差作为路径评分：

$$\mathrm{score}(\mathbf{s}_{0:k}) = \sqrt{\frac{1}{k+1} \sum_{i=0}^{k} \big\| \mathrm{baseFromFlange}_i \cdot T_{fc} \cdot \mathrm{cameraPoint}_{i,s_i} - \mathrm{targetBase} \big\|^2}$$

> **切换逻辑的意义：** 第 1、2 站用的是"自评"（候选自身的几何属性），第 3 站开始用的是"互评"（所有站之间通过标定矩阵耦合的交叉验证）。互评才是真正的目标函数——同时保证了连续性。

#### 4.1.6 完整算法流程（伪代码 + 逐行解释）

```
输入: stations[0..N-1]  每站包含 candidates[]（已按 score 排序）
      targetBase        标定靶在基坐标系下的真实坐标
输出: observations[0..N-1]  每站选定一个候选构成的观测集合

────────────────────────────────────────────────────────
│  Step 0: 初始化
│──────────────────────────────────────────────────────
│  beam = [BeamState{indices=[], score=INF}]    // 空路径作为种子
│  if stations.size() < 3:  返回错误（至少 3 站）
│
│  for stationIndex = 0 to N-1:
│  ┌─────────────────────────────────────────────────
│  │  Step 1: 截取当前站的候选列表
│  │─────────────────────────────────────────────────
│  │  candidateCount = min(kCandidatesPerStation=2, stations[i].candidates.size())
│  │  // 只取评分最好的前 2 个候选
│  │  // 若该站 0 个候选：报错并返回
│  │
│  │  Step 2: 扩展阶段 — 对 beam 中每条路径 × 本站每个候选
│  │─────────────────────────────────────────────────
│  │  next = []   // 下一轮束（空容器）
│  │
│  │  for each state in beam:
│  │    ┌ for each candidateIndex in [0..candidateCount):
│  │    │
│  │    │  Step 2a: 克隆路径并追加选择
│  │    │  newState = state
│  │    │  newState.candidateIndices.push(candidateIndex)
│  │    │
│  │    │  Step 2b: 构建 FeatureObservation 列表
│  │    │  observations = []
│  │    │  for j = 0..stationIndex:
│  │    │    selectedCandidate = stations[j].candidates[state.candidateIndices[j]]
│  │    │    observations[j] = MakeFeatureObservation(stations[j], selectedCandidate)
│  │    │      // 提取 cameraPoint、baseFromFlange、featurePlanes、
│  │    │      //   candidateScore、centroidDistanceOk、cloudPath 等
│  │    │
│  │    │  Step 2c: 评分
│  │    │  geometryPenalty = mean of observations[*].candidateScore
│  │    │
│  │    │  if observations.size() >= 3:
│  │    │    result = Calibrate(observations, targetBase, enableTrihedral=false)
│  │    │    // 内部调用: EstimateFlangeFromCamera(SVD) → Ceres 非线性优化
│  │    │    // 若标定失败（无解）→ continue 丢弃该路径
│  │    │    newState.rmsError = result.rmsError
│  │    │    newState.score     = result.rmsError
│  │    │  else:
│  │    │    newState.rmsError = geometryPenalty
│  │    │    newState.score     = geometryPenalty
│  │    │
│  │    │  next.push(newState)
│  │    └─ end for candidateIndex
│  │  end for state
│  │
│  │  Step 3: 剪枝阶段 — 排序 + 截断
│  │─────────────────────────────────────────────────
│  │  if next.empty():  报错并返回
│  │  sort(next, key=score, ascending=true)
│  │  if next.size() > kBeamWidth:  next.resize(kBeamWidth)
│  │  beam = next
│  └─────────────────────────────────────────────────
│  end for stationIndex
│
│  Step 4: 提取最优解
│──────────────────────────────────────────────────────
│  best = beam[0]   // 经过所有站后，束中排第一的即为全局最优路径
│  for i = 0..N-1:
│    observations[i] = MakeFeatureObservation(stations[i],
│         stations[i].candidates[best.candidateIndices[i]])
│  return observations
```

#### 4.1.7 具体数值示例

假设 5 个扫描站，每站候选评分和标定误差如下：

| 站 | 候选 0 score | 候选 1 score | 备注 |
|----|-------------|-------------|------|
| 0 | 0.3 | 0.5 | 候选 0 更优 |
| 1 | 0.4 | 0.6 | 候选 0 更优 |
| 2 | 0.2 | 0.8 | 候选 0 更优 |
| 3 | 0.7 | 0.3 | 候选 1 更优（注意反转） |
| 4 | 0.9 | 0.1 | 候选 1 更优 |

**站 0 扩展（beam size = 1 × 2 = 2）：**

| 路径 | indices | 观测数 | 评分方式 | score |
|------|---------|--------|---------|-------|
| s₀₀ | [0] | 1 | geometryPenalty | 0.3 |
| s₀₁ | [1] | 1 | geometryPenalty | 0.5 |

**站 1 扩展（beam size = 2 × 2 = 4）：**

| 路径 | indices | 观测数 | 评分方式 | score |
|------|---------|--------|---------|-------|
| s₁₀₀ | [0,0] | 2 | geometryPenalty | (0.3+0.4)/2=0.35 |
| s₁₀₁ | [0,1] | 2 | geometryPenalty | (0.3+0.6)/2=0.45 |
| s₁₁₀ | [1,0] | 2 | geometryPenalty | (0.5+0.4)/2=0.45 |
| s₁₁₁ | [1,1] | 2 | geometryPenalty | (0.5+0.6)/2=0.55 |

> **到此为止**：评分完全是候选自身的几何质量，4 条路径都还保留着（远未达到 10000）。

**站 2 扩展（beam size = 4 × 2 = 8，开始用 Calibrate 评分）：**

| 路径 | indices | 评分方式 | RMS 误差 |
|------|---------|---------|----------|
| s₂₀₀₀ | [0,0,0] | Calibrate | 0.52 mm |
| s₂₀₀₁ | [0,0,1] | Calibrate | 1.87 mm |
| s₂₀₁₀ | [0,1,0] | Calibrate | 3.41 mm |
| s₂₀₁₁ | [0,1,1] | Calibrate | 0.88 mm |
| s₂₁₀₀ | [1,0,0] | Calibrate | 5.12 mm |
| s₂₁₀₁ | [1,0,1] | Calibrate | 2.30 mm |
| s₂₁₁₀ | [1,1,0] | Calibrate | 4.67 mm |
| s₂₁₁₁ | [1,1,1] | Calibrate | 7.81 mm |

> **评分方式切换生效：** `s₂₀₀₀` 虽然每站几何质量都是候选 0，但真正重要的是它产生了最低的标定误差 0.52 mm。`s₂₀₁₀` 的几何评分 0.35 是最低的（候选 0→候选 1→候选 0 各自 score 都很低），但标定误差 3.41 mm 说明它们不是同一物理点的角点——局部几何质量好不等于全局一致性好。

**站 3 和站 4** 继续扩展，每次保留 score 最小的 10000 条（实际始终远小于上限）。最终 `beam[0]` 给出 RMS 误差最小的全路径。

> **关键洞察：** 候选 3 的"最优"是候选 1（score 0.3 vs 候选 0 的 0.7），但如果在站 2 时选了路径 [0,0,0]（几何最优），到了站 3 用候选 0 反而可能比用候选 1 误差更大。束搜索同时保留多条路径，避免了贪婪选择的短视。

#### 4.1.8 时间复杂度分析

| 项目 | 公式 | N=10 时 |
|------|------|---------|
| 树的全节点数 | $\sum_{k=1}^N C^k$ | 2046 |
| 每条路径评估次数 | $\min(W, C^k)$ 次 Calibrate() | 1024 次 |
| 每次 Calibrate() | SVD(3×3) + Ceres(1000 iter) | ~2-5 ms |
| 总计算时间（N=10） | $N \times 2 \times \min(W, 2^N)$ 次 × ~3ms | ~6 秒 |

当 $2^N \ll W$（N ≤ 13）时，束搜索退化为**精确穷举**，保证全局最优。当 N 更大时，剪枝开始生效，复杂度维持 $O(N \times C \times W)$。

#### 4.1.9 评分函数的渐进性质

```
        几何评分 (自评)             标定 RMS 评分 (互评)
        ────────────                ──────────────────
               │                          │
    站 0 ──────┼──── 站 1 ────────────────┼──── 站 2 ───────→ 站 N-1
               │                          │
        candidateScore            Calibrate().rmsError
        (角度残差+质心距)          (SVD + Ceres 重投影误差)
               │                          │
         局部几何一致              全局几何一致（真正目标）
```

这种"先自评后互评"的策略是算法的核心设计智慧：
- 前 2 站无法做 6-DOF 标定，但候选的几何自评已能过滤大量错误；
- 从第 3 站起，切换为真正的标定误差——自动惩罚"局部看上去好但全局不一致"的伪正确组合；
- 每个 `Calibrate()` 调用内部执行完整的 SVD→Ceres 流程（但不做束平差），确保了评分是真实可复现的标定质量。

#### 4.1.10 多靶点扩展 `SelectGloballyConsistentMultiTargetObservations`

当场景中存在多个标定靶点时（如 `targetBases` 包含 3 个靶点），对每个靶点**独立运行**单靶点的束搜索：

```cpp
for each targetIndex in [0..M-1]:
    observations_for_target = SelectGloballyConsistentObservations(
        stations, targetBases[targetIndex])
    // 将每个观测的 targetIndex 标记为当前靶点
    allObservations.merge(observations_for_target)
return allObservations
```

这使得每个靶点可以独立选择最优候选组合，互不干扰。最终所有靶点的观测合并，放入一个大的 Ceres 优化中同时求解——多个靶点的约束共同决定手眼矩阵，提高了标定的鲁棒性和精度。

---

### 4.2 初始位姿估计 `EstimateFlangeFromCamera` (SVD)

在进入非线性优化之前，先用**基于 SVD 的点集配准**给出初始估计：

```cpp
std::optional<Eigen::Isometry3d> EstimateFlangeFromCamera(
    const std::vector<FeatureObservation>& observations,
    const std::vector<Eigen::Vector3d>& targetBases)
```

**数学原理（Kabsch 算法）：**

设每个观测站 i 有：
- `cameraPoint_i`：相机坐标系下的角点坐标
- `flangePoint_i = baseFromFlange_i⁻¹ × targetBase`：法兰坐标系下的目标点

目标：找旋转矩阵 R 和平移向量 t，使：

```
cameraPoint_i ≈ R⁻¹ × (flangePoint_i - t)
```

等价于求：

```
flangePoint_i ≈ R × cameraPoint_i + t
```

**SVD 求解步骤：**

```
1. 计算质心
   cameraMean = mean(cameraPoints)
   flangeMean = mean(flangePoints)

2. 构建协方差矩阵（3×3）
   H = Σ (cameraPoint_i - cameraMean) × (flangePoint_i - flangeMean)ᵀ

3. SVD 分解
   H = U × S × Vᵀ

4. 计算旋转矩阵（考虑反射情况）
   R = V × Uᵀ
   if det(R) < 0:  // 纠正反射为旋转
       V[:,2] *= -1
       R = V × Uᵀ

5. 计算平移
   t = flangeMean - R × cameraMean

6. 构建 SE(3) 变换
   flangeFromCamera = [R | t; 0 0 0 1]
```

**为何需要初始估计：** Ceres 非线性优化对初值敏感，SVD 给出的封闭解虽然精度有限，但足以将优化引导至正确的局部极值附近。

---

### 4.3 非线性优化 `Calibrate`

```cpp
std::optional<CalibrationResult> Calibrate(
    const std::vector<FeatureObservation>& observations,
    const std::vector<Eigen::Vector3d>& targetBases,
    bool enableTrihedralOptimization)
```

**优化问题：**

最小化所有观测站上，预测基坐标点与真实靶点之间的重投影误差：

```
min_{T_fc} Σ_i ρ(‖baseFromFlange_i × T_fc × cameraPoint_i - targetBase_i‖)
```

其中 `ρ(·)` 为 Huber 鲁棒损失函数，`T_fc = flangeFromCamera`。

**Ceres 优化器配置：**

```cpp
HandEyeNonlinearOptimizer::Options options;
options.huberLossMm          = 1.0;   // Huber 损失阈值（mm）
options.maxNumIterations     = 1000;  // 最大迭代次数
options.minimizerProgressToStdout = false;
```

使用 **Ceres Solver** 自动微分求解，内部使用四元数参数化旋转（`EigenQuaternionParameterization`），确保旋转流形上的正确优化。

**Huber 损失函数的作用：**

```
ρ(r) = {  r²/2,         |r| ≤ δ
          δ(|r| - δ/2), |r| > δ
```

当误差小于阈值 δ（1 mm）时退化为 L2 损失，超过阈值时线性增长，有效抑制离群点对优化的干扰。

---

### 4.4 三面束平差优化（可选）

当 `enableTrihedralOptimization = true` 时，在角点优化结果基础上进行**三面束平差（Trihedral Bundle Adjustment）**，进一步精化手眼矩阵。

```cpp
HandEyeTrihedralBundleOptimizer::Options trihedralOptions;
trihedralOptions.maxCornerErrorForTrihedralStageMm = 5.0;  // 只有RMS误差<5mm才执行
trihedralOptions.planeHuberLossMm              = 0.5;
trihedralOptions.targetAlignmentHuberLossMm    = 0.5;
trihedralOptions.targetAlignmentSigmaMm        = 0.5;
trihedralOptions.minMatchedNormalDot           = 0.94;  // 法向匹配阈值（约 ±20°）
trihedralOptions.maxNumIterations              = 5000;
```

**束平差的额外约束：**

除角点位置约束外，还加入**平面约束**：将相机坐标系下的各观测平面变换到基坐标系后，同一物理平面的所有观测应共面。

残差包含两类：
- **角点残差**：预测基坐标角点 vs 已知靶点位置
- **平面残差**：各站相机平面转换到基坐标系后，点到平面距离

**可选目标姿态优化（多靶点模式）：**

当有 ≥2 个靶点时，允许同步优化靶点的真实位置（`optimizeTargetPose = true`），以消除靶点安装误差对标定结果的影响。

---

### 4.5 迭代离群点剔除

```cpp
// 最大允许单站误差（mm），超过则剔除，0 = 不剔除
double maxStationErrorMm = 10.0;

while (observations.size() >= kMinStations) {
    // 找误差最大的站
    worstIdx = argmax(errors)
    worstError = max(errors)

    if worstError <= maxStationErrorMm:
        break  // 所有站都在容差内，停止

    // 剔除该站
    observations.erase(worstIdx)
    removedCount++

    // 用剩余站重新标定
    calibResult = Calibrate(observations, targetBases, ...)
}
```

**最少保留 3 个扫描站**（`kMinStations = 3`），确保 6-DOF 约束仍然有解。

**剔除策略的意义：** 机器人位姿测量误差、点云测量噪声、或错误的角点匹配都可能导致某几个站的误差异常大，迭代剔除避免这些离群站污染最终的标定结果。

---

### 4.6 `RunCalibrate` 顶层流程

```
RunCalibrate(params)
│
├── Step 0: 验证输入目录是否存在
│
├── Step 1: 特征提取
│   ├── 若 params.stations 非空 → 直接使用预提取特征（跳过提取）
│   └── 否则 → 读取 pose 文件，对每个站调用 ExtractStationCandidates
│
├── Step 2: 全局候选选择
│   ├── 单靶点 → SelectGloballyConsistentObservations
│   └── 多靶点 → SelectGloballyConsistentMultiTargetObservations
│
├── Step 3: 标定
│   └── Calibrate(observations, targetBases, enableTrihedral)
│
├── Step 4: 迭代离群点剔除（若 maxStationErrorMm > 0）
│
├── Step 5: 填充结果结构体（误差统计、每站详情）
│
└── Step 6: 输出
    ├── handeye_calibration_result.txt   — 标定报告
    ├── handeye_matrix.txt               — 手眼矩阵（4×4）
    ├── base/*.pcd                       — 所有扫描点云变换到基坐标系（可选）
    └── featureBase/*.pcd                — 特征点云变换到基坐标系（可选）
```

---

## 5. 模块 4 — 多块平面标定（MultiBlock 模式）

### 5.1 多块平面提取

MultiBlock 模式使用**平面约束**而非角点约束，适用于无法稳定检测三面角点的场景（如只有两块平板标定块）。

```cpp
std::optional<MultiBlockPlaneStation> ExtractMultiBlockPlaneStation(
    const PoseRecord& record)
```

提取流程与单站特征提取类似：
1. 加载点云 → 去 NaN → 降采样
2. RANSAC 平面检测
3. 平面尺寸过滤
4. 平面合并
5. 两两构建平面对候选（需满足正交约束）
6. 组合两对构建双块候选

---

### 5.2 跨站匹配策略

MultiBlock 最核心的挑战是：不同扫描站检测到的平面，需要确定哪些平面对应同一个物理平面（标定块的同一面）。

两种匹配策略（根据是否有初始手眼矩阵选择）：

#### 策略 A：基于初始手眼矩阵匹配（`useInitialForMatching = true`）

```
result.matchingStrategy = "initial_handeye"
```

利用已有的手眼矩阵（上一次标定结果或粗估计），将各站相机坐标系的平面法向量变换到基坐标系，然后比较法向量夹角：

```
normalBase_i = (baseFromFlange_i × flangeFromCamera) × normalCamera_i
matchScore = max(|normalBase_ref · normalBase_i|)  // ≥ minMatchedNormalDot(0.70)
```

#### 策略 B：基于 GICP 相邻站配准（`useInitialForMatching = false`）

```
result.matchingStrategy = "adjacent_gicp"
```

当没有可用的初始手眼矩阵时，使用 **Fast GICP（Generalized ICP）** 对相邻扫描站的点云进行配准，从配准变换推算各站平面的对应关系。

```cpp
// GICP 参数
gicpMinPoints = 80;
gicpVoxelLeafMm = 3.0;
gicpMaxCorrespondenceDistanceMm = 2.0;
gicpMaxIterations = 10000;
gicpMaxFitnessMm2 = 200000.0;   // 配准质量阈值
```

---

### 5.3 多块平面优化

```cpp
HandEyeMultiBlockPlaneOptimizer::Result optimizer.Optimize(
    optimizerStations,
    initialFlangeFromCamera)
```

**优化约束：**

对于每个扫描站 i 的每个匹配平面 f，变换到基坐标系后，同一物理平面的所有点应满足：

```
n_base · (baseFromCamera_i × p) - d_base ≈ 0
```

其中 `baseFromCamera_i = baseFromFlange_i × flangeFromCamera`。

**约束完备性检查（平移秩检查）：**

```cpp
options.requireFullTranslationRank = true;  // 要求平移约束满秩（秩=3）
result.translationConstraintRank = ...;
```

平面约束理论上只能约束旋转，平移约束需要不同的平面方向覆盖三个坐标轴方向。`translationConstraintRank` 记录实际约束的秩，必须为 3（满秩）才能唯一确定 6-DOF 变换。

---

## 6. 核心数据结构速查

### 平面观测 `PlaneObservation`

| 字段 | 类型 | 说明 |
|------|------|------|
| `normal` | `Eigen::Vector3d` | 单位法向量（相机坐标系，d>0 确保朝外） |
| `d` | `double` | 原点到平面的有符号距离（mm） |
| `centroid` | `Eigen::Vector3d` | 内点质心（mm） |
| `pointCount` | `size_t` | 支持点数量 |
| `points` | `vector<Vector3d>` | 所有内点坐标（用于束平差） |

### 三面角点候选 `CornerCandidate`

| 字段 | 类型 | 说明 |
|------|------|------|
| `planeIndices` | `array<int,3>` | 构成此角点的三个平面索引 |
| `point` | `Eigen::Vector3d` | 三平面交点坐标（相机坐标系，mm） |
| `score` | `double` | 综合评分（越低越好）|
| `angleResidualDeg` | `double` | 三对正交角度残差之和（deg） |
| `centroidDistanceOk` | `bool` | 质心距离约束是否满足 |
| `usesMergedPlanes` | `bool` | 是否来自合并后平面 |

**评分公式：**

```
score = angleResidualDeg + 0.02 × min(centroidPairResidual, centroidCornerResidual)
        + (10.0 if not centroidDistanceOk else 0)
```

### 标定观测 `FeatureObservation`（内部）

| 字段 | 类型 | 说明 |
|------|------|------|
| `baseFromFlange` | `Isometry3d` | 机器人正运动学结果（base←flange） |
| `cameraPoint` | `Vector3d` | 相机坐标系下的角点坐标 |
| `targetIndex` | `int` | 对应的靶点编号（多靶点场景） |
| `featurePlanes` | `array<PlaneObservation,3>` | 构成角点的三个平面（用于束平差） |

---

## 7. 误差评估指标

标定完成后，对每个扫描站 i 计算重投影误差：

```
predicted_base_i = baseFromFlange_i × flangeFromCamera × cameraPoint_i
error_i          = predicted_base_i - targetBase_{targetIndex_i}
errorNorm_i      = ‖error_i‖  (mm)
```

汇总统计量：

| 指标 | 计算公式 | 说明 |
|------|----------|------|
| `meanErrorMm` | `Σ errorNorm / N` | 平均误差 |
| `rmsErrorMm` | `√(Σ errorNorm² / N)` | 均方根误差（最常用指标）|
| `maxErrorMm` | `max(errorNorm)` | 最大误差 |

**典型接受标准：** RMS 误差 < 1.0 mm 为优秀，< 2.0 mm 为可接受。

---

## 8. 输出文件说明

| 文件 | 内容 |
|------|------|
| `handeye_matrix.txt` | 手眼矩阵 `flangeFromCamera`，4×4 格式，行主序 |
| `handeye_calibration_result.txt` | 完整标定报告（参数、误差、每站详情） |
| `base/*.pcd` | 所有扫描点云变换到机器人基坐标系（用于验证） |
| `featureBase/*.pcd` | 特征点云变换到基坐标系（用于验证角点位置） |
| `handeye_multiblock_result.txt` | MultiBlock 模式报告 |

**`handeye_matrix.txt` 格式示例：**

```
flangeFromCamera:
 0.9998  0.0012 -0.0187  23.456
-0.0012  0.9999  0.0045  -5.123
 0.0187 -0.0045  0.9998  78.901
 0.0000  0.0000  0.0000   1.000
```

---

## 9. 依赖库与坐标约定

### 依赖库

| 库 | 用途 |
|----|------|
| **Eigen** | 矩阵运算、SVD、SE(3) 变换 |
| **PCL** | 点云加载、降采样、NaN 过滤 |
| **RANSAC Shape Detector** | 多平面同时检测 |
| **Ceres Solver** | 非线性最小二乘优化 |
| **Fast GICP** | 点云配准（MultiBlock 模式） |

### 坐标系约定

| 符号 | 含义 |
|------|------|
| `base` | 机器人世界坐标系（固定） |
| `flange` | 机器人末端法兰坐标系（随关节运动） |
| `camera` | 3D 相机坐标系（固定在法兰上） |
| `baseFromFlange` | 从法兰到基坐标系的变换（机器人正运动学） |
| `flangeFromCamera` | 从相机到法兰的变换（**即手眼矩阵，待求量**） |

**变换链：**

```
basePoint = baseFromFlange × flangeFromCamera × cameraPoint
           ↑(机器人提供)   ↑(标定结果)         ↑(相机测量)
```

---

## 10. 完整调用流程图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     RunCalibrate 完整流程                                │
└─────────────────────────────────────────────────────────────────────────┘

机器人移动到各标定位置
        │
        ▼
BeginScanCollection(outputDir)
        │
        ▼
foreach 位置:
  采集点云 + 读取机器人关节角 → 正运动学 → baseFromFlange
  SaveScanStation(.pcd + .pose)
        │
        ▼
RunCalibrate(dataDir)
  │
  ├─► foreach 扫描站:
  │     ExtractStationCandidates
  │       │
  │       ├─ VoxelGrid 降采样 (1mm)
  │       ├─ RANSAC 多平面检测
  │       ├─ 尺寸过滤 [65mm, 150mm]
  │       ├─ MergeSimilarPlanes (angle<5°, dist<12mm)
  │       ├─ BuildCornerCandidates (merged) → candidates A
  │       ├─ BuildCornerCandidates (raw)    → candidates B
  │       └─ 合并排序去重 → top-K candidates
  │
  ├─► SelectGloballyConsistentObservations (Beam Search)
  │     束宽=10000, 每站考虑前2个候选
  │     用 Calibrate() 作为束评分函数
  │     └─ 返回全局最优观测集合
  │
  ├─► EstimateFlangeFromCamera (SVD / Kabsch)
  │     构建协方差矩阵 → SVD → 旋转矩阵 → 平移向量
  │     └─ 初始 flangeFromCamera
  │
  ├─► HandEyeNonlinearOptimizer (Ceres)
  │     最小化 Σ ρ(‖baseFromFlange×T×cameraPoint - targetBase‖)
  │     Huber loss δ=1mm, maxIter=1000
  │     └─ 精化 flangeFromCamera
  │
  ├─► [可选] HandEyeTrihedralBundleOptimizer
  │     同时优化角点约束 + 平面约束 (+ 多靶点时的靶点位置)
  │     Huber loss δ=0.5mm, maxIter=5000
  │     └─ 进一步精化 flangeFromCamera
  │
  ├─► 迭代离群点剔除 (maxStationErrorMm=10mm)
  │     重复：找最大误差站 → 若超限则剔除 → 重新标定
  │     └─ 直到所有站误差 < 阈值 或 站数 < 3
  │
  └─► 输出
        handeye_matrix.txt
        handeye_calibration_result.txt
        base/*.pcd (可选)
```

---

*文档基于 `handeye_calibrator.h` 和 `handeye_calibrator.cpp` 源码生成，如有代码更新请同步维护。*

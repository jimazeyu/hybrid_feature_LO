# basic_operations是对点云的基本操作，包括以下步骤
1. pykitti读取数据集并可视化
2. 基础坐标变换（旋转平移，使用欧拉角和四元数）
3. 读取两帧并使用ICP进行匹配
4. 点云基础操作（降采样、点云访问）
5. 映射range image
6. 点云放入rangenet进行分割

# GT_coor_transformer
该程序中将GT的坐标 transform from cam to velo
# basic_operations

## RangeNet++ Segment
../prediction 中的数据是用rangenet++后得到的，

max_valsxxxx.label 是 某点云属于最可能类别的权重值，也就是贡献度

其中project_xxx.label 是 proj_argmax 是在Range Image上的语义分割结果，

unproj_argmax 是将 proj_argmax 反投影回三维空间的结果，保持了原始点云的结构和形式。


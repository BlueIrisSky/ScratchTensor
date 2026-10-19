# 函数与算子

本页的算子通过输入张量名称取值，再把结果注册到输出名称。输出名称不必事先创建，普通临时结果可以被同名结果替换。运算输出不能覆盖开启“需梯度”的叶子参数，遇到这种情况会报错；需要重建参数时，明确使用创建积木。

以下 `sbdsl` 代码块都是脚本体片段，应放在角色的绿旗事件或自制积木体中。使用前先创建所需的输入张量，并在配置中加载 `scratch-tensor-extension.js`。完整工程、扩展配置和训练示例见[演示](demo.md)，张量的创建和取值见[张量](tensor.md)。

`scratch-tensor-extension` 是扩展文件名对应的命名空间，可直接写在调用前。菜单用 `/'选项'/`，布尔值用 `<true>` 或 `<false>`。所有输入槽都要写，积木面板上的默认值不能在源码中省略。

## 损失函数

四种损失函数都输出标量张量：

```sbdsl
scratch-tensor-extension.[MSELoss 预测: "prediction" 真实标签: "target" 赋值给 "mse_loss"]
scratch-tensor-extension.[CrossEntropyLoss Logits: "logits" 标签: "labels" 赋值给 "ce_loss"]
scratch-tensor-extension.[BCELoss 预测: "probabilities" 标签: "binary_labels" 赋值给 "bce_loss"]
scratch-tensor-extension.[L1Loss 预测: "prediction" 标签: "target" 赋值给 "l1_loss"]
```

| 算子 | 输入要求与计算内容 |
| --- | --- |
| MSELoss | 预测值和标签形状相同，计算平方误差的平均值。 |
| CrossEntropyLoss | `logits` 为 `[N, C]`，`labels` 为长度 `N` 的类别索引向量，索引从 `0` 开始。输入是未经 Softmax 的分数，内部计算 Softmax 和负对数似然。 |
| BCELoss | 输入预测概率和二值标签，形状相同。预测值通常来自 Sigmoid；内部会截断过于接近 `0` 或 `1` 的概率，避免对数溢出。 |
| L1Loss | 预测值和标签形状相同，计算绝对误差的平均值。 |

损失值可通过张量的 `/'标量值'/` 菜单读取。反向传播只更新预测一侧的梯度，标签不参与求导。

## 激活函数与 Softmax

激活函数逐元素计算，输出形状不变。菜单有四项：

```sbdsl
scratch-tensor-extension.[对 Tensor "A" 应用 /'ReLU'/ 激活函数 赋值给 "relu"]
scratch-tensor-extension.[对 Tensor "A" 应用 /'GELU'/ 激活函数 赋值给 "gelu"]
scratch-tensor-extension.[对 Tensor "A" 应用 /'Sigmoid'/ 激活函数 赋值给 "sigmoid"]
scratch-tensor-extension.[对 Tensor "A" 应用 /'Tanh'/ 激活函数 赋值给 "tanh"]
```

Softmax 沿指定维度计算概率，`-1` 表示最后一维：

```sbdsl
scratch-tensor-extension.[Softmax 输入: "A" 沿维度: -1 赋值给 "softmax"]
```

## 缩放点积注意力

SDPA 先计算 Q 与 K 的相似度，经过缩放、掩码和 Softmax，再与 V 相乘。

```sbdsl
scratch-tensor-extension.[Scaled Dot Product Attention
    Q: "Q" K: "K" V: "V"
    因果掩码: <false> 丢弃概率: 0 缩放（0=自动）: 0 赋值给 "attention"]
```

Q、K、V 都必须是四维张量，形状要求如下：

| 张量 | 形状 |
| --- | --- |
| Q | `[B, H, Lq, D]` |
| K | `[B, H, Lk, D]` |
| V | `[B, H, Lk, Dv]` |
| 输出 | `[B, H, Lq, Dv]` |

`B` 是批次大小，`H` 是注意力头数。三者的 `B`、`H` 必须一致；Q 与 K 的最后一维相同，K 与 V 的序列长度相同。Q 的序列长度可以与 K 不同，V 的最后一维也可以与 Q 不同，因此输出不一定与 Q 同形状。

因果掩码设为 `<true>` 时，会屏蔽键索引大于查询索引的位置。丢弃概率作用于注意力权重；关闭计算图追踪后，不再随机丢弃。

缩放槽填 `0` 时使用 `1 / sqrt(D)`，填正数则使用该值。底层函数的缩放参数可选，但 ScratchDSL 调用不能留空；这里用 `0` 选择默认值。缩放值必须是有限数，负数会报错。

底层 TypeScript API 还支持叠加自定义注意力掩码 `attnMask`，当前积木没有这个输入槽，不能通过上述调用传入。

## 四则运算与矩阵乘法

以下示例先创建两个矩阵，再分别计算五种运算：

```sbdsl
scratch-tensor-extension.[创建 Tensor 命名为 "A" 并赋值 "[[1,2],[3,4]]" 需梯度: <false>]
scratch-tensor-extension.[创建 Tensor 命名为 "B" 并赋值 "[[10,20],[30,40]]" 需梯度: <false>]

scratch-tensor-extension.[张量 "A" 与 "B" 进行 /'+'/ 运算 赋值给 "add"]
scratch-tensor-extension.[张量 "A" 与 "B" 进行 /'-'/ 运算 赋值给 "sub"]
scratch-tensor-extension.[张量 "A" 与 "B" 进行 /'*'/ 运算 赋值给 "mul"]
scratch-tensor-extension.[张量 "B" 与 "A" 进行 /'/'/ 运算 赋值给 "div"]
scratch-tensor-extension.[张量 "A" 与 "B" 进行 /'@'/ 运算 赋值给 "matmul"]
```

| 菜单 | 运算 | 上例结果 |
| --- | --- | --- |
| `/'+'/` | 逐元素加法 | `[[11,22],[33,44]]` |
| `/'-'/` | 逐元素减法 | `[[-9,-18],[-27,-36]]` |
| `/'*'/` | 逐元素乘法 | `[[10,40],[90,160]]` |
| `/'/'/` | 逐元素除法 | `[[10,10],[10,10]]` |
| `/'@'/` | 矩阵乘法 | `[[70,100],[150,220]]` |

菜单内容统一用单引号包住，除法选项写成 `/'/'/`。`/'*'/` 里的星号是乘法菜单项，不是注释；ScratchDSL 的单行注释以 `#` 开始。

逐元素运算支持广播。矩阵乘法支持向量点积、二维矩阵乘法、矩阵与向量相乘，以及高维张量的批量矩阵乘法和批次维度广播；参与相乘的维度仍需匹配。除数中的零可能产生 `Infinity` 或 `NaN`。

## 逐元素数学函数

一元数学菜单有 Exp、Log、Sqrt、Abs 四项，分别计算指数、自然对数、平方根和绝对值：

```sbdsl
scratch-tensor-extension.[对 Tensor "A" 计算 /'Exp'/ 赋值给 "exp"]
scratch-tensor-extension.[对 Tensor "A" 计算 /'Log'/ 赋值给 "log"]
scratch-tensor-extension.[对 Tensor "A" 计算 /'Sqrt'/ 赋值给 "sqrt"]
scratch-tensor-extension.[对 Tensor "A" 计算 /'Abs'/ 赋值给 "abs"]
```

幂运算使用单独的积木，下面计算每个元素的平方：

```sbdsl
scratch-tensor-extension.[对 Tensor "A" 计算幂 2 赋值给 "square"]
```

## 形状与结构变换

下面的片段按前文的 `A`、`B` 均为 `[2, 2]` 举例。形状和张量名称列表传入的是 JSON 字符串；名称列表内的双引号需要转义。

```sbdsl
# 改变形状、交换维度、插入或移除大小为 1 的维度。
scratch-tensor-extension.[将 Tensor "A" 重塑(Reshape)为 "[4]" 赋值给 "flat"]
scratch-tensor-extension.[将 Tensor "A" 转置(Transpose) 维度 0 和 1 赋值给 "transposed"]
scratch-tensor-extension.[在 Tensor "A" 的维度 0 插入新维度(Unsqueeze) 赋值给 "with_axis"]
scratch-tensor-extension.[移除 Tensor "with_axis" 大小为 1 的维度(Squeeze) 赋值给 "squeezed"]
scratch-tensor-extension.[将 Tensor "A" 广播(Expand)至目标形状 "[3,2,2]" 赋值给 "expanded"]

# 拼接、堆叠、等分与截取。
scratch-tensor-extension.[拼接张量列表: "[\"A\",\"B\"]" 沿维度: 0 赋值给 "concatenated"]
scratch-tensor-extension.[堆叠张量列表: "[\"A\",\"B\"]" 在新维度: 0 赋值给 "stacked"]
scratch-tensor-extension.[将 Tensor "A" 沿维度 0 均分为 2 份 赋值前缀为 "part"]
scratch-tensor-extension.[截取 Tensor "A" 沿维度 0 从索引 0 到索引 1（不含） 赋值给 "first_row"]

# 条件张量以非零表示真、零表示假。
scratch-tensor-extension.[创建 Tensor 命名为 "mask" 并赋值 "[[1,0],[0,1]]" 需梯度: <false>]
scratch-tensor-extension.[条件选择(Where): 条件 "mask" 为真取 "A" 为假取 "B" 赋值给 "selected"]
scratch-tensor-extension.[将 Tensor "A" 裁剪数值范围至 -1 到 1 之间 赋值给 "clamped"]
```

| 算子 | 行为 |
| --- | --- |
| Reshape | 元素总数不变，最多一维可填 `-1` 让系统推导大小。 |
| Transpose | 交换两个维度。上例交换矩阵的行与列。 |
| Unsqueeze | 在指定位置插入一个大小为 `1` 的维度。 |
| Squeeze | 移除所有大小为 `1` 的维度，没有单独选择维度的输入槽。 |
| Expand | 按广播规则扩展形状，结果持有完整的数据副本；反向传播时把广播后的梯度求和还原到原形状。 |
| Concat | 沿已有维度拼接，维度数不变，其余维度大小须相同。 |
| Stack | 沿新维度堆叠，维度数加一，输入形状须相同。 |
| Split | 将指定维度等分，维度大小必须能被份数整除。上例输出名为 `part_0`、`part_1`。 |
| Slice | 截取 `[start, end)`，包含起点、不包含终点；梯度只回传到被截取的区域。 |
| Where | 按条件逐元素选取 A 或 B，支持广播；梯度回传到被选中的输入分支。 |
| Clamp | 把数值限制在给定范围内，原值超出范围的位置梯度为零。 |

## 维度聚合

聚合菜单有求和、平均值、最大值和最小值四项：

```sbdsl
scratch-tensor-extension.[对 Tensor "A" 沿维度 -1 进行 /'求和(Sum)'/ 保留维度: <false> 赋值给 "sum"]
scratch-tensor-extension.[对 Tensor "A" 沿维度 -1 进行 /'求平均值(Mean)'/ 保留维度: <false> 赋值给 "mean"]
scratch-tensor-extension.[对 Tensor "A" 沿维度 -1 进行 /'求最大值(Max)'/ 保留维度: <false> 赋值给 "max"]
scratch-tensor-extension.[对 Tensor "A" 沿维度 -1 进行 /'求最小值(Min)'/ 保留维度: <false> 赋值给 "min"]
```

维度支持负数索引，`-1` 表示最后一维，并不表示所有维度。保留维度为 `<false>` 时会移除被聚合的轴；设为 `<true>` 时保留该轴，大小变为 `1`。例如对 `[2, 3]` 的最后一维求和，两种输出形状分别为 `[2]` 和 `[2, 1]`。最大值或最小值有多个相同位置时，梯度只回传到第一个位置。

要聚合所有维度，明确填入 `"全部"`：

```sbdsl
scratch-tensor-extension.[对 Tensor "A" 沿维度 "全部" 进行 /'求和(Sum)'/ 保留维度: <false> 赋值给 "all_sum"]
[说 scratch-tensor-extension.(获取 Tensor "all_sum" 的 /'标量值'/)]
```

若 A 为 `[[1,2],[3,4]]`，上例输出 `10`。维度槽只接受整数、整数字符串或 `"全部"`；空文本、`"NaN"`、拼写错误和小数都会报错。底层 TypeScript API 用 `null` 表示全维聚合。

## LayerNorm 与 Dropout

LayerNorm 对最后指定数量的维度做归一化，输出形状不变：

```sbdsl
scratch-tensor-extension.[LayerNorm 输入: "A" 归一化最后 1 维 赋值给 "normalized"]
```

维度数量须为正整数，不能超过输入张量的维度数。当前实现只做归一化，不包含可训练的缩放参数 gamma 和偏置 beta。

Dropout 在计算图追踪开启时随机将元素置零，并按 `1 / (1 - p)` 缩放留下的元素：

```sbdsl
scratch-tensor-extension.[设置计算图追踪: <true>]
scratch-tensor-extension.[Dropout 输入: "A" 丢弃概率: 0.5 赋值给 "dropped"]

# 推理时关闭计算图追踪，Dropout 保持输入数值。
scratch-tensor-extension.[设置计算图追踪: <false>]
scratch-tensor-extension.[Dropout 输入: "A" 丢弃概率: 0.5 赋值给 "unchanged"]
```

概率 `0` 不丢弃，`1` 在追踪开启时将全部元素置零。计算图追踪是全局开关，也影响后续算子的求导和注意力中的 Dropout；继续训练前需要重新开启。

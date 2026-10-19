# Tensor 模块

本页介绍 Tensor 的创建、读取、删除、形状变换和模型参数导入导出。示例使用 ScratchDSL；扩展加载和完整训练工程见 [训练演示](demo.md)。

## 1. 示例怎么运行

先按演示文档创建工程，在配置中加载 `dist/scratch-tensor-extension.js`。调用前写上文件名对应的命名空间 `scratch-tensor-extension`；连字符可以直接写，不需要加引号。

下面是可放入 `demo/src/main.sbdsl` 的完整最小源码。后文短代码块是脚本体片段，应放进这个角色的绿旗事件中，替换或接在已有命令后；各节会说明需要先创建的张量。也可以将片段放入自制积木体中。

```sbdsl
@var result = ""
@var state = ""

@sprite "Tensor演示" {
    @init {}

    当绿旗被点击 {
        # Tensor 名称由扩展统一管理，与 Scratch 变量名不同。
        scratch-tensor-extension.[删除所有 Tensor]
        scratch-tensor-extension.[创建 Tensor 命名为 "A" 并赋值 "[[1,2,3],[4,5,6]]" 需梯度: <false>]
        [将 $result 设为 scratch-tensor-extension.(获取 Tensor "A" 的 /'数值'/)]
        [显示变量 $result]
    }
}
```

- 命令积木用方括号；报告积木用圆括号，并放进其他积木的输入槽，例如上面的变量赋值。
- Tensor 名称和 JSON 文本都是字符串，使用双引号及 JSON 转义。形状 `"[2,3]"` 是字符串输入，不是两层积木。
- 菜单用 `/'选项'/`，例如 `/'数值'/`；布尔输入用 `<true>` 或 `<false>`；普通行注释以 `#` 开始。
- `$result`、`$state` 是上面声明的 Scratch 变量。自制积木中的输入参数用 `@参数名`；完整写法见 [Linear 层示例](demo.md#linear-层与完整训练)。

## 2. 创建与读取

### 创建四类 Tensor

```sbdsl
scratch-tensor-extension.[创建 Tensor 命名为 "A" 并赋值 "[[1,2,3],[4,5,6]]" 需梯度: <true>]
scratch-tensor-extension.[创建形状为 "[2,3]" 的随机正态分布 Tensor 命名为 "R" 需梯度: <true>]
scratch-tensor-extension.[创建形状为 "[2,3]" 的全0 Tensor 命名为 "Z" 需梯度: <false>]
scratch-tensor-extension.[创建形状为 "[2,3]" 的全1 Tensor 命名为 "O" 需梯度: <false>]
```

数据输入可以是 JSON 数字或数组；形状输入是维度数组的 JSON 文本。两者都可由保存了 JSON 文本的 Scratch 变量提供。注册表按名称查找 Tensor，重新创建同名 Tensor 会替换该名称对应的注册项，分配新 ID，并清空旧梯度。运算积木的输出则不能覆盖可学习叶子参数，以免参数被中间结果替换。

这四种创建方式都产生叶子张量。`需梯度: <true>` 表示允许累积梯度，且会被优化器和权重导出识别为叶子参数；它不表示创建时已有梯度值，初始 `grad` 为 `null`。`<false>` 适合输入数据和标签，不参与参数更新。两类叶子张量都不会被反向传播的中间节点清理机制删除。

图形积木面板中，普通创建和随机正态创建的默认梯度选项为真，全 0 和全 1 为假。ScratchDSL 示例仍需明确填写每个输入槽。

### 读取属性

以下片段先创建单元素张量，然后把四种报告值依次赋给 `$result`：

```sbdsl
scratch-tensor-extension.[创建 Tensor 命名为 "Scalar" 并赋值 "3" 需梯度: <true>]
[将 $result 设为 scratch-tensor-extension.(获取 Tensor "Scalar" 的 /'数值'/)]
[将 $result 设为 scratch-tensor-extension.(获取 Tensor "Scalar" 的 /'标量值'/)]
[将 $result 设为 scratch-tensor-extension.(获取 Tensor "Scalar" 的 /'Shape'/)]
[将 $result 设为 scratch-tensor-extension.(获取 Tensor "Scalar" 的 /'梯度'/)]
```

| 菜单 | 返回类型与含义 |
| --- | --- |
| `/'数值'/` | 数据的 JSON **字符串**，例如 `"[[1,2],[3,4]]"`；上述标量返回字符串 `"3"`。 |
| `/'标量值'/` | 恰有一个元素时返回 **数字**，上述例子为 `3`。单元素数组也适用；多元素张量会报错并停止当前脚本。 |
| `/'Shape'/` | 形状的 JSON 字符串，上述标量为 `"[]"`。 |
| `/'梯度'/` | 已有梯度的 JSON 字符串；尚无梯度时返回字符串 `"null"`。 |

名称不存在时，以上报告均报错。只有张量存在但尚无梯度时，梯度报告才返回字符串 `"null"`。数值中的 `NaN`、`Infinity` 会被普通 JSON 报告转换为 `null`；标量报告保留原数字，不能用 JSON 报告判断计算结果是否有限。

## 3. 删除与反向传播后的清理

```sbdsl
scratch-tensor-extension.[删除 Tensor "R"]
scratch-tensor-extension.[删除所有 Tensor]
```

删除单个 Tensor 会移除名称和 ID 注册项；删除不存在的名称不产生效果。计算图若仍持有对象引用，该对象不会因此立即释放，之后也无法再通过被删除的名称查找它。

删除所有 Tensor 会清空注册表和优化器状态缓存，并把全局学习率缩放系数重置为 `1`。它也会删除模型参数，适合在绿旗事件开头重置运行状态。

下面是一个独立的梯度片段，逐元素乘法菜单写成 `/'*'/`：

```sbdsl
scratch-tensor-extension.[设置计算图追踪: <true>]
scratch-tensor-extension.[创建 Tensor 命名为 "W" 并赋值 "[[2]]" 需梯度: <true>]
scratch-tensor-extension.[创建 Tensor 命名为 "X" 并赋值 "[[3]]" 需梯度: <false>]
scratch-tensor-extension.[张量 "W" 与 "X" 进行 /'*'/ 运算 赋值给 "Y"]
[将 $result 设为 scratch-tensor-extension.(获取 Tensor "Y" 的 /'标量值'/)]
scratch-tensor-extension.[对 Tensor "Y" 进行反向传播]
[将 $result 设为 scratch-tensor-extension.(获取 Tensor "W" 的 /'梯度'/)]
scratch-tensor-extension.[清空所有参数梯度]
```

这里 `Y` 的标量值为 `6`，反向传播后 `W` 的梯度为字符串 `"[[3]]"`。清空参数梯度后，再读它会得到字符串 `"null"`。

反向传播通常从损失张量开始，也允许非标量目标；目标尚无梯度时，会以同形状的全 1 梯度作为起点。完成后，引擎清除本次遍历到的非叶子计算节点的运算记录和梯度，并移除这些中间张量的注册项，但保留目标张量本身。叶子参数和输入仍保留，参数梯度会累积，直到明确清空。

因此，应在反向传播前读取所需中间结果；下一轮训练重新执行前向计算。该清理只处理本次反向传播访问到的图，不会删除所有无关张量。更多说明见 [自动求导](autograd.md)。

## 4. 形状与结构变换

本节先准备以下张量，再执行各小节中的片段：

```sbdsl
scratch-tensor-extension.[创建 Tensor 命名为 "A" 并赋值 "[[1,2,3],[4,5,6]]" 需梯度: <false>]
scratch-tensor-extension.[创建 Tensor 命名为 "B" 并赋值 "[[7,8,9],[10,11,12]]" 需梯度: <false>]
scratch-tensor-extension.[创建 Tensor 命名为 "Row" 并赋值 "[[10,20,30]]" 需梯度: <false>]
scratch-tensor-extension.[创建 Tensor 命名为 "Mask" 并赋值 "[[1,0,1],[0,1,0]]" 需梯度: <false>]
```

### 重塑与转置

```sbdsl
scratch-tensor-extension.[将 Tensor "A" 重塑(Reshape)为 "[6]" 赋值给 "Flat"]
scratch-tensor-extension.[将 Tensor "A" 转置(Transpose) 维度 0 和 1 赋值给 "AT"]
```

`Flat` 的形状为 `[6]`，`AT` 的形状为 `[3,2]`。重塑前后元素总数必须一致；目标形状至多有一个 `-1`，用于自动推导该维长度。转置交换指定的两个轴，支持负数轴索引。

### 插入维度、移除维度与广播

```sbdsl
scratch-tensor-extension.[在 Tensor "A" 的维度 0 插入新维度(Unsqueeze) 赋值给 "U"]
scratch-tensor-extension.[移除 Tensor "U" 大小为 1 的维度(Squeeze) 赋值给 "S"]
scratch-tensor-extension.[将 Tensor "Row" 广播(Expand)至目标形状 "[2,3]" 赋值给 "E"]
```

结果形状分别为 `[1,2,3]`、`[2,3]`、`[2,3]`。插入轴支持负数索引；`-1` 或当前维数表示在末尾插入。移除维度积木会去掉所有长度为 `1` 的轴。广播要求原形状与目标形状兼容，例如把 `[1,3]` 扩展为 `[2,3]`。

### 拼接、堆叠与均分

```sbdsl
scratch-tensor-extension.[拼接张量列表: "[\"A\",\"B\"]" 沿维度: 0 赋值给 "C"]
scratch-tensor-extension.[堆叠张量列表: "[\"A\",\"B\"]" 在新维度: 0 赋值给 "Stack"]
scratch-tensor-extension.[将 Tensor "A" 沿维度 0 均分为 2 份 赋值前缀为 "part"]
```

- 拼接沿已有轴合并长度，其他轴应一致；这里 `C` 的形状为 `[4,3]`。
- 堆叠要求输入形状一致，并新增一个轴；这里 `Stack` 的形状为 `[2,2,3]`。
- 均分要求指定轴长度能被份数整除；这里产生 `part_0`、`part_1`，形状均为 `[1,3]`，编号从 `0` 开始。

拼接和堆叠接收的是张量名称列表的 JSON 字符串，里面的双引号需要转义。列表不能为空，每个名称都必须存在；名称写错会报错，不会跳过。允许只传入一个张量。

### 切片与条件选择

```sbdsl
scratch-tensor-extension.[截取 Tensor "A" 沿维度 0 从索引 0 到索引 1（不含） 赋值给 "FirstRow"]
scratch-tensor-extension.[条件选择(Where): 条件 "Mask" 为真取 "A" 为假取 "B" 赋值给 "Chosen"]
```

切片包含起始位置、不包含结束位置，所以上例取第一行，得到 `[[1,2,3]]`。索引范围应满足 `0 <= start < end <= 轴长度`。

条件选择把非零掩码视为真、零视为假，结果为 `[[1,8,3],[10,5,12]]`；三个输入支持广播。在追踪计算图且输入需要梯度时，切片将梯度传回被选区域，条件选择只向选中的数据分支传递梯度，不对掩码求导。

### 裁剪数值范围

```sbdsl
scratch-tensor-extension.[将 Tensor "A" 裁剪数值范围至 -1 到 1 之间 赋值给 "Clamped"]
```

小于下限的值替换为下限，大于上限的值替换为上限。这是裁剪张量数值，不是优化器的梯度裁剪。

## 5. 模型参数导出与恢复

权重导出报告返回 StateDict JSON 字符串；积木本身不读写文件。下面用 `$state` 保存文本，清空注册表后再恢复：

```sbdsl
scratch-tensor-extension.[删除所有 Tensor]
scratch-tensor-extension.[创建 Tensor 命名为 "W" 并赋值 "[[2]]" 需梯度: <true>]
[将 $state 设为 scratch-tensor-extension.(获取当前模型权重字典 JSON 字符串)]
scratch-tensor-extension.[删除所有 Tensor]
scratch-tensor-extension.[从 StateDict 文本 $state 导入并恢复模型权重]
[将 $result 设为 scratch-tensor-extension.(获取 Tensor "W" 的 /'标量值'/)]
```

恢复后 `$result` 为数字 `2`。StateDict 只导出 `requiresGrad = true` 的叶子参数，不包含普通输入、非叶子中间结果、梯度、计算图或优化器缓存。

格式如下，`id` 的实际数值由运行时分配：

```json
{
  "framework": "ScratchTensor",
  "tensors": [
    {
      "name": "W",
      "id": 101,
      "shape": [1, 1],
      "requires_grad": true,
      "data": [[2]]
    }
  ]
}
```

导入按名称匹配，不使用文件里的 ID 来寻找参数：

- 已有同名张量：检查形状一致后覆盖数据、清空梯度，保留原对象、ID、`requiresGrad` 标记和优化器缓存。
- 没有同名张量：按记录创建，使用声明的形状，并分配新的运行时 ID；不会占用文件中记录的 ID。

导入前会检查整个文件，并准备全部数据。任何一项失败，都不会改写已有参数、清空梯度或创建一部分张量。`name` 必须是非空字符串且不能重复，名称保留原样；`requires_grad` 必须是布尔值。`shape` 必须由非负整数构成，并与 `data` 的嵌套结构一致，也支持 `[0,3]` 等含空维度的形状。数据只接受有限数字，拒绝 `null`、布尔值、数字字符串以及超出 float32 有限范围的值。

`framework` 和 `id` 可以省略；提供时，前者必须为 `"ScratchTensor"`，后者必须是安全整数。未知字段会被忽略。导出时遇到 `NaN` 或 `Infinity` 也会报错，并指出对应参数，避免写成 `null` 后丢失数值含义。

导入本身不会重置已有优化器缓存。上例先删除所有 Tensor，再恢复参数，适合开始一次新的训练或推理运行。若保存到文件，需要另外保存 `$state` 中的文本；加载时将完整文本传入导入积木。

## 6. 错误与边界

- 张量名称不存在、形状不兼容、轴或切片范围非法、JSON 格式错误时，积木统一报错，错误信息带有积木 opcode 和具体原因。
- 在 TurboWarp 中，错误会停止当前脚本线程，包括调用自制积木的上层脚本；其他脚本继续运行。角色会显示错误气泡，控制台也可查看错误。
- 失败的运算不会伪装成成功。已有同名输出可能仍是上一次的结果，因此不要在另一个脚本里把它当作本轮结果。
- 删除不存在的名称仍可重复执行；读取现存张量尚无的梯度仍返回字符串 `"null"`。
- StateDict 必须含 `tensors` 数组。空模型可用 `{"framework":"ScratchTensor","tensors":[]}`；普通空对象 `{}` 无效。

这些行为对应当前 [扩展积木实现](../src/scratch-extension.ts)、[Tensor 注册表](../src/tensor.ts) 和 [序列化实现](../src/serialization.ts)。

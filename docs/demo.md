# 用 ScratchDSL 编写训练演示

我们推荐使用 ScratchDSL 编写和构建项目，文档中的积木和示例也采用 ScratchDSL 写法。它与 Scratch 积木基本一一对应；如果您更习惯图形化编辑器，也可以参照示例搭建相应积木。

本页展示带可选偏置的 Linear 层、固定数据的训练循环，以及一个乘法和梯度验证。按 ScratchDSL 方式运行时，需要先安装 ScratchDSL，并按仓库 README 构建出 `dist/scratch-tensor-extension.js`。

## 创建工程

在 ScratchTensor 仓库根目录执行：

```powershell
sbdsl init demo
```

将生成的 `demo/sbdsl.buildsb` 替换为：

```sbdsl-build
entry: "./src/main.sbdsl"
output: "./dist/training.sb3"
extensions: ["../dist/scratch-tensor-extension.js"]
```

配置路径相对于 `demo/sbdsl.buildsb` 所在目录，因此扩展路径中的 `..` 指向仓库根目录。源码中的默认命名空间是去掉末尾 `.js` 的文件名，写成 `scratch-tensor-extension.` 即可，不需要引号。文件名中的点也保留，例如 `my.extension.js` 对应 `my.extension`；这些点不表示多级命名空间。

想缩短前缀时，可把 `extensions` 里的条目写成 `"../dist/scratch-tensor-extension.js" as "tensor"`，再把调用前缀统一改为 `tensor.`。别名只改变源码命名空间，运行时 ID 仍是 `ScratchTensor`；下文继续使用默认文件名。

模块文档中的短代码块是脚本体片段：先加载上述扩展，再把片段放入角色的绿旗事件或自制积木体中，使用前创建所需张量。报告积木放在输入槽中使用，例如赋给 Scratch 变量。下文的训练与基础验证则是完整源码，可以直接保存并编译。

菜单写成 `/'选项'/`，例如 `/'@'/` 表示矩阵乘法，`/'*'/` 表示逐元素乘法；选项本身是除号时写成 `/'/'/`。菜单内必须使用单引号，普通字符串使用双引号，注释以 `#` 开始。比较大小直接写 `<`、`>`，例如 `<$step < 250>`、`<$loss > 0>`。配置和 `@init` 中的字段可按行分隔；普通积木命令在括号内换行时也不需要逗号。

## Linear 层与完整训练

把下面完整代码保存为 `demo/src/main.sbdsl`，替换初始化生成的源码。`@init` 引用的图标由 `sbdsl init` 创建。

模型为 `Y = X @ W + b`。输入为 `[[1],[2],[3]]`，标签为 `[[3],[5],[7]]`；权重和偏置从零开始，使用学习率 `0.05`、无动量、无权重衰减的 SGD，训练 250 步。目标关系是 `y = 2x + 1`，对应理想参数 `W = [[2]]`、`b = [1]`；有限步训练的实际数值应通过运行后的变量读取，不要求浮点结果逐位相等。

```sbdsl
# 拟合 y = 2x + 1；Tensor 名称由扩展统一管理。
@var step = 0
@var initial_loss = 0
@var loss = 0
@var final_loss = 0
@var learned_W = ""
@var learned_b = ""
@var prediction = ""

定义自制积木 Linear层 输入名称: @in_x 权重名称: @weight_w 使用偏置: <@use_bias> 偏置名称: @bias_b 输出名称: @out_name {
    如果 <@use_bias> 那么 {
        scratch-tensor-extension.[张量 @in_x 与 @weight_w 进行 /'@'/ 运算 赋值给 "linear.tmp"]
        scratch-tensor-extension.[张量 "linear.tmp" 与 @bias_b 进行 /'+'/ 运算 赋值给 @out_name]
    } 否则 {
        scratch-tensor-extension.[张量 @in_x 与 @weight_w 进行 /'@'/ 运算 赋值给 @out_name]
    }
}

@sprite "训练演示" {
    @init {
        x: 0
        y: 0
        costumes: [{name: "ScratchDSL", path: "../assets/sbdsl.svg"}]
    }

    当绿旗被点击 {
        scratch-tensor-extension.[删除所有 Tensor]
        scratch-tensor-extension.[设置计算图追踪: <true>]
        scratch-tensor-extension.[设置优化器全局学习率缩放系数为: 1]
        scratch-tensor-extension.[创建 Tensor 命名为 "X" 并赋值 "[[1],[2],[3]]" 需梯度: <false>]
        scratch-tensor-extension.[创建 Tensor 命名为 "label" 并赋值 "[[3],[5],[7]]" 需梯度: <false>]
        scratch-tensor-extension.[创建 Tensor 命名为 "W" 并赋值 "[[0]]" 需梯度: <true>]
        scratch-tensor-extension.[创建 Tensor 命名为 "b" 并赋值 "[0]" 需梯度: <true>]
        [将 $step 设为 0]

        重复执行 250 次 {
            [Linear层 输入名称: "X" 权重名称: "W" 使用偏置: <true> 偏置名称: "b" 输出名称: "Y"]
            scratch-tensor-extension.[MSELoss 预测: "Y" 真实标签: "label" 赋值给 "Loss"]
            # 记录本轮更新参数之前的损失。
            [将 $loss 设为 scratch-tensor-extension.(获取 Tensor "Loss" 的 /'标量值'/)]
            如果 <$step = 0> 那么 {
                [将 $initial_loss 设为 $loss]
            }
            scratch-tensor-extension.[对 Tensor "Loss" 进行反向传播]
            scratch-tensor-extension.[执行 SGD 优化器 学习率: 0.05 动量: 0 权重衰减: 0]
            scratch-tensor-extension.[清空所有参数梯度]
            [将 $step 增加 1]
        }

        # 用更新后的参数重新前向计算，得到最终损失和预测。
        scratch-tensor-extension.[设置计算图追踪: <false>]
        [Linear层 输入名称: "X" 权重名称: "W" 使用偏置: <true> 偏置名称: "b" 输出名称: "Y"]
        scratch-tensor-extension.[MSELoss 预测: "Y" 真实标签: "label" 赋值给 "FinalLoss"]
        [将 $final_loss 设为 scratch-tensor-extension.(获取 Tensor "FinalLoss" 的 /'标量值'/)]
        [将 $learned_W 设为 scratch-tensor-extension.(获取 Tensor "W" 的 /'数值'/)]
        [将 $learned_b 设为 scratch-tensor-extension.(获取 Tensor "b" 的 /'数值'/)]
        [将 $prediction 设为 scratch-tensor-extension.(获取 Tensor "Y" 的 /'数值'/)]
        [显示变量 $initial_loss]
        [显示变量 $final_loss]
        [显示变量 $learned_W]
        [显示变量 $learned_b]
        [显示变量 $prediction]
        [说 (连接 "训练完成，最终 MSE: " 和 $final_loss)]
    }
}
```

在仓库根目录检查并构建：

```powershell
sbdsl check demo/sbdsl.buildsb
sbdsl build demo/sbdsl.buildsb
```

在支持自定义 JavaScript 扩展的 TurboWarp 中打开 `demo/dist/training.sb3`，点击绿旗。查看 `step`、`initial_loss`、`final_loss`、`learned_W`、`learned_b` 和 `prediction`；其中参数与预测是 JSON 字符串，损失是数值。生成的 SB3 已内嵌本地扩展源码。

使用上面的数据和参数生成 SB3，在无图形界面的 TurboWarp VM 中执行 250 步后，`initial_loss` 约为 `27.666666`，`final_loss` 约为 `0.00000344`；`W` 约为 `[[2.002155]]`，`b` 约为 `[0.995101]`，预测约为 `[[2.997256],[4.999411],[7.001566]]`。这些是本次运行的参考值。

### 语法与执行顺序

- `/'@'/` 是矩阵乘法菜单，`/'+'/` 是加法菜单，`/'*'/` 是逐元素乘法菜单。菜单内的符号不是 ScratchDSL 的算术运算符。Tensor 数据和形状均以 JSON **字符串**传入。
- `Linear层` 是命令型自制积木，通过 `@out_name` 指定输出 Tensor 名称；普通参数写 `@参数`，布尔参数写 `<@参数>`。示例顺序调用这一层，复用临时名称 `linear.tmp`；有偏置时才创建这份中间结果。
- `$loss` 记录每轮 SGD 更新前的损失。循环结束后再前向计算一次，`$final_loss` 才对应最后一次更新后的参数。
- `backward(Loss)` 会清理参与这次反向传播的非叶子中间张量，如 `linear.tmp` 和 `Y`；叶子 `X`、`label`、`W`、`b` 以及目标 `Loss` 保留。`Loss` 的数值仍可查询，但其计算图已释放，再次反向传播前应重新构建前向图。
- 参数梯度会累积，所以每轮 SGD 后清空梯度。最后一次预测关闭图追踪，并保留结果供查看；重新点击绿旗时，开头的“删除所有 Tensor”会清空张量和优化器状态。

## 最小乘法与梯度验证

将下面完整代码另存为 `demo/src/basic.sbdsl`。它复用同一份构建配置和扩展：

```sbdsl
@var product = 0
@var gradient = ""
@var cleared_gradient = ""

@sprite "基础验证" {
    @init {
        costumes: [{name: "ScratchDSL", path: "../assets/sbdsl.svg"}]
    }
    当绿旗被点击 {
        scratch-tensor-extension.[删除所有 Tensor]
        scratch-tensor-extension.[设置计算图追踪: <true>]
        scratch-tensor-extension.[创建 Tensor 命名为 "W" 并赋值 "[[2]]" 需梯度: <true>]
        scratch-tensor-extension.[创建 Tensor 命名为 "X" 并赋值 "[[3]]" 需梯度: <false>]
        scratch-tensor-extension.[张量 "W" 与 "X" 进行 /'*'/ 运算 赋值给 "Y"]
        [将 $product 设为 scratch-tensor-extension.(获取 Tensor "Y" 的 /'标量值'/)]
        scratch-tensor-extension.[对 Tensor "Y" 进行反向传播]
        [将 $gradient 设为 scratch-tensor-extension.(获取 Tensor "W" 的 /'梯度'/)]
        scratch-tensor-extension.[清空所有参数梯度]
        [将 $cleared_gradient 设为 scratch-tensor-extension.(获取 Tensor "W" 的 /'梯度'/)]
        [显示变量 $product]
        [显示变量 $gradient]
        [显示变量 $cleared_gradient]
        [说 "基础验证完成"]
    }
}
```

直接指定源码会覆盖配置中的入口；用 `-o` 为基础验证指定单独输出：

```powershell
sbdsl check demo/src/basic.sbdsl
sbdsl build demo/src/basic.sbdsl -o demo/dist/basic.sb3
```

点击基础验证项目的绿旗后，检查以下结果。这些参考值已通过上述 ScratchDSL 编译生成的 SB3，在 TurboWarp VM 中执行确认：

| 变量 | 参考值 | 类型 |
| --- | --- | --- |
| `product` | `6` | 数值 |
| `gradient` | `[[3]]` | JSON 字符串 |
| `cleared_gradient` | `null` | 字符串，不是空值 |

这里验证的是 `Y = W * X`：`W = [[2]]` 需要梯度，`X = [[3]]` 不需要梯度，因此 `dY/dW = [[3]]`。训练示例与基础验证分别运行，各自从空张量注册表开始。

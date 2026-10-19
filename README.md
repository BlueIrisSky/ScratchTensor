# ScratchTensor

[![TurboWarp](https://img.shields.io/badge/Made%20with-TurboWarp-FFAB19?style=flat-square)](https://turbowarp.org/)
[![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat-square&logo=javascript&logoColor=black)](https://developer.mozilla.org/en-US/docs/Web/JavaScript)
[![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?style=flat-square&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![npm](https://img.shields.io/badge/npm-CB3837?style=flat-square&logo=npm&logoColor=white)](https://www.npmjs.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)

一款面向 Scratch 图形化编程的轻量级深度学习张量引擎，提供类 PyTorch 的动态计算图与自动求导能力，并打包为 Scratch 扩展积木。

我们推荐使用 ScratchDSL 编写和构建项目。本仓库文档也使用 ScratchDSL 描述积木和示例。ScratchDSL 与 Scratch 积木基本一一对应；如果您更习惯图形化编辑器，也可以参照文档搭建相应积木。

## 核心特性

- **自动求导**：追踪前向运算，为需要梯度的叶子参数累积梯度；反向传播后释放本次计算图的非叶子中间张量，保留叶子与目标张量。
- **Scratch 扩展**：将张量创建、算子、损失函数、优化器等封装为 Scratch 积木，可在 TurboWarp 等支持自定义 JavaScript 扩展的编辑器中使用。
- **PyTorch 风格 API**：`sum` / `mean` / `max` / `min` 支持 `dim` 与 `keepdim`，`matmul` 支持 1D 点积、2D 矩阵乘、3D 及以上批量矩阵乘与 batch 广播，`where` 支持广播，`scaled_dot_product_attention` 支持 causal mask 与 dropout。
- **模型序列化**：通过 JSON 文本导出 / 导入 StateDict，方便在 Scratch 变量或外部文件中保存权重。
- **内存管理**：提供 `删除 Tensor` / `删除所有 Tensor` 积木，避免长期运行或循环训练时的内存泄漏。

## 快速开始

在仓库根目录执行：

```bash
npm ci
npm run build
npm test
```

构建产物位于 `dist/`：

- `scratch-tensor-extension.js`：Scratch 3 扩展包。

## 在 Scratch 中使用

1. 将仓库中的 `dist/scratch-tensor-extension.js` 作为自定义扩展加载到 TurboWarp。
2. 扩展加载后，积木栏会出现 **ScratchTensor** 分类，包含：
   - 创建 / 查询 / 删除 Tensor
   - 矩阵乘法与四则运算
   - 激活函数与损失函数
   - 维度操作（Reshape、Unsqueeze、Squeeze、Expand、Concat、Stack、Slice、Where、Clamp）
   - 归约操作（Sum、Mean、Max、Min）
   - 归一化与正则化（LayerNorm、Dropout）
   - 注意力机制（Scaled Dot Product Attention）
   - 优化器（SGD、Adam、AdamW）
   - 模型权重导出 / 导入

3. 用 ScratchDSL 编写并编译演示项目，见[Linear 层与完整训练演示](docs/demo.md)。文档提供固定数据的 250 步训练、可选偏置的自制积木，以及乘法和梯度基础验证，所有代码均为完整 ScratchDSL 源码。

   在仓库根目录执行 `sbdsl init demo`，按文档替换生成的配置和源码，再运行：

   ```powershell
   sbdsl check demo/sbdsl.buildsb
   sbdsl build demo/sbdsl.buildsb
   ```

   生成的 SB3 内嵌本地扩展源码，可在 TurboWarp 中打开并点击绿旗运行。文档同时说明损失读取时机与反向传播后的张量清理行为。

## 核心概念

### Tensor 与 requiresGrad

- 所有 Tensor 在系统中通过**全局名称**注册，积木层只传递字符串名称。
- 通过创建积木创建的张量均为叶子张量（`isLeaf = true`）：其中 `需梯度: <true>` 表示**可学习叶子参数**，反向传播时会在其 `.grad` 中累积梯度并由优化器更新；`需梯度: <false>` 表示**常数/数据叶子张量**（如样本与标签），可在训练循环外创建并跨轮次复用。
- 运算产出的张量标记为非叶子节点（`isLeaf = false`）；参与本次反向传播的中间张量会被清理，目标张量保留。

### Autograd

- 执行 `scratch-tensor-extension.[设置计算图追踪: <true>]` 后，涉及可求导输入的前向运算会记录计算图。
- 执行 `scratch-tensor-extension.[对 Tensor "Loss" 进行反向传播]`，从目标张量向输入回传梯度。
- 反向传播完成后，本次计算图被释放，相关的非叶子中间张量从注册表移除；再次求导前需要重新执行前向计算。

### 优化器

- 支持 SGD（含 Momentum、L2 权重衰减）、Adam、AdamW。
- 优化器会自动遍历所有 `requiresGrad = true` 的叶子参数并完成单步更新。
- `scratch-tensor-extension.[对所有参数执行梯度裁剪 最大阈值: 1.0]` 应在反向传播之后、优化器之前调用。

## 开发测试

```bash
npm test
npm run build
npm run test:scratchdsl
```

`test:scratchdsl` 需要 PATH 中有 `sbdsl`，也可以通过 `SCRATCHDSL_BIN` 指定编译器。它会编译文档示例，并在 TurboWarp VM 中检查完整训练、梯度和复杂边界。CI 自动构建固定提交的编译器；本地运行和测试报告见[集成测试说明](test/scratchdsl/README.md)。

## 目录

- `src/`：张量引擎；`ops/` 放算子，`extension/` 放积木定义和运行时错误处理。
- `test/`：单元测试、数值对照测试和 ScratchDSL 集成测试；可编辑的 `.sbdsl` 用例在 `test/scratchdsl/fixtures/`。
- `docs/`：积木说明和训练示例。
- `scripts/`：发布包生成与校验。
- `dist/`：编译结果、测试报告和发布包，均不提交到 Git。

## 文档索引

所有积木示例统一使用 ScratchDSL，扩展调用带 `scratch-tensor-extension` 命名空间，菜单写成 `/'选项'/`。模块页中的短代码块是脚本体片段；工程配置和完整程序见[演示教程](docs/demo.md)。

- [docs/tensor.md](docs/tensor.md)：Tensor 创建、查询、结构变形、序列化
- [docs/autograd.md](docs/autograd.md)：计算图追踪、反向传播、梯度清空
- [docs/fun.md](docs/fun.md)：算子、损失函数、激活函数、维度聚合
- [docs/optim.md](docs/optim.md)：SGD / Adam / AdamW 优化器、梯度裁剪、学习率缩放
- [docs/demo.md](docs/demo.md)：ScratchDSL 工程配置、Linear 层封装、完整训练与基础验证

## 叠甲

亲爱的朋友，这个仓库是我闲暇时间的消遣，是用agent跑出来的玩具，由于能力和精力，大量代码没法人工审核，如果存在一些很幼稚的疏漏也请包容，存在问题也请轻骂，谢谢
export function getExtensionInfo() {
  const colors = {
    tensor: { color1: '#4C97FF', color2: '#3373CC' },
    autograd: { color1: '#5CB85C', color2: '#449D44' },
    operator: { color1: '#F0AD4E', color2: '#ba9158' },
    activation: { color1: '#b1a368', color2: '#4b708a' },
    loss: { color1: '#D9534F', color2: '#C9302C' },
    shape: { color1: '#9B59B6', color2: '#8E44AD' },
    optim: { color1: '#5BC0DE', color2: '#31B0D5' },
    serialization: { color1: '#95A5A6', color2: '#7F8C8D' }
  };

  return {
    id: 'ScratchTensor',
    name: 'ScratchTensor',
    color1: '#4C97FF',
    color2: '#3373CC',
    blocks: [
      // ========== Tensor 创建 ==========
      {
        opcode: 'createTensorBlock',
        blockType: 'command',
        text: '创建 Tensor 命名为 [NAME] 并赋值 [DATA] 需梯度: [REQ_GRAD]',
        ...colors.tensor,
        arguments: {
          NAME: { type: 'string', defaultValue: 'w1' },
          DATA: { type: 'string', defaultValue: '[[1.0, 2.0], [3.0, 4.0]]' },
          REQ_GRAD: { type: 'Boolean', defaultValue: true }
        }
      },
      {
        opcode: 'createZerosTensorBlock',
        blockType: 'command',
        text: '创建形状为 [SHAPE] 的全0 Tensor 命名为 [NAME] 需梯度: [REQ_GRAD]',
        ...colors.tensor,
        arguments: {
          SHAPE: { type: 'string', defaultValue: '[2, 3]' },
          NAME: { type: 'string', defaultValue: 'zeros_1' },
          REQ_GRAD: { type: 'Boolean', defaultValue: false }
        }
      },
      {
        opcode: 'createOnesTensorBlock',
        blockType: 'command',
        text: '创建形状为 [SHAPE] 的全1 Tensor 命名为 [NAME] 需梯度: [REQ_GRAD]',
        ...colors.tensor,
        arguments: {
          SHAPE: { type: 'string', defaultValue: '[2, 3]' },
          NAME: { type: 'string', defaultValue: 'ones_1' },
          REQ_GRAD: { type: 'Boolean', defaultValue: false }
        }
      },
      {
        opcode: 'createRandomNormalTensorBlock',
        blockType: 'command',
        text: '创建形状为 [SHAPE] 的随机正态分布 Tensor 命名为 [NAME] 需梯度: [REQ_GRAD]',
        ...colors.tensor,
        arguments: {
          SHAPE: { type: 'string', defaultValue: '[10, 5]' },
          NAME: { type: 'string', defaultValue: 'randn_1' },
          REQ_GRAD: { type: 'Boolean', defaultValue: true }
        }
      },
      // ========== Tensor 查询 ==========
      {
        opcode: 'getTensorPropertyBlock',
        blockType: 'reporter',
        text: '获取 Tensor [NAME] 的 [PROPERTY]',
        ...colors.tensor,
        arguments: {
          NAME: { type: 'string', defaultValue: 'w1' },
          PROPERTY: { type: 'string', defaultValue: '数值', menu: 'tensorPropertyMenu' }
        }
      },
      // ========== Tensor 删除 ==========
      {
        opcode: 'deleteTensorBlock',
        blockType: 'command',
        text: '删除 Tensor [NAME]',
        ...colors.tensor,
        arguments: {
          NAME: { type: 'string', defaultValue: 'temp' }
        }
      },
      {
        opcode: 'deleteAllTensorsBlock',
        blockType: 'command',
        text: '删除所有 Tensor',
        ...colors.tensor
      },
      // ========== Autograd 计算图控制 ==========
      {
        opcode: 'setGraphTrackingBlock',
        blockType: 'command',
        text: '设置计算图追踪: [ENABLED]',
        ...colors.autograd,
        arguments: {
          ENABLED: { type: 'Boolean', defaultValue: true }
        }
      },
      {
        opcode: 'backwardBlock',
        blockType: 'command',
        text: '对 Tensor [TENSOR] 进行反向传播',
        ...colors.autograd,
        arguments: {
          TENSOR: { type: 'string', defaultValue: 'Loss' }
        }
      },
      {
        opcode: 'clearGradBlock',
        blockType: 'command',
        text: '清空所有参数梯度',
        ...colors.autograd
      },
      // ========== 基础算子 ==========
      {
        opcode: 'binaryOpBlock',
        blockType: 'command',
        text: '张量 [A] 与 [B] 进行 [OP] 运算 赋值给 [OUT]',
        ...colors.operator,
        arguments: {
          A: { type: 'string', defaultValue: 'A' },
          B: { type: 'string', defaultValue: 'B' },
          OP: { type: 'string', defaultValue: '+', menu: 'opMenu' },
          OUT: { type: 'string', defaultValue: 'C' }
        }
      },
      // ========== 激活函数 ==========
      {
        opcode: 'activationBlock',
        blockType: 'command',
        text: '对 Tensor [X] 应用 [ACT] 激活函数 赋值给 [OUT]',
        ...colors.activation,
        arguments: {
          X: { type: 'string', defaultValue: 'H' },
          ACT: { type: 'string', defaultValue: 'ReLU', menu: 'activationMenu' },
          OUT: { type: 'string', defaultValue: 'A' }
        }
      },
      {
        opcode: 'softmaxBlock',
        blockType: 'command',
        text: 'Softmax 输入: [X] 沿维度: [DIM] 赋值给 [OUT]',
        ...colors.activation,
        arguments: {
          X: { type: 'string', defaultValue: 'H' },
          DIM: { type: 'number', defaultValue: -1 },
          OUT: { type: 'string', defaultValue: 'A' }
        }
      },
      {
        opcode: 'scaledDotProductAttentionBlock',
        blockType: 'command',
        text: 'Scaled Dot Product Attention Q: [Q] K: [K] V: [V] 因果掩码: [CAUSAL] 丢弃概率: [DROPOUT] 缩放（0=自动）: [SCALE] 赋值给 [OUT]',
        ...colors.activation,
        arguments: {
          Q: { type: 'string', defaultValue: 'Q' },
          K: { type: 'string', defaultValue: 'K' },
          V: { type: 'string', defaultValue: 'V' },
          CAUSAL: { type: 'Boolean', defaultValue: false },
          DROPOUT: { type: 'number', defaultValue: 0 },
          SCALE: { type: 'number', defaultValue: 0 },
          OUT: { type: 'string', defaultValue: 'Out' }
        }
      },
      //损失函数
      {
        opcode: 'mseLossBlock',
        blockType: 'command',
        text: 'MSELoss 预测: [PRED] 真实标签: [TARGET] 赋值给 [OUT]',
        ...colors.loss,
        arguments: {
          PRED: { type: 'string', defaultValue: 'Y' },
          TARGET: { type: 'string', defaultValue: 'label' },
          OUT: { type: 'string', defaultValue: 'Loss' }
        }
      },
      {
        opcode: 'crossEntropyLossBlock',
        blockType: 'command',
        text: 'CrossEntropyLoss Logits: [LOGITS] 标签: [LABELS] 赋值给 [OUT]',
        ...colors.loss,
        arguments: {
          LOGITS: { type: 'string', defaultValue: 'Logits' },
          LABELS: { type: 'string', defaultValue: 'Labels' },
          OUT: { type: 'string', defaultValue: 'Loss' }
        }
      },
      {
        opcode: 'bceLossBlock',
        blockType: 'command',
        text: 'BCELoss 预测: [PRED] 标签: [TARGET] 赋值给 [OUT]',
        ...colors.loss,
        arguments: {
          PRED: { type: 'string', defaultValue: 'Y' },
          TARGET: { type: 'string', defaultValue: 'label' },
          OUT: { type: 'string', defaultValue: 'Loss' }
        }
      },
      {
        opcode: 'l1LossBlock',
        blockType: 'command',
        text: 'L1Loss 预测: [PRED] 标签: [TARGET] 赋值给 [OUT]',
        ...colors.loss,
        arguments: {
          PRED: { type: 'string', defaultValue: 'Y' },
          TARGET: { type: 'string', defaultValue: 'label' },
          OUT: { type: 'string', defaultValue: 'Loss' }
        }
      },
      //形状操作
      {
        opcode: 'reshapeBlock',
        blockType: 'command',
        text: '将 Tensor [X] 重塑(Reshape)为 [SHAPE] 赋值给 [OUT]',
        ...colors.shape,
        arguments: {
          X: { type: 'string', defaultValue: 'A' },
          SHAPE: { type: 'string', defaultValue: '[6]' },
          OUT: { type: 'string', defaultValue: 'B' }
        }
      },
      {
        opcode: 'unsqueezeBlock',
        blockType: 'command',
        text: '在 Tensor [X] 的维度 [DIM] 插入新维度(Unsqueeze) 赋值给 [OUT]',
        ...colors.shape,
        arguments: {
          X: { type: 'string', defaultValue: 'A' },
          DIM: { type: 'number', defaultValue: 0 },
          OUT: { type: 'string', defaultValue: 'B' }
        }
      },
      {
        opcode: 'squeezeBlock',
        blockType: 'command',
        text: '移除 Tensor [X] 大小为 1 的维度(Squeeze) 赋值给 [OUT]',
        ...colors.shape,
        arguments: {
          X: { type: 'string', defaultValue: 'A' },
          OUT: { type: 'string', defaultValue: 'B' }
        }
      },
      {
        opcode: 'expandBlock',
        blockType: 'command',
        text: '将 Tensor [X] 广播(Expand)至目标形状 [SHAPE] 赋值给 [OUT]',
        ...colors.shape,
        arguments: {
          X: { type: 'string', defaultValue: 'A' },
          SHAPE: { type: 'string', defaultValue: '[3, 4]' },
          OUT: { type: 'string', defaultValue: 'B' }
        }
      },
      {
        opcode: 'concatBlock',
        blockType: 'command',
        text: '拼接张量列表: [NAMES] 沿维度: [DIM] 赋值给 [OUT]',
        ...colors.shape,
        arguments: {
          NAMES: { type: 'string', defaultValue: '["A", "B"]' },
          DIM: { type: 'number', defaultValue: 0 },
          OUT: { type: 'string', defaultValue: 'C' }
        }
      },
      {
        opcode: 'splitBlock',
        blockType: 'command',
        text: '将 Tensor [X] 沿维度 [DIM] 均分为 [N] 份 赋值前缀为 [PREFIX]',
        ...colors.shape,
        arguments: {
          X: { type: 'string', defaultValue: 'A' },
          DIM: { type: 'number', defaultValue: 0 },
          N: { type: 'number', defaultValue: 2 },
          PREFIX: { type: 'string', defaultValue: 'part' }
        }
      },
      {
        opcode: 'stackBlock',
        blockType: 'command',
        text: '堆叠张量列表: [NAMES] 在新维度: [DIM] 赋值给 [OUT]',
        ...colors.shape,
        arguments: {
          NAMES: { type: 'string', defaultValue: '["A", "B"]' },
          DIM: { type: 'number', defaultValue: 0 },
          OUT: { type: 'string', defaultValue: 'C' }
        }
      },
      {
        opcode: 'sliceBlock',
        blockType: 'command',
        text: '截取 Tensor [X] 沿维度 [DIM] 从索引 [START] 到索引 [END]（不含） 赋值给 [OUT]',
        ...colors.shape,
        arguments: {
          X: { type: 'string', defaultValue: 'A' },
          DIM: { type: 'number', defaultValue: 0 },
          START: { type: 'number', defaultValue: 0 },
          END: { type: 'number', defaultValue: 2 },
          OUT: { type: 'string', defaultValue: 'B' }
        }
      },
      {
        opcode: 'whereBlock',
        blockType: 'command',
        text: '条件选择(Where): 条件 [COND] 为真取 [A] 为假取 [B] 赋值给 [OUT]',
        ...colors.shape,
        arguments: {
          COND: { type: 'string', defaultValue: 'mask' },
          A: { type: 'string', defaultValue: 'A' },
          B: { type: 'string', defaultValue: 'B' },
          OUT: { type: 'string', defaultValue: 'C' }
        }
      },
      {
        opcode: 'clampBlock',
        blockType: 'command',
        text: '将 Tensor [X] 裁剪数值范围至 [MIN] 到 [MAX] 之间 赋值给 [OUT]',
        ...colors.shape,
        arguments: {
          X: { type: 'string', defaultValue: 'A' },
          MIN: { type: 'number', defaultValue: -1 },
          MAX: { type: 'number', defaultValue: 1 },
          OUT: { type: 'string', defaultValue: 'B' }
        }
      },
      //逐元素数学函数
      {
        opcode: 'powBlock',
        blockType: 'command',
        text: '对 Tensor [X] 计算幂 [P] 赋值给 [OUT]',
        ...colors.operator,
        arguments: {
          X: { type: 'string', defaultValue: 'A' },
          P: { type: 'number', defaultValue: 2 },
          OUT: { type: 'string', defaultValue: 'B' }
        }
      },
      {
        opcode: 'unaryMathBlock',
        blockType: 'command',
        text: '对 Tensor [X] 计算 [OP] 赋值给 [OUT]',
        ...colors.operator,
        arguments: {
          X: { type: 'string', defaultValue: 'A' },
          OP: { type: 'string', defaultValue: 'Exp', menu: 'unaryMathMenu' },
          OUT: { type: 'string', defaultValue: 'B' }
        }
      },
      //归约与矩阵操作
      {
        opcode: 'reductionBlock',
        blockType: 'command',
        text: '对 Tensor [X] 沿维度 [DIM] 进行 [OP] 保留维度: [KEEPDIM] 赋值给 [OUT]',
        ...colors.shape,
        arguments: {
          X: { type: 'string', defaultValue: 'A' },
          DIM: { type: 'string', defaultValue: '-1' },
          OP: { type: 'string', defaultValue: '求和(Sum)', menu: 'reductionMenu' },
          KEEPDIM: { type: 'Boolean', defaultValue: false },
          OUT: { type: 'string', defaultValue: 'B' }
        }
      },
      {
        opcode: 'transposeBlock',
        blockType: 'command',
        text: '将 Tensor [X] 转置(Transpose) 维度 [DIM1] 和 [DIM2] 赋值给 [OUT]',
        ...colors.shape,
        arguments: {
          X: { type: 'string', defaultValue: 'A' },
          DIM1: { type: 'number', defaultValue: 0 },
          DIM2: { type: 'number', defaultValue: 1 },
          OUT: { type: 'string', defaultValue: 'B' }
        }
      },
      //正则化
      {
        opcode: 'layerNormBlock',
        blockType: 'command',
        text: 'LayerNorm 输入: [X] 归一化最后 [N] 维 赋值给 [OUT]',
        ...colors.shape,
        arguments: {
          X: { type: 'string', defaultValue: 'A' },
          N: { type: 'number', defaultValue: 1 },
          OUT: { type: 'string', defaultValue: 'B' }
        }
      },
      {
        opcode: 'dropoutBlock',
        blockType: 'command',
        text: 'Dropout 输入: [X] 丢弃概率: [P] 赋值给 [OUT]',
        ...colors.shape,
        arguments: {
          X: { type: 'string', defaultValue: 'A' },
          P: { type: 'number', defaultValue: 0.5 },
          OUT: { type: 'string', defaultValue: 'B' }
        }
      },
      //优化器
      {
        opcode: 'sgdFullBlock',
        blockType: 'command',
        text: '执行 SGD 优化器 学习率: [LR] 动量: [MOMENTUM] 权重衰减: [WD]',
        ...colors.optim,
        arguments: {
          LR: { type: 'number', defaultValue: 0.01 },
          MOMENTUM: { type: 'number', defaultValue: 0.9 },
          WD: { type: 'number', defaultValue: 0 }
        }
      },
      {
        opcode: 'adamBlock',
        blockType: 'command',
        text: '执行 Adam 优化器 学习率: [LR] Beta1: [B1] Beta2: [B2] 权重衰减: [WD]',
        ...colors.optim,
        arguments: {
          LR: { type: 'number', defaultValue: 0.001 },
          B1: { type: 'number', defaultValue: 0.9 },
          B2: { type: 'number', defaultValue: 0.999 },
          WD: { type: 'number', defaultValue: 0 }
        }
      },
      {
        opcode: 'adamWBlock',
        blockType: 'command',
        text: '执行 AdamW 优化器 学习率: [LR] Beta1: [B1] Beta2: [B2] 权重衰减: [WD]',
        ...colors.optim,
        arguments: {
          LR: { type: 'number', defaultValue: 0.001 },
          B1: { type: 'number', defaultValue: 0.9 },
          B2: { type: 'number', defaultValue: 0.999 },
          WD: { type: 'number', defaultValue: 0.01 }
        }
      },
      {
        opcode: 'clipGradNormBlock',
        blockType: 'command',
        text: '对所有参数执行梯度裁剪 最大阈值: [MAX_NORM]',
        ...colors.optim,
        arguments: {
          MAX_NORM: { type: 'number', defaultValue: 1.0 }
        }
      },
      {
        opcode: 'setLrScaleBlock',
        blockType: 'command',
        text: '设置优化器全局学习率缩放系数为: [SCALE]',
        ...colors.optim,
        arguments: {
          SCALE: { type: 'number', defaultValue: 1.0 }
        }
      },
      //序列化
      {
        opcode: 'exportStateDictBlock',
        blockType: 'reporter',
        text: '获取当前模型权重字典 JSON 字符串',
        ...colors.serialization
      },
      {
        opcode: 'loadStateDictBlock',
        blockType: 'command',
        text: '从 StateDict 文本 [JSON_STR] 导入并恢复模型权重',
        ...colors.serialization,
        arguments: {
          JSON_STR: { type: 'string', defaultValue: '{"framework":"ScratchTensor","tensors":[]}' }
        }
      }
    ],
    menus: {
      opMenu: {
        items: ['+', '-', '*', '/', '@'],
        acceptReporters: false
      },
      activationMenu: {
        items: ['ReLU', 'GELU', 'Sigmoid', 'Tanh'],
        acceptReporters: false
      },
      unaryMathMenu: {
        items: ['Exp', 'Log', 'Sqrt', 'Abs'],
        acceptReporters: false
      },
      tensorPropertyMenu: {
        items: ['数值', '标量值', 'Shape', '梯度'],
        acceptReporters: false
      },
      reductionMenu: {
        items: ['求和(Sum)', '求平均值(Mean)', '求最大值(Max)', '求最小值(Min)'],
        acceptReporters: false
      }
    }
  };
}

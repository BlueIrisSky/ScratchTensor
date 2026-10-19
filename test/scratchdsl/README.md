# ScratchDSL 集成测试

这组测试使用真实的 ScratchDSL 编译器和本次构建的扩展包。文档片段检查编译与积木映射，完整演示和边界工程编译出的 SB3 交给 TurboWarp VM 执行，检查数值、梯度和状态。运行器不渲染舞台；资源文件只检查归档内容和哈希。

CI 使用 [ScratchDSL](https://github.com/BlueIrisSky/ScratchDSL) 的 `62c45299acf126eb6d148ed5f14bcd1322c9e4d6` 提交，以及 [TurboWarp scratch-vm](https://github.com/TurboWarp/scratch-vm) 的 `e9c6e122bff25a13b57059e4fd32544e7d9560ca` 提交。两者固定版本，升级时需要显式修改工作流或依赖锁文件并重新验证。

CI 使用 Node.js 22 / npm 10。npm 12 安装时需加 `--allow-remote=all`，用于下载固定提交的 TurboWarp 依赖。

在仓库根目录安装依赖并构建扩展，再指定编译器运行：

```sh
npm ci
npm run build
SCRATCHDSL_BIN=/absolute/path/to/sbdsl npm run test:scratchdsl
```

PowerShell 中先设置 `$env:SCRATCHDSL_BIN = 'C:\path\to\sbdsl.exe'`，然后运行同样的 npm 命令。如果编译器已在 PATH 中，可以省略这个环境变量。找不到编译器、编译失败或执行超时都会使测试失败，不会跳过。

`docs.check.mjs` 每次直接读取根目录 README 和 `docs` 中的 Markdown，编译全部 `sbdsl`、`sbdsl-build` 代码块及行内的完整扩展调用。文档中的扩展别名也会实际加载验证。短片段只补角色、绿旗事件和报告积木的输入位置，不改写积木调用；训练与基础验证使用文档中的完整源码。

配置示例会放进独立的仓库形状目录，保留 `../dist/scratch-tensor-extension.js` 等相对路径。每个示例使用独立目录，避免其他示例生成的文件掩盖缺失的入口或资源。

文档测试要求六个现有页面都存在，各页的示例数量不能少于当前基线，完整训练与基础程序也必须保留。还会从扩展实际注册的元信息读取公开 opcode，检查编译后的文档是否全部覆盖。增加积木后需要补充对应示例；删减文档示例时需要同时审查测试基线。

生成的源码、配置、SB3、编译日志及 JSON 报告保存在 `dist/scratchdsl-tests/`，该目录已被 Git 排除。`documentation.json` 记录每个示例所在文档和行号，失败时可以用留下的配置单独重现。GitHub Actions 会保留这些文件供下载。

`fixtures/*.sbdsl` 保存运行测试的完整源码，包括计算图生命周期、参数恢复、广播与自制积木、错误隔离和数值边界。常规文件可以作为 `.buildsb` 的入口独立编译；`common.sbdsl` 是导入测试使用的模块，`boundary-duplicate-source.sbdsl` 需要将扩展别名配置为 `tensor`。`compile-*.sbdsl` 用来测试非法语法或配置，其中部分文件预期编译失败。测试只调整命名空间别名等必要配置，不在 JavaScript 中拼接运行程序。文档测试仍直接读取 Markdown，不以这些 fixture 替代文档内容。

`runtime.mjs` 默认在遇到扩展错误时让测试失败。`allowExtensionErrors` 用于检查预期失败，记录错误后停止本次工程；`continueAfterExtensionErrors` 专门验证线程隔离，只在错误来自扩展且该线程已经被扩展完整终止时继续 VM 调度。它不会忽略其他 VM 错误或重试仍活动的失败积木，并始终受超时限制。隔离用例检查错误气泡、opcode、失败线程后续语句未执行，以及其他线程完成。

# 可复现性与检查范围

本公开包含完整独立V2核心、独立Temporal R1核心、可移植CLI、冻结配置、测试和总体结果。科学核心与本地冻结源码字节一致，`.gitattributes`禁止自动换行转换，以维持SHA256。

|对象|证据|
|---|---|
|冻结V2源代码|`configs/frozen_v2_source.json`；源指纹e47d39fc6f58c8d009af9531f197a66f3d0ce8e28b5b2e37983de1747c121261|
|R1源代码/配置/测试|`configs/repair_freeze.json`；组合指纹c92de5185a049f051df4f8aebd93baa4fcd62b051f9f6de4271642de99951d78|
|固定开发/评价列表|`configs/split.json`，开发列表在R1修订前已固定|
|继承与检测器不变|`configs/frozen_dependency_manifest.json`记录原V2依赖字节哈希|
|精确环境|`requirements.lock.txt`；Python3.12.14|
|公开测试|`results/public_tests.xml`：94 passed；22条MNE/NumPy deprecation warnings|
|新增时序测试|`results/temporal_r1_development/synthetic_tests.xml`：31 passed|
|10晚真实回放|`results/temporal_r1_development/validation_summary.json`：事件/证据/V1V3精确一致|
|公开CLI真实烟雾验证|`results/portable_cli_validation.json`：1晚真实EDF与封存R1事件、预测、primary/full指标一致|
|公共内容|`PUBLIC_MANIFEST.json`，仅清单中的文件属于交付候选|

`scripts/audit.py`核查冻结科学核心及公共清单；`scripts/recheck_results.py`从混淆矩阵独立复算56组总体指标，不读取临床行。测试验证数值、边界与实现合同，不代表事件级临床有效性。

机器路径绑定的私有`replay.py`没有公开。`scripts/predict.py`与`evaluate.py`是新可移植适配层；科学核心相同，编排文件不作逐字节相同声明。新CLI经过额外真实EDF烟雾测试才交付。

重新运行需要本地合法持有HMC v1.1。预测不读取评分；评价只读取指定运行记录的TXT/EDF+，两格式一致且完整覆盖的30秒进入full，首末可靠非Wake范围进入primary。查看README中的精确命令。

主结果V3、V0–V4消融与后续R1开发结果分别冻结。R1没有最终141条/151条结果。不得把只包含总体矩阵的公共目录当作可恢复的完整原运行缓存。

公开JUnit仅移除了机器hostname属性，测试名称、数量、结果和耗时保留；原始与导出哈希在`results/junit_export_provenance.json`。冻结协议中的原始测试XML哈希指向本地原始文件，不能与去除hostname后的公开副本混同。

本地临床证据不在公共清单中；逐记录差异、事件来源、真实波形与详细trace仅在私人研究工程保留。未上传任何文件，当前为本地发布候选分支。

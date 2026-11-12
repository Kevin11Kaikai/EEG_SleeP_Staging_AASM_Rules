# 十二项改进：实现、证据与局限

全部工作包有可运行分支、测试/实际状态与真实PSG产物。**没有宣称12项均获临床验证。**
表内‘Used’表示进入固定实验管线，不等于每条规则都曾获足够证据触发。

| Improvement | Implemented | Validated | Used in experiment | Observed impact | Limitations |
|---|---|---|---|---|---|
| WP01 Blink | 已实现（部分为proxy） | 单测+真实PSG；无临床金标准 | 是；可用性逐录音记录 | 增强blink候选60311；独立Accuracy贡献未识别；W.2全量触发0 | 双EOG形态、真实边界、局部共同伪迹排除；HMC无眼睑真值 |
| WP02 LAMF | 已实现（部分为proxy） | 单测+真实PSG；无临床金标准 | 是；可用性逐录音记录 | 增强lamf候选309375；独立Accuracy贡献未识别 | 逐样点与1s/0.5s局部RMS对照；20µV及theta1%是工程约定 |
| WP03 REM | 已实现（部分为proxy） | 单测+真实PSG；无临床金标准 | 是；可用性逐录音记录 | 增强rem候选83574；独立Accuracy贡献未识别 | 真实双EOG YASA+反相检查；零事件与失效分开 |
| WP04 Low EMG | 已实现（部分为proxy） | 单测+真实PSG；无临床金标准 | 是；可用性逐录音记录 | 增强low_emg候选214295；独立Accuracy贡献未识别 | Q20/Q80对照稳定30s背景Q10个体floor；未经临床标定 |
| WP05 K-complex | 已实现（部分为proxy） | 单测+真实PSG；无临床金标准 | 是；可用性逐录音记录 | 增强kcomplex候选264693；独立Accuracy贡献未识别 | F4双相/过零/最小持续/1s arousal关联；仍会与普通慢波混淆 |
| WP06 Arousal | 已实现（部分为proxy） | 单测+真实PSG；无临床金标准 | 是；可用性逐录音记录 | 增强arousal候选13932；独立Accuracy贡献未识别 | 固定前基线、alpha/theta/beta变化、睡眠代理与REM-EMG条件；非专家确认背景 |
| WP07 N1 completeness | 已实现（部分为proxy） | 单测+真实PSG；无临床金标准 | 是；可用性逐录音记录 | 增强vertex候选307357；独立Accuracy贡献未识别 | alpha衰减+非alpha SEM/Vertex/slowing；Cz缺失，非alpha slowing参考常不可用 |
| WP08 Transition/inheritance | 已实现（部分为proxy） | 单测+真实PSG；无临床金标准 | 是；可用性逐录音记录 | 见规则触发、配对错误与包级消融；未做逐WP因果分离 | 继承/拒绝/最大链2条件继承；锚点、链长、不兼容证据均保存 |
| WP09 Wake | 已实现（部分为proxy） | 单测+真实PSG；无临床金标准 | 是；可用性逐录音记录 | 见规则触发、配对错误与包级消融；未做逐WP因果分离 | W.1/W.2/W.3独立且不相加；Blink/W3覆盖条件不保证在真实数据触发 |
| WP10 N2 transitions | 已实现（部分为proxy） | 单测+真实PSG；无临床金标准 | 是；可用性逐录音记录 | 见规则触发、配对错误与包级消融；未做逐WP因果分离 | onset/continuation/N3→N2/termination/re-entry显式状态；阴性使用显式候选代理 |
| WP11 REM termination | 已实现（部分为proxy） | 单测+真实PSG；无临床金标准 | 是；可用性逐录音记录 | 见规则触发、配对错误与包级消融；未做逐WP因果分离 | 终止目标分支W/N3/N1/N2，先终止再onset；个体tone尚未临床验证 |
| WP12 Major body movement | 已实现（部分为proxy） | 单测+真实PSG；无临床金标准 | 是；可用性逐录音记录 | 增强mbm候选882；独立Accuracy贡献未识别；MBM分期规则全量触发0 | 与QC独立，>15s门槛和确定性离线回填；抽查仅短运动候选 |

物理单位、导联与版本一致性是强制约束，未计为第13项独立收益。
已发现并在开发阶段记录的负例：共同伪迹Blink、快速起始SEM、慢波/滤波混淆Vertex。
全部原始计数/拒绝/失效、覆盖率、规则触发和每个预测trace均保留，不以单元测试代替临床验证。

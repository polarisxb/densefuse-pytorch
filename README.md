# DenseFuse — Modern PyTorch Reproduction

本仓库是 [Hui Li 的 PyTorch 实现](https://github.com/hli1221/densefuse-pytorch)
的科研复现与兼容性迁移分支。先保存原始行为，再完成推理、训练和评价复现，
随后才进行工程重构、性能优化及独立算法实验。

## 1. Original paper and repository

H. Li, X. J. Wu, **DenseFuse: A Fusion Approach to Infrared and Visible Images**,
IEEE Transactions on Image Processing, vol. 28, no. 5, pp. 2614–2623, May 2019.

- [IEEE Xplore](https://ieeexplore.ieee.org/document/8580578)
- [arXiv](https://arxiv.org/abs/1804.08361)
- [Original TensorFlow implementation](https://github.com/hli1221/imagefusion_densefuse)
- Original snapshot: `upstream-original`, commit `4394b63e9295db1c6b7a5c3664551c90f0605f2b`.

## 2. Project boundaries and branches

| Branch / tag | Purpose |
| --- | --- |
| `master` | 已验证成果的稳定分支 |
| `feat/modern-pytorch` | 当前兼容性迁移工作 |
| `feat/reproduction` | 后续从已验证迁移版本分出，用于训练和论文实验；目前不预先创建 |
| `upstream-original` | 原始源码快照，保持不移动 |

当前保留原模型文件及 checkpoint 参数名。架构、职责和阶段验收见
[architecture](docs/architecture.md)。训练和评价尚未验证，详见
[known differences](docs/known-differences.md)。

```text
net.py / fusion_strategy.py       原始网络与融合实现
pytorch_msssim/                   原始训练 loss
utils.py                         图像兼容层及历史训练工具
test_image.py                    单对图像推理、设备、CLI、运行记录
train_densefuse.py / args_fusion.py  原始训练入口与配置
tests/                           输入像素、权重、特征和输出回归
docs/                            架构、差异、实验状态
models/                          随仓库提供的灰度/RGB权重
images/                          随仓库提供的样例ZIP
outputs/                         运行产生的PNG和JSON（Git忽略）
```

## 3. Environment

选定服务器基线：Linux x86_64、glibc 2.28、NVIDIA A10、驱动550.54.14，
Python 3.11.16、torch 2.7.1+cu118、torchvision 0.22.1+cu118。
用户已在该服务器完成回归测试（17项，1项按预期跳过）、第一对图像的TF32对照，
以及全部21对灰度图的TF32关闭后CPU/GPU比较；详见[服务器实测记录](docs/reproduction-status.md#server-a10-evidence)。

新建环境，在仓库根目录执行：

```bash
conda env create -f environment.yml
conda activate densefuse
python -m pip install -r requirements-cu118.txt
python -m pip install -r requirements.txt
python -m pip check
```

如果已按本项目配置安装好 `densefuse` 环境，直接激活并检查依赖即可。
Conda只负责Python环境，torch使用[官方CUDA wheels](https://pytorch.org/get-started/previous-versions/#v271)。
两个requirements文件分别安装，避免把科学计算包交给PyTorch专用索引解析。
这里锁定的是直接依赖，并非完整的传递依赖锁文件。环境验证后可将
`python -m pip freeze` 保存到自己的实验记录中。

CPU机器可使用匹配的官方CPU wheels，再安装相同通用依赖：

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

不用额外安装第三方 `pytorch-msssim`；训练loss来自仓库自身。

## 4. Migration changes

- 移除失效且未使用的 `load_lua` 导入；以Pillow替换当前uint8路径的SciPy图像API。
- 保留0–255量纲、RGB转换路径、nearest训练resize和输出uint8截断。
- 推理统一CPU/CUDA设备，严格加载权重，显式CUDA不可用时报告错误。
- 修正灰度样例路径为PNG，默认路径与仓库位置关联，可从其他工作目录启动。
- 新增单对CLI、尺寸检查、PNG输出和JSON运行记录。
- 网络、融合、训练脚本、参数配置和MS-SSIM源码不变。

JSON记录输入/权重/输出哈希、代码哈希、Git状态、实际设备、版本和后端标志。
它是运行记录，不是论文评价报告；checkpoint训练参数未知，记录为null。

## 5. Dataset preparation

在仓库根目录解压已有样例，无需下载COCO来做预训练推理：

```bash
python -m zipfile -e images/IV_images.zip images
python -m zipfile -e images/test-RGB.zip images
```

灰度样例是 `images/IV_images/IR1.png` 和 `VIS1.png`，第一对为360×270；
ZIP共有21对。RGB样例在 `images/test-RGB/`，扩展名为.jpg，内容实际为PNG。
保持文件原样，勿重新编码。

自己的输入需已配准、同尺寸、每边至少2像素；当前支持L/RGB转换路径。
推理不自动配准、resize、crop或归一化。

## 6. Pretrained model inference

灰度第一对，CPU：

```bash
python -B test_image.py --device cpu
```

学校服务器，在已分配GPU的作业中：

```bash
conda activate densefuse
python -B test_image.py --device cuda --output outputs/a10-gray-1.png
```

保留调度器设置的 `CUDA_VISIBLE_DEVICES`。`cuda:0` 指当前作业可见设备中的第0张，
不一定是整台服务器的物理第0张。

自定义图像与权重：

```bash
python -B test_image.py --ir images/IV_images/IR1.png --vis images/IV_images/VIS1.png --model models/densefuse_gray.model --device cuda --output outputs/custom-gray.png
```

RGB样例：

```bash
python -B test_image.py --mode RGB --device cuda --output outputs/rgb-1.png
```

默认策略始终为原始代码中的特征算术平均。`--mode`决定输入通道和默认权重。
`--device auto`在配置允许且CUDA可用时用CUDA，否则CPU；可显式指定设备。
完整参数见 `python -B test_image.py --help`。

每次CLI运行写入一个PNG和相邻的 `.png.json`。默认文件名中的`1e2`只是保留的
历史标签，不能据此证明权重训练lambda。重复使用同一输出路径会覆盖此前结果，
不同实验应使用不同文件名。

## 7. Training

尚未完成现代环境训练复现。原脚本意图使用MSCOCO 2014灰度图做自编码重建，
默认先取文件名排序前40000张，再shuffle，batch=4、epoch=4、lr=1e-4。
`args.dataset`仍是占位路径。没有IR/VIS配对训练。

当前loss为MSE加100倍的本地归一化MS-SSIM损失。原始MS-SSIM聚合、训练设备迁移、
RGB reshape及resume存在已记录问题。不要因为推理可运行就把原训练脚本当成已验证方案。
下一阶段先固定数据清单、loss/梯度基线及optimizer行为，再做完整训练。

## 8. Evaluation

论文指标、评测集合对应关系及原始TensorFlow数值对照尚未完成。
未来评测需固定输入清单、指标实现/参数、像素范围和保存规则，并保留逐图结果。
目前的运行JSON不包含论文指标，也不宣称论文复现成功。

## 9. Verification and reproduction results

```bash
python -B -m unittest discover -s tests -v
```

测试使用标准库unittest，无需额外测试依赖。包括灰度/RGB像素、nearest resize、
两份权重严格加载、编码/融合/解码参考张量、CLI与运行记录，以及无效输入检查。
参考张量来自原始源码在现代CPU运行时的执行，见[fixture说明](tests/fixtures/README.md)。

验证环境、实际结果和未验证项目见[reproduction status](docs/reproduction-status.md)。

第一对灰度图的A10实测：默认GPU与CPU相比，1221/97200个保存像素相差1灰度级；
关闭TF32后只剩1个像素相差1灰度级，最大原始浮点差为0.000244140625。

随后21对灰度图全部完成TF32关闭后的CPU/GPU对照：共6,199,556个像素，最大原始
浮点差0.000396728515625，按像素加权的平均绝对差约4.17165e-5。保存图像中仅40个
像素不同（0.0006452%），均相差1灰度级；7对保存图像完全一致。
详见[批量记录](docs/reproduction-status.md#all-21-gray-pairs-with-tf32-disabled)。
这是当前样例集的数值观察基线，不是论文质量指标或事先设定的验收阈值。
后续数值对齐采用显式关闭TF32的独立对照，当前CLI默认行为不变。

已对用户提供的第2、10、21对IR/VIS/Fused对照图做视觉抽检：未见明显错图、
全黑/全白或大幅几何错位；第10、21对存在明显对比度减弱。该观察保留为后续
参考实现对照的问题，不通过调整亮度、融合公式或输出层来改善外观。
这仅是三对缩放对照图的定性核验，尚未与论文参考融合图比较。

## 10. Known differences and limitations

见[完整差异记录](docs/known-differences.md)。尤其不要把平均融合改成求和、
删除末层ReLU、将输入除以255或替换MS-SSIM而仍称其为纯兼容性改动。
原始未启用attention模块的字符串`is`会产生SyntaxWarning，当前保留以维持源码快照。

## 11. Acknowledgements and licensing

感谢Hui Li、Xiao-Jun Wu及原实现作者。论文和原始仓库归属保持不变。
当前上游快照没有LICENSE文件，本分支没有给继承代码追加新的许可证。
需要独立再发布或重新授权时应先明确原作者许可。

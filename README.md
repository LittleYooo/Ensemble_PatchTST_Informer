# Usage

### 1. 修改配置
在 `config.py` 中修改相关配置，如数据路径、模型路径等。

### 2. 脚本执行
在项目根目录下有 `exp.sh` 脚本，可以直接运行该脚本进行训练和测试。

```shell
sh exp.sh
```
之后，日志会保存在 `logs` 文件夹中，结果会保存在 `results` 文件夹中。

---

如果要自定义训练和测试，可以使用以下命令。

#### 训练模型
须指定数据集名称。

可以指定训练选择器的数据集路径，默认是 `./dataset/5138`。
```shell
python run.py --train --dset s0.3548_m907

python run.py --train --dset CAV-H

python run.py --train --dset HTV2
```

#### 测试模型
须指定数据集名称和数据集大小。

```shell
python run.py --test --dset s0.3548_m907 --dset_size 5138

python run.py --test --dset HTV2 --dset_size 10276

python run.py --test --dset CAV-H --dset_size 20552
```
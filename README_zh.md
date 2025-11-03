# AGDA-DDI
**AGDA-DDI：基于属性增强图结构的药物相互作用预测方法**
## 项目介绍
本项目旨在预测药物之间的相互作用（DDI）。项目提供了多种模型架构，通过 5 折交叉验证和多模型比较，并结合药物特征（如化学相似性、靶点相似性等）和图结构信息，模型能够有效预测药物之间的相互作用类型。

## 文件结构
- `main.py`：主程序文件，用于训练和测试模型。
- `merge_csv.py`：用于合并多个 CSV 文件的结果并计算最大值。
- `model.py`：定义了多种模型架构，包括基础模型、GAT、GCN 等变体。
- `run_script.py`：批量运行实验脚本。
- `train_on_fold.py`：实现 5 折交叉验证的训练和测试。
- `utils.py`：提供数据处理和图预处理工具。
- `data/`：存放数据文件，包括药物相互作用数据和特征矩阵。

## 数据准备
数据文件应放在 `data/` 文件夹中，具体包括以下文件：
- `ddi_class_65.csv`：药物相互作用数据，包含药物对和标签。
- `chem_Jacarrd_sim.csv`：药物分子结构相似性矩阵。
- `target_Jacarrd_sim.csv`：药物靶点相似性矩阵。
- `enzyme_Jacarrd_sim.csv`：药物酶相似性矩阵。
- `pathway_Jacarrd_sim.csv`：药物通路相似性矩阵。


## 环境依赖
本项目基于 Python 3.8+ 开发，依赖以下主要库：
- torch==2.4.0+cu124
- torch-geometric==2.5.3
- numpy==1.24.1
- pandas==2.0.3
- scikit-learn==1.3.2
- matplotlib==3.7.5


## 安装方法
1. 创建并激活虚拟环境（推荐使用 Conda ）：
   ```bash
   conda create -n ddi_env python=3.8
   conda activate ddi_env
   ```
2. 安装依赖库：
   ```bash
   pip install -r requirements.txt
   ```


## 运行方法
### 单次训练与测试
运行主程序，指定模型类型和超参数：
   ```bash
   python Deng_main.py --model 1
   ```
#### 参数说明

| 参数名          | 说明                             | 默认值  |
|-----------------|----------------------------------|---------|
| `--hidden_dim`   | 隐藏层的维度                     | 512     |
| `--out_dim`      | 输出层的维度                     | 256     |
| `--dropout`      | Dropout率                        | 0.3     |
| `--rels`         | 关系的数量                       | 65      |
| `--n_epochs`     | 训练的轮数                       | 400     |
| `--batch_size`   | 批量大小                         | 1024    |
| `--threshold`    | 边的阈值                         | 0.4     |
| `--model`        | 模型类型                         | 1       |
**以下是可用的模型类型及其对应的命令行参数值：**
* Model (默认)：--model 1
* Model_avg：--model avg
* Model_sum：--model sum
* Model_wo_feat：--model feat
* Model_wo_diffgraph：--model dg
* Model_wo_relgraph：--model rg
* Model_wo_att：--model att
* Model_GAT：--model gat
* Model_GCN：--model gcn

**其他参数（如隐藏层维度、训练轮数等）可在 main.py 中通过命令行参数设置。**

### 5 折交叉验证
运行 5 折交叉验证脚本：
   ```bash
   python Deng_train_on_fold.py
   ```
* 该脚本会自动对数据进行 5 折分割，并分别训练和测试模型，最终保存每个折的结果。


### 合并结果
运行以下脚本，将多个 CSV 文件的结果合并并计算最大值：
   ```bash
   python merge_csv.py
   ```
**输出结果:**
* 训练和测试结果将保存到 results.csv 文件中。
* 每个类别的 AUCPR 和 F1 分数将保存到 class_metrics_fold_k.csv 文件中。
* 交叉验证结果将保存到 cross_validation.csv 文件中。
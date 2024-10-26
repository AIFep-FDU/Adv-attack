# 针对视觉模型的对抗攻击
```bash
git clone https://github.com/AIFep-FDU/Adv-attack.git
```

### 环境设置:

```bash
pip install -r requirements.txt
```

### 模型权重下载:
https://codalab.lisn.upsaclay.fr/my/datasets/download/c0c23b23-dd43-457a-865d-49195be63ff0

### 任务说明
1. 修改data_loader.py和evaluation.py中的路径
2. 填写my_adv_attack.py中的缺失代码
3. 执行evaluation.py，并对评估结果进行截图

#### 补充
可以尝试替换其他的[攻击方式](https://github.com/Harry24k/adversarial-attacks-pytorch)，选择效果最好的。

### 提交说明
提交完整的my_adv_attack.py和评估结果的截图
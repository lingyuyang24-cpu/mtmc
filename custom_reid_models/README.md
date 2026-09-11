# 自定义 ReID 模型

将每个自定义 ReID 模型放在本目录下的独立文件夹中。此功能适用于命令行入口，Web 工作台目前固定使用 TransReID。

目录示例：

```text
custom_reid_models/
  my_reid_model/
    adapter.py
    config.json
    weights.pth
```

运行方式：

```bash
python demo.py --videos videos/init/view-HC2.mp4 videos/init/view-HC3.mp4 --reid-backend custom --custom-reid-name my_reid_model
```

`adapter.py` 必须提供以下工厂函数或类之一：

```text
create_extractor(...)
create_reid_extractor(...)
build_extractor(...)
ReIDExtractor(...)
CustomReIDExtractor(...)
ReIDAdapter(...)
```

返回对象必须提供 `extract` 方法，也可提供以下属性：

```python
extract(images, camera_id=None, view_id=None)  # 返回 [N, feature_dim] 特征矩阵
feature_dim                                   # 可选，建议提供
uses_camera_id                                # 可选，默认 False
```

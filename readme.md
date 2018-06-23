- 用于模型训练的代码
- run_doom.py
    - 命令行参数、示例
    
      - --env upfloor(用于训练上楼策略)
      - --ent 0.01(policy entrophy) 
      - --lr 1e-4 
      - --policy lstm 
      - --save_name upfloor512 (optional; default = --env) 
    
    - 代码中的一些参数
      - 继续训练: True 首次训练新模型请改为False
      - 并行游戏实例数: 16
      - 输入地图路径、输出模型(.dat)路径 请自行更改
    
    - 测试环境
      - Tensorflow 1.1.0 with GPU
      - Python 3.6.4
      
    - 其他
      - 使用的训练地图
      - 验证相关请见viz2018/doorandkey

- anogenerator.py
    - 手动控制、自动存储图片+标注。对窗体点击鼠标，换下一张地图(编号+1)；空格键开门。
    - 需要加载修改过的地图和标记txt。这里提供1000张用于生成标记的地图（尽可能多地包含门和钥匙，但并不保证每一张），均为之前生成标记时没有用过的。下载链接 https://cloud.tsinghua.edu.cn/f/67d2016cea5c4fdc93c5/?dl=1
    - 运行前请设置地图路径、地图起止编号、生成图片及label路径、开始编号等。
    - 物体代号(地图标记txt文件中的，也是label中的):
      - 'reddoor' 0
      - 'bluedoor' 1
      - 'yellowdoor' 2
      - 'dooropen' 3
      - 'teleport' 4
      - 'exit' 5
      - 'redkey' 6
      - 'bluekey' 7
      - 'yellowkey' 8
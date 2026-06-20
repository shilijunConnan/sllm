baseline使用文件
- model_base
- runner_base
- engine_base

无kv cache
| 配置                                   | 加速后端口 | 测试结果                                |                                                                                           
|--------------------------------------|-------|-------------------------------------|
| input_100, output_100, batch_size_4  | mps   | batch generation 58.330702 seconds  |
| input_100, output_100, batch_size_32 | mps   | batch generation 596.267757 seconds |


kvcache v1配置
tensor cat模拟kvcache的增加

kv cache v1
| 配置                                   | 加速后端口 | 测试结果                                |                                                                                           
|--------------------------------------|-------|-------------------------------------|
| input_100, output_100, batch_size_4  | mps   | batch generation 6.566058 seconds  |
| input_100, output_100, batch_size_32 | mps   | batch generation 26.549186 seconds |


from opencc import OpenCC

# 创建繁体转简体转换器
cc = OpenCC('t2s')  # t2s 表示 Traditional Chinese 转 Simplified Chinese

# 输入繁体文本
traditional_text = "這是繁體字範例"

# 转换为简体
simplified_text = cc.convert(traditional_text)
print(simplified_text)
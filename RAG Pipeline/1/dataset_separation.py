import pandas as pd
import re

# 读取 CSV 文件
input_csv = r".\gpt4_response_v2_by1.csv"  # 输入文件路径
output_csv = r".\separated_gpt4_response_v2_by1.csv"  # 输出文件路径

# 读取 CSV 文件
try:
    df = pd.read_csv(input_csv)
except FileNotFoundError:
    print(f"Error: File {input_csv} not found. Please check the path.")
    exit(1)
except pd.errors.EmptyDataError:
    print("Error: The CSV file is empty.")
    exit(1)

# 确保 CSV 文件包含 "text" 列
assert "text" in df.columns, "CSV must have 'text' column"

# 定义正则表达式来提取用户输入和模型响应
pattern = r"\[INST\](.*?)\[/INST\](.*?)</s>"

# 初始化两个列表来存储提取的内容
user_inputs = []
responses = []

# 遍历每一行并提取内容
for index, row in df.iterrows():
    text = row["text"]
    match = re.search(pattern, text, re.DOTALL)  # 使用 re.DOTALL 匹配多行
    if match:
        user_input = match.group(1).strip()  # 提取用户输入
        response = match.group(2).strip()  # 提取模型响应
        user_inputs.append(user_input)
        responses.append(response)
    else:
        print(f"Warning: No match found in row {index + 1}")
        user_inputs.append("")  # 如果没有匹配到，留空
        responses.append("")

# 创建新的 DataFrame
new_df = pd.DataFrame({"User Input": user_inputs, "Response": responses})

# 保存到新的 CSV 文件
new_df.to_csv(output_csv, index=False)
print(f"Separated data saved to {output_csv}")
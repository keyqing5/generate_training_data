import ast
import os
import json
import inspect
from typing import List, Dict, Any, Optional
from openai import OpenAI
import getpass  # 用于安全输入API Key，防止泄露
import datetime  # 用于生成时间戳

# --- Default Configurations ---
DEFAULT_CODE_REPO_PATH = "./my_python_repo"  # 默认的本地Python代码仓库路径
DEFAULT_OUTPUT_FILE = "scenario1_qa_data_with_llm.jsonl"
LLM_MODEL_NAME = "qwen-plus"  # 使用您指定的模型

# --- Global LLM Client (将会在 main 函数中初始化) ---
llm_client = None


# --- Helper Functions (unchanged from previous example) ---
def extract_function_info(file_path: str) -> List[Dict[str, Any]]:
    """
    从单个Python文件中提取函数及其文档字符串和代码片段。
    """
    functions_info = []
    with open(file_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=file_path)

    source_lines = open(file_path, "r", encoding="utf-8").readlines()

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function_name = node.name
            docstring = ast.get_docstring(node)

            try:
                start_line = node.lineno
                end_line = node.end_lineno if node.end_lineno is not None else start_line
                snippet_lines = source_lines[start_line - 1:end_line]
                snippet = "".join(snippet_lines)
            except Exception as e:
                snippet = f"# Error extracting snippet for {function_name}: {e}\n"
                print(f"Warning: Could not extract snippet for {function_name} in {file_path}. Error: {e}")

            functions_info.append({
                "function_name": function_name,
                "docstring": docstring if docstring else "",
                "code_snippet": snippet,
                "start_line": start_line,
                "end_line": end_line
            })
    return functions_info


# --- LLM Integration Function ---
def call_llm_for_qa(function_name: str, docstring: str, code_snippet: str, file_path: str) -> Dict[str, str]:
    """
    调用LLM生成问答对的答案和推理trace。
    """
    global llm_client  # 声明使用全局的 llm_client 变量
    if not llm_client:
        return {
            "answer": "LLM客户端未初始化，无法生成答案。",
            "inference_trace": "LLM客户端初始化失败。"
        }

    # 构建Prompt
    system_prompt = (
        "你是一个专业的代码分析助手，能够理解Python代码并解释其功能。\n"
        "请根据提供的函数信息，首先一步步思考，然后给出函数的主要功能解释和相关的推理过程。\n"
        "你的回答应包含两部分：'answer' 和 'inference_trace'。"
    )

    user_prompt = (
        f"请分析以下Python函数，并回答它的主要功能是什么？\n\n"
        f"--- 函数信息 ---\n"
        f"文件路径: {file_path}\n"
        f"函数名: {function_name}\n"
        f"文档字符串:\n```\n{docstring if docstring else '无文档字符串'}\n```\n"
        f"代码片段:\n```python\n{code_snippet}\n```\n\n"
        f"--- 输出格式 ---\n"
        f"请以JSON格式返回你的回答，其中包含 'answer' (对函数功能的解释) 和 'inference_trace' (你思考并得出答案的步骤)。\n"
        f"例如：\n"
        f"```json\n"
        f"{{\n"
        f"  \"answer\": \"函数的主要功能是...\",\n"
        f"  \"inference_trace\": \"1. 首先我识别到...\\n2. 接着我分析了...\\n3. 最终我得出结论...\"\n"
        f"}}\n"
        f"```"
    )

    try:
        completion = llm_client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            response_format={"type": "json_object"}  # 明确要求LLM以JSON格式返回
        )
        llm_response_content = completion.choices[0].message.content

        # 尝试解析LLM的JSON响应
        try:
            parsed_response = json.loads(llm_response_content)
            return {
                "answer": parsed_response.get("answer", "LLM未提供答案。"),
                "inference_trace": parsed_response.get("inference_trace", "LLM未提供推理trace。")
            }
        except json.JSONDecodeError:
            print(f"Warning: LLM 返回的不是有效的 JSON 格式。\n原始响应: {llm_response_content}")
            return {
                "answer": f"LLM返回无效格式，原始响应：{llm_response_content}",
                "inference_trace": "LLM返回无效JSON。"
            }

    except Exception as e:
        print(f"LLM 调用失败: {e}")
        return {
            "answer": f"LLM调用失败，错误：{e}",
            "inference_trace": f"LLM调用失败，错误：{e}"
        }


def generate_qa_for_function(
        file_path: str,
        func_info: Dict[str, Any],
        qa_id_counter: List[int],
        code_repo_path: str  # 新增参数，用于 relativize 文件路径
) -> Optional[Dict[str, Any]]:
    """
    为单个函数生成一个简化的问答对，并集成LLM调用。
    """
    function_name = func_info["function_name"]
    docstring = func_info["docstring"]
    code_snippet = func_info["code_snippet"]
    start_line = func_info["start_line"]
    end_line = func_info["end_line"]

    if not code_snippet.strip():  # 如果代码片段为空，则跳过
        return None

    qa_id_counter[0] += 1
    qa_id = f"qa_py_{qa_id_counter[0]:05d}"

    question = f"函数 `{function_name}` (在 `{os.path.basename(file_path)}` 中) 的主要功能是什么？"

    # 调用LLM生成答案和推理trace
    llm_output = call_llm_for_qa(function_name, docstring, code_snippet, file_path)
    answer = llm_output["answer"]
    inference_trace = llm_output["inference_trace"]

    # 模拟业务规则（这里是通用的，实际应从项目文档或特定注释中提取）
    business_rules = []
    if "save" in function_name.lower() or "update" in function_name.lower():
        business_rules.append("数据持久化操作需考虑事务一致性。")
    if "auth" in function_name.lower() or "login" in function_name.lower():
        business_rules.append("用户认证和授权操作需遵循安全最佳实践。")

    qa_data = {
        "id": qa_id,
        "question": question,
        "answer": answer,
        "code_context": [
            {
                # 使用传入的 code_repo_path 参数
                "file_path": os.path.relpath(file_path, code_repo_path),
                "line_start": start_line,
                "line_end": end_line,
                "snippet": code_snippet.strip()
            }
        ],
        "business_rules_context": business_rules,
        "inference_trace": inference_trace,
        "metadata": {
            # 使用传入的 code_repo_path 参数
            "source_module": os.path.dirname(os.path.relpath(file_path, code_repo_path)),
            "language": "python",
            "difficulty": "llm_generated",  # 标记为LLM生成
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),  # 使用 datetime 模块生成 ISO 格式的时间戳
            "version_control_hash": "dummy_hash_for_example"
        }
    }
    return qa_data


def get_user_inputs():
    """
    提示用户输入配置值，并提供默认值。
    返回: (code_repo_path, output_file, dashscope_api_key)
    """
    print("\n--- 配置信息输入 ---")

    # 获取代码仓库路径
    repo_path = input(f"请输入您的Python代码仓库路径 (留空使用默认值: '{DEFAULT_CODE_REPO_PATH}'): ").strip()
    if not repo_path:
        repo_path = DEFAULT_CODE_REPO_PATH
    print(f"使用的代码仓库路径: {repo_path}")

    # 获取输出文件名
    output_file = input(f"请输入输出文件名 (留空使用默认值: '{DEFAULT_OUTPUT_FILE}'): ").strip()
    if not output_file:
        output_file = DEFAULT_OUTPUT_FILE
    print(f"使用的输出文件: {output_file}")

    # 获取 DASHSCOPE_API_KEY，优先从环境变量获取，否则安全输入
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        api_key = getpass.getpass("请输入您的 DASHSCOPE_API_KEY：")

    return repo_path, output_file, api_key


# --- Main Script ---
def main():
    global llm_client  # 声明使用全局的 llm_client 变量

    # 1. 获取用户配置
    code_repo_path, output_file, dashscope_api_key = get_user_inputs()

    # 2. 初始化 LLM 客户端
    if not dashscope_api_key:
        print("\nDASHSCOPE_API_KEY 未提供，LLM 客户端将无法初始化。")
        llm_client = None
    else:
        try:
            llm_client = OpenAI(
                api_key=dashscope_api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 北京地域base_url
            )
            print("Qwen-plus LLM 客户端初始化成功。")
        except Exception as e:
            print(f"错误信息：LLM 客户端初始化失败 - {e}")
            print("请参考文档：https://help.aliyun.com/zh/model-studio/developer-reference/error-code")
            llm_client = None  # 将客户端设置为None，后续调用会跳过

    if not llm_client:
        print("警告: 由于LLM客户端未成功初始化，LLM相关功能将无法使用。")

    # 3. 检查代码仓库路径，并创建虚拟仓库（如果不存在）
    if not os.path.exists(code_repo_path):
        print(f"\n错误: 代码仓库路径 `{code_repo_path}` 不存在。")
        print("尝试创建一个简单的虚拟仓库用于演示...")
        os.makedirs(code_repo_path, exist_ok=True)
        with open(os.path.join(code_repo_path, "my_module.py"), "w", encoding="utf-8") as f:
            f.write("""
def calculate_sum(a: int, b: int) -> int:
    \"\"\"
    计算两个整数的和。
    这个函数接受两个整数作为输入，并返回它们的和。
    \"\"\"
    return a + b

def process_data(data_list: list):
    # This function processes a list of data without a docstring.
    print(f"Processing {len(data_list)} items.")
    for item in data_list:
        if item % 2 == 0:
            print(f"Even item: {item}")
        else:
            print(f"Odd item: {item}")

class MyManager:
    \"\"\"
    一个用于管理资源的类。
    提供了资源的创建、读取、更新和删除(CRUD)操作。
    \"\"\"
    def __init__(self, resource_name: str):
        self.resource_name = resource_name
        self.resources = []

    def create_resource(self, resource_data: dict):
        \"\"\"
        创建一个新资源并添加到管理器中。
        :param resource_data: 资源的字典数据。
        :return: None
        \"\"\"
        self.resources.append(resource_data)
        print(f"Resource created: {resource_data}")

    def get_resource(self, resource_id: str) -> Optional[dict]:
        # 从管理器中获取指定ID的资源。
        # 这是一个查找资源的示例方法。
        for res in self.resources:
            if res.get("id") == resource_id:
                return res
        return None
""")
        print(f"已创建虚拟仓库于 `{code_repo_path}`。")

    # 4. 遍历代码仓库并生成问答数据
    all_qa_data = []
    qa_id_counter = [0]

    for root, _, files in os.walk(code_repo_path):
        for file_name in files:
            if file_name.endswith(".py"):
                file_path = os.path.join(root, file_name)
                print(f"正在处理文件: {file_path}")
                functions_info = extract_function_info(file_path)
                for func_info in functions_info:
                    # 将 code_repo_path 作为参数传递
                    qa_entry = generate_qa_for_function(file_path, func_info, qa_id_counter, code_repo_path)
                    if qa_entry:
                        all_qa_data.append(qa_entry)

    # 5. 将生成的问答数据写入文件
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in all_qa_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n训练数据已生成并保存到 `{output_file}`。共生成 {len(all_qa_data)} 条记录。")


if __name__ == "__main__":
    main()
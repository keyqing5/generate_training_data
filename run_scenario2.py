import ast
import os
import json
from typing import List, Dict, Any, Optional
from openai import OpenAI
import getpass
import datetime

# --- Default Configurations ---
DEFAULT_CODE_REPO_PATH = "./my_python_repo"
DEFAULT_OUTPUT_FILE = "scenario2_design_data_with_llm.jsonl"
LLM_MODEL_NAME = "qwen-plus"  # 使用您指定的模型

# 默认需求列表，如果用户不提供自定义需求，将使用此列表
DEFAULT_REQUIREMENTS = [
    {"req": "为现有订单系统增加一个异步的库存扣减服务，以提高订单处理的响应速度，并确保库存数据最终一致性。",
     "keywords": ["order", "inventory", "async"]},
    {"req": "实现一个用户权限管理模块，支持角色-权限分配，并提供API进行权限校验。",
     "keywords": ["user", "auth", "permission", "role"]},
    {"req": "优化支付流程，引入重试机制和幂等性处理，提高支付成功率。", "keywords": ["payment", "retry", "idempotent"]},
    {"req": "为系统添加一个统一的错误日志和监控报警机制。", "keywords": ["log", "monitor", "error"]},
    {"req": "将用户注册和登录功能从现有用户管理模块中独立出来，形成一个独立的认证服务。",
     "keywords": ["user", "register", "login", "auth", "service"]},
]

# --- Global LLM Client (将会在 main 函数中初始化) ---
llm_client = None


# --- Helper Functions ---
def get_repo_file_list(repo_path: str, keywords: List[str]) -> List[str]:
    """
    获取代码仓库中所有Python文件的相对路径，
    并模拟根据关键词筛选相关文件。
    在真实场景中，这里会是复杂的知识图谱查询或向量搜索。
    """
    relevant_files = []
    all_files = []
    if not os.path.exists(repo_path):
        return []  # 如果仓库不存在，直接返回空列表

    for root, _, files in os.walk(repo_path):
        for file_name in files:
            if file_name.endswith(".py"):
                relative_path = os.path.relpath(os.path.join(root, file_name), repo_path)
                all_files.append(relative_path)

                # 模拟关键词匹配：如果文件路径或文件名包含关键词，则认为相关
                is_relevant = False
                for keyword in keywords:
                    if keyword.lower() in relative_path.lower() or keyword.lower() in file_name.lower():
                        is_relevant = True
                        break
                if is_relevant:
                    relevant_files.append(relative_path)

    # 如果没有找到相关文件，则返回所有文件中的一小部分作为通用上下文
    if not relevant_files and all_files:
        return all_files[:min(5, len(all_files))]  # 返回最多前5个文件
    elif relevant_files:
        return relevant_files
    return []


def read_file_content(repo_path: str, relative_file_path: str) -> str:
    """读取指定文件的内容。"""
    full_path = os.path.join(repo_path, relative_file_path)
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"# Error reading {relative_file_path}: {e}"


# --- LLM Integration Function for Scenario 2 ---
def call_llm_for_design_solution(
        requirement: str,
        codebase_context: List[str],  # 相关文件列表
        code_contents: Dict[str, str]  # 相关文件内容
) -> Dict[str, str]:
    """
    调用LLM生成架构设计方案、解释和推理trace。
    """
    global llm_client  # 声明使用全局的 llm_client 变量
    if not llm_client:
        return {
            "design_solution": "LLM客户端未初始化，无法生成设计方案。",
            "explanation": "LLM客户端初始化失败。",
            "inference_trace": "LLM客户端初始化失败。"
        }

    # 构建Prompt
    system_prompt = (
        "你是一个资深的软件架构师和代码专家，能够根据给定的需求和现有代码仓信息，提供详细、合理且可扩展的架构设计方案。\n"
        "请先一步步思考，分析需求和现有上下文，然后给出设计方案、解释和推理过程。\n"
        "你的回答应包含三部分：'design_solution' (设计方案), 'explanation' (设计方案的解释), 和 'inference_trace' (你思考并得出方案的步骤)。"
    )

    context_str = ""
    if codebase_context:
        context_str += "\n--- 现有代码仓相关文件 ---\n"
        for i, file_path in enumerate(codebase_context):
            context_str += f"文件 {i + 1}: {file_path}\n"
            # 实际中这里不会直接放所有文件内容，而是通过知识图谱提取关键信息或使用RAG获取摘要
            # 为了演示，这里假设可以提供部分文件内容
            if file_path in code_contents:
                # 仅展示部分内容，避免Prompt过长
                content = code_contents[file_path]
                context_str += f"```python\n{content[:500]}...\n```\n"  # 截断
            context_str += "---\n"
    else:
        context_str += "\n--- 现有代码仓信息 ---\n"
        context_str += "未找到与需求直接相关的代码文件，请基于通用设计原则和最佳实践进行设计。\n"

    user_prompt = (
        f"请根据以下需求和提供的现有代码仓上下文，设计一个架构方案。请注意：\n"
        f"1. 你的设计应考虑现有Python代码仓的特点和可能的扩展方向。\n"
        f"2. 方案应结构化、清晰，并包含必要的解释和推理过程。\n\n"
        f"--- 需求 ---\n"
        f"{requirement}\n"
        f"{context_str}\n\n"
        f"--- 输出格式 ---\n"
        f"请以JSON格式返回你的回答，其中包含 'design_solution' (详细的架构设计方案，使用Markdown格式),\n"
        f" 'explanation' (对设计方案的解释) 和 'inference_trace' (你思考并得出方案的步骤)。\n"
        f"例如：\n"
        f"```json\n"
        f"{{\n"
        f"  \"design_solution\": \"# 方案标题\\n1. ...\\n2. ...\",\n"
        f"  \"explanation\": \"此方案的优点是...\",\n"
        f"  \"inference_trace\": \"1. 首先我分析了需求...\\n2. 接着我评估了现有系统...\\n3. 最终我提出了...\"\n"
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
            response_format={"type": "json_object"}
        )
        llm_response_content = completion.choices[0].message.content

        try:
            parsed_response = json.loads(llm_response_content)
            return {
                "design_solution": parsed_response.get("design_solution", "LLM未提供设计方案。"),
                "explanation": parsed_response.get("explanation", "LLM未提供解释。"),
                "inference_trace": parsed_response.get("inference_trace", "LLM未提供推理trace。")
            }
        except json.JSONDecodeError:
            print(f"Warning: LLM 返回的不是有效的 JSON 格式。\n原始响应: {llm_response_content}")
            return {
                "design_solution": f"LLM返回无效格式，原始响应：{llm_response_content}",
                "explanation": "LLM返回无效JSON。",
                "inference_trace": "LLM返回无效JSON。"
            }

    except Exception as e:
        print(f"LLM 调用失败: {e}")
        return {
            "design_solution": f"LLM调用失败，错误：{e}",
            "explanation": f"LLM调用失败，错误：{e}",
            "inference_trace": f"LLM调用失败，错误：{e}"
        }


def generate_design_data(
        design_id_counter: List[int],
        requirement_text: str,  # 明确这是需求文本
        repo_path: str,
        requirement_keywords: List[str]
) -> Optional[Dict[str, Any]]:
    """
    生成一个架构设计方案的数据点。
    """
    if not requirement_text.strip():
        return None

    design_id_counter[0] += 1
    design_id = f"design_py_{design_id_counter[0]:05d}"

    # 模拟上下文提取：获取与需求关键词相关的代码文件列表
    codebase_context_files = get_repo_file_list(repo_path, requirement_keywords)

    # 为了LLM调用，需要读取这些文件的内容 (这里仅截取部分以避免过长Prompt)
    code_contents_for_llm = {
        f: read_file_content(repo_path, f) for f in codebase_context_files
    }

    # 调用LLM生成设计方案
    llm_output = call_llm_for_design_solution(
        requirement_text,  # 传入需求文本
        codebase_context_files,
        code_contents_for_llm
    )

    design_data = {
        "id": design_id,
        "requirement": requirement_text,
        "design_solution": llm_output["design_solution"],
        "explanation": llm_output["explanation"],
        "inference_trace": llm_output["inference_trace"],
        "codebase_context": codebase_context_files,  # 记录 LLM 看到的相关文件列表
        "metadata": {
            "source_project": os.path.basename(os.path.abspath(repo_path)),  # 使用绝对路径的basename
            "design_type": "feature_extension",  # 默认类型，实际可由LLM判断或根据需求定义
            "language": "python",
            "difficulty": "llm_generated",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "version_control_hash": "dummy_hash_for_example"  # 实际应获取git commit hash
        }
    }
    return design_data


def get_user_inputs():
    """
    提示用户输入配置值，并提供默认值。
    返回: (requirements_list, code_repo_path, output_file, dashscope_api_key)
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

    # 获取需求列表
    requirements_input = input(
        f"请输入需求列表的 JSON 字符串 (留空使用默认值)。\n"
        f"格式示例: '[{{\"req\": \"需求描述\", \"keywords\": [\"关键词1\", \"关键词2\"]}}, ...]' \n"
        f"您的输入: "
    ).strip()

    requirements_list = []
    if not requirements_input:
        requirements_list = DEFAULT_REQUIREMENTS
        print("使用默认需求列表。")
    else:
        try:
            parsed_reqs = json.loads(requirements_input)
            # 简单验证格式
            if isinstance(parsed_reqs, list) and all(
                    isinstance(item, dict) and "req" in item and "keywords" in item for item in parsed_reqs):
                requirements_list = parsed_reqs
                print("已成功解析自定义需求列表。")
            else:
                print("警告: 自定义需求列表格式不正确，将使用默认需求列表。")
                requirements_list = DEFAULT_REQUIREMENTS
        except json.JSONDecodeError:
            print("警告: 需求列表输入不是有效的 JSON 格式，将使用默认需求列表。")
            requirements_list = DEFAULT_REQUIREMENTS

    # 获取 DASHSCOPE_API_KEY，优先从环境变量获取，否则安全输入
    # api_key = os.getenv("DASHSCOPE_API_KEY")
    api_key = input(f"请输入您的 DASHSCOPE_API_KEY: ").strip()
    if not api_key:
        api_key = getpass.getpass("请输入您的 DASHSCOPE_API_KEY：")

    return requirements_list, repo_path, output_file, api_key


# --- Main Script ---
def main():
    global llm_client  # 声明使用全局的 llm_client 变量

    # 1. 获取用户配置
    requirements_config, code_repo_path, output_file, dashscope_api_key = get_user_inputs()

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
        with open(os.path.join(code_repo_path, "__init__.py"), "w") as f: f.write("")
        with open(os.path.join(code_repo_path, "user_management.py"), "w", encoding="utf-8") as f:
            f.write("""
def register_user(username, password, email):
    # This registers a new user
    print(f"Registering {username}")
    # ... database logic
    return {"id": 1, "username": username}

def get_user_profile(user_id):
    # Retrieves user profile from DB
    return {"id": user_id, "username": "test_user"}
""")
        with open(os.path.join(code_repo_path, "order_service.py"), "w", encoding="utf-8") as f:
            f.write("""
def create_order(user_id, items):
    # Creates a new order
    print(f"Creating order for user {user_id}")
    # ... inventory check, payment processing
    return {"order_id": "ORD001", "status": "pending"}

def update_order_status(order_id, new_status):
    # Updates an existing order's status
    print(f"Updating order {order_id} to {new_status}")
    return True
""")
        with open(os.path.join(code_repo_path, "payment_gateway.py"), "w", encoding="utf-8") as f:
            f.write("""
def process_payment(order_id, amount, payment_method):
    # Integrates with external payment provider
    print(f"Processing payment for order {order_id}, amount {amount}")
    return {"success": True, "transaction_id": "TXN123"}
""")
        print(f"已创建虚拟仓库于 `{code_repo_path}`。")

    # 4. 遍历需求并生成设计方案
    all_design_data = []
    design_id_counter = [0]

    for req_item in requirements_config:  # 使用从用户或默认配置中获取的需求列表
        print(f"\n正在为需求生成设计方案: {req_item['req']}")
        design_entry = generate_design_data(
            design_id_counter,
            req_item["req"],
            code_repo_path,  # 传递代码仓库路径
            req_item["keywords"]
        )
        if design_entry:
            all_design_data.append(design_entry)

    # 5. 将生成的问答数据写入文件
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in all_design_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n训练数据已生成并保存到 `{output_file}`。共生成 {len(all_design_data)} 条记录。")


if __name__ == "__main__":
    main()

# 本地代码仓的智能训练数据生成与处理 Automatically generate and process training data from repository


### 使用示例：
仓库中提供jupyterbook文件```demo.ipynb```作为demo参考，您可以直接修改其中的变量内容```CODE_REPO_PATH、OUTPUT_FILE、LLM_MODEL_NAME、REQUIREMENTS（仅在场景2中）```，以满足自定义需求。

如无修改，则按照默认参数运行，即提供默认仓库```my_python_repo```作为示例。

您也可以使用如```python [Python文件名]```的方式运行python文件。如无修改，则按照默认参数运行。

例如，当运行```python run_scenario1.py```时，若无其它输入，屏幕显示为：
```angular2html
--- 配置信息输入 ---
请输入您的Python代码仓库路径 (留空使用默认值: '.\my_python_repo'): 
使用的代码仓库路径: .\my_python_repo
请输入输出文件名 (留空使用默认值: 'scenario1_qa_data_with_llm.jsonl'): 
使用的输出文件: scenario1_qa_data_with_llm.jsonl
请输入您的 DASHSCOPE_API_KEY: sk-***
Qwen-plus LLM 客户端初始化成功。
正在处理文件: .\my_python_repo\my_module.py
训练数据已生成并保存到 `scenario1_qa_data_with_llm.jsonl`。共生成 5 条记录。
```

即可得到和jupyterbook中相同结果。如果需要自定义路径等，可以在出现提示之后输入相应信息。

若运行，自定义需求（```Add new function module: log file generation```），内容如下：
```angular2html
--- 配置信息输入 ---
请输入您的Python代码仓库路径 (留空使用默认值: './my_python_repo'): 
使用的代码仓库路径: ./my_python_repo
请输入输出文件名 (留空使用默认值: 'scenario2_design_data_with_llm.jsonl'): scenario2_design_data_with_llm2.jsonl
使用的输出文件: scenario2_design_data_with_llm2.jsonl
请输入需求列表的 JSON 字符串 (留空使用默认值)。
格式示例: '[{"req": "需求描述", "keywords": ["关键词1", "关键词2"]}, ...]' 
您的输入: [{"req":"Add new function module: log file generation","keywords":["log","record errors"]}]
已成功解析自定义需求列表。
请输入您的 DASHSCOPE_API_KEY: sk-***
Qwen-plus LLM 客户端初始化成功。

正在为需求生成设计方案: Add new function module: log file generation
训练数据已生成并保存到 `scenario2_design_data_with_llm2.jsonl`。共生成 1 条记录。
```

#### 输出格式：xx.jsonl文件

#### 其它提示：
重新运行时，需提供大模型的API key（即DASHSCOPE_API_KEY)。

当代码仓文件较多或代码量较大时，运行可能需要较多时间，请耐心等待。

若第一次运行时长时间没有产生输出文件，可以停止并重新运行，请务必保证输入参数合法。

同样提供```demo_en.ipynb```文件作为英语版本示例。

---
### 功能描述：
#### 概述
本仓库旨在提出一个全面的系统框架，用于自动化生成和处理高质量的智能训练数据。这些数据将专门用于微调Qwen 2.5系列大型语言模型（LLM），使其能够深入理解并高效利用本地代码仓的信息。
具体而言，模型将获得以下核心能力：

- 回答关于本地代码仓的业务流程和规则： 能够解释代码中实现的特定功能、数据流转和业务逻辑。
- 基于给定需求生成代码架构设计方案： 能够分析现有代码仓结构，并根据新需求提出合理、可扩展的系统设计。

核心设计原则：

- 自动化与智能化： 最大限度地减少人工干预，通过智能算法和LLM协同完成数据生成。
- 数据质量与结构化： 确保生成数据的准确性、逻辑正确性，并以结构化格式存储，便于模型消费。
- 可解释性与可追溯性： 为每个训练数据点提供清晰的原文代码片段和详细的推理过程（inference trace），增强透明度和信任度。
- 系统完整性与可扩展性： 构建一个模块化、分层的架构，易于维护、迭代和未来功能扩展。
#### 架构设计
采用分层、模块化的设计思想，旨在实现高内聚、低耦合，并为未来的功能扩展提供坚实基础。

- 模块功能描述：

  1. 数据源层：提供原始代码、文档、注释、Issue、PR等作为输入。

     - SCM集成模块： 负责与GitHub等版本控制系统交互，实现代码仓库的克隆、拉取最新版本、文件读取等操作。
     原始数据输入： 提供代码文件 (.java, .py, .go等)、项目文档 (README.md, CONTRIBUTING.md, Wiki)、代码注释以及Issues和Pull Requests等信息作为系统输入。
     代码解析与知识提取层 (共享能力)： 这是两个场景的共同基石。

     - AST解析器： 使用tree-sitter（支持多语言）或特定语言的解析器（如Java的JDT、Python的ast），将代码转换为抽象语法树（AST），提取函数、类、变量、调用关系、继承关系、依赖关系等结构化信息。
     - 文档/注释解析器： 解析各类文档（Markdown、Wiki）和代码注释（JSDoc、JavaDoc、Python Docstrings），提取高层业务逻辑、功能描述、设计意图和潜在的业务规则。
     - Issue解析器： 解析GitHub Issue和Pull Request，提取用户需求、bug报告、设计讨论等，作为生成场景2需求的灵感或补充上下文。
     - 知识图谱构建器： 将上述解析出的实体（文件、类、函数、变量、业务概念）和关系（调用、继承、依赖、描述、讨论）构建成一个可查询的代码知识图谱 (KG)，便于复杂查询和推理。
     - 代码嵌入生成器： 利用预训练的代码语言模型（如CodeBERT、UniXCoder）生成代码片段的向量表示（代码语义嵌入），用于语义相似性搜索和RAG（Retrieval Augmented Generation）。
  2. 训练数据生成引擎层： 
     - 场景1生成模块： 专注于利用KG、CE和LLM，生成业务流程和规则相关的问答对。
     - 场景2生成模块： 专注于利用KG、CE和LLM，结合需求生成架构设计方案。
     LLM集成 (CoT Prompting)： 通过API调用集成强大的LLM（如Qwen 2.5、GPT-4）。关键在于设计高效的Prompt模板，特别是引入Chain-of-Thought (CoT) Prompting，引导LLM输出详细的推理过程（inference_trace），而非直接给出结果。
  3. 数据存储与管理层：
     - 数据存储 (JSONL)： 将生成的训练数据以JSONL（每行一个JSON对象）格式存储，易于处理和导入。

- 核心流程
  - 代码仓获取： 通过SCM集成模块，克隆或更新目标GitHub代码仓库。
  - 代码解析与知识提取： 运行AST解析器、文档解析器等，从代码仓中提取所有相关信息，构建代码知识图谱和代码语义嵌入。
  - 场景驱动的数据生成： 根据当前要生成的场景（场景1或场景2），调用对应的生成模块。
  - LLM交互与推理： 生成模块构造Prompt，包含从知识图谱中检索到的相关上下文，发送给LLM。LLM执行推理并生成初步的答案/方案和推理 trace。
  - 数据后处理： 对LLM的原始输出进行清洗、格式化，并补充必要的元数据（如code_context中的具体行号）。
  - 数据验证： 将生成的数据提交给数据验证与评估层进行检查。
  - 存储： 将通过验证的最终训练数据存储到JSONL文件，并进行版本管理。
- 场景解析
  - 场景 1: 业务流程与规则问答对生成
  目标： 自动化生成关于本地代码仓内部实现的业务流程和规则的问答对，并提供详细的代码片段和推理过程。
  - 场景 2: 架构设计方案生成
  目标： 根据给定的需求描述，生成基于本地代码仓现状的架构设计方案，并提供详细的解释和推理 trace。

- 数据多样性与代表性策略
为确保生成数据的多样性、全面性和代表性，将采取以下策略：

  - 覆盖代码仓广度：
  模块遍历： 从代码仓的不同功能模块（如用户认证、订单管理、支付服务、库存系统、消息队列集成等）和不同技术栈组件中均匀提取数据。
  文件类型： 不仅限于主要代码文件，也包括配置文件、API定义、数据库schema、测试文件等。

  - 多语言支持：
  采用tree-sitter作为核心AST解析器，其天然支持多种编程语言。
  在知识图谱中存储语言类型元数据。
  针对不同语言，在代码嵌入生成、Prompt Engineering阶段可能需要进行特定调整。
---
### 联系方式：
如有疑问，欢迎联系邮箱```liangqs@tongji.edu.cn```或```qiusliang2-c@my.cityu.edu.hk```。



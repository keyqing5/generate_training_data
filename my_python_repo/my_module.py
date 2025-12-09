
def calculate_sum(a: int, b: int) -> int:
    """
    计算两个整数的和。
    这个函数接受两个整数作为输入，并返回它们的和。
    """
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
    """
    一个用于管理资源的类。
    提供了资源的创建、读取、更新和删除(CRUD)操作。
    """
    def __init__(self, resource_name: str):
        self.resource_name = resource_name
        self.resources = []

    def create_resource(self, resource_data: dict):
        """
        创建一个新资源并添加到管理器中。
        :param resource_data: 资源的字典数据。
        :return: None
        """
        self.resources.append(resource_data)
        print(f"Resource created: {resource_data}")

    def get_resource(self, resource_id: str) -> Optional[dict]:
        # 从管理器中获取指定ID的资源。
        # 这是一个查找资源的示例方法。
        for res in self.resources:
            if res.get("id") == resource_id:
                return res
        return None

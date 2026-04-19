import ast

class SecurityException(Exception):
    pass

class CodeScanner(ast.NodeVisitor):
    def __init__(self):
        # 黑名单机制
        self.forbidden_modules = {'os', 'sys', 'subprocess', 'shutil', 'socket', 'requests'}
        self.forbidden_fuctions = {'open', 'exec', 'eval', 'compile', '__import__'}

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name in self.forbidden_modules:
                raise SecurityException(f"触发安全策略：禁止导入高危模块'{alias.name}'")
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module in self.forbidden_modules:
            raise SecurityException(f"触发安全策略：禁止从高危模块 '{node.module}' 导入")
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id in self.forbidden_fuctions:
            raise SecurityException(f"触发安全策略：禁止调用高危函数 '{node.func.id}()'")
        self.generic_visit(node)

def check_code_security(code_string):
    try:
        tree = ast.parse(code_string)
        scanner = CodeScanner()
        scanner.visit(tree)
    except SyntaxError as e:
        raise SecurityException(f"代码存在语法错误，无法解析: {str(e)}")

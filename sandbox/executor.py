import io
import contextlib
from  .ast_scanner import check_code_security, SecurityException

def execute_python_code(code_string):
    """
    接收 Python 代码字符串，先进行 AST 扫描，通过后再执行，并捕获标准输出。
    返回: (is_success: bool, result: str)
     """
    print("🛡️ [沙盒] 正在进行 AST 静态安全扫描...")
    try:
        check_code_security(code_string)
        print("🛡️ [沙盒] 安全扫描通过。")
    except SecurityException as e:
        return False, f"SecurityException: {str(e)}"

    output_buffer = io.StringIO()
    with contextlib.redirect_stdout(output_buffer):
        try:
            exec(code_string, {"__builtins__":__builtins__})
            return True, output_buffer.getvalue()
        except SecurityException as e:
            return False, f"{type(e).__name__}: {str(e)}"
import contextlib
import io

#! beware this executes arbitrary code, only use with trusted input
# ? it is very suggestable to put some guardrail or check before executing code, for example check if it contains some forbidden keywords or patterns, or execute it in a sandboxed environment with limited resources and permissions, etc.

def execute_code(code: str, state: dict) -> str:
    output_buffer = io.StringIO()
    
    with (
        contextlib.redirect_stdout(output_buffer),
        contextlib.redirect_stderr(output_buffer)
    ):
        try:
            exec(code, state)
        except Exception as e:
            print(f"Error executing code: {e}")
    
    output = output_buffer.getvalue()
    return str(output)
    
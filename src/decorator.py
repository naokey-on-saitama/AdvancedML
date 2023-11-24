def add_print(pattern="=", len=30):
    """文字の出力前後に表示を行うデコレータ

    Args:
        pattern (str, optional): 文字のパターン. Defaults to "=".
        len (int, optional): 長さ. Defaults to 30.
    """
    def _add_print(func):
        def wrapper(*args, **keywords):
            print()
            print(pattern*len)
            print()
            
            v = func(*args, **keywords)
            
            print()
            print(pattern*len)
            print()
            
            return v
        return wrapper
    return _add_print

@add_print("=", 15)
def output():
    print("Hello")
    
if __name__ == "__main__":
    output()

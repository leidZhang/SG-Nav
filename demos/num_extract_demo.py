import re

def extract_number(input: str) -> str:
    match = re.search(r"\d+\.\d+|\d+", input)
    if match:
        return match.group()
    return None


if __name__ == "__main__":
    string = "6.33^&^()"
    string = extract_number(string)
    num = float(string)
    print(num)
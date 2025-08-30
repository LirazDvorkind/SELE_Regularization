import os
import pyperclip

def copy_py_files_to_clipboard(dir_path):
    output = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                output.append(f"\n{'='*30}\n{file_path}\n{'='*30}")
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        output.append(f.read())
                except Exception as e:
                    output.append(f"Error reading {file_path}: {e}")

    final_text = "\n".join(output)
    pyperclip.copy(final_text)
    print("Copied all .py file contents to clipboard.")


if __name__ == '__main__':
    copy_py_files_to_clipboard(r"/")

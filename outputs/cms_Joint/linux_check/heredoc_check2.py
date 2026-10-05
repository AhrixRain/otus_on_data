import ast, pathlib, re, subprocess, sys
src = pathlib.Path("scripts_joint/train_readiness.sh").read_text(encoding="utf-8")
blocks = re.findall(r"<<'PYEOF'[^\n]*\n(.*?)\nPYEOF", src, re.S)
print("embedded python blocks:", len(blocks))
for i, block in enumerate(blocks, 1):
    ast.parse(block)
    print(f"  block {i}: parses OK ({len(block.splitlines())} lines)")
    if i == 1:
        print("  --- executing block 1 (read-only env probe) ---")
        result = subprocess.run([sys.executable, "-c", block], capture_output=True, text=True)
        print("   exit", result.returncode)
        for line in (result.stdout + result.stderr).strip().splitlines():
            print("   ", line)

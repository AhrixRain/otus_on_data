import ast, pathlib, re
src = pathlib.Path("scripts_joint/train_readiness.sh").read_text(encoding="utf-8")
blocks = re.findall(r"<<'PYEOF'\n(.*?)\nPYEOF", src, re.S)
print("embedded python blocks:", len(blocks))
for i, block in enumerate(blocks, 1):
    tree = ast.parse(block)
    names = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    print(f"  block {i}: parses OK, {len(block.splitlines())} lines, defs={names}")
# also confirm the two new features are present
for needle in ("EXTRAPOLATED full run", "Slurm suggestion", "peak CUDA reserved", "_traincheck"):
    print(f"  contains {needle!r}:", needle in src)

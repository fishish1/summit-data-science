
import os

file_path = 'src/summit_housing/dashboard/app.py'

with open(file_path, 'r') as f:
    lines = f.readlines()

new_lines = []

# Keep header and setup (Lines 1-134, index 0-133)
# Actually let's look at key markers.
# Marker 1: "story_tab, methods_tab = st.tabs(["
# Marker 2: The closing "])" of that call.

keep_until_idx = -1
start_dedent_idx = -1

for i, line in enumerate(lines):
    if "story_tab, methods_tab = st.tabs([" in line:
        # Find closing bracket
        for j in range(i, i+10):
            if "])" in lines[j]:
                keep_until_idx = j
                break
        break

if keep_until_idx == -1:
    print("Could not find start marker")
    exit(1)

# Add the kept lines
new_lines.extend(lines[:keep_until_idx+1])
new_lines.append("\n") # Add a spacer

# Find where to resume (Lines 197ish)
# We look for "with story_tab:" that is indented.
resume_idx = -1
for i in range(keep_until_idx + 1, len(lines)):
    if "with story_tab:" in line and line.strip() == "with story_tab:":
        # Check actual indentation
        if line.startswith("                    with story_tab:"): # 20 spaces
             resume_idx = i
             break
    # Also check if I missed finding it by simple string match
    if "with story_tab:" in lines[i] and len(lines[i]) - len(lines[i].lstrip()) >= 16:
        resume_idx = i
        break

if resume_idx == -1:
    print("Could not find resume marker")
    # Fallback to line number 196 (0-indexed) -> 197
    resume_idx = 196 

print(f"Keeping lines 0 to {keep_until_idx}")
print(f"Resuming at line {resume_idx}")

# Dedent function
def dedent_line(line, count=20):
    if len(line.strip()) == 0:
        return "\n"
    if line.startswith(" " * count):
        return line[count:]
    else:
        # If less than 20 spaces but non-empty, warn/handle?
        # Maybe it's a multiline string content?
        # We'll just strip whatever common indent we have, but for now strict 20.
        # Actually, multiline strings might be tricky.
        return line # Retain original if can't dedent (risky)

# Refined dedent:
# Check if 20 spaces exist.
for i in range(resume_idx, len(lines)):
    line = lines[i]
    if line.startswith("                    "):
        new_lines.append(line[20:])
    elif len(line.strip()) == 0:
         new_lines.append("\n")
    else:
        # This is likely a line inside a multiline string that wasn't indented relative to code?
        # Or I miscalculated the indent.
        # Let's check strictness.
        new_lines.append(line)

with open(file_path, 'w') as f:
    f.writelines(new_lines)

print("Fixed app.py")

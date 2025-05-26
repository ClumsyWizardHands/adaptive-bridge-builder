"""Fix indentation issue in universal_agent_connector.py"""

# Read the file
with open('universal_agent_connector.py', 'r') as f:
    lines = f.readlines()

# Find the problematic line around 735
print(f"Total lines: {len(lines)}")
print("\nChecking around line 735...")

# Look for try blocks with no content
for i in range(730, min(740, len(lines))):
    line = lines[i].rstrip()
    print(f"Line {i+1}: {repr(line)}")
    
    # Check if this is a try: line followed by non-indented content
    if line.strip() == 'try:':
        next_line_idx = i + 1
        if next_line_idx < len(lines):
            next_line = lines[next_line_idx]
            # Check if next line is not properly indented
            if next_line.strip() and not next_line.startswith('    ') and not next_line.startswith('\t'):
                print(f"\nFound issue at line {i+1}: try block without indented content")
                print(f"Next line {next_line_idx+1}: {repr(next_line.rstrip())}")

# The error mentions line 735, so let's specifically check that area
if len(lines) > 735:
    print(f"\nLine 735: {repr(lines[734].rstrip())}")
    print(f"Line 736: {repr(lines[735].rstrip())}")
    print(f"Line 737: {repr(lines[736].rstrip())}")
    print(f"Line 738: {repr(lines[737].rstrip())}")

# Look for the __all__ line that seems to be the issue
for i, line in enumerate(lines):
    if '__all__' in line and i > 730:
        print(f"\nFound __all__ at line {i+1}: {repr(line.rstrip())}")
        # Check the line before
        if i > 0:
            print(f"Previous line {i}: {repr(lines[i-1].rstrip())}")
        # Check the line after
        if i < len(lines) - 1:
            print(f"Next line {i+2}: {repr(lines[i+1].rstrip())}")
        
        # If __all__ is indented but there's an empty try block before it
        if line.startswith('    '):
            # Look backward for a try: statement
            for j in range(i-1, max(0, i-10), -1):
                if lines[j].strip() == 'try:':
                    print(f"\nFound try: at line {j+1}")
                    # Fix: add a pass statement after try:
                    lines.insert(j+1, '        pass\n')
                    print("Added 'pass' statement after try:")
                    break

# Write the fixed file
with open('universal_agent_connector_fixed.py', 'w') as f:
    f.writelines(lines)

print("\nFixed file written to universal_agent_connector_fixed.py")
"""Script to check and fix syntax errors in universal_agent_connector.py"""

def check_and_fix_syntax_errors():
    # Read the file
    with open('universal_agent_connector.py', 'r') as f:
        lines = f.readlines()
    
    # Print context around line 805
    print("Lines 795-815:")
    for i in range(max(0, 795), min(len(lines), 815)):
        print(f"{i+1}: {lines[i]}", end='')
    
    # Look for the unclosed parenthesis
    print("\n\nSearching for unclosed parentheses...")
    
    # Track parentheses
    paren_stack = []
    for i, line in enumerate(lines):
        for j, char in enumerate(line):
            if char == '(':
                paren_stack.append((i+1, j+1))
            elif char == ')':
                if paren_stack:
                    paren_stack.pop()
    
    if paren_stack:
        print(f"\nFound unclosed parentheses at:")
        for line_num, col_num in paren_stack[-5:]:  # Show last 5 unclosed
            print(f"  Line {line_num}, Column {col_num}")
            if line_num > 0 and line_num <= len(lines):
                print(f"  Context: {lines[line_num-1].strip()}")

if __name__ == "__main__":
    try:
        check_and_fix_syntax_errors()
    except Exception as e:
        print(f"Error: {e}")
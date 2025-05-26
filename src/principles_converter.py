#!/usr/bin/env python3
"""
Principles Converter

This script converts a plain text list of principles into the structured JSON format
required by the PrincipleEngine. It helps generate appropriate IDs, weights, and
evaluation criteria templates for each principle.
"""

import json
import re
import sys
import argparse
from typing import List, Dict, Any


def generate_id(name: str) -> str:
    """
    Generate a snake_case ID from a principle name.
    
    Args:
        name: The principle name
        
    Returns:
        A snake_case ID
    """
    # Convert to lowercase
    id_base = name.lower()
    
    # Replace non-alphanumeric characters with underscores
    id_base = re.sub(r'[^a-z0-9]+', '_', id_base)
    
    # Remove leading/trailing underscores
    id_base = id_base.strip('_')
    
    # Ensure uniqueness by adding timestamp if needed
    return id_base


def generate_evaluation_criteria(description: str) -> List[str]:
    """
    Generate template evaluation criteria based on the principle description.
    
    Args:
        description: The principle description
        
    Returns:
        List of evaluation criteria
    """
    # Default criteria template
    return [
        "Action respects this principle's core values",
        "No clear violations of the principle are present",
        "Alternatives that better honor this principle were considered",
        "The chosen approach represents a reasonable balance of concerns"
    ]


def convert_principle(text: str) -> Dict[str, Any]:
    """
    Convert a plain text principle into structured format.
    
    Args:
        text: Plain text description of the principle
        
    Returns:
        Structured principle dictionary
    """
    # Extract name and description
    lines = text.strip().split('\n')
    name = lines[0].strip()
    description = ' '.join([line.strip() for line in lines[1:] if line.strip()])
    
    # Generate other fields
    principle_id = generate_id(name)
    
    return {
        "id": principle_id,
        "name": name,
        "description": description,
        "weight": 1.0,
        "example": f"Example application of the '{name}' principle.",
        "evaluation_criteria": generate_evaluation_criteria(description)
    }


def convert_principles(principles_text: str) -> List[Dict[str, Any]]:
    """
    Convert a list of principles from plain text to structured format.
    
    Args:
        principles_text: Plain text principles list
        
    Returns:
        List of structured principle dictionaries
    """
    # Split text into individual principles
    # Assuming each principle starts with a number or bullet point
    principle_pattern = r'(?:\d+\.|\*|\-|\•|\–)\s*(.*?)(?=(?:\d+\.|\*|\-|\•|\–)|$)'
    principles_matches = re.findall(principle_pattern, principles_text, re.DOTALL)
    
    if not principles_matches:
        # If no matches found with pattern, try splitting by empty lines
        principles_matches = [p for p in principles_text.split('\n\n') if p.strip()]
    
    # Convert each principle
    return [convert_principle(match.strip()) for match in principles_matches if match.strip()]


def write_principles_to_file(principles: List[Dict[str, Any]], output_file: str) -> None:
    """
    Write principles to a JSON file.
    
    Args:
        principles: List of principle dictionaries
        output_file: Path to output JSON file
    """
    with open(output_file, 'w') as f:
        json.dump(principles, f, indent=2)
    
    print(f"Saved {len(principles)} principles to {output_file}")


def read_principles_from_file(input_file: str) -> str:
    """
    Read principles from a text file.
    
    Args:
        input_file: Path to input text file
        
    Returns:
        String containing principle text
    """
    with open(input_file, 'r') as f:
        return f.read()


def interactive_mode() -> str:
    """
    Get principles from user input in interactive mode.
    
    Returns:
        String containing user's principles
    """
    print("Enter your principles (one principle per line, with blank lines between principles).")
    print("Type 'END' on a line by itself when done:")
    
    lines = []
    while True:
        line = input()
        if line == "END":
            break
        lines.append(line)
    
    return '\n'.join(lines)


def main() -> int:
    """Main function to run the converter."""
    parser = argparse.ArgumentParser(description="Convert plain text principles to structured JSON format.")
    parser.add_argument("--input", "-i", help="Input text file containing principles")
    parser.add_argument("--output", "-o", default="custom_principles.json", help="Output JSON file (default: custom_principles.json)")
    parser.add_argument("--interactive", "-t", action="store_true", help="Interactive mode (type principles directly)")
    
    args = parser.parse_args()
    
    # Get principles text from file or interactive input
    if args.interactive:
        principles_text = interactive_mode()
    elif args.input:
        principles_text = read_principles_from_file(args.input)
    else:
        print("Error: Please specify an input file or use interactive mode.")
        return 1
    
    # Convert principles
    principles = convert_principles(principles_text)
    
    # Print summary
    print(f"\nConverted {len(principles)} principles:")
    for p in principles:
        print(f"- {p['name']}")
    
    # Save to file
    write_principles_to_file(principles, args.output)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

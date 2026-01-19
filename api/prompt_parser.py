"""Natural language prompt parser for video editing instructions."""

import re
from typing import Dict, Any, Optional, List


def parse_prompt(prompt: str) -> Dict[str, Any]:
    """
    Parse natural language prompt into edit instructions dictionary.
    
    Args:
        prompt: Natural language string with editing instructions
        
    Returns:
        Dictionary with edit instructions compatible with pipeline
        
    Examples:
        >>> parse_prompt("make the expression more intense, increase lip movement")
        {'expression_scale': 1.3, 'lip_intensity': 1.5}
        
        >>> parse_prompt("subtle expression, less lip movement")
        {'expression_scale': 0.7, 'lip_intensity': 0.5}
    """
    prompt_lower = prompt.lower()
    instructions = {}
    
    # Expression intensity patterns
    expression_patterns = [
        (r'(?:more|increase|enhance|intensify|stronger|exaggerate).*expression', 1.3),
        (r'(?:less|decrease|reduce|subtle|weaker|minimize).*expression', 0.7),
        (r'(?:very|extremely|highly).*expression', 1.5),
        (r'(?:slight|slightly|mild).*expression', 0.8),
        (r'expression.*(?:more|increase|enhance)', 1.3),
        (r'expression.*(?:less|decrease|reduce)', 0.7),
    ]
    
    for pattern, scale in expression_patterns:
        if re.search(pattern, prompt_lower):
            instructions['expression_scale'] = scale
            break
    
    # Lip movement patterns
    lip_patterns = [
        (r'(?:more|increase|enhance|intensify|stronger|exaggerate).*lip', 1.5),
        (r'(?:less|decrease|reduce|subtle|weaker|minimize).*lip', 0.5),
        (r'(?:very|extremely|highly).*lip', 2.0),
        (r'(?:slight|slightly|mild).*lip', 0.7),
        (r'lip.*(?:more|increase|enhance)', 1.5),
        (r'lip.*(?:less|decrease|reduce)', 0.5),
        (r'(?:more|increase|enhance).*mouth.*movement', 1.5),
        (r'(?:less|decrease|reduce).*mouth.*movement', 0.5),
    ]
    
    for pattern, intensity in lip_patterns:
        if re.search(pattern, prompt_lower):
            instructions['lip_intensity'] = intensity
            break
    
    # Head pose patterns (basic implementation)
    head_pose_patterns = [
        (r'(?:tilt|turn).*head.*(?:left|right)', 'horizontal'),
        (r'(?:nod|tilt).*head.*(?:up|down)', 'vertical'),
        (r'(?:rotate|turn).*head', 'rotation'),
    ]
    
    for pattern, pose_type in head_pose_patterns:
        if re.search(pattern, prompt_lower):
            instructions['head_pose'] = pose_type
            break
    
    # Numeric extraction for specific values
    # Look for patterns like "expression scale 1.5" or "lip intensity 2.0"
    numeric_patterns = [
        (r'expression.*scale.*?(\d+\.?\d*)', 'expression_scale', float),
        (r'lip.*intensity.*?(\d+\.?\d*)', 'lip_intensity', float),
        (r'expression.*(\d+\.?\d*)', 'expression_scale', float),
        (r'lip.*(\d+\.?\d*)', 'lip_intensity', float),
    ]
    
    for pattern, key, converter in numeric_patterns:
        match = re.search(pattern, prompt_lower)
        if match:
            try:
                value = converter(match.group(1))
                instructions[key] = value
            except (ValueError, IndexError):
                pass
    
    # Default values if nothing found but prompt exists
    if prompt.strip() and not instructions:
        # If prompt is provided but no patterns match, use moderate defaults
        instructions = {
            'expression_scale': 1.0,
            'lip_intensity': 1.0
        }
    
    return instructions


def validate_instructions(instructions: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize edit instructions.
    
    Args:
        instructions: Raw instructions dictionary
        
    Returns:
        Validated and normalized instructions
    """
    validated = {}
    
    # Validate expression_scale (should be between 0.1 and 3.0)
    if 'expression_scale' in instructions:
        scale = float(instructions['expression_scale'])
        validated['expression_scale'] = max(0.1, min(3.0, scale))
    
    # Validate lip_intensity (should be between 0.1 and 3.0)
    if 'lip_intensity' in instructions:
        intensity = float(instructions['lip_intensity'])
        validated['lip_intensity'] = max(0.1, min(3.0, intensity))
    
    # Validate head_pose (should be a string or list)
    if 'head_pose' in instructions:
        pose = instructions['head_pose']
        if isinstance(pose, (str, list)):
            validated['head_pose'] = pose
    
    return validated


def parse_and_validate(prompt: str) -> Dict[str, Any]:
    """
    Parse prompt and validate the resulting instructions.
    
    Args:
        prompt: Natural language prompt string
        
    Returns:
        Validated edit instructions dictionary
    """
    instructions = parse_prompt(prompt)
    return validate_instructions(instructions)

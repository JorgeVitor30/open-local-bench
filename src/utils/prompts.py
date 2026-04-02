"""
Utilities for loading and managing prompts from .md files.
"""
import os
from pathlib import Path
from typing import Optional, Dict, List


def get_prompt_path(prompt_name: str, category: Optional[str] = None) -> Path:
    """
    Get the full path to a prompt file.
    
    Args:
        prompt_name: Name of the prompt file (with or without .md extension)
        category: Optional category subdirectory (e.g., 'reasoning', 'code', 'math')
    
    Returns:
        Path to the prompt file
    """
    if not prompt_name.endswith('.md'):
        prompt_name += '.md'
    
    base_dir = Path(__file__).parent.parent / "prompts"
    
    if category:
        return base_dir / category / prompt_name
    else:
        return base_dir / prompt_name


def load_prompt(prompt_name: str, category: Optional[str] = None) -> str:
    """
    Load a prompt from a .md file.
    
    Args:
        prompt_name: Name of the prompt file (with or without .md extension)
        category: Optional category subdirectory (e.g., 'reasoning', 'code', 'math')
    
    Returns:
        The prompt content as a string
    
    Raises:
        FileNotFoundError: If the prompt file doesn't exist
    """
    prompt_path = get_prompt_path(prompt_name, category)
    
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Prompt not found: {prompt_path}\n"
            f"Make sure the file exists in src/prompts/{category or ''}"
        )
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


def list_prompts(category: Optional[str] = None) -> List[str]:
    """
    List all available prompt files.
    
    Args:
        category: Optional category subdirectory to filter by
    
    Returns:
        List of prompt file paths (relative to prompts directory)
    """
    base_dir = Path(__file__).parent.parent / "prompts"
    
    if category:
        search_dir = base_dir / category
        if not search_dir.exists():
            return []
        pattern = "*.md"
        prompts = list(search_dir.glob(pattern))
        return [str(p.relative_to(base_dir)) for p in prompts]
    else:
        prompts = list(base_dir.rglob("*.md"))
        return [str(p.relative_to(base_dir)) for p in sorted(prompts)]


def load_all_prompts(category: Optional[str] = None) -> Dict[str, str]:
    """
    Load all prompts from a category or all prompts.
    
    Args:
        category: Optional category subdirectory. If None, loads all prompts.
    
    Returns:
        Dictionary mapping prompt names to their content
    """
    prompts = {}
    prompt_files = list_prompts(category)
    
    for prompt_file in prompt_files:
        name = Path(prompt_file).stem
        try:
            prompts[name] = load_prompt(prompt_file)
        except FileNotFoundError:
            continue
    
    return prompts


def get_prompt_info(prompt_name: str, category: Optional[str] = None) -> Dict:
    """
    Get information about a prompt without loading its content.
    
    Args:
        prompt_name: Name of the prompt file
        category: Optional category subdirectory
    
    Returns:
        Dictionary with prompt metadata
    """
    prompt_path = get_prompt_path(prompt_name, category)
    
    info = {
        "name": Path(prompt_name).stem,
        "path": str(prompt_path),
        "exists": prompt_path.exists(),
        "category": category,
    }
    
    if prompt_path.exists():
        stat = prompt_path.stat()
        info["size"] = stat.st_size
        info["modified"] = stat.st_mtime
    
    return info

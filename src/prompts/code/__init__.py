from prompts.prompt_abstract import PromptAbstract


class CodeGenerationPrompt(PromptAbstract):
    """Generate code from description."""
    
    @property
    def name(self) -> str:
        return "code_generation"
    
    @property
    def prompt(self) -> str:
        return "Write code to solve the following problem. Use Python."
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "Generate code from natural language description"
    
    @property
    def tags(self) -> list[str]:
        return ["code", "generation", "python"]


class CodeExplanationPrompt(PromptAbstract):
    """Explain code."""
    
    @property
    def name(self) -> str:
        return "code_explanation"
    
    @property
    def prompt(self) -> str:
        return "Explain what this code does in simple terms."
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "Explain code to non-programmers"
    
    @property
    def tags(self) -> list[str]:
        return ["code", "explanation", "comprehension"]

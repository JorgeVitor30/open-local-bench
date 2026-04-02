from prompts.prompt_abstract import PromptAbstract


class SimpleMathPrompt(PromptAbstract):
    """Simple math questions."""
    
    @property
    def name(self) -> str:
        return "simple_math"
    
    @property
    def prompt(self) -> str:
        return "Solve the following math problem."
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "Basic arithmetic problems"
    
    @property
    def tags(self) -> list[str]:
        return ["math", "arithmetic", "basic"]


class WordProblemPrompt(PromptAbstract):
    """Word problems that need reasoning."""
    
    @property
    def name(self) -> str:
        return "word_problem"
    
    @property
    def prompt(self) -> str:
        return "Solve this word problem step by step."
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "Word problems with real world scenarios"
    
    @property
    def tags(self) -> list[str]:
        return ["math", "reasoning", "word-problem"]

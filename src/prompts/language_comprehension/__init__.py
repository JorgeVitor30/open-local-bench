from prompts.prompt_abstract import PromptAbstract


class ConnectionsPrompt(PromptAbstract):
    """NYT Connections game prompt."""
    
    @property
    def name(self) -> str:
        return "connections_game"
    
    @property
    def prompt(self) -> str:
        return """You are playing the Connections game (like the NYT game).

Given 16 words, find 4 groups of 4 words that belong together.
Each group has a common theme.

Rules:
- Find exactly 4 groups
- Each word belongs to only one group
- Groups get harder (group 1 = easiest, group 4 = hardest)

Explain your reasoning for each group."""
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "Word grouping puzzle game"
    
    @property
    def tags(self) -> list[str]:
        return ["language", "puzzle", "grouping"]


class SummarizationPrompt(PromptAbstract):
    """Text summarization."""
    
    @property
    def name(self) -> str:
        return "summarize_text"
    
    @property
    def prompt(self) -> str:
        return "Summarize the following text in 2-3 sentences."
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "Summarize long text to key points"
    
    @property
    def tags(self) -> list[str]:
        return ["language", "summarization", "comprehension"]

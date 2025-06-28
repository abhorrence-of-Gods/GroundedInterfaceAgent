from abc import ABC, abstractmethod

class BaseSkill(ABC):
    """
    Abstract base class for all skills.
    A skill represents a low-level capability of the agent, such as
    interacting with the mouse or keyboard.
    """
    
    @abstractmethod
    def execute(self, *args, **kwargs):
        """
        Executes the skill.
        """
        pass 
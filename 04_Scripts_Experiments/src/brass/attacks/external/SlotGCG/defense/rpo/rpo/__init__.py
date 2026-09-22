__version__ = '0.0.1'

from .gcg import GCGAttackPrompt as AttackPrompt
from .gcg import GCGMultiPromptAttack as MultiPromptAttack
from .gcg import GCGPromptManager as PromptManager
from .suffix_manager import (
    AttackPrompt,
    MultiPromptAttack,
    ProgressiveMultiPromptAttack,
    PromptManager,
    get_embedding_layer,
    get_embedding_matrix,
    get_embeddings,
    get_goals_and_targets,
    get_nonascii_toks,
    get_workers,
)

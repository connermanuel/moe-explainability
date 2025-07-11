"""Dataset configurations for MoE routing analysis."""

from dataclasses import dataclass
from typing import List, Optional, Dict, Callable, Any
import pandas as pd
from datasets import load_dataset


@dataclass
class DatasetConfig:
    """Configuration for dataset processing.
    
    Defines the schema and processing parameters for a specific dataset,
    including field mappings for different data granularities.
    """
    
    name: str
    text_field: str
    word_fields: List[str]
    words_field: str
    sentence_fields: List[str]
    loader_fn: Callable[..., List[Dict[str, Any]]]
    
    # Optional extensions for future use
    alignment_strategy: str = "character_based"
    nested_structure: Optional[Dict[str, str]] = None
    
    def __post_init__(self):
        """Validate configuration."""
        if not self.text_field:
            raise ValueError("text_field cannot be empty")
        if not self.words_field:
            raise ValueError("words_field cannot be empty")
        if self.words_field not in self.word_fields:
            raise ValueError(f"words_field '{self.words_field}' must be in word_fields")
        if not self.word_fields:
            raise ValueError("word_fields cannot be empty")
        if not self.loader_fn:
            raise ValueError("loader_fn cannot be empty")
    
    def load_data(self, **kwargs) -> List[Dict[str, Any]]:
        """Load data using the configured loader function."""
        return self.loader_fn(**kwargs)


def load_ud_data(dataset_name: str = "universal_dependencies", 
                 dataset_config: str = "en_ewt", 
                 split: str = "train[:100]") -> List[Dict[str, Any]]:
    """Load Universal Dependencies data."""
    dataset = load_dataset(dataset_name, dataset_config, split=split)
    return [dict(item) for item in dataset]


def load_wordsim_data(csv_path: str = "data/wordsim353crowd.csv") -> List[Dict[str, Any]]:
    """Load WordSim data and return unique words as individual texts."""
    df = pd.read_csv(csv_path)
    
    # Get unique words from both columns
    words = set()
    words.update(df["Word 1"].str.lower().unique())
    words.update(df["Word 2"].str.lower().unique())
    
    # Create individual "texts" for each word
    data = []
    for word in sorted(words):
        data.append({
            "text": word,
            "tokens": [word],  # Single token for word-level analysis
            "word_id": word,
        })
    
    return data


# Predefined dataset configurations
UD_ENGLISH_EWT = DatasetConfig(
    name="ud_english_ewt",
    text_field="text",
    word_fields=["tokens", "upos", "xpos", "lemmas", "head", "deprel", "feats"],
    words_field="tokens",
    sentence_fields=[],
    loader_fn=load_ud_data,
)

WORDSIM_353 = DatasetConfig(
    name="wordsim_353",
    text_field="text",
    word_fields=["tokens", "word_id"],
    words_field="tokens",
    sentence_fields=[],
    loader_fn=load_wordsim_data,
)
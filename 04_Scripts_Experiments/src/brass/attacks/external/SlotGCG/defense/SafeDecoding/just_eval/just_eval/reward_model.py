
import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, PreTrainedModel


class LlamaRewardModel(PreTrainedModel):
    config_class = LlamaConfig
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.regression_head = nn.Linear(self.config.hidden_size, 1, bias=False)
        self._no_split_modules = []

    def forward( # args are the same as LlamaForCausalLM
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ):

        transformer_outputs = self.model(
                                input_ids,
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                past_key_values=past_key_values,
                                inputs_embeds=inputs_embeds,
                            )

        hidden_states = transformer_outputs[0]
        rewards = self.regression_head(hidden_states).squeeze(-1)

        ends = attention_mask.cumsum(dim=1).argmax(dim=1).view(-1,1)
        rewards = torch.gather(rewards, 1, ends)

        return rewards

if __name__ == "__main__":
    ultrarm_template = """Human: {instruction}\n Assistant: {completion}"""

    dataset = [ # cases in webgpt
    {"chosen":"Human: \"Who orders martinis \"\"shaken, not stirred\"\"?\"\nAssistant: Sean Connery's character, fictional British Secret Service agent James Bond, in the movie Goldfinger, stated that he preferred his martini to be \"shaken, not stirred\". [1] Some believe that Bond ordered his martini shaken because of the vodka it contained, as vodka was, for the most part, refined from potatoes (cheaper brands) which made the vodka oily. To disperse the oil, Bond ordered his martinis shaken. [2]","rejected":"Human: \"Who orders martinis \"\"shaken, not stirred\"\"?\"\nAssistant: Fleming's fictional British Secret Service agent James Bond orders his martini cocktail shaken, not stirred [1]. Bond's preferences for his martini are carried over to the films, where his orders are seen in both the 1961 film Dr. No and the 2006 film Casino Royale [1, 2]. In both films, Bond's subordinates copy his order, telling the bartender to keep the fruit with their drinks [2]. However, in the 2006 film, Bond appears irritated when the bartender asks if he would like his drink shaken or stirred [2]."},
    {"chosen":"Human: Sural relates to which part of the body?\nAssistant: The sural region is the muscular swelling of the back of the leg below the knee, formed chiefly by the bellies of the gastrocnemius and soleus muscles [1,2].","rejected":"Human: Sural relates to which part of the body?\nAssistant: The Sural nerve runs down the side of the leg near the small saphenous vein, then passes forward below the lateral malleolus and continues on the outside of the foot as the lateral dorsal cutaneous nerve, which then communicates with the intermediate dorsal cutaneous nerve, which branches off to the side of the foot. [1]"}
    ]


    tokenizer = LlamaTokenizer.from_pretrained("openbmb/UltraRM-13b")
    model = LlamaRewardModel.from_pretrained("openbmb/UltraRM-13b")

    for example in dataset:
        inputs = tokenizer(example["chosen"], return_tensors="pt")
        chosen_reward = model(**inputs).item()
        inputs = tokenizer(example["rejected"], return_tensors="pt")
        rejected_reward = model(**inputs).item()
        print(chosen_reward - rejected_reward)

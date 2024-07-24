import json
import os
from typing import Dict, List, Optional

from qwen_agent.llm import BaseChatModel, get_chat_model

from mem0.configs.llms.base import BaseLlmConfig
from mem0.llms.base import LLMBase


class OpenAILikeLLM(LLMBase):
    def __init__(self, config: Optional[BaseLlmConfig] = None):
        super().__init__(config)

        if not self.config.model:
            self.config.model = "gpt-4o"
        cfg = {
            "model": self.config.model,
            "model_server": os.getenv("OPENAI_BASE_URL"),
            "api_key": os.getenv("OPENAI_API_KEY"),
            "generate_cfg": {
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "top_p": self.config.top_p,
            },
        }
        self.client: BaseChatModel = get_chat_model(cfg)

    def _parse_response(self, response, tools):
        """
        Process the response based on whether tools are used or not.

        Args:
            response: The raw response from API.
            tools: The list of tools provided in the request.

        Returns:
            str or dict: The processed response.
        """
        if tools:
            processed_response = {
                "content": response[0]['content'],
                "tool_calls": [],
            }
            for msg in response:
                if msg.get("function_call"):
                    processed_response["tool_calls"].append(
                        {
                            "name": msg["function_call"]["name"],
                            "arguments": json.loads(msg["function_call"]["arguments"]),
                        }
                    )

            return processed_response
        else:
            return response[-1]["content"]

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format=None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
    ):
        """
        Generate a response based on the given messages using OpenAI.

        Args:
            messages (list): List of message dicts containing 'role' and 'content'.
            response_format (str or object, optional): Format of the response. Defaults to "text".
            tools (list, optional): List of tools that the model can call. Defaults to None.
            tool_choice (str, optional): Tool choice method. Defaults to "auto".

        Returns:
            str: The generated response.
        """
        params = {
            "messages": messages,
            "stream": False,
            "delta_stream": False
        }
        if response_format:
            params["response_format"] = response_format
        
        if tools:
            params["functions"] = [i["function"] for i in tools]

        response = self.client.chat(**params)
        return self._parse_response(response, tools)

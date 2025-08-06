#!/usr/bin/env python3
"""
OpenAI Harmony prompt template format implementation for gpt-oss models.

This module implements the official Harmony response format that gpt-oss models
were trained on and require for proper functionality.
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json


class Role(str, Enum):
    """Harmony role hierarchy (system > developer > user > assistant > tool)."""
    SYSTEM = "system"
    DEVELOPER = "developer"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Channel(str, Enum):
    """Harmony channels for different types of content."""
    FINAL = "final"          # User-facing responses
    ANALYSIS = "analysis"    # Internal chain-of-thought (not for user display)
    COMMENTARY = "commentary"  # Function calls, tool interactions


# Special token mappings for harmony format
HARMONY_TOKENS = {
    "start": "<|start|>",      # ID: 200006 - Message beginning
    "end": "<|end|>",          # ID: 200007 - Message end
    "message": "<|message|>",  # ID: 200008 - Content transition
    "channel": "<|channel|>",  # ID: 200005 - Channel information
    "return": "<|return|>",    # ID: 200002 - Completion stop
    "call": "<|call|>"         # ID: 200012 - Tool call stop
}


@dataclass
class HarmonyMessage:
    """Represents a single message in the harmony format."""
    role: Role
    content: str = ""
    channel: Optional[Channel] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_harmony_format(self) -> str:
        """Convert message to harmony format string."""
        # Build header
        header_parts = [f"role:{self.role.value}"]
        
        if self.channel:
            header_parts.append(f"channel:{self.channel.value}")
        
        # Add metadata
        for key, value in self.metadata.items():
            if isinstance(value, (str, int, float)):
                header_parts.append(f"{key}:{value}")
            else:
                header_parts.append(f"{key}:{json.dumps(value)}")
        
        header = " ".join(header_parts)
        
        # Format message
        return f"{HARMONY_TOKENS['start']}{header}{HARMONY_TOKENS['message']}{self.content}{HARMONY_TOKENS['end']}"


@dataclass
class HarmonyConversation:
    """Represents a complete conversation in harmony format."""
    messages: List[HarmonyMessage] = field(default_factory=list)
    
    def add_message(
        self, 
        role: Role, 
        content: str, 
        channel: Optional[Channel] = None,
        **metadata
    ) -> None:
        """Add a message to the conversation."""
        message = HarmonyMessage(
            role=role,
            content=content,
            channel=channel,
            metadata=metadata
        )
        self.messages.append(message)
    
    def add_system_message(self, content: str, **metadata) -> None:
        """Add a system message."""
        self.add_message(Role.SYSTEM, content, **metadata)
    
    def add_developer_message(self, content: str, **metadata) -> None:
        """Add a developer message."""
        self.add_message(Role.DEVELOPER, content, **metadata)
    
    def add_user_message(self, content: str, **metadata) -> None:
        """Add a user message."""
        self.add_message(Role.USER, content, **metadata)
    
    def add_assistant_message(
        self, 
        content: str, 
        channel: Channel = Channel.FINAL,
        **metadata
    ) -> None:
        """Add an assistant message with specified channel."""
        self.add_message(Role.ASSISTANT, content, channel, **metadata)
    
    def add_assistant_analysis(self, content: str, **metadata) -> None:
        """Add assistant analysis (chain-of-thought)."""
        self.add_assistant_message(content, Channel.ANALYSIS, **metadata)
    
    def add_assistant_final(self, content: str, **metadata) -> None:
        """Add assistant final response."""
        self.add_assistant_message(content, Channel.FINAL, **metadata)
    
    def add_assistant_commentary(self, content: str, **metadata) -> None:
        """Add assistant commentary (tool calls, etc.)."""
        self.add_assistant_message(content, Channel.COMMENTARY, **metadata)
    
    def add_tool_message(self, content: str, tool_name: str, **metadata) -> None:
        """Add a tool response message."""
        self.add_message(Role.TOOL, content, metadata={"tool": tool_name, **metadata})
    
    def to_harmony_format(self) -> str:
        """Convert entire conversation to harmony format."""
        return "".join(message.to_harmony_format() for message in self.messages)
    
    def to_tokens(self, tokenizer) -> List[int]:
        """Convert conversation to tokens using provided tokenizer."""
        harmony_text = self.to_harmony_format()
        return tokenizer.encode(harmony_text)


class HarmonyTemplateRenderer:
    """Main renderer for harmony format conversations."""
    
    def __init__(self):
        """Initialize harmony renderer."""
        pass
    
    def create_conversation(self) -> HarmonyConversation:
        """Create a new harmony conversation."""
        return HarmonyConversation()
    
    def render_simple_prompt(self, prompt: str, system_message: Optional[str] = None) -> str:
        """Render a simple prompt in harmony format."""
        conversation = self.create_conversation()
        
        if system_message:
            conversation.add_system_message(system_message)
        
        conversation.add_user_message(prompt)
        
        return conversation.to_harmony_format()
    
    def render_chat_format(self, messages: List[Dict[str, str]]) -> str:
        """Render OpenAI-style chat messages in harmony format."""
        conversation = self.create_conversation()
        
        for msg in messages:
            role_str = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role_str == "system":
                conversation.add_system_message(content)
            elif role_str == "user":
                conversation.add_user_message(content)
            elif role_str == "assistant":
                conversation.add_assistant_final(content)
            elif role_str == "developer":
                conversation.add_developer_message(content)
            elif role_str == "tool":
                tool_name = msg.get("name", "unknown_tool")
                conversation.add_tool_message(content, tool_name)
        
        return conversation.to_harmony_format()
    
    def render_with_chain_of_thought(
        self, 
        prompt: str, 
        analysis: str, 
        final_response: str,
        system_message: Optional[str] = None
    ) -> str:
        """Render prompt with explicit chain-of-thought reasoning."""
        conversation = self.create_conversation()
        
        if system_message:
            conversation.add_system_message(system_message)
        
        conversation.add_user_message(prompt)
        conversation.add_assistant_analysis(analysis)
        conversation.add_assistant_final(final_response)
        
        return conversation.to_harmony_format()
    
    def render_tool_calling(
        self,
        prompt: str,
        tool_calls: List[Dict[str, Any]],
        tool_responses: List[Dict[str, Any]],
        final_response: str,
        system_message: Optional[str] = None
    ) -> str:
        """Render conversation with tool calling."""
        conversation = self.create_conversation()
        
        if system_message:
            conversation.add_system_message(system_message)
        
        conversation.add_user_message(prompt)
        
        # Add tool calls as commentary
        for tool_call in tool_calls:
            tool_content = json.dumps(tool_call)
            conversation.add_assistant_commentary(tool_content)
        
        # Add tool responses
        for tool_response in tool_responses:
            tool_name = tool_response.get("name", "unknown_tool")
            content = tool_response.get("content", "")
            conversation.add_tool_message(content, tool_name)
        
        # Add final response
        conversation.add_assistant_final(final_response)
        
        return conversation.to_harmony_format()


def create_harmony_template(
    prompt: str,
    system_message: Optional[str] = None,
    chain_of_thought: bool = False,
    analysis: Optional[str] = None
) -> str:
    """
    Convenience function to create harmony template.
    
    Args:
        prompt: User prompt
        system_message: Optional system message
        chain_of_thought: Whether to enable chain-of-thought
        analysis: Chain-of-thought analysis content
    
    Returns:
        Harmony formatted string ready for model input
    """
    renderer = HarmonyTemplateRenderer()
    
    if chain_of_thought and analysis:
        return renderer.render_with_chain_of_thought(
            prompt=prompt,
            analysis=analysis,
            final_response="",  # Model will generate this
            system_message=system_message
        )
    else:
        return renderer.render_simple_prompt(prompt, system_message)


# Example usage and templates
HARMONY_SYSTEM_TEMPLATES = {
    "default": "You are a helpful, harmless, and honest assistant.",
    "coding": "You are an expert programmer who writes clean, efficient, and well-documented code.",
    "reasoning": "You think step by step and show your reasoning process clearly.",
    "creative": "You are creative and imaginative while staying grounded in reality.",
    "analytical": "You analyze problems systematically and provide detailed explanations."
}


def demo_harmony_format():
    """Demonstrate harmony format usage."""
    print("🎵 OpenAI Harmony Format Demo")
    print("=" * 50)
    
    renderer = HarmonyTemplateRenderer()
    
    # 1. Simple prompt
    print("\n1. Simple Prompt:")
    simple = renderer.render_simple_prompt(
        "What is machine learning?",
        system_message=HARMONY_SYSTEM_TEMPLATES["default"]
    )
    print(simple)
    
    # 2. Chat format
    print("\n2. Chat Format:")
    chat_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing"},
        {"role": "assistant", "content": "Quantum computing leverages quantum mechanics..."}
    ]
    chat_format = renderer.render_chat_format(chat_messages)
    print(chat_format)
    
    # 3. Chain of thought
    print("\n3. Chain of Thought:")
    cot = renderer.render_with_chain_of_thought(
        prompt="Solve: 2x + 5 = 17",
        analysis="I need to solve for x. First, I'll subtract 5 from both sides: 2x = 12. Then divide by 2: x = 6.",
        final_response="The solution is x = 6.",
        system_message=HARMONY_SYSTEM_TEMPLATES["reasoning"]
    )
    print(cot)
    
    print("\n✅ Harmony format demonstration complete!")


if __name__ == "__main__":
    demo_harmony_format()
"""
Dream model wrapper to ensure hidden_states are extracted and returned properly.
"""
from typing import Optional, Dict, Any
import torch
from torch import nn
from transformers import PreTrainedModel, AutoModel


class DreamModelLM(PreTrainedModel):
    """
    Wrapper for Dream diffusion models that ensures hidden_states are properly
    extracted and returned in the model output.
    """
    
    def __init__(self, model=None, config=None, **kwargs):
        if config is None:
            raise ValueError("config must be provided")
        super().__init__(config)
        
        if model is None:
            # If no model provided, we'll load it later
            self.model = None
        else:
            self.model = model
    
    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs):
        """
        Load a Dream model and wrap it to ensure hidden_states extraction.
        """
        # Ensure trust_remote_code is set
        if 'trust_remote_code' not in kwargs:
            kwargs['trust_remote_code'] = True
        
        # Load the base model using AutoModel (for diffusion models with custom code)
        load_kwargs = {k: v for k, v in kwargs.items() if k != 'torch_dtype'}
        model = AutoModel.from_pretrained(model_id, **load_kwargs)
        
        # Apply dtype if provided
        if 'torch_dtype' in kwargs:
            model = model.to(kwargs['torch_dtype'])
        
        # Get config
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        
        # Create wrapper
        wrapper = cls(model=model, config=config)
        return wrapper
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **kwargs
    ):
        """
        Forward pass that ensures hidden_states are extracted and returned.
        """
        # Store original hook to extract hidden states if needed
        hidden_states_list = []
        
        def get_hidden_states_hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states_list.append(output[0] if len(output) > 0 else output)
            else:
                hidden_states_list.append(output)
        
        # Register hook to capture hidden states if model doesn't return them
        hook_handle = None
        if output_hidden_states and hasattr(self.model, 'model'):
            if hasattr(self.model.model, 'layers'):
                # Register hook on last layer
                hook_handle = self.model.model.layers[-1].register_forward_hook(get_hidden_states_hook)
        
        try:
            # Forward through the underlying Dream model
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=output_hidden_states,
                return_dict=True,
                **kwargs
            )
            
            # If hidden_states not in output but requested, use captured hidden states
            if output_hidden_states:
                if not hasattr(outputs, 'hidden_states') or outputs.hidden_states is None:
                    if hidden_states_list:
                        # Create a simple object with hidden_states attribute
                        outputs.hidden_states = hidden_states_list[-1]
                    else:
                        # Fallback: use layer norm output if available
                        if hasattr(self.model, 'model') and hasattr(self.model.model, 'norm'):
                            # Get the output from model
                            outputs.hidden_states = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs.get('last_hidden_state', None)
            
            return outputs
        finally:
            if hook_handle is not None:
                hook_handle.remove()
    
    def to(self, device):
        """Move model to device."""
        if self.model is not None:
            self.model = self.model.to(device)
        return self
    
    def eval(self):
        """Set model to eval mode."""
        if self.model is not None:
            self.model = self.model.eval()
        return self
    
    @property
    def device(self):
        """Get device of the model."""
        if self.model is not None:
            return next(self.model.parameters()).device
        return None
